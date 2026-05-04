import torch
import os
from flash_attn.ops.rms_norm import RMSNorm
from megatron.core import mpu
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron.core.transformer.enums import AttnMaskType, AttnType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from torch import nn

from galvatron.core import get_args
from galvatron.core.runtime.tensor_parallel import ParallelMLP
from galvatron.core.runtime.tensor_parallel.mlp import MLPSubmodules
from galvatron.core.runtime.tensor_parallel.attention import SelfAttention, SelfAttentionSubmodules
from galvatron.core.runtime.tensor_parallel.attention_impl import DotProductAttention, FlashSelfOrCrossAttention, DistributedAttention
from galvatron.core.runtime.moe.mlp import GroupedMLP, SequentialMLP
from galvatron.core.runtime.moe.router import TopKRouter
from galvatron.core.runtime.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from galvatron.core.runtime.moe.smart_routing import MoEAlltoAllSmartTokenDispatcher
import moe_all_to_all_kernels

class MoEAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        super().__init__()
        args = get_args()
        self.idx = layer_number
        self.sequence_parallel = args.sequence_parallel
        self.use_ulysses = sp_group.size > 1
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        if args.qk_layernorm:
            q_layernorm = RMSNorm
            k_layernorm = RMSNorm
        else:
            q_layernorm = None
            k_layernorm = None
        self.attention = SelfAttention(
            megatron_config,
            SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                flash_attention=FlashSelfOrCrossAttention,
                dist_attention=DistributedAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=q_layernorm,
                k_layernorm=k_layernorm,
            ),
            layer_number,
            attn_mask_type=AttnMaskType.causal,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
        )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.LayerNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_pos_emb = RotaryEmbedding(
            self.head_dim, args.rotary_percent, 
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            rotary_base=args.rotary_base
        )

        self.MLPLayerNorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, rotary_embedding):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        if self.sequence_parallel:
            if self.use_ulysses:
                rotary_pos_emb = self.rotary_pos_emb(
                    hidden_states.shape[0], offset=hidden_states.shape[0] * torch.distributed.get_rank(self.sp_group)
                )
            else:
                if rotary_embedding is not None:
                    rotary_pos_emb = rotary_embedding
                else:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] * torch.distributed.get_world_size(self.tp_group)
                    )
        else:
            if rotary_embedding is not None:
                rotary_pos_emb = rotary_embedding
            else:
                rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0])
        hidden_states, bias = self.attention(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + input_tensor

        mlp_residual = hidden_states
        hidden_states = self.MLPLayerNorm(hidden_states)
        return hidden_states, mlp_residual

class MoERouter(nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        args = get_args()
        megatron_config = core_transformer_config_from_args(args)
        self.idx = layer_number
        self.router = TopKRouter(config=megatron_config)
        self.router.layer_number = layer_number
        # Under --static_input we want the dispatcher to see a bit-identical
        # routing decision iter-after-iter so the LAER token-count history
        # stabilises and the dispatch pattern is reproducible. Zeroing gate
        # gradients alone is not enough — the *input* to the router drifts
        # as upstream attention / expert weights are updated by
        # optimizer.step, and the top-k decision flips assignments for any
        # token whose 8th-vs-9th expert margin is smaller than the input
        # drift. We therefore:
        #   1. Always call ``self.router(hidden_states)`` so its forward
        #      compute is included in the wall-time / memory measurement.
        #   2. Cache (probs, routing_map) at iter 0 and substitute the
        #      cached tensors into the dispatcher input on subsequent
        #      iters. Cached tensors are detached, so the dispatcher's
        #      backward path doesn't reach the gate — gate gradients drop
        #      to whatever the auxiliary load-balance loss contributes
        #      (still computed, still measured).
        #   3. Log how many routing-map cells the freshly-computed router
        #      would have flipped relative to iter 0, gated by a small
        #      budget so we have a quantitative drift trace.
        self._static_input = bool(getattr(args, "static_input", False))
        self._static_routing_cache = None
        self._static_check_count = 0
        self._static_check_budget = 20

    def forward(self, hidden_states):
        if hasattr(self, "predict"):
            self.predict(hidden_states)
        # Always call the underlying router so its forward cost (gate Linear,
        # softmax, top-k, aux loss registration) is part of the measurement,
        # even when we substitute the cached output below.
        probs, routing_map = self.router(hidden_states)
        if self._static_input:
            if self._static_routing_cache is None:
                self._static_routing_cache = (probs.detach(), routing_map.detach())
            else:
                cached_probs, cached_routing_map = self._static_routing_cache
                if self._static_check_count < self._static_check_budget:
                    with torch.no_grad():
                        diff = int((routing_map != cached_routing_map).sum().item())
                        total = int(routing_map.numel())
                    if torch.distributed.get_rank() == 0:
                        tag = "MATCH" if diff == 0 else "MISMATCH"
                        print(
                            f"[router_drift] layer={self.idx} "
                            f"check={self._static_check_count}: "
                            f"{diff}/{total} routing_map cells differ ({tag})",
                            flush=True,
                        )
                    self._static_check_count += 1
                # Substitute cached so the dispatcher sees iter-0 routing.
                probs, routing_map = cached_probs, cached_routing_map
        return probs, routing_map

# TODO: Add shared expert support
class MoEMLP_tp(nn.Module):
    def __init__(self, config, token_dispatcher, layer_number, tp_group=None, ep_group=None, tp_of_ep_group=None, tp_and_ep_group=None, test_mode=False):
        super().__init__()
        self._test_mode = test_mode
        if test_mode:
            # Build the real per-rank MoE expert stack so that
            # --profile_unit mlp measures the same per-layer model state and
            # compute as the FSEP-off real path. The previous short-circuit
            # to a single dense ParallelMLP under-built each layer by the
            # full (num_local_experts - 1) experts of optimizer state and
            # collapsed the per-token expert FFN compute to a single GEMM,
            # which made the cost model's mlp slope wrong by ~50× for
            # 128-expert models. See profile_computation.sh / profile_memory.sh
            # for how this slope feeds the cost model's per-component split.
            args = get_args()
            megatron_config = core_transformer_config_from_args(args)
            self.config = megatron_config
            self.tp_group = tp_group.group if tp_group is not None else None
            self.ep_group = ep_group.group if ep_group is not None else None
            self.tp_of_ep_group = tp_of_ep_group.group if tp_of_ep_group is not None else None
            self.tp_and_ep_group = tp_and_ep_group.group if tp_and_ep_group is not None else None
            self.expert_parallel_size = mpu.get_expert_model_parallel_world_size(self.ep_group)
            assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"
            self.num_global_experts = self.config.num_moe_experts
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            self.num_experts_per_tok = args.num_experts_per_tok
            self.experts = SequentialMLP(
                self.num_local_experts,
                self.config,
                MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
                self.tp_of_ep_group,
                self.tp_and_ep_group,
            )
            return
        args = get_args()
        self.recompute_communication = args.recompute_communication
        self.use_fsep = args.use_fsep
        self.is_moe_layer = self.use_fsep
        self.idx = layer_number
        megatron_config = core_transformer_config_from_args(args)
        self.config = megatron_config
        self.tp_group = tp_group.group if tp_group is not None else None
        self.ep_group = ep_group.group if ep_group is not None else None
        self.tp_of_ep_group = tp_of_ep_group.group if tp_of_ep_group is not None else None
        self.tp_and_ep_group = tp_and_ep_group.group if tp_and_ep_group is not None else None

        self.expert_parallel_size = mpu.get_expert_model_parallel_world_size(self.ep_group)
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        # assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_global_experts = self.config.num_moe_experts
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        self.expert_capacity_per_device = args.expert_capacity_per_device
        expert_capacity_offset = (
            mpu.get_expert_model_parallel_rank(self.ep_group) * self.expert_capacity_per_device
        )
        self.expert_capacity_indices = [
            expert_capacity_offset + i for i in range(self.expert_capacity_per_device)
        ]
        local_expert_indices_offset = (
            mpu.get_expert_model_parallel_rank(self.ep_group) * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        # assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))

        if self.use_fsep:
            # TODO: Update solver to modify global_expert_indices
            self.global_expert_indices = torch.tensor(range(self.num_global_experts),dtype=torch.int32,device=torch.cuda.current_device())
            self.global_expert_indices = self.global_expert_indices.repeat(self.expert_parallel_size * self.expert_capacity_per_device // self.num_global_experts).reshape(self.expert_parallel_size, -1).contiguous()
            self.global_expert_locations = torch.full((self.num_global_experts, self.expert_parallel_size * self.expert_capacity_per_device // self.num_global_experts), -1, dtype=torch.int32, device=torch.cuda.current_device())
            for i in range(self.num_global_experts):
                self.global_expert_locations[i, :(self.expert_parallel_size * self.expert_capacity_per_device // self.num_global_experts)] = torch.arange(i, self.expert_parallel_size * self.expert_capacity_per_device, self.num_global_experts, dtype=torch.int32)
            self.inverse_expert_map = torch.tensor(range(self.num_global_experts),dtype=torch.int32,device=torch.cuda.current_device())
            self.inverse_expert_map = self.inverse_expert_map.reshape(-1,1).repeat(1, self.expert_parallel_size * self.expert_capacity_per_device // self.num_global_experts).contiguous()
            # self.token_dispatcher = MoEAlltoAllTokenDispatcher(
            #     self.num_local_experts, self.local_expert_indices, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group,
            #     layer_number = self.idx
            # )
            self.forward_event = torch.cuda.Event()
            self.backward_event = torch.cuda.Event()
            self.token_dispatcher = token_dispatcher
            self.token_dispatcher.forward_event = self.forward_event
            self.token_dispatcher.backward_event = self.backward_event
            self.token_dispatcher.global_expert_indices_numpy = self.global_expert_indices.cpu().numpy()
        else:
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            self.token_dispatcher = token_dispatcher
        
        if args.moe_grouped_gemm:
            assert self.use_fsep is False, "Grouped gemm does not support fsep."
            self.experts = GroupedMLP(self.num_local_experts, self.config, self.tp_of_ep_group)
        else:
            # TODO: TE Tensor Parallel Adaptation
            if self.use_fsep:
                self.experts = SequentialMLP(
                    self.num_global_experts,
                    self.config,
                    MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                    ),
                    self.tp_of_ep_group,
                    self.tp_and_ep_group,
                )
                self.real_experts = SequentialMLP(
                    self.expert_capacity_per_device,
                    self.config,
                    MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                    ),
                    self.tp_of_ep_group,
                    self.tp_and_ep_group,
                )
                self.real_experts.layer_number = layer_number
            else:
                self.experts = SequentialMLP(
                    self.num_local_experts,
                    self.config,
                    MLPSubmodules(
                        linear_fc1=ColumnParallelLinear,
                        linear_fc2=RowParallelLinear,
                    ),
                    self.tp_of_ep_group,
                    self.tp_and_ep_group,
                )

    def forward(self, hidden_states, tokens_per_expert, probs=None):
        if getattr(self, "_test_mode", False):
            # --profile_unit mlp synthetic dispatch:
            # mimic the real bsz×seq×top_k dispatched-token volume that a
            # balanced router with capacity_factor=1 produces, evenly split
            # across all num_local_experts local experts. Each token is
            # replicated num_experts_per_tok times along the token dim, the
            # replicas are split evenly across local experts, and the
            # expert outputs are then summed across the top_k axis to
            # recover one output per original token (mirrors what the real
            # token unpermutation + weighted-sum path produces, modulo the
            # routing weights — fine for compute / memory profiling).
            orig_shape = hidden_states.shape
            hidden = orig_shape[-1]
            flat = hidden_states.reshape(-1, hidden)
            if self.num_experts_per_tok > 1:
                flat = flat.repeat_interleave(self.num_experts_per_tok, dim=0)
            total_routed = flat.shape[0]
            base = total_routed // self.num_local_experts
            tpe = torch.full(
                (self.num_local_experts,), base,
                dtype=torch.long, device=flat.device,
            )
            tpe[-1] += total_routed - base * self.num_local_experts
            expert_output, mlp_bias = self.experts(flat, tpe)
            if self.num_experts_per_tok > 1:
                expert_output = expert_output.view(-1, self.num_experts_per_tok, hidden).sum(dim=1)
            expert_output = expert_output.view(*orig_shape)
            return expert_output, mlp_bias
        if tokens_per_expert is not None:
            if self.use_fsep:
                if self.recompute_communication:
                    attention_output, routing_map = hidden_states, tokens_per_expert
                    (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                        attention_output, probs, routing_map
                    )
                    expert_output, mlp_bias = self.real_experts(dispatched_input, tokens_per_expert)
                    mlp_output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
                    return mlp_output, mlp_bias
                else:
                    expert_output, mlp_bias = self.real_experts(hidden_states, tokens_per_expert)
            else:
                expert_output, mlp_bias = self.experts(hidden_states, tokens_per_expert)
        else:
            # Only for test
            expert_output, mlp_bias = self.mlp(hidden_states)
        return expert_output, mlp_bias


class MoELayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None, ep_group=None, tp_of_ep_group=None, tp_and_ep_group=None):
        super().__init__()
        self.attention = MoEAttention_tp(config, layer_number, tp_group, sp_group)
        self.router = MoERouter(layer_number)

        args = get_args()
        self.recompute_communication = args.recompute_communication
        self.use_fsep = args.use_fsep
        self.is_moe_layer = self.use_fsep
        self.idx = layer_number
        megatron_config = core_transformer_config_from_args(args)
        self.config = megatron_config
        self.tp_group = tp_group.group if tp_group is not None else None
        self.ep_group = ep_group.group if ep_group is not None else None
        self.tp_of_ep_group = tp_of_ep_group.group if tp_of_ep_group is not None else None
        self.tp_and_ep_group = tp_and_ep_group.group if tp_and_ep_group is not None else None

        self.expert_parallel_size = mpu.get_expert_model_parallel_world_size(self.ep_group)
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        # assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_global_experts = self.config.num_moe_experts
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        self.expert_capacity_per_device = args.expert_capacity_per_device
        expert_capacity_offset = (
            mpu.get_expert_model_parallel_rank(self.ep_group) * self.expert_capacity_per_device
        )
        self.expert_capacity_indices = [
            expert_capacity_offset + i for i in range(self.expert_capacity_per_device)
        ]
        local_expert_indices_offset = (
            mpu.get_expert_model_parallel_rank(self.ep_group) * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        if self.use_fsep:
            # TODO: Update solver to modify global_expert_indices
            self.global_expert_indices = torch.tensor(range(self.num_global_experts),dtype=torch.int32,device=torch.cuda.current_device())
            self.global_expert_indices = self.global_expert_indices.repeat(self.expert_parallel_size * self.expert_capacity_per_device // self.num_global_experts).reshape(self.expert_parallel_size, -1).contiguous()
            self.global_expert_locations = torch.full((self.num_global_experts, self.expert_parallel_size * self.expert_capacity_per_device // self.num_global_experts), -1, dtype=torch.int32, device=torch.cuda.current_device())
            for i in range(self.num_global_experts):
                self.global_expert_locations[i, :(self.expert_parallel_size * self.expert_capacity_per_device // self.num_global_experts)] = torch.arange(i, self.expert_parallel_size * self.expert_capacity_per_device, self.num_global_experts, dtype=torch.int32)
            self.inverse_expert_map = torch.tensor(range(self.num_global_experts),dtype=torch.int32,device=torch.cuda.current_device())
            self.inverse_expert_map = self.inverse_expert_map.reshape(-1,1).repeat(1, self.expert_parallel_size * self.expert_capacity_per_device // self.num_global_experts).contiguous()
            # self.token_dispatcher = MoEAlltoAllTokenDispatcher(
            #     self.num_local_experts, self.local_expert_indices, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group,
            #     layer_number = self.idx
            # )
            self.token_dispatcher = MoEAlltoAllSmartTokenDispatcher(
                self.expert_capacity_per_device, self.expert_capacity_indices, self.global_expert_indices, self.global_expert_locations, self.inverse_expert_map, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group,
                layer_number = self.idx
            )
        else:
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            if self.config.moe_token_dispatcher_type == "allgather":
                self.token_dispatcher = MoEAllGatherTokenDispatcher(
                    self.num_local_experts, self.local_expert_indices, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group
                )
            elif self.config.moe_token_dispatcher_type == "alltoall":
                self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                    self.num_local_experts, self.local_expert_indices, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group,
                    layer_number = self.idx
                )
            elif self.config.moe_token_dispatcher_type == "alltoall_seq":
                assert False, "alltoall_seq is deprecated"
            elif self.config.moe_token_dispatcher_type == "flex":
                self.token_dispatcher = MoEFlexTokenDispatcher(
                    self.num_local_experts, self.local_expert_indices, config=self.config, ep_group=self.ep_group, tp_of_ep_group=self.tp_of_ep_group, tp_and_ep_group=self.tp_and_ep_group
                )
        self.mlp = MoEMLP_tp(config, self.token_dispatcher, layer_number, tp_group, ep_group, tp_of_ep_group, tp_and_ep_group)

        # if os.getenv("ENABLE_HIERARCHICAL", "0") == "0":
        #     rank = torch.distributed.get_rank(ep_group.group)
        #     world_size = torch.distributed.get_world_size(ep_group.group)
        #     moe_all_to_all_kernels.init_nccl_comm(ep_group.group, rank, world_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rotary_embedding=None,
    ):
        attention_output, mlp_residual = self.attention(
            hidden_states,
            attention_mask,
            rotary_embedding,
        )
        probs, routing_map = self.router(attention_output)
        if self.use_fsep:
            routing_map, probs = self.token_dispatcher.get_smart_routing(
                routing_map, probs
            )
        if self.recompute_communication:
            mlp_output, mlp_bias = self.mlp(attention_output, routing_map, probs)
        else:
            dispatched_input, tokens_per_expert = (
                self.token_dispatcher.token_permutation(
                    attention_output, probs, routing_map
                )
            )
            expert_output, mlp_bias = self.mlp(dispatched_input, tokens_per_expert)
            mlp_output, mlp_bias = self.token_dispatcher.token_unpermutation(
                expert_output, mlp_bias
            )
        if self.use_fsep:
            self.token_dispatcher.sync_lp_solver()
        layer_output = mlp_output + mlp_residual
        return layer_output


class MoELayer_attention(nn.Module):
    def __init__(
        self,
        config,
        layer_number,
        tp_group=None,
        sp_group=None,
        ep_group=None,
        tp_of_ep_group=None,
        tp_and_ep_group=None,
    ):
        super().__init__()
        self.attention = MoEAttention_tp(config, layer_number, tp_group, sp_group)
        self.idx = layer_number

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rotary_embedding=None,
    ):
        attention_output, mlp_residual = self.attention(
            hidden_states,
            attention_mask,
            rotary_embedding,
        )
        return attention_output


class MoELayer_mlp(nn.Module):
    def __init__(
        self,
        config,
        layer_number,
        tp_group=None,
        sp_group=None,
        ep_group=None,
        tp_of_ep_group=None,
        tp_and_ep_group=None,
    ):
        super().__init__()
        self.mlp = MoEMLP_tp(
            config,
            token_dispatcher=None,
            layer_number=layer_number,
            tp_group=tp_group,
            ep_group=ep_group,
            tp_of_ep_group=tp_of_ep_group,
            tp_and_ep_group=tp_and_ep_group,
            test_mode=True,
        )
        self.idx = layer_number

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rotary_embedding=None,
    ):
        expert_output, _mlp_bias = self.mlp(hidden_states, None)
        return expert_output


def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc, ep_groups_enc, tp_of_ep_groups_enc, tp_and_ep_groups_enc):
    args = get_args()
    if args.profile_unit == "attention":
        layers_tp = nn.ModuleList(
            [
                MoELayer_attention(config, i, 
                    tp_group=tp_groups_enc[i + 1], 
                    sp_group=sp_groups_enc[i + 1], 
                    ep_group=ep_groups_enc[i + 1], 
                    tp_of_ep_group=tp_of_ep_groups_enc[i + 1], 
                    tp_and_ep_group=tp_and_ep_groups_enc[i + 1])
                for i in range(config.num_hidden_layers)
            ]
        )
    elif args.profile_unit == "mlp":
        layers_tp = nn.ModuleList(
            [
                MoELayer_mlp(config, i, 
                    tp_group=tp_groups_enc[i + 1], 
                    sp_group=sp_groups_enc[i + 1], 
                    ep_group=ep_groups_enc[i + 1], 
                    tp_of_ep_group=tp_of_ep_groups_enc[i + 1], 
                    tp_and_ep_group=tp_and_ep_groups_enc[i + 1])
                for i in range(config.num_hidden_layers)
            ]
        )
    else: # "all"
        layers_tp = nn.ModuleList(
            [
                MoELayer_tp(config, i, 
                    tp_group=tp_groups_enc[i + 1], 
                    sp_group=sp_groups_enc[i + 1], 
                    ep_group=ep_groups_enc[i + 1], 
                    tp_of_ep_group=tp_of_ep_groups_enc[i + 1], 
                    tp_and_ep_group=tp_and_ep_groups_enc[i + 1])
                for i in range(config.num_hidden_layers)
            ]
        )
        for i in range(len(layers_tp)-1):
            layers_tp[i].router.predict = layers_tp[i+1].router.router.predict
            layers_tp[i].mlp.token_dispatcher.nxt_dispatcher = layers_tp[i+1].mlp.token_dispatcher
        for i in range(1,len(layers_tp)):
            layers_tp[i].mlp.token_dispatcher.pre_dispatcher = layers_tp[i-1].mlp.token_dispatcher
        # layers_tp[len(layers_tp)-1].mlp.token_dispatcher.nxt_dispatcher = layers_tp[0].mlp.token_dispatcher

    setattr(model.model, "layers", layers_tp)
    
    megatron_config = core_transformer_config_from_args(get_args())
    setattr(
        model.model,
        "embed_tokens",
        VocabParallelEmbedding(
            args.padded_vocab_size,
            megatron_config.hidden_size,
            config=megatron_config,
            init_method=megatron_config.init_method,
            reduce_scatter_embeddings=args.sequence_parallel,
            tp_group=tp_groups_enc[0].group,
            sp_group=sp_groups_enc[0].group,
        ),
    )
    setattr(
        model,
        "lm_head",
        ColumnParallelLinear(
            megatron_config.hidden_size,
            args.padded_vocab_size,
            config=megatron_config,
            init_method=megatron_config.init_method,
            bias=False,
            tp_group=tp_groups_enc[-1].group,
            sp_group=sp_groups_enc[-1].group,
        ),
    )

    loss_scale = torch.ones(1, device=torch.cuda.current_device())
    MoEAuxLossAutoScaler.set_loss_scale(loss_scale / args.chunks)

    return model
