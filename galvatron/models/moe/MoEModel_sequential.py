import torch
import torch.nn as nn

# from transformers.models.llama.modeling_llama import LlamaRMSNorm
# from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm
from flash_attn.ops.rms_norm import RMSNorm
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from galvatron.core import get_args
from galvatron.core.runtime import ModelInfo, mixed_precision_dtype
from galvatron.core.runtime.pipeline import PipeSequential
from galvatron.core.runtime.tensor_parallel import colummn_row_reset_parameters


def get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""
    micro_batch_size, seq_length = data.size()
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # position_ids = torch.arange(seq_length, dtype=torch.long,
    #                             device=data.device)
    # position_ids = position_ids.unsqueeze(0).expand_as(data)
    attention_mask = attention_mask < 0.5

    return attention_mask  # , position_ids


class MoEEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.model
        self.embed_tokens = model.embed_tokens
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.tp_group = self.embed_tokens.tp_group
        self.sp_group = self.embed_tokens.sp_group
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                args.seq_length,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group),
            )

    def forward(
        self,
        tokens,
        position_ids=None,
        attention_mask=None,
        labels=None,
        rotary_embedding=None,
    ):
        if self.vocab_sp:
            tokens = tokens[:, self.seq_start_index : self.seq_end_index].contiguous()

        # Fix 5: drain the embedding's TP all-reduce on the current stream
        # and barrier on the TP group before continuing. Without this the
        # TP all-reduce can stay in flight while subsequent layer collectives
        # are queued, causing tp=4 dp=1 to hang at the next MoE layer's
        # internal sync. The synchronize+barrier here is intentional — do
        # not remove without re-validating tp=4 ep=1 end-to-end.
        hidden_states = self.embed_tokens(tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.distributed.barrier(group=self.tp_group)

        # VocabParallelEmbedding emits SBH only when sequence_parallel=True
        # (via scatter_to_sequence_parallel_region). Without SP it returns BSH,
        # which breaks every downstream module that assumes SBH (attention
        # rotary lookup, cross-entropy reduction, etc.). Coerce to SBH here so
        # all profile units and the full training path see the same layout.
        if not self.sequence_parallel:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        return hidden_states


class MoELayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.model
        self.layer = model.layers[layer_idx]

    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        labels=None,
        rotary_embedding=None,
    ):
        hidden_states = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            rotary_embedding=rotary_embedding,
        )
        return hidden_states


class MoEPreNorm_(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None):
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MoELoss_(nn.Module):
    def __init__(self, weight, sequence_parallel, tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        world_size = mpu.get_tensor_model_parallel_world_size(tp_group)
        if self.sequence_parallel and world_size <= 1:
            self.sequence_parallel = False
            # disable sp to avoid global buffer

    def forward(self, hidden_states):
        logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
            input=hidden_states,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
        )
        return logits_parallel


class MoECls_(nn.Module):
    def __init__(self, model, parallel_loss=True, half_entropy=False):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = model.lm_head.tp_group
        self.sp_group = model.lm_head.sp_group
        self.lm_head = MoELoss_(model.lm_head.weight, self.sequence_parallel, self.tp_group)
        self.clone_scatter_output_in_embedding = get_args().clone_scatter_output_in_embedding
        self.parallel_loss = parallel_loss
        self.half_entropy = half_entropy
        args = get_args()
        if args.entropy_in_fp32:
            self.half_entropy = False
        self.seq_length = args.seq_length
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                self.seq_length,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group),
            )

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None):
        if self.vocab_sp:
            labels = labels[:, self.seq_start_index : self.seq_end_index].contiguous()

        if not self.sequence_parallel:
            hidden_states = copy_to_tensor_model_parallel_region(hidden_states, self.tp_group)

        logits_parallel = self.lm_head(hidden_states)

        # [b s] -> [s b]
        labels = labels.transpose(0, 1).contiguous()

        # loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), input_ids)
        if not self.parallel_loss:
            output = gather_from_tensor_model_parallel_region(logits_parallel, self.tp_group)
            if not self.half_entropy:
                logits = output.float()
            else:
                logits = output
            loss = None
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous()  # logits[:-1, ..., :].contiguous()
            shift_labels = labels.contiguous()  # input_ids[1:, ...].contiguous()
            # Flatten the tokens
            from torch.nn import CrossEntropyLoss

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = fused_vocab_parallel_cross_entropy(logits_parallel, labels, self.half_entropy, tp_group=self.tp_group)
            # loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel, labels, self.half_entropy, tp_group=self.tp_group)
            if self.vocab_sp:
                loss = gather_from_tensor_model_parallel_region(loss, self.sp_group)
            # loss = loss.mean()
        loss = loss.transpose(0, 1).contiguous()
        return loss


def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module("embeddings", MoEEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = MoELayers_(model, i)
        model_.add_module("layer_%d" % i, enc)
    model_.add_module("prenorm", MoEPreNorm_(model, config))
    model_.add_module("cls", MoECls_(model))
    MoELoss_.reset_parameters = colummn_row_reset_parameters
    return model_


class MoEModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(MoEModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len, hidden_size = config.max_position_embeddings, config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        if args.shape_order == "SBH":
            layer_shapes_list = [[[seq_len, -1, hidden_size]]]
        else:
            layer_shapes_list = [[[-1, seq_len, hidden_size]]]
        layer_dtypes_list = [[mixed_precision]]
        module_types = ["embed"] + ["gpt_dec"] * config.num_hidden_layers + ["norm", "cls"]
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)
