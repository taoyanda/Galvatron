"""Intra-stage MoE cost model: cost of running N transformer layers on a
single pipeline stage.

Knows nothing about cross-stage pipelining — ``pp`` must equal 1. The
model accounts for compute (per-layer fwd+bwd, optionally embedding
and/or lm-head), DP communication, EP all-to-all, Adam optimizer step,
and per-rank peak memory. Used directly for ``pp == 1`` configs and as
the building block for :class:`PPCostModel` under the 1F1B schedule.

Profile artifacts loaded at construction time
(produced by ``scripts/profile_*.{py,sh}``):

  - ``configs/computation_profiling_<prec>_<model>[_tp<TP>_ep<EP>].json``
        forward-only per-layer time profile.
  - ``configs/non-solver/memory_profiling_<prec>_<model>.json``
        per-layer parameter / activation memory and ``other_memory_*``.
  - ``configs/network_config.json``  intra/inter-node bandwidth (GB/s).
  - ``configs/optimizer_step_profiling_<prec>_<model>.json``
        Adam step throughput + empirical optimizer/params ratio.
  - ``configs/runtime_profiling_<prec>_<model>.json``
        full-iter ``fwd_bwd_ms`` / ``opt_ms`` / cuda_peak per
        ``(tp, ep, bsz, fsep)`` shape — preferred over the analytical
        path when shape matches.
  - ``configs/embedding_lmhead_profiling_<prec>_<model>.json``
        standalone embedding + LM-head fwd/bwd timings (split, so PP
        can place each on the correct boundary stage).
  - ``meta_configs/<model>.json``  static model dims.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .base import CostEstimate, ICostModel

_HERE = os.path.dirname(os.path.abspath(__file__))
# Profile JSONs / meta configs live one level up (galvatron/models/moe/).
_MOE_DIR = os.path.normpath(os.path.join(_HERE, ".."))


class IntraCostModel(ICostModel):
    """Cost of running ``num_layers`` transformer layers on a single
    pipeline stage. Use directly when ``pp == 1`` or as a building block
    inside :class:`PPCostModel`.

    Empirical-calibration profiles are loaded if present and used in
    preference to the analytical formulas. Falls back gracefully when
    a profile is missing.
    """

    DEFAULT_OPTIMIZER_TO_PARAMS_RATIO = 3.0
    MODEL_STATE_MULT = 4  # legacy; new code reads ``optimizer_to_params_ratio``

    # ---------- construction ----------

    def __init__(
        self,
        model_name: str,
        mixed_precision: str = "bf16",
        configs_dir: Optional[str] = None,
        meta_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.mixed_precision = mixed_precision
        self.configs_dir = configs_dir or os.path.join(_MOE_DIR, "configs")
        self.meta_dir = meta_dir or os.path.join(_MOE_DIR, "meta_configs")

        self.network = self._load_json(
            os.path.join(self.configs_dir, "network_config.json")
        )
        self.meta = self._load_json(
            os.path.join(self.meta_dir, f"{model_name}.json")
        )
        self.memory_profile = self._load_json_first_existing([
            os.path.join(
                self.configs_dir,
                f"memory_profiling_{mixed_precision}_{model_name}.json",
            ),
            os.path.join(
                self.configs_dir, "non-solver",
                f"memory_profiling_{mixed_precision}_{model_name}.json",
            ),
        ])
        self.optimizer_step_profile: Optional[Dict[str, Any]] = self._try_load(
            os.path.join(
                self.configs_dir,
                f"optimizer_step_profiling_{mixed_precision}_{model_name}.json",
            )
        )
        self.optimizer_to_params_ratio: float = float(
            (self.optimizer_step_profile or {}).get(
                "optimizer_to_params_ratio_median",
                self.DEFAULT_OPTIMIZER_TO_PARAMS_RATIO,
            )
        )
        self.runtime_profile: Optional[Dict[str, Any]] = self._try_load(
            os.path.join(
                self.configs_dir,
                f"runtime_profiling_{mixed_precision}_{model_name}.json",
            )
        )
        self.embedding_lmhead_profile: Optional[Dict[str, Any]] = self._try_load(
            os.path.join(
                self.configs_dir,
                f"embedding_lmhead_profiling_{mixed_precision}_{model_name}.json",
            )
        )
        # FSEP per-MoE-layer overhead profile: time + memory deltas between
        # fsep=on and fsep=off real measurements at matching shapes. Used by
        # the analytical fall-back when the user passes ``fsep=True`` and no
        # runtime-profile entry is available (otherwise the runtime-profile
        # measurement already includes the FSEP cost).
        self.fsep_overhead_profile: Optional[Dict[str, Any]] = self._try_load(
            os.path.join(
                self.configs_dir,
                f"fsep_overhead_profiling_{mixed_precision}_{model_name}.json",
            )
        )

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path) as json_file:
            return json.load(json_file)

    @staticmethod
    def _load_json_first_existing(paths) -> Dict[str, Any]:
        for path in paths:
            if os.path.isfile(path):
                with open(path) as json_file:
                    return json.load(json_file)
        raise FileNotFoundError(f"None of: {paths}")

    @staticmethod
    def _try_load(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(path):
            return None
        with open(path) as json_file:
            return json.load(json_file)

    # ---------- compute / memory profile lookups ----------

    def _compute_profile_path(self, tp: int, ep: int, seq_len: int) -> str:
        suffix = "" if (tp == 1 and ep == 1) else f"_tp{tp}_ep{ep}"
        return os.path.join(
            self.configs_dir,
            f"computation_profiling_{self.mixed_precision}_{self.model_name}_seqlen{seq_len}{suffix}.json",
        )

    def _compute_processed_path(self, seq_len: int) -> str:
        return os.path.join(
            self.configs_dir, "non-solver",
            f"computation_profiling_{self.mixed_precision}_{self.model_name}_seqlen{seq_len}.json",
        )

    def _layer_time_ms(self, tp: int, ep: int, micro_bsz: int, seq_len: int
                       ) -> Tuple[float, float]:
        """Return (per-layer fwd+bwd ms, processed-profile "other" ms)."""
        raw_path = self._compute_profile_path(tp, ep, seq_len)
        per_layer = None
        if os.path.isfile(raw_path):
            raw = self._load_json(raw_path)
            per_layer = self._interp_per_layer_time(raw, micro_bsz, seq_len)
        if per_layer is None:
            proc_path = self._compute_processed_path(seq_len)
            if os.path.isfile(proc_path):
                proc = self._load_json(proc_path)
                key = f"layertype_0_bsz{micro_bsz}_seq{seq_len}"
                if key in proc:
                    per_layer = proc[key]
        if per_layer is None:
            raise KeyError(
                f"No computation profile found for tp={tp}, ep={ep}, "
                f"micro_bsz={micro_bsz}, seq_len={seq_len}. Looked in {raw_path}"
            )
        other_ms = 0.0
        proc_path = self._compute_processed_path(seq_len)
        if os.path.isfile(proc_path):
            proc = self._load_json(proc_path)
            other_ms = float(
                proc.get(f"layertype_other_bsz{micro_bsz}_seq{seq_len}", 0.0)
            )
        return per_layer, other_ms

    @staticmethod
    def _interp_per_layer_time(raw_profile, micro_bsz, seq_len):
        """Extract per-layer fwd-only time (ms) from a raw computation
        profile by linear regression across the available ``layernum``
        samples at the requested (micro_bsz, seq_len). Profiling sweeps
        report total time at multiple ``layernum[N]`` values; the
        per-layer slope is robust to per-iteration "other" overhead."""
        suffix = f"_bsz{micro_bsz}_seq{seq_len}"
        layer_count_to_time: List[Tuple[int, float]] = []
        for key, total_ms in raw_profile.items():
            if not key.startswith("layernum[") or not key.endswith(suffix):
                continue
            try:
                num_layers = int(key[len("layernum["):key.index("]")])
            except ValueError:
                continue
            layer_count_to_time.append((num_layers, total_ms))
        if len(layer_count_to_time) < 2:
            if len(layer_count_to_time) == 1:
                num_layers, total_ms = layer_count_to_time[0]
                return total_ms / num_layers
            return None
        layer_count_to_time.sort()
        n_low, ms_low = layer_count_to_time[0]
        n_high, ms_high = layer_count_to_time[-1]
        return (ms_high - ms_low) / (n_high - n_low)

    def _memory_for_seq(self, seq_len, sequence_parallel=True):
        sp = "_sp" if sequence_parallel else ""
        layer_key = f"layertype_0{sp}"
        if layer_key not in self.memory_profile:
            raise KeyError(f"Memory profile missing key {layer_key!r}")
        layer = self.memory_profile[layer_key][str(seq_len)]
        return {
            "param_per_layer_unsharded_mb": float(layer["parameter_size"]),
            "act_per_bsz_by_tp": {
                int(k): float(v)
                for k, v in layer["tp_activation_per_bsz_dict"].items()
                if k != "checkpoint"
            },
            "act_per_bsz_checkpoint": float(
                layer["tp_activation_per_bsz_dict"]["checkpoint"]
            ),
            "other_off": self.memory_profile[f"other_memory_pp_off{sp}"][str(seq_len)],
            "other_first": self.memory_profile[f"other_memory_pp_on_first{sp}"][str(seq_len)],
            "other_last": self.memory_profile[f"other_memory_pp_on_last{sp}"][str(seq_len)],
        }

    # ---------- embed / lm-head split ----------

    def _emb_lm_split_ms(self, micro_bsz: int, seq_len: int
                          ) -> Tuple[float, float, float]:
        """Return (embedding_fwd_bwd_ms, lmhead_fwd_bwd_ms, total_ms).

        Reads the standalone profile when available; otherwise falls back
        to the forward-only ``layertype_other_*`` from the computation
        profile multiplied by ``(1 + bwd_mult)`` (caller handles that).
        Returning zeros means "no profile" — caller should use the
        analytical fall-back.
        """
        if self.embedding_lmhead_profile is None:
            return 0.0, 0.0, 0.0
        key = f"bsz{micro_bsz}_seq{seq_len}"
        entry = self.embedding_lmhead_profile.get("by_shape", {}).get(key)
        if entry is None:
            # Linear scaling from closest profiled bsz at this seq_len.
            same_seq = [
                e for e in self.embedding_lmhead_profile.get("samples", [])
                if e.get("seq") == seq_len
            ]
            if not same_seq:
                return 0.0, 0.0, 0.0
            closest = min(same_seq, key=lambda e: abs(e["bsz"] - micro_bsz))
            scale = micro_bsz / closest["bsz"]
            emb = float(closest["embedding"]["fwd_bwd_ms"]) * scale
            lm = float(closest["lmhead"]["fwd_bwd_ms"]) * scale
            return emb, lm, emb + lm
        emb = float(entry["embedding"]["fwd_bwd_ms"])
        lm = float(entry["lmhead"]["fwd_bwd_ms"])
        return emb, lm, emb + lm

    # ---------- helpers ----------

    def _fsep_overhead_per_layer_ms(self, tp: int, ep: int,
                                     micro_bsz: int, seq_len: int) -> float:
        """Per-MoE-layer FSEP time overhead from the empirical profile.

        Looks up the (tp, ep, micro_bsz, seq) shape; falls back to the
        ``default_time_overhead_per_layer_ms`` median when the requested
        shape isn't profiled. Returns 0.0 when no profile is loaded.
        """
        if self.fsep_overhead_profile is None:
            return 0.0
        # Profile keys omit `_pp1` suffix and the FSEP overhead at pp=1
        # generalises to any pp on the same per-stage shape. Try the
        # exact key first (with no _pp suffix for back-compat), then
        # fall back to the global median.
        key = f"tp{tp}_ep{ep}_bsz{micro_bsz}_seq{seq_len}"
        entry = self.fsep_overhead_profile.get("by_shape", {}).get(key)
        if entry is not None:
            return float(entry.get("time_overhead_per_layer_ms", 0.0))
        return float(
            self.fsep_overhead_profile.get("default_time_overhead_per_layer_ms", 0.0)
        )

    def _fsep_memory_overhead_per_layer_mb(self, tp: int, ep: int,
                                            micro_bsz: int, seq_len: int) -> float:
        """Per-MoE-layer FSEP memory overhead (expert replication)."""
        if self.fsep_overhead_profile is None:
            return 0.0
        key = f"tp{tp}_ep{ep}_bsz{micro_bsz}_seq{seq_len}"
        entry = self.fsep_overhead_profile.get("by_shape", {}).get(key)
        if entry is not None:
            v = entry.get("memory_overhead_per_layer_mb")
            if v is not None:
                return float(v)
        return float(
            self.fsep_overhead_profile.get("default_memory_overhead_per_layer_mb", 0.0)
        )

    def _expert_param_share(self) -> float:
        """Fraction of per-layer parameters that live in the expert MLPs
        (and thus shard with EP). The rest live in attention + the
        always-replicated layernorm/router/gating, which shard with TP
        only."""
        hidden = self.meta.get("hidden_size")
        ffn = self.meta.get("intermediate_size")
        num_experts = self.meta.get("num_local_experts", 1)
        if not hidden or not ffn or num_experts <= 1:
            return 0.0
        # Mixtral-style SwiGLU expert: 3 GEMMs each (hidden × ffn).
        expert_params = num_experts * 3 * hidden * ffn
        # Attention: q + k + v + o, each (hidden × hidden).
        attention_params = 4 * hidden * hidden
        return expert_params / (expert_params + attention_params)

    def _dp_bandwidth_gbps(self, group_size: int, gpus_per_node: int) -> float:
        if group_size <= gpus_per_node:
            return float(self.network.get("intra_node", 10.0))
        return float(self.network.get("inter_node", 4.5))

    # ---------- main estimator ----------

    def estimate(
        self,
        *,
        num_layers: int,
        num_gpus: int,
        dp: int,
        pp: int,
        tp: int,
        ep: int = 1,
        micro_batch_size: int = 1,
        global_batch_size: Optional[int] = None,
        seq_len: int = 4096,
        gpus_per_node: int = 8,
        recompute: bool = False,
        zero_stage: int = 1,
        sdp: bool = False,
        sequence_parallel: bool = True,
        bwd_mult: float = 2.0,
        fsep: bool = False,
        # Single-stage extras: PP wrappers set these to place embed/lm-head
        # on the first/last stages, and to pass the 1F1B in-flight count.
        has_embedding: bool = True,
        has_lmhead: bool = True,
        in_flight_microbatches: Optional[int] = None,
        **_kwargs: Any,
    ) -> CostEstimate:
        """Compute the cost of one pipeline stage holding ``num_layers``
        transformer layers. ``pp`` must be 1; PP is handled by
        :class:`PPCostModel`.

        Returns a :class:`CostEstimate` whose ``total_iter_ms`` is the
        single-microbatch stage cost (compute + DP comm + EP a2a + Adam),
        i.e. **not** multiplied by ``(n_micro + pp - 1)``. The orchestrator
        applies that multiplier.
        """
        if pp != 1:
            raise ValueError(
                f"IntraCostModel.estimate requires pp=1; got pp={pp}. "
                f"Use PPCostModel for pp > 1."
            )
        if dp * tp * ep != num_gpus:
            raise ValueError(
                f"dp({dp})*tp({tp})*ep({ep})={dp*tp*ep} != num_gpus({num_gpus})"
            )
        if global_batch_size is None:
            global_batch_size = micro_batch_size * dp
        if global_batch_size % (dp * micro_batch_size) != 0:
            raise ValueError(
                f"global_batch_size {global_batch_size} not divisible by "
                f"dp*micro_bsz ({dp}*{micro_batch_size})"
            )
        n_micro = global_batch_size // (dp * micro_batch_size)
        if in_flight_microbatches is None:
            in_flight_microbatches = n_micro

        # ---------- TIME ----------
        fsep_key = (
            f"tp{tp}_ep{ep}_bsz{micro_batch_size}_seq{seq_len}"
            f"_fsep{'on' if fsep else 'off'}"
        )
        rt_entry = None
        if self.runtime_profile is not None:
            rt_entry = self.runtime_profile.get("by_shape", {}).get(fsep_key)

        emb_ms, lm_ms, emb_lm_total_ms = self._emb_lm_split_ms(
            micro_batch_size, seq_len
        )
        included_emb_lm_ms = (
            (emb_ms if has_embedding else 0.0)
            + (lm_ms if has_lmhead else 0.0)
        )

        if rt_entry is not None:
            n_layers_profiled = int(self.runtime_profile.get("num_layers_profiled", 1))
            fwd_bwd_total_ms = float(rt_entry["fwd_bwd_ms"])
            # Peel off embed+lm-head measured separately so the per-layer
            # cost isn't inflated. The runtime profile was always taken
            # with both ends on a single stage, so subtract the *full*
            # emb+lm-head bundle even when this stage has only one of them.
            if emb_lm_total_ms > 0 and fwd_bwd_total_ms > emb_lm_total_ms:
                pure_layers_ms = fwd_bwd_total_ms - emb_lm_total_ms
            else:
                pure_layers_ms = fwd_bwd_total_ms
            per_layer_ms = pure_layers_ms / max(1, n_layers_profiled)
            other_ms_fwdonly = 0.0
            time_source = f"runtime_profile[{fsep_key}]"
        else:
            fwd_per_layer_ms, other_ms_fwdonly = self._layer_time_ms(
                tp, ep, micro_batch_size, seq_len
            )
            per_layer_ms = fwd_per_layer_ms * (1.0 + bwd_mult)
            if recompute:
                per_layer_ms *= 4.0 / 3.0
            time_source = "computation_profile_forward_only"

        # Stage compute = layers + (embed if first) + (lm-head if last).
        # When the embed/lm-head profile is missing, fall back to the
        # forward-only "other" × (1+bwd_mult), which historically applies
        # to whichever stage is asked to include it.
        if emb_lm_total_ms > 0:
            stage_emb_lm_ms = included_emb_lm_ms
        elif rt_entry is None and (has_embedding or has_lmhead):
            stage_emb_lm_ms = other_ms_fwdonly * (1.0 + bwd_mult)
        else:
            stage_emb_lm_ms = 0.0
        # FSEP per-MoE-layer overhead from the empirical profile. Skipped
        # when ``rt_entry is not None`` because the runtime profile already
        # measures the full FSEP-on iter time. Skipped when fsep=False.
        # The overhead value is fwd+bwd (already reflects recompute and
        # whatever else was active at calibration), so it's added once
        # per layer per microbatch — no (1+bwd_mult) multiplier here.
        fsep_overhead_ms = 0.0
        if fsep and rt_entry is None and self.fsep_overhead_profile is not None:
            fsep_overhead_ms = self._fsep_overhead_per_layer_ms(
                tp, ep, micro_batch_size, seq_len
            ) * num_layers
        stage_compute_ms = num_layers * per_layer_ms + stage_emb_lm_ms + fsep_overhead_ms

        # ---------- Param sharding (used by DP comm + memory model) ----------
        mem = self._memory_for_seq(seq_len, sequence_parallel=sequence_parallel)
        param_per_layer_mb = mem["param_per_layer_unsharded_mb"] / tp
        expert_share = self._expert_param_share()
        if ep > 1:
            param_per_layer_mb = (
                param_per_layer_mb * (1 - expert_share + expert_share / ep)
            )
        params_on_stage_mb = num_layers * param_per_layer_mb
        bw_dp = self._dp_bandwidth_gbps(dp, gpus_per_node)

        # ---------- DP comm ----------
        dp_comm_ms = 0.0
        if dp > 1 and bwd_mult > 0 and rt_entry is None:
            ar_volume_ms = (
                2 * (dp - 1) / dp
                * params_on_stage_mb / 1024.0 / bw_dp * 1000.0
            )
            dp_comm_ms = ar_volume_ms * (1.5 if zero_stage >= 3 else 1.0)

        # ---------- EP all-to-all ----------
        ep_comm_ms = 0.0
        if ep > 1 and rt_entry is None:
            hidden = self.meta.get("hidden_size", 0)
            topk = self.meta.get("num_experts_per_tok", 1)
            tokens = micro_batch_size * seq_len
            n_dirs = 2 if bwd_mult > 0 else 1
            bytes_per_layer = n_dirs * tokens * hidden * topk * 2
            bw_ep = self._dp_bandwidth_gbps(ep, gpus_per_node)
            ep_comm_ms = num_layers * bytes_per_layer / 1e6 / bw_ep

        # ---------- Param vs optimizer shard factors ----------
        param_shard = 1
        optim_shard = 1
        if dp > 1:
            if zero_stage >= 3:
                param_shard = dp
                optim_shard = dp
            elif zero_stage >= 2 and sdp:
                optim_shard = dp

        # ---------- Adam optimizer step ----------
        opt_mb_per_rank = (
            num_layers * param_per_layer_mb * self.optimizer_to_params_ratio
            / optim_shard
        )
        opt_step_ms = 0.0
        if bwd_mult > 0 and self.optimizer_step_profile is not None:
            thr = float(self.optimizer_step_profile.get("throughput_mb_per_ms_median", 0.0))
            if thr > 0:
                opt_step_ms = opt_mb_per_rank / thr

        # Stage iter time: one microbatch through this stage + post-bwd terms.
        stage_iter_ms = stage_compute_ms + dp_comm_ms + ep_comm_ms + opt_step_ms

        # ---------- MEMORY ----------
        params_per_layer_sharded_mb = param_per_layer_mb / param_shard
        optim_per_layer_sharded_mb = (
            param_per_layer_mb * self.optimizer_to_params_ratio / optim_shard
        )
        ms_per_layer_mb = params_per_layer_sharded_mb + optim_per_layer_sharded_mb

        act_dict = mem["act_per_bsz_by_tp"]
        if recompute:
            act_per_layer_per_bsz = mem["act_per_bsz_checkpoint"]
        elif tp in act_dict:
            act_per_layer_per_bsz = act_dict[tp]
        else:
            closest = min(act_dict.keys(), key=lambda k: abs(k - tp))
            act_per_layer_per_bsz = act_dict[closest] * closest / tp

        # "Other" memory bucket: depends on which stage ends are present.
        # For a single-stage (pp=1) with both embed+lm-head, use ``other_off``;
        # for first-only or last-only stages from PPCostModel, use the
        # corresponding pp_on_first / pp_on_last buckets.
        if has_embedding and has_lmhead:
            other = mem["other_off"]
        elif has_embedding:
            other = mem["other_first"]
        elif has_lmhead:
            other = mem["other_last"]
        else:
            other = {"model_states": {}, "activation": {}}
        tp_key = (
            str(tp)
            if str(tp) in (other.get("model_states") or {})
            else "1"
        )
        other_ms_total = float((other.get("model_states") or {}).get(tp_key, 0.0))
        if zero_stage >= 3 and dp > 1:
            other_ms_total /= dp
        ms_total_mult = 1.0 + self.optimizer_to_params_ratio
        other_param_unit = other_ms_total / ms_total_mult
        other_optim_unit = other_ms_total * self.optimizer_to_params_ratio / ms_total_mult
        if dp > 1 and zero_stage == 2 and sdp:
            other_optim_unit /= dp

        # Analytical stage memory.
        ms_layers = num_layers * ms_per_layer_mb
        act_layers = (
            num_layers * act_per_layer_per_bsz
            * micro_batch_size * in_flight_microbatches
        )
        other_act_unit = float((other.get("activation") or {}).get(tp_key, 0.0))
        other_act_mb = other_act_unit if recompute else other_act_unit * micro_batch_size
        analytical_peak_mb = ms_layers + act_layers + other_ms_total + other_act_mb

        # Memory source priority:
        #   1. Empirical alpha-beta fit from ≥ 2 runtime-profile N points.
        #      Each component is fit as ``y = α + β × num_layers`` and
        #      evaluated at the requested num_layers. Captures the true
        #      per-layer cost — including FSDP per-layer all-gather buffers
        #      and grad staging — that single-point extrapolation misses.
        #   2. Single-point runtime profile + analytical β derived from
        #      per-layer params/optim/activation. Less accurate but works
        #      when only one N has been profiled.
        #   3. Analytical stage_memory() — fully analytical fall-back.
        memory_source = "analytical_stage_memory"
        ab_fit = (rt_entry or {}).get("alpha_beta_fit") or {}
        cuda_fit = ab_fit.get("cuda_peak_mb") if ab_fit else None
        if (cuda_fit is not None
                and ab_fit.get("params_mb") is not None
                and ab_fit.get("optimizer_mb") is not None
                and ab_fit.get("activation_peak_mb") is not None):
            params_mb_total = ab_fit["params_mb"]["alpha"] + ab_fit["params_mb"]["beta"] * num_layers
            optimizer_mb_total = ab_fit["optimizer_mb"]["alpha"] + ab_fit["optimizer_mb"]["beta"] * num_layers
            activations_mb_total = ab_fit["activation_peak_mb"]["alpha"] + ab_fit["activation_peak_mb"]["beta"] * num_layers
            peak_mb = cuda_fit["alpha"] + cuda_fit["beta"] * num_layers
            memory_source = (
                f"runtime_profile[{fsep_key}]+alpha_beta_fit"
                f"(N_pts={cuda_fit.get('n_points', 0)})"
            )
        elif (rt_entry is not None
                and rt_entry.get("cuda_peak_mb") is not None
                and rt_entry.get("params_mb") is not None
                and rt_entry.get("optimizer_mb") is not None
                and rt_entry.get("activation_peak_mb") is not None):
            # Single-N: fall back to the previous calibration-point + linear
            # extrapolation derived from per-layer constants. Less accurate
            # because β-of-activation-pool is taken from a static analytical
            # estimate rather than from data — but no other choice with one N.
            n_profiled = max(1, int(self.runtime_profile.get("num_layers_profiled", 1)))
            profiled_params = float(rt_entry["params_mb"])
            profiled_optim = float(rt_entry["optimizer_mb"])
            profiled_act_peak = float(rt_entry["activation_peak_mb"])
            per_layer_params_calibrated = (profiled_params - other_param_unit) / n_profiled
            per_layer_optim_calibrated = (profiled_optim - other_optim_unit) / n_profiled
            hidden = float(self.meta.get("hidden_size", 0))
            tp_act_div = float(tp) if sequence_parallel and tp > 1 else 1.0
            layer_boundary_mb = (
                micro_batch_size * seq_len * hidden * 2 / 1024**2 / tp_act_div
                if recompute else
                act_per_layer_per_bsz * micro_batch_size
            )
            framework_overhead_mb = max(
                0.0, profiled_act_peak - n_profiled * layer_boundary_mb
            )
            params_mb_total = per_layer_params_calibrated * num_layers + other_param_unit
            optimizer_mb_total = per_layer_optim_calibrated * num_layers + other_optim_unit
            activations_mb_total = (
                framework_overhead_mb + num_layers * layer_boundary_mb * in_flight_microbatches
            )
            peak_mb = params_mb_total + optimizer_mb_total + activations_mb_total
            memory_source = f"runtime_profile[{fsep_key}]+linear_extrapolation"
        else:
            params_mb_total = num_layers * params_per_layer_sharded_mb + other_param_unit
            optimizer_mb_total = num_layers * optim_per_layer_sharded_mb + other_optim_unit
            activations_mb_total = (
                analytical_peak_mb - num_layers * ms_per_layer_mb - other_ms_total
            )
            peak_mb = analytical_peak_mb
            # FSEP memory overhead (per-MoE-layer expert replication) on
            # the analytical path. Same provenance as the time overhead.
            if fsep and self.fsep_overhead_profile is not None:
                fsep_mem_overhead_mb = self._fsep_memory_overhead_per_layer_mb(
                    tp, ep, micro_batch_size, seq_len
                ) * num_layers
                activations_mb_total += fsep_mem_overhead_mb
                peak_mb += fsep_mem_overhead_mb

        return CostEstimate(
            total_iter_ms=stage_iter_ms,
            peak_memory_mb=peak_mb,
            breakdown={
                "stage_compute_ms": stage_compute_ms,
                "stage_iter_ms": stage_iter_ms,
                "per_layer_ms": per_layer_ms,
                "embed_ms": emb_ms if has_embedding else 0.0,
                "lmhead_ms": lm_ms if has_lmhead else 0.0,
                "dp_allreduce_ms": dp_comm_ms,
                "ep_alltoall_ms": ep_comm_ms,
                "opt_step_ms": opt_step_ms,
                "time_source": time_source,
                "memory_source": memory_source,
                "n_microbatches": float(n_micro),
                "in_flight_microbatches": float(in_flight_microbatches),
                "num_layers": float(num_layers),
                "parameters_mb": params_mb_total,
                "optimizer_mb": optimizer_mb_total,
                "activations_mb": activations_mb_total,
                "model_states_mb": params_mb_total + optimizer_mb_total,
                "param_per_layer_mb": param_per_layer_mb,
                "act_per_layer_per_bsz_mb": act_per_layer_per_bsz,
                "has_embedding": has_embedding,
                "has_lmhead": has_lmhead,
            },
        )
