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
        per-layer fwd+bwd iteration time profile (full block under the
        ``all`` pass; per-component slopes under the ``attention`` /
        ``mlp`` passes of the three-pass loop in
        ``profile_computation.sh``).
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
from typing import Any, Dict, List, Literal, Optional, Tuple

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
        # ``memory_profile`` is the legacy ``profile_memory.sh`` (Step 4)
        # output. The base ``IntraCostModel`` doesn't read it — the
        # methods that consumed it (``_memory_for_seq``,
        # ``per_microbatch_activation_mb``, ``per_layer_act_alpha_beta``,
        # ``attention_mlp_act_ratio``) are stubbed at this layer and
        # implemented by :class:`IntraCostModelMeasuredAct` (which loads
        # ``memory_profile`` in its own ``__init__``). The base class'
        # corresponding analytical / chunks_overhead path is delivered
        # in step 2b.2.
        self.memory_profile: Optional[Dict[str, Any]] = None
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
        # Per-microbatch overhead derived from chunks=1 vs chunks=2
        # calibration pairs. Captures synchronous grad-reduce cost +
        # extra PP send/recv + scheduler overhead — quantities the
        # chunks=1-only Alpa formula can't see. ``None`` if no
        # chunks-overhead JSON has been generated yet (analytical Alpa
        # path is used as fallback).
        self.chunks_overhead_profile: Optional[Dict[str, Any]] = self._try_load(
            os.path.join(
                self.configs_dir,
                f"chunks_overhead_profiling_{mixed_precision}_{model_name}.json",
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
        per_layer_full, _attn, _mlp = self._layer_time_ms_split(
            tp, ep, micro_bsz, seq_len
        )
        if per_layer_full is None:
            raw_path = self._compute_profile_path(tp, ep, seq_len)
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
        return per_layer_full, other_ms

    def _layer_time_ms_split(
        self, tp: int, ep: int, micro_bsz: int, seq_len: int,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return ``(full, attention, mlp)`` per-layer fwd+bwd ms.

        Each component is the linear-fit slope across ``layernum[N]``
        samples in the raw computation profile (or, when the raw file
        only has one N value, the single-point ratio). The values are
        full fwd+bwd+opt iteration time per layer because the profiler
        drives ``train_dist_frozen.py`` which runs a complete training
        step per iteration; the slope across N peels off the constant
        per-iteration overhead. ``attention`` and ``mlp`` are populated
        only when the three-pass profiling was run
        (``profile_computation.sh`` writes ``_attention`` and ``_mlp``
        suffixed keys); otherwise they are ``None`` and the caller
        falls back to the full-block slope.
        """
        raw_path = self._compute_profile_path(tp, ep, seq_len)
        full = attn = mlp = None
        # Linear interpolation from raw profile
        if os.path.isfile(raw_path):
            raw = self._load_json(raw_path)
            full = self._interp_per_layer_time(raw, micro_bsz, seq_len, suffix="")
            attn = self._interp_per_layer_time(
                raw, micro_bsz, seq_len, suffix="_attention"
            )
            mlp = self._interp_per_layer_time(
                raw, micro_bsz, seq_len, suffix="_mlp"
            )
        if full is None:
            # Fall back to the processed (non-solver/) JSON which holds
            # only the full-block slope.
            proc_path = self._compute_processed_path(seq_len)
            if os.path.isfile(proc_path):
                proc = self._load_json(proc_path)
                key = f"layertype_0_bsz{micro_bsz}_seq{seq_len}"
                if key in proc:
                    full = float(proc[key])
                # Processed JSON may also carry per-component slopes when
                # _process_computation_data preserved them.
                if attn is None:
                    attn_v = proc.get(f"{key}_attention")
                    attn = float(attn_v) if attn_v is not None else None
                if mlp is None:
                    mlp_v = proc.get(f"{key}_mlp")
                    mlp = float(mlp_v) if mlp_v is not None else None
        return full, attn, mlp

    def attention_mlp_fwd_ratio(
        self, tp: int, ep: int, micro_bsz: int, seq_len: int,
    ) -> Optional[Tuple[float, float]]:
        """Return ``(ratio_attn, ratio_mlp)`` summing to 1, or ``None``
        when per-component data isn't available at this shape.

        Kept for back-compat callers (e.g. cost_model_split_regression.py).
        The cost-model time path now prefers
        :meth:`attention_per_layer_time_ms` (direct slope) over multiplying
        a small-N ratio by a large-N calibration anchor.
        """
        _full, attn, mlp = self._layer_time_ms_split(tp, ep, micro_bsz, seq_len)
        if attn is None or mlp is None:
            return None
        if attn <= 0 or mlp <= 0:
            return None
        total = attn + mlp
        return attn / total, mlp / total

    def attention_per_layer_time_ms(
        self, tp: int, ep: int, micro_bsz: int, seq_len: int,
    ) -> Optional[float]:
        """Direct fwd+bwd ms per attention layer from the FSEP-off
        three-pass profile, or ``None`` when the per-component sweep
        hasn't been run at this shape.

        Preferred over the ratio×full-block path: the per-component
        slope is fit across the same small-N sweep that produced it,
        so the ratio-vs-N conflation (small-N split applied to
        large-N full-block calibration) doesn't apply. Attention is
        FSEP-invariant by construction, so this slope holds for both
        FSEP=on and FSEP=off queries.
        """
        _full, attn, _mlp = self._layer_time_ms_split(
            tp, ep, micro_bsz, seq_len
        )
        if attn is None or attn <= 0:
            return None
        return attn

    @staticmethod
    def _interp_per_layer_time(raw_profile, micro_bsz, seq_len, suffix=""):
        """Extract per-layer fwd+bwd ms from a raw computation profile by
        linear regression across the available ``layernum`` samples at
        the requested (micro_bsz, seq_len). Profiling sweeps report
        total iteration time at multiple ``layernum[N]`` values; the
        per-layer slope is robust to per-iteration "other" overhead.

        ``suffix`` selects the profile pass: ``""`` for full-block
        (``layernum[N]_bsz<B>_seq<S>``), ``"_attention"`` for the
        attention-only pass, ``"_mlp"`` for the MoE-only pass.
        """
        full_suffix = f"_bsz{micro_bsz}_seq{seq_len}{suffix}"
        layer_count_to_time: List[Tuple[int, float]] = []
        for key, total_ms in raw_profile.items():
            if not key.startswith("layernum[") or not key.endswith(full_suffix):
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

    # Per-microbatch slope below this is treated as noise floor and
    # discarded in favor of the closed-form analytical path. Empirically
    # PP=2 chunks_overhead measurements at 4-GPU calibration land near 0
    # (or slightly negative) due to allocator-state variance dominating
    # the small per-microbatch delta — see findings 1 and 2 in the
    # step-2b sweep analysis. PP=1 measurements are typically ~hundreds
    # of MB per microbatch per layer-equivalent.
    _CHUNKS_OVERHEAD_NOISE_FLOOR_MB = 50.0

    def per_microbatch_activation_mb(
        self, *,
        num_layers: int,
        per_rank_micro_bsz: int,
        seq_len: int,
        tp: int,
        recompute: bool,
        sequence_parallel: bool = True,
    ) -> float:
        """Activation memory (MB) held by one stage of ``num_layers``
        layers per microbatch in flight, at the given shape. Used by
        :class:`PPCostModel`'s ``num_stages_behind`` reserve calculation.

        Analytical implementation — boundary-tensor formula under
        recompute. For higher-precision measured values, callers with
        shape context (tp, ep, micro_bsz, fsep, pp, dp_mode) should
        consult ``chunks_overhead_profile`` directly before falling back
        to this method (the function signature is shape-agnostic by
        design — pp.py owns the chunks_overhead lookup).

        Returns ``num_layers × per_rank_micro_bsz × per_layer_act_per_bsz_mb``
        where ``per_layer_act_per_bsz_mb`` is the boundary tensor at
        (seq_len, hidden, bf16) divided by tp under SP. Under no-recompute
        this under-predicts (interior activations not modeled
        analytically); the calibration sweep always runs with recompute,
        so the recompute=True path is the production case.
        """
        mem = self._memory_for_seq(seq_len, sequence_parallel=sequence_parallel)
        act_dict = mem["act_per_bsz_by_tp"]
        if recompute:
            act_per_layer_per_bsz = mem["act_per_bsz_checkpoint"]
        elif tp in act_dict:
            act_per_layer_per_bsz = act_dict[tp]
        else:
            closest = min(act_dict.keys(), key=lambda k: abs(k - tp))
            act_per_layer_per_bsz = act_dict[closest] * closest / tp
        return num_layers * act_per_layer_per_bsz * per_rank_micro_bsz

    def _memory_for_seq(self, seq_len, sequence_parallel=True):
        """Analytical memory profile derived from ``self.meta``.

        Schema matches the legacy (memory_profile-based) version so
        consumers in :meth:`stage_memory` don't need code changes:

          - ``param_per_layer_unsharded_mb``: bf16 params per transformer
            block (attention QKV+O + experts × 3 × hidden × intermediate
            + router + 2 layer norms).
          - ``act_per_bsz_by_tp``: dict ``{tp: per-bsz boundary tensor}``
            for tp ∈ {1, 2, 4, 8}. Boundary-only — interior activations
            (no-recompute regime) aren't modeled.
          - ``act_per_bsz_checkpoint``: per-bsz boundary tensor at tp=1
            (recompute regime; SP scaling applied in
            :meth:`per_microbatch_activation_mb`).
          - ``other_off`` / ``other_first`` / ``other_last``: embed +
            lmhead bucket (model_states + activation per tp). Embed
            sharded by tp (≈ vocab_tp); LM-head similarly.

        Replaces the legacy ``profile_memory.sh`` (Step 4) JSON read.
        For consumers that need higher-precision interior-activation
        values under no-recompute queries (rare), use
        :class:`IntraCostModelMeasuredAct` instead.
        """
        hidden = float(self.meta["hidden_size"])
        intermediate = float(self.meta["intermediate_size"])
        num_attn_heads = int(self.meta["num_attention_heads"])
        num_kv_heads = int(self.meta.get("num_key_value_heads", num_attn_heads))
        head_dim = int(self.meta.get("head_dim", int(hidden) // num_attn_heads))
        num_experts = int(self.meta.get("num_local_experts", 1))
        vocab = int(self.meta["vocab_size"])
        elem_bytes = 2  # bf16

        # Per-layer unsharded params. Same parameter accounting as
        # MoEModel's transformer block: GQA attention (Q wide, KV
        # narrower), 128-expert SwiGLU MLP, router, 2 RMSNorm.
        attn_params = (
            hidden * num_attn_heads * head_dim       # Q
            + hidden * num_kv_heads * head_dim       # K
            + hidden * num_kv_heads * head_dim       # V
            + num_attn_heads * head_dim * hidden     # O
        )
        expert_params = num_experts * 3 * hidden * intermediate
        router_params = hidden * num_experts
        layer_norm_params = 2 * hidden  # input_norm + post_attn_norm
        per_layer_params_unsharded = (
            attn_params + expert_params + router_params + layer_norm_params
        )
        param_per_layer_unsharded_mb = (
            per_layer_params_unsharded * elem_bytes / 1024.0**2
        )

        # Boundary tensor under recompute: bsz × seq × hidden × elem_bytes.
        # Per-bsz, per-layer, before TP/SP sharding.
        boundary_per_bsz_no_sp_mb = (
            seq_len * hidden * elem_bytes / 1024.0**2
        )
        # tp_activation_per_bsz_dict: legacy schema returns one value
        # per tp shard. Under SP, divide by tp; without SP, no shard.
        act_per_bsz_by_tp = {
            tp_v: (
                boundary_per_bsz_no_sp_mb / tp_v
                if sequence_parallel else boundary_per_bsz_no_sp_mb
            )
            for tp_v in (1, 2, 4, 8)
        }
        # ``act_per_bsz_checkpoint`` mirrors the legacy field —
        # boundary tensor at tp=1 (the recompute path's per-layer
        # cost). The :meth:`per_microbatch_activation_mb` consumer
        # divides by ``tp_sp_div`` itself when SP applies.
        act_per_bsz_checkpoint = boundary_per_bsz_no_sp_mb

        # Embed / lm-head bucket. Model states under each tp shard:
        # bf16 params (vocab × hidden / tp) + Adam state. Activation:
        # boundary tensor between embed/lm-head and the adjacent
        # transformer block (per-bsz; consumer scales by per_rank_bsz).
        embed_unsharded_params_bytes = vocab * hidden * elem_bytes
        lmhead_unsharded_params_bytes = embed_unsharded_params_bytes  # untied

        def _other_block(params_bytes: float) -> Dict[str, Dict[str, float]]:
            block: Dict[str, Dict[str, float]] = {
                "model_states": {}, "activation": {},
            }
            for tp_v in (1, 2, 4, 8):
                sharded_params_mb = params_bytes / tp_v / 1024.0**2
                # Model state = sharded params × (1 + optimizer/params ratio).
                ms_mb = sharded_params_mb * (
                    1.0 + self.optimizer_to_params_ratio
                )
                act_mb = (
                    boundary_per_bsz_no_sp_mb / tp_v
                    if sequence_parallel else boundary_per_bsz_no_sp_mb
                )
                block["model_states"][str(tp_v)] = ms_mb
                block["activation"][str(tp_v)] = act_mb
            return block

        return {
            "param_per_layer_unsharded_mb": param_per_layer_unsharded_mb,
            "act_per_bsz_by_tp": act_per_bsz_by_tp,
            "act_per_bsz_checkpoint": act_per_bsz_checkpoint,
            # PP=1: both embed and lm-head live on the same stage.
            "other_off": _other_block(
                embed_unsharded_params_bytes + lmhead_unsharded_params_bytes
            ),
            # PP first stage: embed only.
            "other_first": _other_block(embed_unsharded_params_bytes),
            # PP last stage: lm-head only.
            "other_last": _other_block(lmhead_unsharded_params_bytes),
        }

    def attention_mlp_act_ratio(
        self, tp: int, ep: int, micro_bsz: int, seq_len: int,
        sequence_parallel: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Per-component activation split as ``(ratio_attn, ratio_mlp)``.

        Derived from :meth:`per_layer_act_alpha_beta` β values for
        each component. Returns ``None`` only if both components miss
        a slope (chunks_overhead absent and analytical fallback fails).
        """
        attn_ab = self.per_layer_act_alpha_beta(
            unit="attention", tp=tp, ep=ep, micro_bsz=micro_bsz,
            seq_len=seq_len, recompute=True,
            sequence_parallel=sequence_parallel,
        )
        mlp_ab = self.per_layer_act_alpha_beta(
            unit="mlp", tp=tp, ep=ep, micro_bsz=micro_bsz,
            seq_len=seq_len, recompute=True,
            sequence_parallel=sequence_parallel,
        )
        if attn_ab is None or mlp_ab is None:
            return None
        attn_beta = attn_ab[1]
        mlp_beta = mlp_ab[1]
        if attn_beta <= 0 or mlp_beta <= 0:
            return None
        total = attn_beta + mlp_beta
        return attn_beta / total, mlp_beta / total

    def per_layer_act_alpha_beta(
        self, *, unit: Literal["attention", "mlp"],
        tp: int, ep: int, micro_bsz: int, seq_len: int,
        recompute: bool = False, sequence_parallel: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """``(α, β)`` for per-layer activation MB of one component.

        Source priority:
          1. ``chunks_overhead_profile.{unit}_alloc_per_extra_microbatch_mb``
             at the matching ``shape_key`` and ``dp_mode``. The slope is
             a measured per-microbatch reserve (chunks=2 vs chunks=1
             cuda_peak delta on the unit-only model), which under 1F1B
             is the per-microbatch activation cost stripped of
             grad-bucket noise. Divided by the calibration ``num_layers``
             to give per-layer-per-microbatch slope, then by ``micro_bsz``
             to give per-layer-per-bsz (the schema this method returns).
             Discarded if below ``_CHUNKS_OVERHEAD_NOISE_FLOOR_MB`` or
             non-positive — empirically PP=2 entries land near zero on
             4-GPU calibration.
          2. Analytical closed-form: boundary tensor under recompute.
             For ``unit=attention``, this is the same boundary as for
             ``unit=mlp`` under recompute (no per-component split when
             only the boundary is retained). The closed-form is
             ``(seq × hidden × 2 / tp_sp_div) × micro_bsz`` MB.

        Returns ``(α, β)`` where ``α=0`` (no per-iter offset under
        recompute calibration) and ``β`` is the per-layer slope.

        Note: under recompute, attention and mlp boundary tensors are
        identical (both retain only input-to-layer activation), so the
        analytical fallback returns the same β for both. The
        chunks_overhead-anchored path picks up the real per-component
        difference (mlp's grad bucket + dispatcher buffers vs attention's
        smaller footprint) when measured data is available.
        """
        # Schema reminder: legacy returned β = per-layer activation MB
        # AT the requested ``micro_bsz`` (not per-bsz). Consumer at
        # intra.py:1121 divides β by ``per_rank_micro_bsz`` to recover
        # per-bsz, then multiplies back by per_rank_micro_bsz × N at
        # intra.py:1176. We must match that contract: return
        # ``β = per_layer_per_bsz × micro_bsz``.

        # --- Path 1: chunks_overhead per-component slope ---
        co = self.chunks_overhead_profile or {}
        # Resolve dp_mode preference: zero2sdp first, then zero3 — the
        # main matrix is zero2sdp-only, so zero2sdp is the canonical
        # source. Try fsep=on first (matrix is FSEP-on except gbsz=1),
        # then fsep=off.
        for fsep in ("on", "off"):
            shape_key = (
                f"tp{tp}_ep{ep}_micro_bsz{micro_bsz}_seq{seq_len}_fsep{fsep}"
            )
            shape_block = co.get("by_shape", {}).get(shape_key)
            if shape_block is None:
                continue
            for dp_mode in ("zero2sdp", "zero3"):
                dp_block = shape_block.get("per_dp_mode", {}).get(dp_mode)
                if dp_block is None:
                    continue
                slope_per_mb = dp_block.get(
                    f"{unit}_alloc_per_extra_microbatch_mb"
                )
                if slope_per_mb is None:
                    continue
                if slope_per_mb < self._CHUNKS_OVERHEAD_NOISE_FLOOR_MB:
                    continue
                nl_cal = dp_block.get("num_layers", 1) or 1
                # slope_per_mb is the per-microbatch reserve for
                # ``nl_cal`` layers at THIS shape's micro_bsz (the
                # shape_key matched on micro_bsz). Per-layer at the
                # calibration micro_bsz = slope_per_mb / nl_cal —
                # which IS β at the consumer's expected micro_bsz
                # (the lookup was keyed on it).
                per_layer_at_bsz = slope_per_mb / float(nl_cal)
                if per_layer_at_bsz <= 0:
                    continue
                return 0.0, per_layer_at_bsz

        # --- Path 2: analytical fallback (boundary tensor under recompute) ---
        hidden = float(self.meta.get("hidden_size", 0))
        if hidden <= 0:
            return None
        elem_bytes = 2  # bf16
        tp_sp_div = float(tp) if sequence_parallel and tp > 1 else 1.0
        per_bsz_per_layer_mb = (
            seq_len * hidden * elem_bytes / 1024.0**2 / tp_sp_div
        )
        if per_bsz_per_layer_mb <= 0:
            return None
        # Return β at micro_bsz (consumer divides by micro_bsz to
        # recover per-bsz). Matches legacy contract.
        return 0.0, per_bsz_per_layer_mb * float(max(micro_bsz, 1))

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

    @staticmethod
    def _pick_overhead(entry: Dict[str, Any], primary: str,
                       legacy: str) -> Optional[float]:
        """Read ``primary`` field (newer name) with a fallback to
        ``legacy`` field (older name kept for back-compat). Returns
        ``None`` when neither is present or both are explicitly None.
        """
        v = entry.get(primary)
        if v is None:
            v = entry.get(legacy)
        return float(v) if v is not None else None

    def _fsep_overhead_per_expert_layer_ms(self, tp: int, ep: int,
                                            micro_bsz: int, seq_len: int) -> float:
        """Per-MoE-layer FSEP time overhead (fwd+bwd, ms).

        FSEP (Fully Sharded Expert Parallel) shards expert weights and
        activations across the EP group; attention compute is unchanged
        because it doesn't touch the expert sharding. So this overhead
        multiplies the **expert** layer count under asymmetric-layer
        queries (uniform-FSEP rule). The profile field
        was historically named ``time_overhead_per_layer_ms`` because
        calibration was 1:1 — we accept either name and prefer the
        newer ``time_overhead_per_expert_layer_ms``.

        Looks up the (tp, ep, micro_bsz, seq) shape; falls back to the
        median across all calibrated shapes when the requested shape
        isn't profiled. Returns 0.0 when no profile is loaded.
        """
        if self.fsep_overhead_profile is None:
            return 0.0
        # Profile keys omit `_pp1` suffix and the FSEP overhead at pp=1
        # generalises to any pp on the same per-stage shape. Try the
        # exact key first (with no _pp suffix for back-compat), then
        # fall back to the global median.
        key = f"tp{tp}_ep{ep}_micro_bsz{micro_bsz}_seq{seq_len}"
        entry = self.fsep_overhead_profile.get("by_shape", {}).get(key)
        if entry is not None:
            value = self._pick_overhead(
                entry,
                "time_overhead_per_expert_layer_ms",
                "time_overhead_per_layer_ms",
            )
            if value is not None:
                return value
        return float(self._pick_overhead(
            self.fsep_overhead_profile,
            "default_time_overhead_per_expert_layer_ms",
            "default_time_overhead_per_layer_ms",
        ) or 0.0)

    def _fsep_memory_overhead_per_expert_layer_mb(
        self, tp: int, ep: int, micro_bsz: int, seq_len: int,
    ) -> float:
        """Per-MoE-layer FSEP memory overhead (expert replication, MB)."""
        if self.fsep_overhead_profile is None:
            return 0.0
        key = f"tp{tp}_ep{ep}_micro_bsz{micro_bsz}_seq{seq_len}"
        entry = self.fsep_overhead_profile.get("by_shape", {}).get(key)
        if entry is not None:
            value = self._pick_overhead(
                entry,
                "memory_overhead_per_expert_layer_mb",
                "memory_overhead_per_layer_mb",
            )
            if value is not None:
                return value
        return float(self._pick_overhead(
            self.fsep_overhead_profile,
            "default_memory_overhead_per_expert_layer_mb",
            "default_memory_overhead_per_layer_mb",
        ) or 0.0)

    # Back-compat aliases — old call sites used the "_per_layer_" names,
    # which under 1:1 mean the same thing as "_per_expert_layer_". Keep
    # them so internal callers don't churn while the asymmetric path
    # rolls out.
    _fsep_overhead_per_layer_ms = _fsep_overhead_per_expert_layer_ms
    _fsep_memory_overhead_per_layer_mb = _fsep_memory_overhead_per_expert_layer_mb

    def _expert_param_share(self) -> float:
        """Fraction of per-layer parameters that live in the expert MLPs
        (and thus shard with EP). The rest live in attention + the
        always-replicated layernorm/router/gating, which shard with TP
        only.

        Pair invariant: ``_attention_param_share() + _expert_param_share()
        == 1.0``.
        """
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

    def _attention_param_share(self) -> float:
        """Sibling of :meth:`_expert_param_share`. Returns ``1 - expert_share``
        for the asymmetric-layer split arithmetic. Defined as a separate
        method so callers don't have to compute the complement inline."""
        return 1.0 - self._expert_param_share()

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
        # Asymmetric-layer kwargs. Default to symmetric (= num_layers).
        # The runtime always trains 1:1, so these are cost-model-only;
        # the search uses them to project costs for hypothetical
        # n_attn ≠ n_expert layouts.
        num_attention_layers: Optional[int] = None,
        num_expert_layers: Optional[int] = None,
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
        i.e. **not** multiplied by ``(num_microbatches + pp - 1)``. The orchestrator
        applies that multiplier.

        Asymmetric layer counts: pass ``num_attention_layers`` and/or
        ``num_expert_layers`` to project a hypothetical layout where the
        block is no longer 1:1 (attention + MoE). Defaults to ``num_layers``
        for both → same result as today's full-block path. When the values
        diverge, a per-component computation profile is required at the
        requested ``(tp, ep, micro_bsz, seq)`` shape; otherwise we raise
        rather than silently using a 50/50 split.
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
        # When global_batch_size isn't passed, assume deactivated PP.
        # Otherwise, there will be an outer PP loop. num_microbatches > 1.
        if global_batch_size is None:
            global_batch_size = micro_batch_size
        if global_batch_size % micro_batch_size != 0:
            raise ValueError(
                f"global_batch_size {global_batch_size} not divisible by "
                f"micro_batch_size {micro_batch_size}"
            )
        # Feasibility checks
        if dp * ep > micro_batch_size:
            raise ValueError(
                f"dp*ep ({dp}*{ep}={dp*ep}) > micro_batch_size "
                f"({micro_batch_size}); per-rank batch would be < 1 "
                f"sample per microbatch step. Increase micro_batch_size "
                f"or reduce dp/ep."
            )
        if micro_batch_size % (dp * ep) != 0:
            raise ValueError(
                f"micro_batch_size ({micro_batch_size}) must be divisible "
                f"by dp*ep ({dp*ep}); per-rank batch must be a whole number."
            )
        per_rank_micro_bsz = micro_batch_size // (dp * ep)
        num_microbatches = global_batch_size // micro_batch_size
        if in_flight_microbatches is None:
            in_flight_microbatches = num_microbatches

        # Resolve asymmetric counts.
        # only then do we need the per-component time/memory ratios.
        if num_attention_layers is None:
            num_attention_layers = num_layers
        if num_expert_layers is None:
            num_expert_layers = num_layers
        asymmetric = num_attention_layers != num_expert_layers

        # Per-component time: read the direct attention fwd+bwd slope
        # from the FSEP-off three-pass profile. Required (no silent
        # fallback): the per-component diagnostic surface — exposed in
        # the breakdown and consumed by drift / regression scripts —
        # is meaningful only when this slope was actually measured.
        attention_time_per_layer_ms = self.attention_per_layer_time_ms(
            tp, ep, per_rank_micro_bsz, seq_len
        )
        if attention_time_per_layer_ms is None:
            raise KeyError(
                f"Per-component attention time slope missing at tp={tp}, "
                f"ep={ep}, per_rank_bsz={per_rank_micro_bsz}, "
                f"seq_len={seq_len}. Run profile_computation.sh (three-pass "
                f"loop over all/attention/mlp) to populate the `_attention` "
                f"and `_mlp` keys."
            )

        # ---------- TIME ----------
        # Runtime-profile shape key. The format is shipped in
        # runtime_profiling_*.json — don't churn it without
        # re-aggregating; the profile_cost_model_terms.py aggregator
        # writes the same string. ``bsz`` here is the per-rank micro
        # batch size used at calibration time.
        # Shape key: (tp, ep, micro_bsz, seq_len, fsep). ``micro_bsz`` is
        # the per-PP-stage compute batch size (= trainer's
        # global_train_batch_size at chunks=1 calibration), NOT the
        # per-rank value. DP and EP only enter as a feasibility guard
        # (DP*EP ≤ micro_bsz, enforced upstream); they don't affect the
        # lookup key.
        runtime_shape_key = (
            f"tp{tp}_ep{ep}_micro_bsz{micro_batch_size}_seq{seq_len}"
            f"_fsep{'on' if fsep else 'off'}"
        )
        runtime_entry = None
        if self.runtime_profile is not None:
            runtime_entry = (
                self.runtime_profile.get("by_shape", {}).get(runtime_shape_key)
            )

        embedding_ms, lmhead_ms, embedding_plus_lmhead_ms = self._emb_lm_split_ms(
            per_rank_micro_bsz, seq_len
        )
        stage_emb_lm_included_ms = (
            (embedding_ms if has_embedding else 0.0)
            + (lmhead_ms if has_lmhead else 0.0)
        )

        if runtime_entry is not None:
            num_layers_in_calibration = int(
                self.runtime_profile.get("num_layers_profiled", 1)
            )
            calibrated_fwd_bwd_ms = float(runtime_entry["fwd_bwd_ms"])
            # Peel off embedding + lm-head measured separately so the
            # per-layer cost isn't inflated. The runtime profile was
            # always taken with both ends on a single stage, so we
            # subtract the *full* embed+lm-head bundle even when the
            # stage we're scoring has only one of them.
            if (embedding_plus_lmhead_ms > 0
                    and calibrated_fwd_bwd_ms > embedding_plus_lmhead_ms):
                pure_layers_total_ms = (
                    calibrated_fwd_bwd_ms - embedding_plus_lmhead_ms
                )
            else:
                pure_layers_total_ms = calibrated_fwd_bwd_ms
            per_layer_ms = pure_layers_total_ms / max(1, num_layers_in_calibration)
            forward_only_other_ms = 0.0
            time_source = f"runtime_profile[{runtime_shape_key}]"
        else:
            forward_only_per_layer_ms, forward_only_other_ms = self._layer_time_ms(
                tp, ep, per_rank_micro_bsz, seq_len
            )
            per_layer_ms = forward_only_per_layer_ms * (1.0 + bwd_mult)
            if recompute:
                per_layer_ms *= 4.0 / 3.0
            time_source = "computation_profile_forward_only"

        # Stage compute = layers + (embed if first) + (lm-head if last).
        # When the embed/lm-head profile is missing, fall back to the
        # forward-only "other" × (1+bwd_mult), which historically applies
        # to whichever stage is asked to include it.
        if embedding_plus_lmhead_ms > 0:
            stage_embedding_lmhead_ms = stage_emb_lm_included_ms
        elif runtime_entry is None and (has_embedding or has_lmhead):
            stage_embedding_lmhead_ms = forward_only_other_ms * (1.0 + bwd_mult)
        else:
            stage_embedding_lmhead_ms = 0.0

        # Per-component fwd+bwd from the runtime profile when available.
        # Step 8's per-unit calibration writes ``unit_breakdown`` under each
        # shape with measured ``attention_fwd_bwd_ms`` (FSEP-invariant) and
        # ``mlp_fwd_bwd_ms`` (FSEP-aware). When present we use those slopes
        # directly: no analytical ``× (1 + bwd_mult)`` and no residual
        # subtraction, so the per-attention/per-expert breakdown matches
        # what was actually measured.
        #
        # DP modes are kept isolated in ``unit_breakdown["per_dp_mode"]``
        # (zero3 vs zero2sdp differ measurably in fwd+bwd), so we pick the
        # block matching the caller's ``(zero_stage, sdp)``. Fall back to
        # whichever dp_mode was profiled if the requested one is missing.
        unit_breakdown = (
            runtime_entry.get("unit_breakdown")
            if runtime_entry is not None else None
        )
        unit_block: Optional[Dict[str, Any]] = None
        unit_dp_label: Optional[str] = None
        if unit_breakdown is not None:
            per_dp_blocks = unit_breakdown.get("per_dp_mode") or {}
            requested_dp_mode = "zero3" if zero_stage == 3 else "zero2sdp"
            unit_block = per_dp_blocks.get(requested_dp_mode)
            if unit_block is not None:
                unit_dp_label = requested_dp_mode
            elif per_dp_blocks:
                # Calibration didn't cover the requested mode at this shape
                # — fall back to the other mode rather than the analytical
                # path, since the per-component measurement is still closer
                # to ground truth than ``× (1 + bwd_mult)``.
                unit_dp_label, unit_block = next(iter(per_dp_blocks.items()))
            if (unit_block is not None
                    and (unit_block.get("attention_fwd_bwd_ms") is None
                         or unit_block.get("mlp_fwd_bwd_ms") is None)):
                unit_block = None

        if unit_block is not None:
            num_layers_unit = max(
                1, int(unit_block.get("unit_num_layers", 1))
            )
            attention_time_per_layer_ms = (
                float(unit_block["attention_fwd_bwd_ms"]) / num_layers_unit
            )
            expert_time_per_layer_ms = (
                float(unit_block["mlp_fwd_bwd_ms"]) / num_layers_unit
            )
            time_source = (
                f"runtime_profile[{runtime_shape_key}]"
                f"+unit_breakdown[{unit_dp_label}]"
            )
        else:
            # Direct attention slope (FSEP-invariant by construction);
            # expert is the residual of per_layer_ms so split-and-recombine
            # is identity under symmetric counts and the FSEP delta lands
            # on the expert component under FSEP=on (uniform-FSEP rule).
            expert_time_per_layer_ms = max(0.0, per_layer_ms - attention_time_per_layer_ms)
        # Symmetric identity: when n_attn == n_expert == num_layers, the
        # sum equals ``num_layers × per_layer_ms`` regardless of how we
        # divided the split between attention_time_per_layer_ms and
        # expert_time_per_layer_ms — i.e. split-and-recombine is identity
        # (regression invariant 1).
        layers_compute_ms = (
            num_attention_layers * attention_time_per_layer_ms
            + num_expert_layers * expert_time_per_layer_ms
        )
        # FSEP per-MoE-layer overhead from the empirical profile. Skipped
        # when ``runtime_entry is not None`` because the runtime profile
        # already measures the full FSEP-on iter time. Skipped when
        # fsep=False. The overhead value is fwd+bwd (already reflects
        # recompute and whatever else was active at calibration), so
        # it's added once per *expert* layer per microbatch —
        # uniform-FSEP rule: attention is fsep-invariant.
        fsep_overhead_ms = 0.0
        if (fsep and runtime_entry is None
                and self.fsep_overhead_profile is not None):
            # FSEP overhead profile is keyed by micro_bsz (= trainer's
            # global_train_batch_size at chunks=1 calibration), matching
            # the runtime_profile key convention.
            fsep_overhead_ms = self._fsep_overhead_per_expert_layer_ms(
                tp, ep, micro_batch_size, seq_len
            ) * num_expert_layers
        stage_compute_ms = (
            layers_compute_ms + stage_embedding_lmhead_ms + fsep_overhead_ms
        )

        # ---------- Param sharding (used by DP comm + memory model) ----------
        # Per-attention-layer and per-expert-layer params after TP sharding.
        # Attention shards with TP only; experts shard with TP × EP.
        mem = self._memory_for_seq(seq_len, sequence_parallel=sequence_parallel)
        unsharded_per_layer_mb = mem["param_per_layer_unsharded_mb"]
        attn_share = self._attention_param_share()
        expert_share = self._expert_param_share()
        per_attn_layer_mb = unsharded_per_layer_mb * attn_share / tp
        per_expert_layer_mb = unsharded_per_layer_mb * expert_share / tp
        if ep > 1:
            per_expert_layer_mb /= ep
        # Symmetric per-block param size (= today's `param_per_layer_mb`).
        # Used for the runtime-profile-anchored memory extrapolation path
        # below, where the calibration anchor is full-block.
        param_per_layer_mb = per_attn_layer_mb + per_expert_layer_mb
        # Asymmetric stage param footprint: attention layers carry no EP
        # shard; expert layers do.
        params_on_stage_mb = (
            num_attention_layers * per_attn_layer_mb
            + num_expert_layers * per_expert_layer_mb
        )
        bw_dp = self._dp_bandwidth_gbps(dp, gpus_per_node)

        # ---------- DP comm ----------
        dp_comm_ms = 0.0
        if dp > 1 and bwd_mult > 0 and runtime_entry is None:
            allreduce_volume_ms = (
                2 * (dp - 1) / dp
                * params_on_stage_mb / 1024.0 / bw_dp * 1000.0
            )
            dp_comm_ms = allreduce_volume_ms * (1.5 if zero_stage >= 3 else 1.0)

        # ---------- EP all-to-all ----------
        # Per-MoE-layer cost; scales with num_expert_layers under the
        # uniform-FSEP / asymmetric model (attention layers don't dispatch
        # tokens across the EP group).
        ep_comm_ms = 0.0
        if ep > 1 and runtime_entry is None:
            hidden = self.meta.get("hidden_size", 0)
            topk = self.meta.get("num_experts_per_tok", 1)
            tokens = per_rank_micro_bsz * seq_len
            n_dirs = 2 if bwd_mult > 0 else 1
            bytes_per_expert_layer = n_dirs * tokens * hidden * topk * 2
            bw_ep = self._dp_bandwidth_gbps(ep, gpus_per_node)
            ep_comm_ms = num_expert_layers * bytes_per_expert_layer / 1e6 / bw_ep

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
        # Optimizer state size scales with the asymmetric param total (as
        # opposed to today's symmetric `num_layers × param_per_layer`).
        opt_mb_per_rank = (
            params_on_stage_mb * self.optimizer_to_params_ratio / optim_shard
        )
        opt_step_ms = 0.0
        if bwd_mult > 0 and self.optimizer_step_profile is not None:
            thr = float(self.optimizer_step_profile.get("throughput_mb_per_ms_median", 0.0))
            if thr > 0:
                opt_step_ms = opt_mb_per_rank / thr

        # Stage iter time: one microbatch through this stage + post-bwd terms.
        stage_iter_ms = stage_compute_ms + dp_comm_ms + ep_comm_ms + opt_step_ms

        # ---------- MEMORY ----------
        # Sharded per-layer model state (params + Adam) for each component.
        per_attn_layer_params_sharded_mb = per_attn_layer_mb / param_shard
        per_expert_layer_params_sharded_mb = per_expert_layer_mb / param_shard
        per_attn_layer_optim_sharded_mb = (
            per_attn_layer_mb * self.optimizer_to_params_ratio / optim_shard
        )
        per_expert_layer_optim_sharded_mb = (
            per_expert_layer_mb * self.optimizer_to_params_ratio / optim_shard
        )
        # Symmetric "per-layer" aggregates kept for the runtime-profile-
        # anchored extrapolation paths below.
        params_per_layer_sharded_mb = (
            per_attn_layer_params_sharded_mb + per_expert_layer_params_sharded_mb
        )
        optim_per_layer_sharded_mb = (
            per_attn_layer_optim_sharded_mb + per_expert_layer_optim_sharded_mb
        )
        ms_per_layer_mb = params_per_layer_sharded_mb + optim_per_layer_sharded_mb

        # Per-component activation memory: direct β slopes per
        # component, fit independently from the three-pass profile.
        # Each component is treated as a separate sublayer with its
        # own α/β — no ratio, no residual against a shared full-block
        # anchor. The recompute flag selects the matching profile
        # regime (checkpointed vs no-recompute samples).
        attn_ab = self.per_layer_act_alpha_beta(
            unit="attention", tp=tp, ep=ep,
            micro_bsz=per_rank_micro_bsz, seq_len=seq_len,
            recompute=recompute, sequence_parallel=sequence_parallel,
        )
        expert_ab = self.per_layer_act_alpha_beta(
            unit="mlp", tp=tp, ep=ep,
            micro_bsz=per_rank_micro_bsz, seq_len=seq_len,
            recompute=recompute, sequence_parallel=sequence_parallel,
        )
        if attn_ab is None or expert_ab is None:
            missing = []
            if attn_ab is None:
                missing.append("attention")
            if expert_ab is None:
                missing.append("mlp")
            raise KeyError(
                f"Per-component activation slope(s) missing for unit(s) "
                f"{missing} at tp={tp}, ep={ep}, "
                f"per_rank_bsz={per_rank_micro_bsz}, seq_len={seq_len}, "
                f"recompute={recompute}. Run profile_memory.sh (three-pass "
                f"loop over all/attention/mlp); the model_profiler sweeps "
                f"checkpoint=0,1 internally so a single sweep covers both "
                f"recompute regimes."
            )
        _alpha_attn, beta_attn = attn_ab
        _alpha_expert, beta_expert = expert_ab
        # β is per-layer activation MB at the profiled per-rank bsz;
        # convert to per-rank-per-sample so it composes with the
        # ``per_rank_micro_bsz × in_flight`` term below.
        act_attn_per_bsz_mb = beta_attn / max(1, per_rank_micro_bsz)
        act_expert_per_bsz_mb = beta_expert / max(1, per_rank_micro_bsz)
        # Full-block per-bsz figure used by some downstream tooling
        # (breakdown, alpha-beta extrapolation paths). Sum the per-
        # component β values rather than the legacy full-block
        # processed value; under direct-direct this is the cost
        # model's own sum-of-parts.
        act_per_layer_per_bsz = act_attn_per_bsz_mb + act_expert_per_bsz_mb

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

        # Analytical stage memory — asymmetric. Each per-block quantity
        # splits into attention and expert components; under symmetric
        # (n_attn = n_expert = num_layers) the sum equals today's
        # ``num_layers × per_layer`` exactly.
        ms_layers_mb = (
            num_attention_layers * (
                per_attn_layer_params_sharded_mb
                + per_attn_layer_optim_sharded_mb
            )
            + num_expert_layers * (
                per_expert_layer_params_sharded_mb
                + per_expert_layer_optim_sharded_mb
            )
        )
        # Per-component activation memory: scale by the per-rank micro
        # batch size (the actual sample count each rank processes per
        # microbatch step), not the global micro_batch_size — only DP
        # × EP ranks share the data dim, and each holds
        # ``per_rank_micro_bsz`` samples.
        act_layers_mb = (
            (num_attention_layers * act_attn_per_bsz_mb
             + num_expert_layers * act_expert_per_bsz_mb)
            * per_rank_micro_bsz * in_flight_microbatches
        )
        other_act_unit = float((other.get("activation") or {}).get(tp_key, 0.0))
        other_act_mb = (
            other_act_unit if recompute
            else other_act_unit * per_rank_micro_bsz
        )
        analytical_peak_mb = ms_layers_mb + act_layers_mb + other_ms_total + other_act_mb
        # Compatibility aliases for the runtime-profile-anchored paths
        # below, which still phrase β-extrapolation in terms of
        # "per-layer" quantities under the 1:1 calibration assumption.
        ms_layers = ms_layers_mb
        act_layers = act_layers_mb

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
        # Paths 1 and 2 anchor on a 1:1 calibration; their N-extrapolation
        # is meaningful only at symmetric n_attn = n_expert. Asymmetric
        # queries skip these anchors and route through the analytical
        # path, which already understands the split.
        memory_source = "analytical_stage_memory"
        alpha_beta_fits = (runtime_entry or {}).get("alpha_beta_fit") or {}
        cuda_peak_fit = (
            alpha_beta_fits.get("cuda_peak_mb") if alpha_beta_fits else None
        )
        if (not asymmetric and cuda_peak_fit is not None
                and alpha_beta_fits.get("params_mb") is not None
                and alpha_beta_fits.get("optimizer_mb") is not None
                and alpha_beta_fits.get("activation_peak_mb") is not None):
            params_fit = alpha_beta_fits["params_mb"]
            optimizer_fit = alpha_beta_fits["optimizer_mb"]
            activation_fit = alpha_beta_fits["activation_peak_mb"]
            params_mb_total = params_fit["alpha"] + params_fit["beta"] * num_layers
            optimizer_mb_total = (
                optimizer_fit["alpha"] + optimizer_fit["beta"] * num_layers
            )
            activations_mb_total = (
                activation_fit["alpha"] + activation_fit["beta"] * num_layers
            )
            peak_mb = cuda_peak_fit["alpha"] + cuda_peak_fit["beta"] * num_layers
            memory_source = (
                f"runtime_profile[{runtime_shape_key}]+alpha_beta_fit"
                f"(N_pts={cuda_peak_fit.get('n_points', 0)})"
            )
        elif (not asymmetric and runtime_entry is not None
                and runtime_entry.get("cuda_peak_mb") is not None
                and runtime_entry.get("params_mb") is not None
                and runtime_entry.get("optimizer_mb") is not None
                and runtime_entry.get("activation_peak_mb") is not None):
            # Single-N: fall back to the previous calibration-point + linear
            # extrapolation derived from per-layer constants. Less accurate
            # because β-of-activation-pool is taken from a static analytical
            # estimate rather than from data — but no other choice with one N.
            num_layers_in_calibration = max(
                1, int(self.runtime_profile.get("num_layers_profiled", 1))
            )
            calibrated_params_mb = float(runtime_entry["params_mb"])
            calibrated_optimizer_mb = float(runtime_entry["optimizer_mb"])
            calibrated_activation_peak_mb = float(
                runtime_entry["activation_peak_mb"]
            )
            calibrated_params_per_layer_mb = (
                (calibrated_params_mb - other_param_unit)
                / num_layers_in_calibration
            )
            calibrated_optim_per_layer_mb = (
                (calibrated_optimizer_mb - other_optim_unit)
                / num_layers_in_calibration
            )
            hidden = float(self.meta.get("hidden_size", 0))
            tp_activation_divisor = (
                float(tp) if sequence_parallel and tp > 1 else 1.0
            )
            per_layer_boundary_mb = (
                per_rank_micro_bsz * seq_len * hidden * 2 / 1024**2 / tp_activation_divisor
                if recompute else
                act_per_layer_per_bsz * per_rank_micro_bsz
            )
            framework_overhead_mb = max(
                0.0,
                calibrated_activation_peak_mb
                - num_layers_in_calibration * per_layer_boundary_mb,
            )
            params_mb_total = (
                calibrated_params_per_layer_mb * num_layers + other_param_unit
            )
            optimizer_mb_total = (
                calibrated_optim_per_layer_mb * num_layers + other_optim_unit
            )
            activations_mb_total = (
                framework_overhead_mb
                + num_layers * per_layer_boundary_mb * in_flight_microbatches
            )
            peak_mb = params_mb_total + optimizer_mb_total + activations_mb_total
            memory_source = (
                f"runtime_profile[{runtime_shape_key}]+linear_extrapolation"
            )
        else:
            # Analytical path — already asymmetric-aware via the per-
            # component model-state and activation totals computed above.
            params_mb_total = (
                num_attention_layers * per_attn_layer_params_sharded_mb
                + num_expert_layers * per_expert_layer_params_sharded_mb
                + other_param_unit
            )
            optimizer_mb_total = (
                num_attention_layers * per_attn_layer_optim_sharded_mb
                + num_expert_layers * per_expert_layer_optim_sharded_mb
                + other_optim_unit
            )
            activations_mb_total = (
                analytical_peak_mb - ms_layers_mb - other_ms_total
            )
            peak_mb = analytical_peak_mb
            # FSEP memory overhead — uniform-FSEP rule: scales with
            # n_expert (attention layers don't carry expert replicas).
            if fsep and self.fsep_overhead_profile is not None:
                fsep_mem_overhead_mb = self._fsep_memory_overhead_per_expert_layer_mb(
                    tp, ep, micro_batch_size, seq_len
                ) * num_expert_layers
                activations_mb_total += fsep_mem_overhead_mb
                peak_mb += fsep_mem_overhead_mb
            if asymmetric:
                memory_source = "analytical_stage_memory[asymmetric]"

        return CostEstimate(
            total_iter_ms=stage_iter_ms,
            peak_memory_mb=peak_mb,
            breakdown={
                "stage_compute_ms": stage_compute_ms,
                "stage_iter_ms": stage_iter_ms,
                "per_layer_ms": per_layer_ms,
                "per_attention_layer_ms": attention_time_per_layer_ms,
                "per_expert_layer_ms": expert_time_per_layer_ms,
                "embed_ms": embedding_ms if has_embedding else 0.0,
                "lmhead_ms": lmhead_ms if has_lmhead else 0.0,
                "fsep_overhead_ms": fsep_overhead_ms,
                "dp_allreduce_ms": dp_comm_ms,
                "ep_alltoall_ms": ep_comm_ms,
                "opt_step_ms": opt_step_ms,
                "time_source": time_source,
                "memory_source": memory_source,
                "num_attention_layers": float(num_attention_layers),
                "num_expert_layers": float(num_expert_layers),
                "asymmetric": asymmetric,
                "n_microbatches": float(num_microbatches),
                "per_rank_micro_bsz": float(per_rank_micro_bsz),
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


class IntraCostModelMeasuredAct(IntraCostModel):
    """Legacy variant that sources per-layer activation memory and
    per-component activation splits from ``profile_memory.sh``'s output
    (``memory_profiling_*.json`` and the per-(tp, ep) raw files).

    This subclass preserves the pre-refactor behavior. The base
    :class:`IntraCostModel`'s analytical / chunks_overhead path
    (step 2b.2) is the recommended path going forward; this subclass
    remains for parity comparisons and for environments that still
    have ``profile_memory.sh`` data on disk.

    Selected by ``PPCostModel(..., use_measured_memory_profile=True)``,
    which is the current default for backward compatibility.
    """

    def __init__(
        self,
        model_name: str,
        mixed_precision: str = "bf16",
        configs_dir: Optional[str] = None,
        meta_dir: Optional[str] = None,
    ):
        super().__init__(model_name, mixed_precision, configs_dir, meta_dir)
        # Path search order mirrors the compute-profile loader at
        # ``_compute_profile_path``: try the seqlen-suffixed file first
        # (the profiler's ``model_name(config, args)`` appends
        # ``_seqlen{max_position_embeddings}`` whenever profile_mode !=
        # "sequence", per ``meta_configs/config_utils.py:164``), then
        # fall back to the legacy seqlen-less path so existing bundles
        # built from older sweeps keep working.
        seqlen_for_path = int(self.meta.get("max_position_embeddings", 4096))
        self.memory_profile = self._load_json_first_existing([
            os.path.join(
                self.configs_dir,
                f"memory_profiling_{mixed_precision}_{model_name}_seqlen{seqlen_for_path}.json",
            ),
            os.path.join(
                self.configs_dir,
                f"memory_profiling_{mixed_precision}_{model_name}.json",
            ),
            os.path.join(
                self.configs_dir, "non-solver",
                f"memory_profiling_{mixed_precision}_{model_name}_seqlen{seqlen_for_path}.json",
            ),
            os.path.join(
                self.configs_dir, "non-solver",
                f"memory_profiling_{mixed_precision}_{model_name}.json",
            ),
        ])

    def per_microbatch_activation_mb(
        self, *,
        num_layers: int,
        per_rank_micro_bsz: int,
        seq_len: int,
        tp: int,
        recompute: bool,
        sequence_parallel: bool = True,
    ) -> float:
        """Activation memory (MB) held by one stage of ``num_layers``
        layers per microbatch in flight, at the given shape.

        ``per_rank_micro_bsz`` is the per-actual-rank sample count per
        microbatch step (= ``micro_batch_size // (dp * ep)`` since both
        DP and EP shard the data dim). Used by :class:`PPCostModel`'s
        ``num_stages_behind`` reserve calculation.
        """
        mem = self._memory_for_seq(seq_len, sequence_parallel=sequence_parallel)
        act_dict = mem["act_per_bsz_by_tp"]
        if recompute:
            act_per_layer_per_bsz = mem["act_per_bsz_checkpoint"]
        elif tp in act_dict:
            act_per_layer_per_bsz = act_dict[tp]
        else:
            closest = min(act_dict.keys(), key=lambda k: abs(k - tp))
            act_per_layer_per_bsz = act_dict[closest] * closest / tp
        return num_layers * act_per_layer_per_bsz * per_rank_micro_bsz

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

    def attention_mlp_act_ratio(
        self, tp: int, ep: int, micro_bsz: int, seq_len: int,
        sequence_parallel: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Return ``(ratio_attn, ratio_mlp)`` for activation memory at this
        shape, or ``None`` when per-component memory data isn't present.

        Reads the raw per-rank ``layernum[N]_bsz<B>_seq<S>_<unit>_rank0_act``
        keys produced by ``profile_memory.sh`` running the three passes
        (``all``, ``attention``, ``mlp``).
        """
        raw_path = self._raw_memory_path(tp, ep)
        if not os.path.isfile(raw_path):
            return None
        raw = self._load_json(raw_path)
        attn_acts: List[float] = []
        mlp_acts: List[float] = []
        sp_marker = "_sp" if sequence_parallel else ""
        for strategy_key, entries in raw.items():
            if not strategy_key.startswith("1_"):
                continue
            ends_sp = strategy_key.endswith("_sp")
            if sp_marker and not ends_sp:
                continue
            if not sp_marker and ends_sp:
                continue
            for layernum in (1, 2):
                attn_key = (
                    f"layernum[{layernum}]_bsz{micro_bsz}_seq{seq_len}"
                    f"_attention_rank0_act"
                )
                mlp_key = (
                    f"layernum[{layernum}]_bsz{micro_bsz}_seq{seq_len}"
                    f"_mlp_rank0_act"
                )
                if attn_key in entries and mlp_key in entries:
                    attn_acts.append(float(entries[attn_key]))
                    mlp_acts.append(float(entries[mlp_key]))
        if not attn_acts or not mlp_acts:
            return None
        attn_avg = sum(attn_acts) / len(attn_acts)
        mlp_avg = sum(mlp_acts) / len(mlp_acts)
        if attn_avg <= 0 or mlp_avg <= 0:
            return None
        total = attn_avg + mlp_avg
        return attn_avg / total, mlp_avg / total

    def per_layer_act_alpha_beta(
        self, *, unit: Literal["attention", "mlp"],
        tp: int, ep: int, micro_bsz: int, seq_len: int,
        recompute: bool = False, sequence_parallel: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """OLS ``(α, β)`` for the per-layer activation MB of one
        component (``unit="attention"`` or ``"mlp"``) at the profiled
        ``micro_bsz``.

        Returns ``None`` when the three-pass profile hasn't been run
        at this shape (or under the requested recompute mode).
        """
        raw_path = self._raw_memory_path(tp, ep)
        if not os.path.isfile(raw_path):
            return None
        raw = self._load_json(raw_path)
        sp_marker = "_sp" if sequence_parallel else ""
        cpt_marker = "_c" if recompute else ""
        samples_by_n: Dict[int, List[float]] = {}
        for strategy_key, entries in raw.items():
            if not strategy_key.startswith("1_"):
                continue
            base = (
                strategy_key[:-3]
                if strategy_key.endswith("_sp") else strategy_key
            )
            ends_cpt = base.endswith("_c")
            if cpt_marker and not ends_cpt:
                continue
            if not cpt_marker and ends_cpt:
                continue
            ends_sp = strategy_key.endswith("_sp")
            if (sp_marker and not ends_sp) or (not sp_marker and ends_sp):
                continue
            expected_tail = (
                f"_bsz{micro_bsz}_seq{seq_len}_{unit}_rank0_act"
            )
            for key, value in entries.items():
                if not key.startswith("layernum["):
                    continue
                if not key.endswith(expected_tail):
                    continue
                try:
                    n = int(key[len("layernum["):key.index("]")])
                except ValueError:
                    continue
                samples_by_n.setdefault(n, []).append(float(value))
        if not samples_by_n:
            return None
        averaged = sorted(
            (float(n), sum(vals) / len(vals))
            for n, vals in samples_by_n.items()
        )
        if len(averaged) >= 2:
            mean_n = sum(n for n, _ in averaged) / len(averaged)
            mean_y = sum(y for _, y in averaged) / len(averaged)
            cov = sum((n - mean_n) * (y - mean_y) for n, y in averaged)
            var = sum((n - mean_n) ** 2 for n, _ in averaged)
            if var <= 0:
                return None
            beta = cov / var
            alpha = mean_y - beta * mean_n
        else:
            n, y = averaged[0]
            if n <= 0:
                return None
            alpha, beta = 0.0, y / n
        if beta <= 0:
            return None
        return alpha, beta

    def _raw_memory_path(self, tp: int, ep: int) -> str:
        """Path to the raw per-(tp, ep) memory profile JSON written by
        ``utils.save_profiled_memory``."""
        suffix = "" if (tp == 1 and ep == 1) else f"_tp{tp}_ep{ep}"
        return os.path.join(
            self.configs_dir,
            f"memory_profiling_{self.mixed_precision}_{self.model_name}"
            f"_seqlen{self.meta.get('max_position_embeddings', 4096)}"
            f"{suffix}.json",
        )
