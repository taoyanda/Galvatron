"""Pipeline-parallel cost model (1F1B / PipeDream-Flush).

Composes per-stage costs from :class:`IntraCostModel` into a full-pipeline
cost. Pure orchestration; all real cost computation lives in
:class:`IntraCostModel`.

1F1B critical-path formula
--------------------------

Under 1F1B with ``pp`` uniform stages and ``num_microbatches`` microbatches, the
critical path of one training iteration is::

    pipeline_iter_ms = (num_microbatches + pp - 1) × max_stage_compute_ms

where ``max_stage_compute_ms`` is the slowest single-microbatch stage
compute time. Post-backward terms (DP all-reduce, EP all-to-all if not
already on the critical path, Adam) run in parallel across stages, so the
total iteration time is::

    total_iter_ms = pipeline_iter_ms + max_stage_post_bwd_ms

In-flight memory under 1F1B steady-state: stage ``k`` (1-indexed from the
first stage) holds ``min(pp - k + 1, num_microbatches)`` microbatches of activation
in flight. The first stage is the memory hot-spot.

Stage placement of embedding / lm-head:
  - first stage: embedding (no lm-head)
  - last stage:  lm-head (no embedding)
  - middle:      neither
"""

from __future__ import annotations

from typing import Any, Optional

from .base import CostEstimate, ICostModel
from .intra import IntraCostModel


class PPCostModel(ICostModel):
    """1F1B pipeline-parallel cost model.

    Wraps an :class:`IntraCostModel` (auto-constructed if not provided)
    and combines per-stage costs into a full-iteration estimate. For
    ``pp == 1`` this is just a passthrough to the wrapped model.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        intra: Optional[IntraCostModel] = None,
        **intra_kwargs: Any,
    ):
        if intra is None:
            if model_name is None:
                raise ValueError(
                    "PPCostModel requires either ``intra`` or ``model_name``"
                )
            intra = IntraCostModel(model_name, **intra_kwargs)
        self.intra = intra
        # Re-expose loaded profiles for convenience (so callers can read
        # e.g. ``cm.runtime_profile`` without reaching through ``cm.intra``).
        self.model_name = self.intra.model_name
        self.mixed_precision = self.intra.mixed_precision
        self.runtime_profile = self.intra.runtime_profile
        self.optimizer_step_profile = self.intra.optimizer_step_profile
        self.embedding_lmhead_profile = self.intra.embedding_lmhead_profile
        self.memory_profile = self.intra.memory_profile
        self.meta = self.intra.meta
        self.network = self.intra.network
        self.optimizer_to_params_ratio = self.intra.optimizer_to_params_ratio

    @staticmethod
    def _stage_post_bwd_ms(estimate: CostEstimate) -> float:
        """Return the post-backward portion of this stage's iter_ms — the
        part that runs in parallel across pipeline stages after the bubble
        empties (DP all-reduce + EP all-to-all + Adam step)."""
        breakdown = estimate.breakdown
        return float(
            breakdown.get("dp_allreduce_ms", 0.0)
            + breakdown.get("ep_alltoall_ms", 0.0)
            + breakdown.get("opt_step_ms", 0.0)
        )

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
        # Asymmetric layer counts; default to symmetric (= num_layers).
        # Both must be divisible by pp; per-stage we get
        # num_attention_layers // pp attention sublayers and
        # num_expert_layers // pp expert sublayers (uniform layout).
        num_attention_layers: Optional[int] = None,
        num_expert_layers: Optional[int] = None,
        # Extra activation reserve hyperparameter — see docstring below.
        num_stages_behind: int = 0,
        **kwargs: Any,
    ) -> CostEstimate:
        """Estimate full-iteration cost under 1F1B.

        ``num_gpus`` describes the **total** allocation; per-stage GPU
        count is ``num_gpus // pp = dp × tp × ep``.

        Asymmetric layer counts: pass ``num_attention_layers`` and/or
        ``num_expert_layers`` to project a hypothetical layout where the
        block isn't 1:1. Both must divide pp evenly; the runtime-profile
        shortcut (anchored on 1:1 calibrations) is skipped under
        asymmetric queries and the per-stage composition runs instead.

        ``num_stages_behind`` is a hyperparameter for "how many extra
        microbatches of activation memory each stage reserves for
        downstream stages." The reserve is **pure-additive**: every
        stage's effective in-flight count becomes
        ``(pp − k + 1) + num_stages_behind`` — uniform across all
        stages including the last one, and even at ``pp == 1`` (the
        single stage gets ``+num_stages_behind`` extra microbatches).

        At ``num_stages_behind == 0`` (default) the cost model behaves
        exactly as today.

        Use cases:

          - Modelling framework-specific buffer overheads (Megatron's
            send/recv keep-alive buffers, FSDP per-stage scratch, etc.)
            that aren't captured by the analytical in-flight count.
          - Probing the search space — "what if every stage held one
            more microbatch's worth of activations? What optima get
            filtered out by the OOM check?"

        ``num_stages_behind`` is a memory-only knob — ``iter_ms`` and
        ``max_stage_ms`` stay anchored on the calibrated runtime
        profile when available; only the activation reserve delta is
        added analytically.
        """
        if dp * pp * tp * ep != num_gpus:
            raise ValueError(
                f"dp({dp})*pp({pp})*tp({tp})*ep({ep})={dp*pp*tp*ep} "
                f"!= num_gpus({num_gpus})"
            )
        if num_layers % pp != 0:
            raise ValueError(
                f"num_layers ({num_layers}) must be divisible by pp ({pp})"
            )
        if num_attention_layers is None:
            num_attention_layers = num_layers
        if num_expert_layers is None:
            num_expert_layers = num_layers
        if num_attention_layers % pp != 0 or num_expert_layers % pp != 0:
            raise ValueError(
                f"num_attention_layers ({num_attention_layers}) and "
                f"num_expert_layers ({num_expert_layers}) must each be "
                f"divisible by pp ({pp}). Heterogeneous per-stage layouts "
                f"are not yet supported."
            )
        if num_stages_behind < 0:
            raise ValueError(
                f"num_stages_behind={num_stages_behind} must be ≥ 0 "
                f"(count of extra microbatches to reserve)"
            )
        asymmetric = (num_attention_layers != num_expert_layers)
        # ``num_stages_behind`` is a pure-additive count: every stage's
        # effective n_behind is ``natural_n_behind + num_stages_behind``,
        # i.e. every stage (including the last and the single pp == 1
        # stage) reserves +num_stages_behind microbatches of activation
        # memory on top of its standard 1F1B in-flight count.
        reserve_active = (num_stages_behind > 0)
        # ``micro_batch_size`` is the GLOBAL microbatch size; both DP
        # and EP shard the data dim, so per-actual-rank sample count is
        # ``micro_batch_size // (dp * ep)``. Validate divisibility and
        # the ≥1-sample-per-rank constraint here — IntraCostModel re-
        # validates per-stage but PPCostModel's runtime-profile shortcut
        # path also needs the per-rank value to form lookup keys.
        if global_batch_size is None:
            global_batch_size = micro_batch_size
        if dp * ep > micro_batch_size:
            raise ValueError(
                f"dp*ep ({dp}*{ep}={dp*ep}) > micro_batch_size "
                f"({micro_batch_size}); per-rank batch < 1 sample."
            )
        if micro_batch_size % (dp * ep) != 0:
            raise ValueError(
                f"micro_batch_size ({micro_batch_size}) must be divisible "
                f"by dp*ep ({dp*ep})."
            )
        per_rank_micro_bsz = micro_batch_size // (dp * ep)
        num_microbatches = global_batch_size // micro_batch_size
        layers_per_stage = num_layers // pp
        attn_layers_per_stage = num_attention_layers // pp
        expert_layers_per_stage = num_expert_layers // pp
        num_gpus_per_stage = dp * tp * ep

        # ---------- Whole-pipeline runtime-profile shortcut ----------
        # When the runtime profile has an entry keyed by the per-stage
        # shape × this pp, with an α/β fit across num_layers, use the fit
        # to compute totals at the requested ``num_layers``. This captures
        # PP send/recv comm + bubble overhead + per-rank framework memory
        # under 1F1B that the per-stage compositional model misses.
        # Skipped under asymmetric queries: the calibration was 1:1, so
        # extrapolating the whole-iter ms by num_layers can't tell us
        # what would happen with extra/fewer expert layers. We route
        # through the per-stage composition instead.
        # Shape key: (tp, ep, micro_bsz, seq_len, fsep[, pp]). ``micro_bsz``
        # is the per-PP-stage compute batch size, matching the cost model's
        # ``micro_batch_size`` argument (= trainer's
        # global_train_batch_size at chunks=1 calibration). DP and EP are
        # feasibility guards only (DP*EP ≤ micro_bsz), not key dimensions.
        pp_key = (
            f"tp{tp}_ep{ep}_micro_bsz{micro_batch_size}_seq{seq_len}"
            f"_fsep{'on' if fsep else 'off'}" + ("" if pp == 1 else f"_pp{pp}")
        )
        runtime_pp_entry = None
        if not asymmetric and self.runtime_profile is not None:
            runtime_pp_entry = self.runtime_profile.get("by_shape", {}).get(pp_key)
        alpha_beta_fits = (runtime_pp_entry or {}).get("alpha_beta_fit") or {}
        # Two-tier lookup: if the requested ``num_layers`` matches one of
        # the profiled samples exactly, use the sample directly (avoids
        # the OLS noise of fitting potentially non-linear data — iter_ms
        # in particular grows super-linearly with N, so a line through
        # three N points has ~20 % residuals at each point). Otherwise
        # extrapolate via the α/β fit when ≥ 2 N points are available.
        samples = (runtime_pp_entry or {}).get("samples_by_num_layers") or []
        exact_sample = next(
            (s for s in samples if int(s.get("num_layers", -1)) == num_layers),
            None,
        )

        def _from_alpha_beta_fit(field: str) -> Optional[float]:
            fit = alpha_beta_fits.get(field)
            if not fit:
                return None
            return fit["alpha"] + fit["beta"] * num_layers

        def _resolve(field: str) -> Optional[float]:
            if exact_sample is not None and exact_sample.get(field) is not None:
                return float(exact_sample[field])
            return _from_alpha_beta_fit(field)

        iter_ms_resolved = _resolve("iter_ms")
        opt_ms_resolved = _resolve("opt_ms")
        peak_mb_resolved = _resolve("cuda_peak_mb")
        params_mb_resolved = _resolve("params_mb")
        optim_mb_resolved = _resolve("optimizer_mb")
        activation_mb_resolved = _resolve("activation_peak_mb")

        if (
            iter_ms_resolved is not None
            and peak_mb_resolved is not None
            and params_mb_resolved is not None
            and optim_mb_resolved is not None
            and activation_mb_resolved is not None
        ):
            resolution_mode = "sample" if exact_sample is not None else "alpha_beta_fit"
            num_data_points = (
                len(samples)
                if exact_sample is None
                else exact_sample.get("n_dp_samples", 1)
            )
            iter_ms_fit = alpha_beta_fits.get("iter_ms", {})
            cuda_peak_fit = alpha_beta_fits.get("cuda_peak_mb", {})
            # Alpa-style 1F1B critical path
            # (https://arxiv.org/pdf/2201.12023):
            #
            #   T_pipeline = bottleneck × (num_mb − 1) + Σ_stages stage_compute
            #   T_iter = T_pipeline + opt_step
            #
            # ``cost_model_real_test.sh`` runs with chunks=1 + global_bsz
            # tuned so num_microbatches_at_calibration == 1, so the
            # calibrated ``iter_ms_resolved`` is exactly
            # ``Σ_stages_compute + opt_ms`` (the (num_mb − 1) × bottleneck
            # term vanishes). That gives us Σ stage_compute directly:
            #
            #   sum_stages = iter_ms_cal − opt_ms_cal
            #
            # Bottleneck is recovered by assuming a uniform layout (the
            # only layout the calibration matrix covers): bottleneck =
            # sum_stages / pp. Under the symmetric per-stage layout
            # this is exact; an asymmetric query routes through the
            # analytical path instead (which predicts each stage
            # individually).
            #
            # Pre-fix this branch did a divide-then-multiply by the same
            # divisor, collapsing to ``pipeline_iter_ms = iter_ms_resolved``
            # — i.e. the predicted iter time was independent of query
            # ``num_microbatches``, which under-predicted any query with
            # num_mb > 1.
            opt_ms_for_scaling = (opt_ms_resolved
                                  if opt_ms_resolved is not None else 0.0)
            sum_stage_compute_ms_cal = max(
                0.0, iter_ms_resolved - opt_ms_for_scaling
            )
            stage_bottleneck_ms = sum_stage_compute_ms_cal / max(1, pp)
            # Empirical per-microbatch slope from the chunks=1 vs chunks=2
            # calibration pair, when available for this shape + dp_mode.
            # Bundles the analytical bottleneck term (which the Alpa
            # formula gets right at chunks=1) with the per-microbatch
            # overhead the analytical path misses (sync grad reduce, PP
            # send/recv, scheduler — see Phase 0f). Validation at
            # chunks=32 closed the gap from +34% to ~−2%.
            chunks_slope_ms: Optional[float] = None
            chunks_slope_source = ""
            if self.intra.chunks_overhead_profile is not None:
                cs_entry = (
                    self.intra.chunks_overhead_profile
                    .get("by_shape", {}).get(pp_key)
                )
                cs_dp_mode = "zero3" if zero_stage == 3 else "zero2sdp"
                if cs_entry is not None:
                    per_dp = cs_entry.get("per_dp_mode") or {}
                    cs_block = per_dp.get(cs_dp_mode) or next(
                        iter(per_dp.values()), None
                    )
                    if cs_block is not None and cs_block.get(
                        "time_per_extra_microbatch_ms"
                    ) is not None:
                        chunks_slope_ms = float(
                            cs_block["time_per_extra_microbatch_ms"]
                        )
                        chunks_slope_source = (
                            f"chunks_overhead[{pp_key}/"
                            f"{cs_dp_mode if cs_dp_mode in per_dp else 'fallback'}]"
                        )

            if chunks_slope_ms is not None:
                # Use the empirical slope directly (it already includes
                # bottleneck × 1 plus the per-microbatch overhead Alpa
                # misses).
                pipeline_iter_ms = (
                    iter_ms_resolved
                    + chunks_slope_ms * max(0, num_microbatches - 1)
                )
            else:
                # Fall back to the Alpa analytical path: bottleneck
                # recovered by assuming a uniform layout (the only layout
                # the calibration matrix covers).
                pipeline_iter_ms = (
                    stage_bottleneck_ms * max(0, num_microbatches - 1)
                    + sum_stage_compute_ms_cal
                    + opt_ms_for_scaling
                )

            # Two activation-reserve contributions on top of the
            # calibrated peak:
            #
            #   (a) Natural 1F1B stacking delta. The calibration was
            #       captured at ``num_microbatches=1`` (chunks=1 +
            #       global_bsz tuned so global_bsz / dp / micro_bsz == 1
            #       in cost_model_real_test.sh). At runtime, the
            #       bottleneck stage holds
            #       ``min(pp, num_microbatches)`` microbatches in flight,
            #       so any delta above the calibration's 1 must be
            #       analytically reserved on top of the calibrated peak.
            #       This is what makes the cost model PP-depth-aware
            #       even for shapes whose calibrated entry was profiled
            #       at lower microbatch concurrency than the request.
            #
            #   (b) The user-supplied ``num_stages_behind`` knob — a
            #       pure-additive count layered on top of (a).
            #
            # Time stays calibrated — these are memory-only adjustments.
            NUM_MICROBATCHES_AT_CALIBRATION = 1
            natural_n_behind = max(
                0,
                min(pp, num_microbatches) - NUM_MICROBATCHES_AT_CALIBRATION,
            )
            total_extra_microbatches = natural_n_behind + num_stages_behind
            memory_source = (
                f"runtime_profile[{pp_key}]+{resolution_mode}"
                f"(N_pts={num_data_points})"
            )
            peak_mb_total = peak_mb_resolved
            activation_mb_total = activation_mb_resolved
            extra_reserve_mb = 0.0
            if total_extra_microbatches > 0:
                per_microbatch_act_mb = self.intra.per_microbatch_activation_mb(
                    num_layers=layers_per_stage,
                    per_rank_micro_bsz=per_rank_micro_bsz,
                    seq_len=seq_len, tp=tp, recompute=recompute,
                    sequence_parallel=sequence_parallel,
                )
                extra_reserve_mb = (
                    total_extra_microbatches * per_microbatch_act_mb
                )
                peak_mb_total += extra_reserve_mb
                activation_mb_total += extra_reserve_mb
                tag_parts: List[str] = []
                if natural_n_behind > 0:
                    tag_parts.append(f"natural_n_behind({natural_n_behind})")
                if num_stages_behind > 0:
                    tag_parts.append(f"num_stages_behind({num_stages_behind})")
                memory_source = memory_source + "+" + "+".join(tag_parts)

            return CostEstimate(
                # ``total_iter_ms`` reports the predicted wall-clock
                # iteration time at the QUERY's num_microbatches, not the
                # calibration's. Pre-fix this was pinned to
                # ``iter_ms_resolved`` (the calibrated num_mb=1 value),
                # making the headline number invariant under
                # num_microbatches scaling. We now report the scaled
                # ``pipeline_iter_ms`` so callers see the right value.
                total_iter_ms=pipeline_iter_ms,
                peak_memory_mb=peak_mb_total,
                breakdown={
                    "schedule": "1f1b",
                    "pp": pp,
                    "num_microbatches": float(num_microbatches),
                    "layers_per_stage": float(layers_per_stage),
                    "stage_compute_ms": stage_bottleneck_ms,
                    "stage_bottleneck_ms": stage_bottleneck_ms,
                    "pipeline_iter_ms": pipeline_iter_ms,
                    "bottleneck_stage": "single" if pp == 1 else "uniform",
                    "memory_stage": "single" if pp == 1 else "first",
                    # Layout — at the shortcut path we're on the
                    # calibrated 1:1 anchor so attn == expert == num_layers.
                    "num_attention_layers": float(num_attention_layers),
                    "num_expert_layers": float(num_expert_layers),
                    "asymmetric": asymmetric,
                    "num_stages_behind": int(num_stages_behind),
                    "natural_n_behind": int(natural_n_behind),
                    "num_stages_behind_extra_mb": extra_reserve_mb,
                    "time_source": (
                        f"runtime_profile[{pp_key}]+{resolution_mode}"
                        f"(N_pts={num_data_points})"
                        + (f"+{chunks_slope_source}" if chunks_slope_source else "")
                    ),
                    "memory_source": memory_source,
                    "parameters_mb": params_mb_resolved,
                    "optimizer_mb": optim_mb_resolved,
                    "activations_mb": activation_mb_total,
                    "model_states_mb": params_mb_resolved + optim_mb_resolved,
                    "iter_ms_alpha": iter_ms_fit.get("alpha"),
                    "iter_ms_beta": iter_ms_fit.get("beta"),
                    "peak_mb_alpha": cuda_peak_fit.get("alpha"),
                    "peak_mb_beta": cuda_peak_fit.get("beta"),
                },
            )

        intra_kwargs = dict(
            num_gpus=num_gpus_per_stage,
            dp=dp,
            pp=1,
            tp=tp,
            ep=ep,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            seq_len=seq_len,
            gpus_per_node=gpus_per_node,
            recompute=recompute,
            zero_stage=zero_stage,
            sdp=sdp,
            sequence_parallel=sequence_parallel,
            bwd_mult=bwd_mult,
            fsep=fsep,
        )

        # ---------- pp == 1: degenerate to single-stage ----------
        if pp == 1:
            stage = self.intra.estimate(
                num_layers=num_layers,
                num_attention_layers=num_attention_layers,
                num_expert_layers=num_expert_layers,
                has_embedding=True,
                has_lmhead=True,
                in_flight_microbatches=num_microbatches,
                **intra_kwargs,
            )
            breakdown = dict(stage.breakdown)
            breakdown.update(
                {
                    "pp": 1,
                    "num_microbatches": float(num_microbatches),
                    "pipeline_iter_ms": stage.breakdown["stage_compute_ms"],
                    "stage_bottleneck_ms": stage.breakdown["stage_compute_ms"],
                    "memory_stage": "single",
                    "schedule": "1f1b",
                }
            )
            return CostEstimate(
                total_iter_ms=stage.total_iter_ms,
                peak_memory_mb=stage.peak_memory_mb,
                breakdown=breakdown,
            )

        # ---------- pp >= 2 ----------
        # Each stage gets ``attn_layers_per_stage`` attention sublayers and
        # ``expert_layers_per_stage`` expert sublayers (uniform layout —
        # heterogeneous per-stage layouts are rejected at the divisibility
        # check above).
        per_stage_split_kwargs = dict(
            num_layers=layers_per_stage,
            num_attention_layers=attn_layers_per_stage,
            num_expert_layers=expert_layers_per_stage,
        )
        # Per-stage prediction. 1F1B steady-state holds ``pp − i``
        # microbatches in flight at stage ``i`` (0-indexed), capped by
        # ``num_microbatches`` when there are fewer microbatches than
        # stages. Every stage additionally reserves ``num_stages_behind``
        # microbatches of activation memory — pure-additive uniform
        # reserve, applied to every stage including the last (its
        # natural n_behind is 0 but the reserve adds num_stages_behind
        # on top).
        #
        # We compute every stage individually rather than first/middle/
        # last representatives, so the bottleneck and Σ stage_compute
        # are exact under any per-stage layout (including the asymmetric
        # layouts the Phase-2 search probes).
        def _in_flight_at(stage_idx: int) -> int:
            return min(pp - stage_idx, num_microbatches) + num_stages_behind

        stage_estimates: List[CostEstimate] = []
        for stage_idx in range(pp):
            stage_estimates.append(self.intra.estimate(
                has_embedding=(stage_idx == 0),
                has_lmhead=(stage_idx == pp - 1),
                in_flight_microbatches=_in_flight_at(stage_idx),
                **per_stage_split_kwargs,
                **intra_kwargs,
            ))
        stages = [(f"stage_{i}", est) for i, est in enumerate(stage_estimates)]
        first_stage = stage_estimates[0]
        last_stage = stage_estimates[-1]

        # Bottleneck stage = max compute across ALL stages (not just
        # representative ones).
        bottleneck_name, bottleneck_compute_ms = max(
            ((name, est.breakdown["stage_compute_ms"]) for name, est in stages),
            key=lambda name_and_compute: name_and_compute[1],
        )

        # Alpa-style 1F1B critical path
        # (https://arxiv.org/pdf/2201.12023, eq. for pipeline latency):
        #
        #   T_pipeline = bottleneck × (num_mb − 1) + Σ_stages stage_compute
        #
        # The Σ term covers warmup (the first microbatch traversing every
        # stage's forward+backward path) and cooldown (the last microbatch
        # returning through every stage); the (num_mb − 1) × bottleneck
        # term is the steady-state cost of pumping the remaining
        # microbatches through the bottleneck stage.
        #
        # Equivalent to ``(num_mb + pp − 1) × bottleneck`` ONLY when
        # stages are uniform (Σ_stages = pp × bottleneck). Under
        # asymmetric per-stage layer counts, the Σ-form gives the
        # correct critical path; the uniform shortcut would over- or
        # under-count.
        sum_stage_compute_ms = sum(
            est.breakdown["stage_compute_ms"] for est in stage_estimates
        )
        pipeline_iter_ms = (
            bottleneck_compute_ms * max(0, num_microbatches - 1)
            + sum_stage_compute_ms
        )

        # Post-backward terms run in parallel across stages → take the max.
        max_post_bwd_ms = max(self._stage_post_bwd_ms(est) for est in stage_estimates)
        total_iter_ms = pipeline_iter_ms + max_post_bwd_ms

        # Memory: peak across stages. First stage usually wins under 1F1B.
        peak_stage_name, peak_stage = max(
            stages, key=lambda name_and_stage: name_and_stage[1].peak_memory_mb
        )
        peak_memory_mb = peak_stage.peak_memory_mb

        # Aggregate breakdown.
        breakdown: dict = {
            "schedule": "1f1b",
            "pp": pp,
            "num_microbatches": float(num_microbatches),
            "layers_per_stage": float(layers_per_stage),
            "num_attention_layers": float(num_attention_layers),
            "num_expert_layers": float(num_expert_layers),
            "asymmetric": asymmetric,
            "attention_layers_per_stage": float(attn_layers_per_stage),
            "expert_layers_per_stage": float(expert_layers_per_stage),
            "num_stages_behind": int(num_stages_behind),
            "pipeline_iter_ms": pipeline_iter_ms,
            "stage_bottleneck_ms": bottleneck_compute_ms,
            "bottleneck_stage": bottleneck_name,
            "max_post_bwd_ms": max_post_bwd_ms,
            "memory_stage": peak_stage_name,
            "peak_stage_memory_mb": peak_memory_mb,
            "first_stage_compute_ms": first_stage.breakdown["stage_compute_ms"],
            "last_stage_compute_ms": last_stage.breakdown["stage_compute_ms"],
            "first_stage_peak_mb": first_stage.peak_memory_mb,
            "last_stage_peak_mb": last_stage.peak_memory_mb,
            # Per-stage diagnostics — one entry per PP stage, in stage order.
            "per_stage_compute_ms": [
                est.breakdown["stage_compute_ms"] for est in stage_estimates
            ],
            "per_stage_peak_mb": [
                est.peak_memory_mb for est in stage_estimates
            ],
            "sum_stage_compute_ms": sum_stage_compute_ms,
            "time_source": first_stage.breakdown.get("time_source", "?"),
            "memory_source": peak_stage.breakdown.get("memory_source", "?"),
            # Memory categories from the peak stage (so totals are
            # consistent with peak_memory_mb).
            "parameters_mb": peak_stage.breakdown["parameters_mb"],
            "optimizer_mb": peak_stage.breakdown["optimizer_mb"],
            "activations_mb": peak_stage.breakdown["activations_mb"],
            "model_states_mb": peak_stage.breakdown["model_states_mb"],
            # Comm / opt are stage-local and uniform across stages here.
            "dp_allreduce_ms": peak_stage.breakdown.get("dp_allreduce_ms", 0.0),
            "ep_alltoall_ms": peak_stage.breakdown.get("ep_alltoall_ms", 0.0),
            "opt_step_ms": peak_stage.breakdown.get("opt_step_ms", 0.0),
            "per_layer_ms": peak_stage.breakdown.get("per_layer_ms", 0.0),
        }
        return CostEstimate(
            total_iter_ms=total_iter_ms,
            peak_memory_mb=peak_memory_mb,
            breakdown=breakdown,
        )
