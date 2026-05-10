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
from .intra import IntraCostModel, IntraCostModelMeasuredAct


class PPCostModel(ICostModel):
    """1F1B pipeline-parallel cost model.

    Wraps an :class:`IntraCostModel` (auto-constructed if not provided)
    and combines per-stage costs into a full-iteration estimate. For
    ``pp == 1`` this is just a passthrough to the wrapped model.

    The ``use_measured_memory_profile`` flag selects between two
    intra-stage variants when ``intra`` is auto-constructed:

      * ``False`` (default): instantiate :class:`IntraCostModel`
        directly. Per-layer params + embed/lm-head come from analytical
        formulas (model dimensions in ``meta``); per-component
        activation slope comes from ``chunks_overhead_profile`` (Step 8b
        per-component data) when available, with closed-form
        boundary-tensor fallback. Step 4 (``profile_memory.sh``) is not
        required for this path.
      * ``True``: instantiate :class:`IntraCostModelMeasuredAct`, which
        sources per-layer activation memory from Step 4's
        ``memory_profile`` JSON. Use this for parity comparisons or
        when working with environments that still have Step 4 data on
        disk. Raises ``FileNotFoundError`` at construction time if no
        ``memory_profiling_*.json`` is present.

    The flag is ignored if ``intra`` is provided directly (the caller
    has already chosen the variant).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        intra: Optional[IntraCostModel] = None,
        use_measured_memory_profile: bool = False,
        **intra_kwargs: Any,
    ):
        if intra is None:
            if model_name is None:
                raise ValueError(
                    "PPCostModel requires either ``intra`` or ``model_name``"
                )
            intra_cls = (
                IntraCostModelMeasuredAct if use_measured_memory_profile
                else IntraCostModel
            )
            intra = intra_cls(model_name, **intra_kwargs)
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
        # Shape key: ``tp{T}_ep{E}_micro_bsz{M}_seq{S}_fsep{ON|OFF}
        # [_pp{P}][_w{W}]`` — pp omitted when 1, world omitted when 4.
        # At pp > 1 we prefer the matching-world entry (PP=1 at
        # world=num_gpus/pp): same per-rank compute as a PP=k stage
        # but in saturation regime. Falls back to same-world entry,
        # then to per-stage compositional path.
        # Skipped for asymmetric queries (1:1 calibration only).
        DEFAULT_WORLD = 4

        def _shape_key(pp_eff: int, world_eff: int) -> str:
            key = (f"tp{tp}_ep{ep}_micro_bsz{micro_batch_size}"
                   f"_seq{seq_len}_fsep{'on' if fsep else 'off'}")
            if pp_eff != 1:
                key += f"_pp{pp_eff}"
            if world_eff != DEFAULT_WORLD:
                key += f"_w{world_eff}"
            return key

        def _entry(key: str) -> Optional[Dict]:
            if asymmetric or self.runtime_profile is None:
                return None
            return self.runtime_profile.get("by_shape", {}).get(key)

        pp_key = _shape_key(pp, num_gpus)
        runtime_pp_entry = _entry(pp_key)

        time_entry, time_source = runtime_pp_entry, pp_key
        if pp > 1:
            matching_key = _shape_key(1, num_gpus // pp)
            matching_entry = _entry(matching_key)
            if matching_entry and matching_entry.get("alpha_beta_fit"):
                time_entry, time_source = matching_entry, matching_key

        alpha_beta_fits = (runtime_pp_entry or {}).get("alpha_beta_fit") or {}
        time_alpha_beta_fits = (time_entry or {}).get("alpha_beta_fit") or {}
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

        # When predicting PP > 1 with matching-world calibration
        # available, override the time AND memory fields from the
        # matching-world α/β fits.
        #
        # Time override: α + β·N from ``fwd_bwd_ms`` and ``opt_ms``.
        # ``matching_world_alpha_beta`` also signals that the downstream
        # max_stage_compute formula should use α-on-last-stage
        # (``α + β·(N/pp)``) instead of the legacy ``sum/pp`` divisor.
        # Validation (validate_unseen_drift.py): PP=2 nl=12 drift drops
        # from +13% to ±4% across chunks ∈ {1, 2, 4}.
        #
        # Memory override: per-stage memory state (params, optimizer,
        # activation, cuda_peak) at PP=k 4-GPU stage equals
        # ``α/pp + β·(N/pp)`` where α and β come from PP=1 at world=4/pp
        # measurements. The α/pp split assumes embedding params (on
        # stage 0) equals lm-head params (on stage pp-1), which holds
        # exactly at vocab×hidden for our model. Validated within 1-2%
        # against direct PP=2 4-GPU measurement.
        matching_world_alpha_beta = None
        if pp > 1 and time_entry is not None and time_entry is not runtime_pp_entry:
            fb_fit = time_alpha_beta_fits.get("fwd_bwd_ms")
            opt_fit_match = time_alpha_beta_fits.get("opt_ms")
            if fb_fit and opt_fit_match:
                matching_world_alpha_beta = (fb_fit, opt_fit_match)
                sum_stage_at_N = fb_fit["alpha"] + fb_fit["beta"] * num_layers
                # Per-stage opt time: matching-world opt is for a PP=1
                # rank holding all N layers' optimizer state. At PP=k
                # each rank holds ~1/pp of that state (transformer
                # blocks split across stages; emb/lm-head per stage
                # adds back ~half of α-on-each-end roughly cancelled by
                # the loss). Divide by pp to get the per-rank wall-clock
                # opt time at PP>1.
                opt_at_N = (
                    opt_fit_match["alpha"] + opt_fit_match["beta"] * num_layers
                ) / pp
                iter_ms_resolved = sum_stage_at_N + opt_at_N
                opt_ms_resolved = opt_at_N

                # Memory override:
                #   - params_mb / optimizer_mb: α/pp + β·(N/pp). α here
                #     captures emb+lm-head footprint, which truly splits
                #     across stages (emb on stage 0, lm-head on stage k-1,
                #     equal param counts at vocab×hidden).
                #   - cuda_peak_mb / activation_peak_mb: α + β·(N/pp).
                #     α here is dominated by per-rank framework constants
                #     (FSDP all-gather workspace, expandable-segments
                #     slack, dispatcher buffers) that do NOT shrink with
                #     PP — every PP stage retains the full framework
                #     footprint. Splitting α/pp under-predicts cuda_peak
                #     by ~half of α at PP=2, which produced the OOM at
                #     gbsz=128 nl=24 PP=2 tp=2 ep=1 (predicted 49 GB,
                #     measured 77 GB → trainer OOM).
                # Falls back to same-world if a field isn't in the
                # matching-world fit.
                _SPLIT_ALPHA = {"params_mb", "optimizer_mb"}

                def _matching_world_memory(field: str) -> Optional[float]:
                    fit = time_alpha_beta_fits.get(field)
                    if not fit:
                        return None
                    alpha_term = fit["alpha"] / pp if field in _SPLIT_ALPHA else fit["alpha"]
                    return alpha_term + fit["beta"] * (num_layers / pp)

                for field, var_set in [
                    ("cuda_peak_mb", "peak"),
                    ("params_mb", "params"),
                    ("optimizer_mb", "optim"),
                    ("activation_peak_mb", "activation"),
                ]:
                    val = _matching_world_memory(field)
                    if val is not None:
                        if var_set == "peak":
                            peak_mb_resolved = val
                        elif var_set == "params":
                            params_mb_resolved = val
                        elif var_set == "optim":
                            optim_mb_resolved = val
                        elif var_set == "activation":
                            activation_mb_resolved = val

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
            # Analytical 1F1B critical-path formula:
            #
            #   pipeline_iter_ms = sum_stage_compute
            #                    + max_stage_compute × (num_microbatches − 1)
            #                    + optimizer_step_time
            #
            # ``sum_stage_compute`` = total fwd+bwd through ALL stages
            # for one microbatch (= ``fwd_bwd_ms`` at chunks=1 under
            # the calibration recipe, recovered here as
            # ``iter_ms_resolved − opt_ms_for_scaling``).
            # ``max_stage_compute`` = slowest stage's compute time
            # (uniform-layout approximation: sum_stage / pp).
            #
            # Empirically tested at nl ∈ {8, 12} PP ∈ {1, 2} chunks ∈
            # {1, 2, 4}: this aggregated slope-only path keeps drift
            # within ±13% across all axes (max |·| = 9.5% at nl=8,
            # 13.3% at nl=12). Per-component slope-only sourced from
            # ``unit_breakdown`` was tried and reverted because it
            # over-predicts at PP > 1 — canonical nl=4 calibration is
            # in the under-saturated regime at PP=2, while the
            # aggregated {nl=2, nl=4} slope is closer to the saturated
            # asymptote.
            sum_stage_compute_ms = max(
                0.0, iter_ms_resolved - opt_ms_for_scaling
            )
            if matching_world_alpha_beta is not None:
                # Matching-world calibration: max_stage = α + β·(N/pp).
                # Attributes the per-iter overhead α (embedding fwd+bwd
                # + lm-head fwd+bwd + dataloader/bookkeeping) entirely
                # to the last stage — lm-head's hidden×vocab matmul
                # dominates over the embedding gather, and the last
                # stage's max_stage_compute is what gates the chunks
                # scaling under 1F1B. β·(N/pp) is the per-stage
                # transformer-block portion (uniform layout).
                fb_fit, _ = matching_world_alpha_beta
                stage_bottleneck_ms = (
                    fb_fit["alpha"] + fb_fit["beta"] * (num_layers / pp)
                )
            else:
                # Same-world calibration fallback: assume balanced
                # per-stage compute (sum / pp).
                stage_bottleneck_ms = sum_stage_compute_ms / max(1, pp)
            pipeline_iter_ms = (
                sum_stage_compute_ms
                + stage_bottleneck_ms * max(0, num_microbatches - 1)
                + opt_ms_for_scaling
            )
            # Diagnostic-only: still surface the chunks_overhead slope
            # alongside the analytical prediction so cost_model_drift /
            # cost_model_pp_drift can compare. The chunks_overhead
            # profile remains useful for validating that the analytical
            # formula matches measured chunks=2 behavior at calibration N.
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
                        chunks_slope_source = (
                            f"chunks_overhead_observed["
                            f"{pp_key}/"
                            f"{cs_dp_mode if cs_dp_mode in per_dp else 'fallback'}]"
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
                        f"runtime_profile[{time_source}]+{resolution_mode}"
                        f"(N_pts={num_data_points})"
                        + ("+matching_world+α-on-last-stage"
                           if matching_world_alpha_beta is not None else "")
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
