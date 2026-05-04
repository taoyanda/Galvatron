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
        pp_key = (
            f"tp{tp}_ep{ep}_bsz{per_rank_micro_bsz}_seq{seq_len}"
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
            # The shortcut anchors on a calibrated whole-iter measurement;
            # we don't have post-bwd terms broken out, so the best we can do
            # for ``stage_bottleneck_ms`` is the structural inverse of the
            # 1F1B critical path: iter_ms ≈ (num_microbatches + pp − 1) × bottleneck.
            # Under-counts post-bwd by absorbing it into the bottleneck;
            # documented in cost_model_guide.md.
            divisor = max(1, num_microbatches + pp - 1)
            stage_bottleneck_ms = iter_ms_resolved / divisor
            pipeline_iter_ms = stage_bottleneck_ms * divisor

            # ``num_stages_behind`` adds reserve activation memory on
            # top of the calibrated peak; the calibrated value didn't
            # see that reserve so we compute the delta analytically and
            # stack it. Time stays calibrated — this is a memory-only
            # knob.
            memory_source = (
                f"runtime_profile[{pp_key}]+{resolution_mode}"
                f"(N_pts={num_data_points})"
            )
            peak_mb_total = peak_mb_resolved
            activation_mb_total = activation_mb_resolved
            extra_reserve_mb = 0.0
            if reserve_active:
                # Peak stage under 1F1B is the first stage; it has
                # n_behind = pp − 1 ≥ 1 (active). The reserve is
                # ``num_stages_behind`` extra microbatches of full-stage
                # activation memory — uniform additive, no per-stage
                # scaling.
                per_microbatch_act_mb = self.intra.per_microbatch_activation_mb(
                    num_layers=layers_per_stage,
                    per_rank_micro_bsz=per_rank_micro_bsz,
                    seq_len=seq_len, tp=tp, recompute=recompute,
                    sequence_parallel=sequence_parallel,
                )
                extra_reserve_mb = num_stages_behind * per_microbatch_act_mb
                peak_mb_total += extra_reserve_mb
                activation_mb_total += extra_reserve_mb
                memory_source = (
                    memory_source + f"+num_stages_behind({num_stages_behind})"
                )

            return CostEstimate(
                total_iter_ms=iter_ms_resolved,
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
                    "num_stages_behind_extra_mb": extra_reserve_mb,
                    "time_source": (
                        f"runtime_profile[{pp_key}]+{resolution_mode}"
                        f"(N_pts={num_data_points})"
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
        # 1F1B steady-state in-flight count for stage k (1-indexed) is
        # ``pp − k + 1``; we use the first/middle/last representatives
        # below. Every stage additionally reserves ``num_stages_behind``
        # microbatches of activation memory — pure-additive uniform
        # reserve, applied to every stage including the last (its
        # natural n_behind is 0 but the reserve adds num_stages_behind
        # on top).
        def _in_flight(base: int) -> int:
            return base + num_stages_behind

        # First stage holds pp microbatches in flight at 1F1B steady state
        # (capped by num_microbatches when there are fewer microbatches than stages).
        first_stage = self.intra.estimate(
            has_embedding=True,
            has_lmhead=False,
            in_flight_microbatches=_in_flight(min(pp, num_microbatches)),
            **per_stage_split_kwargs,
            **intra_kwargs,
        )
        # Last stage holds 1 microbatch in flight under 1F1B; the
        # reserve still applies under the pure-additive rule.
        last_stage = self.intra.estimate(
            has_embedding=False,
            has_lmhead=True,
            in_flight_microbatches=_in_flight(1),
            **per_stage_split_kwargs,
            **intra_kwargs,
        )
        # Representative middle stage (when pp >= 3): holds ~pp/2
        # microbatches at steady state. Skipped for pp == 2.
        middle_stage: Optional[CostEstimate] = None
        if pp >= 3:
            middle_stage = self.intra.estimate(
                has_embedding=False,
                has_lmhead=False,
                in_flight_microbatches=_in_flight(max(1, pp // 2)),
                **per_stage_split_kwargs,
                **intra_kwargs,
            )

        stages = [("first", first_stage), ("last", last_stage)]
        if middle_stage is not None:
            stages.append(("middle", middle_stage))

        # Bottleneck stage compute (per-microbatch) drives the bubble math.
        bottleneck_name, bottleneck_compute_ms = max(
            ((name, stage.breakdown["stage_compute_ms"]) for name, stage in stages),
            key=lambda name_and_compute: name_and_compute[1],
        )
        pipeline_iter_ms = (num_microbatches + pp - 1) * bottleneck_compute_ms

        # Post-backward terms run in parallel across stages → take the max.
        max_post_bwd_ms = max(self._stage_post_bwd_ms(stage) for _, stage in stages)
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
        if middle_stage is not None:
            breakdown["middle_stage_compute_ms"] = middle_stage.breakdown[
                "stage_compute_ms"
            ]
            breakdown["middle_stage_peak_mb"] = middle_stage.peak_memory_mb

        return CostEstimate(
            total_iter_ms=total_iter_ms,
            peak_memory_mb=peak_memory_mb,
            breakdown=breakdown,
        )
