"""Pipeline-parallel cost model (1F1B / PipeDream-Flush).

Composes per-stage costs from :class:`IntraCostModel` into a full-pipeline
cost. Pure orchestration; all real cost computation lives in
:class:`IntraCostModel`.

1F1B critical-path formula
--------------------------

Under 1F1B with ``pp`` uniform stages and ``n_micro`` microbatches, the
critical path of one training iteration is::

    pipeline_iter_ms = (n_micro + pp - 1) × max_stage_compute_ms

where ``max_stage_compute_ms`` is the slowest single-microbatch stage
compute time. Post-backward terms (DP all-reduce, EP all-to-all if not
already on the critical path, Adam) run in parallel across stages, so the
total iteration time is::

    total_iter_ms = pipeline_iter_ms + max_stage_post_bwd_ms

In-flight memory under 1F1B steady-state: stage ``k`` (1-indexed from the
first stage) holds ``min(pp - k + 1, n_micro)`` microbatches of activation
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
        **kwargs: Any,
    ) -> CostEstimate:
        """Estimate full-iteration cost under 1F1B.

        ``num_gpus`` describes the **total** allocation; per-stage GPU
        count is ``num_gpus // pp = dp × tp × ep``.
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
        if global_batch_size is None:
            global_batch_size = micro_batch_size * dp
        n_micro = global_batch_size // (dp * micro_batch_size)
        layers_per_stage = num_layers // pp
        num_gpus_per_stage = dp * tp * ep

        # ---------- Whole-pipeline runtime-profile shortcut ----------
        # When the runtime profile has an entry keyed by the per-stage
        # shape × this pp, with an α/β fit across num_layers, use the fit
        # to compute totals at the requested ``num_layers``. This captures
        # PP send/recv comm + bubble overhead + per-rank framework memory
        # under 1F1B that the per-stage compositional model misses.
        pp_key = (
            f"tp{tp}_ep{ep}_bsz{micro_batch_size}_seq{seq_len}"
            f"_fsep{'on' if fsep else 'off'}" + ("" if pp == 1 else f"_pp{pp}")
        )
        runtime_pp_entry = None
        if self.runtime_profile is not None:
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
            return CostEstimate(
                total_iter_ms=iter_ms_resolved,
                peak_memory_mb=peak_mb_resolved,
                breakdown={
                    "schedule": "1f1b",
                    "pp": pp,
                    "n_microbatches": float(n_micro),
                    "layers_per_stage": float(layers_per_stage),
                    "time_source": (
                        f"runtime_profile[{pp_key}]+{resolution_mode}"
                        f"(N_pts={num_data_points})"
                    ),
                    "memory_source": (
                        f"runtime_profile[{pp_key}]+{resolution_mode}"
                        f"(N_pts={num_data_points})"
                    ),
                    "parameters_mb": params_mb_resolved,
                    "optimizer_mb": optim_mb_resolved,
                    "activations_mb": activation_mb_resolved,
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
                has_embedding=True,
                has_lmhead=True,
                in_flight_microbatches=n_micro,
                **intra_kwargs,
            )
            breakdown = dict(stage.breakdown)
            breakdown.update(
                {
                    "pp": 1,
                    "n_microbatches": float(n_micro),
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
        # First stage holds pp microbatches in flight at 1F1B steady state
        # (capped by n_micro when there are fewer microbatches than stages).
        first_stage = self.intra.estimate(
            num_layers=layers_per_stage,
            has_embedding=True,
            has_lmhead=False,
            in_flight_microbatches=min(pp, n_micro),
            **intra_kwargs,
        )
        # Last stage holds 1 microbatch in flight under 1F1B.
        last_stage = self.intra.estimate(
            num_layers=layers_per_stage,
            has_embedding=False,
            has_lmhead=True,
            in_flight_microbatches=1,
            **intra_kwargs,
        )
        # Representative middle stage (when pp >= 3): holds ~pp/2 microbatches
        # at steady state. Skipped for pp == 2.
        middle_stage: Optional[CostEstimate] = None
        if pp >= 3:
            middle_stage = self.intra.estimate(
                num_layers=layers_per_stage,
                has_embedding=False,
                has_lmhead=False,
                in_flight_microbatches=max(1, pp // 2),
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
        pipeline_iter_ms = (n_micro + pp - 1) * bottleneck_compute_ms

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
            "n_microbatches": float(n_micro),
            "layers_per_stage": float(layers_per_stage),
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
