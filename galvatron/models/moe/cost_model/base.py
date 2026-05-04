"""Abstract cost-model interface and shared result containers.

The MoE cost model is split into two layers:

  - :class:`IntraCostModel` (in ``intra.py``): cost of running ``num_layers``
    transformer layers on a *single pipeline stage* — compute, DP comm,
    EP all-to-all, Adam step, memory. Knows nothing about cross-stage
    pipelining; ``pp`` must equal 1.

  - :class:`PPCostModel` (in ``pp.py``): wraps an :class:`IntraCostModel`
    and assembles the full-pipeline cost under the 1F1B (PipeDream-Flush)
    schedule for ``pp >= 1``. For ``pp == 1`` it delegates straight through;
    for ``pp > 1`` it computes per-stage costs (first/middle/last) and
    composes via the standard ``(n_micro + pp - 1) × bottleneck_stage``
    critical-path formula.

Both implement :class:`ICostModel`, which exposes two query entry points:

  - :meth:`ICostModel.estimate` returns a full :class:`CostEstimate`
    with the diagnostic breakdown — used by drift scripts, regression
    tests, and the cost-model itself for stage composition.
  - :meth:`ICostModel.query` returns a focused :class:`CostQuery` with
    just the three primary metrics (iter_ms, max_stage_ms,
    peak_memory_mb) plus provenance — the right shape for callers
    (search drivers, planners, external interfaces) that don't need
    to dig into the breakdown.

New cost models (e.g. for a different schedule, or a different model
family) implement this interface so downstream callers can substitute
them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CostEstimate:
    """Standard return shape of :meth:`ICostModel.estimate`.

    - ``total_iter_ms``: end-to-end iteration time on the bottleneck rank.
    - ``peak_memory_mb``: per-rank memory high-water mark.
    - ``breakdown``: free-form diagnostic dictionary; concrete cost models
      add their own keys (``per_layer_ms``, ``dp_allreduce_ms``,
      ``parameters_mb``, ``time_source``, etc.). Useful for sanity-checking
      and for the drift / comparison scripts.

    For most callers :class:`CostQuery` (the focused 3-metric façade
    returned by :meth:`ICostModel.query`) is the right type to consume;
    :class:`CostEstimate` exposes the full diagnostic breakdown.
    """

    total_iter_ms: float
    peak_memory_mb: float
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        bd = ", ".join(
            f"{k}={v:.1f}" if isinstance(v, (int, float)) else f"{k}={v!r}"
            for k, v in self.breakdown.items()
        )
        return (
            f"CostEstimate(total_iter_ms={self.total_iter_ms:.1f}, "
            f"peak_memory_mb={self.peak_memory_mb:.1f}, [{bd}])"
        )


@dataclass
class CostQuery:
    """Focused query result: just the three primary cost metrics plus
    enough provenance to reason about confidence.

    Returned by :meth:`ICostModel.query`. Use this when the caller is a
    planner / search loop / external interface that needs the headline
    numbers without unpacking :attr:`CostEstimate.breakdown`.

    Primary metrics:

    - ``iter_ms``: end-to-end iteration time on the bottleneck rank
      (1F1B critical path + post-bwd terms).
    - ``max_stage_ms``: per-microbatch cost of the slowest pipeline
      stage. At ``pp == 1`` this is the single stage's compute; at
      ``pp > 1`` the bubble formula gives
      ``iter_ms ≈ (n_micro + pp − 1) × max_stage_ms + post_bwd``.
      A high ``max_stage_ms / iter_ms`` ratio at large ``pp`` flags a
      bubble-bound config; a low ratio at small ``pp`` flags
      comm/opt-bound.
    - ``peak_memory_mb``: per-rank memory high-water mark.

    Provenance:

    - ``time_source`` / ``memory_source``: where each estimate came from
      (``runtime_profile[...]+sample(...)``,
      ``computation_profile_forward_only``,
      ``analytical_stage_memory[asymmetric]``, etc.). Useful for the
      ``--trust-source`` filter in the search driver.
    - ``bottleneck_stage`` / ``memory_stage``: which pipeline stage
      drove the bottleneck (``first`` / ``middle`` / ``last`` /
      ``single`` / ``uniform``).

    Asymmetric layout (when ``num_attention_layers != num_expert_layers``):

    - ``num_attention_layers`` / ``num_expert_layers`` / ``asymmetric``:
      the layout this query was scored against. Defaults to ``num_layers``
      for both with ``asymmetric=False`` for the standard 1:1 case.

    The full diagnostic breakdown is still available via
    :attr:`breakdown` for callers that want the per-layer / per-stage /
    per-component drill-down.
    """

    iter_ms: float
    max_stage_ms: float
    peak_memory_mb: float

    bottleneck_stage: str = "single"
    memory_stage: str = "single"
    time_source: str = "?"
    memory_source: str = "?"

    num_attention_layers: int = 0
    num_expert_layers: int = 0
    asymmetric: bool = False

    breakdown: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_estimate(
        cls,
        estimate: CostEstimate,
        *,
        num_attention_layers: Optional[int] = None,
        num_expert_layers: Optional[int] = None,
    ) -> "CostQuery":
        """Build a :class:`CostQuery` from a :class:`CostEstimate`.

        ``num_attention_layers`` / ``num_expert_layers`` are read from
        ``estimate.breakdown`` first (the cost model populates them);
        the kwargs are a fall-back for the symmetric back-compat case
        where the breakdown carries no explicit asymmetric counts.
        """
        breakdown = estimate.breakdown
        max_stage_ms = breakdown.get(
            "stage_bottleneck_ms",
            breakdown.get("stage_compute_ms", math.nan),
        )
        n_attn_breakdown = breakdown.get("num_attention_layers")
        n_exp_breakdown = breakdown.get("num_expert_layers")
        n_attn = int(
            n_attn_breakdown if n_attn_breakdown is not None
            else (num_attention_layers or 0)
        )
        n_exp = int(
            n_exp_breakdown if n_exp_breakdown is not None
            else (num_expert_layers or 0)
        )
        asymmetric = bool(breakdown.get("asymmetric", n_attn != n_exp))
        return cls(
            iter_ms=float(estimate.total_iter_ms),
            max_stage_ms=float(max_stage_ms),
            peak_memory_mb=float(estimate.peak_memory_mb),
            bottleneck_stage=str(breakdown.get("bottleneck_stage", "single")),
            memory_stage=str(breakdown.get("memory_stage", "single")),
            time_source=str(breakdown.get("time_source", "?")),
            memory_source=str(breakdown.get("memory_source", "?")),
            num_attention_layers=n_attn,
            num_expert_layers=n_exp,
            asymmetric=asymmetric,
            breakdown=dict(breakdown),
        )

    def __repr__(self) -> str:
        layout = (
            f"num_layers={self.num_attention_layers}"
            if not self.asymmetric
            else (
                f"num_attention_layers={self.num_attention_layers}, "
                f"num_expert_layers={self.num_expert_layers}"
            )
        )
        return (
            f"CostQuery(iter_ms={self.iter_ms:.1f}, "
            f"max_stage_ms={self.max_stage_ms:.1f}, "
            f"peak_memory_mb={self.peak_memory_mb:.1f}, "
            f"{layout}, "
            f"bottleneck={self.bottleneck_stage}, "
            f"mem_stage={self.memory_stage})"
        )


class ICostModel(ABC):
    """Abstract interface implemented by every cost model.

    Concrete classes load whatever profile artifacts they need at
    construction time (memory profile, runtime profile, optimizer-step
    profile, embedding/lm-head profile, network bandwidth, meta config),
    and answer ``estimate(...)`` / ``query(...)`` calls about a particular
    ``(num_layers, num_gpus, dp, pp, tp, ep, micro_batch_size, seq_len, ...)``
    parallelization config.

    Implementations may accept additional kwargs (e.g. ``has_embedding``
    on :class:`IntraCostModel`) but must accept the keyword arguments
    listed in :meth:`estimate` below.
    """

    @abstractmethod
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
        """Return per-iteration time and per-rank peak memory plus the
        full diagnostic breakdown.

        See :class:`PPCostModel.estimate` and :class:`IntraCostModel.estimate`
        for the meaning of each keyword.
        """

    def query(self, **kwargs: Any) -> CostQuery:
        """Return the focused :class:`CostQuery` (iter_ms, max_stage_ms,
        peak_memory_mb + provenance) for one parallelization config.

        Thin wrapper over :meth:`estimate`; ``**kwargs`` are forwarded
        verbatim. Subclasses can override for cheaper paths but the
        default delegation is the recommended implementation.

        This is the entry point external interfaces (search drivers,
        planners, evaluation harnesses) should call. ``estimate(...)``
        is the lower-level surface for callers that need the full
        diagnostic breakdown.
        """
        estimate = self.estimate(**kwargs)
        return CostQuery.from_estimate(
            estimate,
            num_attention_layers=kwargs.get("num_attention_layers"),
            num_expert_layers=kwargs.get("num_expert_layers"),
        )
