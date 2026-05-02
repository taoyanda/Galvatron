"""Abstract cost-model interface and shared result container.

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

Both implement :class:`ICostModel`. New cost models (e.g. for a different
schedule, or a different model family) should also implement this interface
so any downstream caller can substitute them.
"""

from __future__ import annotations

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


class ICostModel(ABC):
    """Abstract interface implemented by every cost model.

    Concrete classes load whatever profile artifacts they need at
    construction time (memory profile, runtime profile, optimizer-step
    profile, embedding/lm-head profile, network bandwidth, meta config),
    and answer ``estimate(...)`` queries about a particular
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
        """Return per-iteration time and per-rank peak memory.

        See :class:`PPCostModel.estimate` and :class:`IntraCostModel.estimate`
        for the meaning of each keyword.
        """
        ...
