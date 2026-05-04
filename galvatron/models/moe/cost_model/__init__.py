"""MoE cost-model package.

Public surface
--------------

- :class:`ICostModel` (``base``): abstract interface every cost model
  implements.
- :class:`CostEstimate` (``base``): full diagnostic return value, with
  per-layer / per-stage / per-component breakdown.
- :class:`CostQuery` (``base``): focused 3-metric façade
  (``iter_ms``, ``max_stage_ms``, ``peak_memory_mb``) plus provenance —
  the right shape for external callers.
- :class:`IntraCostModel` (``intra``): single-stage cost (``pp == 1``);
  consumes profile artifacts + runs the analytical fall-back.
- :class:`PPCostModel` (``pp``): 1F1B-aware orchestrator over
  :class:`IntraCostModel`; the recommended public class for general use.

Programmatic usage
------------------

The cost model holds its profile artifacts in memory once you construct
it, so external interfaces (planners, search loops, evaluation
harnesses) should build a :class:`PPCostModel` once and reuse it across
many ``query`` calls::

    from galvatron.models.moe.cost_model import PPCostModel

    cm = PPCostModel("mixtral-8x7b-e8k2")
    result = cm.query(
        num_layers=4, num_gpus=4,
        dp=1, pp=1, tp=1, ep=4,
        micro_batch_size=4, global_batch_size=4, seq_len=4096,
        sequence_parallel=True, zero_stage=2, sdp=True,
        recompute=True, bwd_mult=2.0, fsep=False,
    )
    print(result.iter_ms, result.max_stage_ms, result.peak_memory_mb)

For a one-shot CLI-style call where construction overhead doesn't
matter, :func:`query_cost` and :func:`estimate_cost` are
top-level wrappers that build the model on each call.

Backwards compatibility
-----------------------

Older callers import ``CostModel`` directly::

    >>> from galvatron.models.moe.cost_model import CostModel

This name is preserved as an alias for :class:`PPCostModel`, which
transparently delegates to :class:`IntraCostModel` when ``pp == 1``.
"""
from .base import CostEstimate, CostQuery, ICostModel
from .intra import IntraCostModel
from .pp import PPCostModel
from .search import MoESearcher, RankedSearch, SearchResult, enumerate_configs

# Public alias used by older scripts (cost_model_sweep.py, etc.).
CostModel = PPCostModel


def estimate_cost(
    model_name: str,
    *,
    num_layers: int,
    num_gpus: int,
    dp: int,
    pp: int,
    tp: int,
    ep: int = 1,
    **kwargs,
) -> CostEstimate:
    """One-shot wrapper around :meth:`PPCostModel.estimate`. Returns a
    :class:`CostEstimate` with the full diagnostic breakdown.

    Builds a fresh :class:`PPCostModel` each call (re-loading profile
    JSONs); if you're going to call it many times, construct
    :class:`PPCostModel` once and call ``cm.estimate(...)`` directly.
    """
    mixed_precision = kwargs.pop("mixed_precision", "bf16")
    return PPCostModel(model_name, mixed_precision=mixed_precision).estimate(
        num_layers=num_layers, num_gpus=num_gpus,
        dp=dp, pp=pp, tp=tp, ep=ep, **kwargs,
    )


def query_cost(
    model_name: str,
    *,
    num_layers: int,
    num_gpus: int,
    dp: int,
    pp: int,
    tp: int,
    ep: int = 1,
    **kwargs,
) -> CostQuery:
    """One-shot wrapper around :meth:`PPCostModel.query`. Returns a
    :class:`CostQuery` with the three primary metrics plus provenance.

    Equivalent to::

        PPCostModel(model_name).query(
            num_layers=..., num_gpus=..., dp=..., pp=..., tp=..., ep=..., ...
        )

    Same caveat as :func:`estimate_cost`: re-builds the model each call.
    """
    mixed_precision = kwargs.pop("mixed_precision", "bf16")
    return PPCostModel(model_name, mixed_precision=mixed_precision).query(
        num_layers=num_layers, num_gpus=num_gpus,
        dp=dp, pp=pp, tp=tp, ep=ep, **kwargs,
    )


__all__ = [
    "CostEstimate",
    "CostQuery",
    "ICostModel",
    "IntraCostModel",
    "PPCostModel",
    "CostModel",
    "estimate_cost",
    "query_cost",
    # Search layer over PPCostModel.
    "MoESearcher",
    "RankedSearch",
    "SearchResult",
    "enumerate_configs",
]
