"""MoE cost-model package.

  - :class:`ICostModel` (``base``): abstract interface every cost model
    implements.
  - :class:`CostEstimate` (``base``): standard return value.
  - :class:`IntraCostModel` (``intra``): single-stage cost (``pp == 1``);
    consumes profile artifacts + runs the analytical fall-back.
  - :class:`PPCostModel` (``pp``): 1F1B-aware orchestrator over
    :class:`IntraCostModel`; the recommended public class for general use.

Backwards compatibility
-----------------------

Older callers import ``CostModel`` directly:

    >>> from galvatron.models.moe.cost_model import CostModel

This name is preserved as an alias for :class:`PPCostModel`, which
transparently delegates to :class:`IntraCostModel` when ``pp == 1``.
"""
from .base import CostEstimate, ICostModel
from .intra import IntraCostModel
from .pp import PPCostModel

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
    """One-shot wrapper around :class:`PPCostModel`. See
    :meth:`PPCostModel.estimate` for the full kwargs."""
    mixed_precision = kwargs.pop("mixed_precision", "bf16")
    return PPCostModel(model_name, mixed_precision=mixed_precision).estimate(
        num_layers=num_layers, num_gpus=num_gpus,
        dp=dp, pp=pp, tp=tp, ep=ep, **kwargs,
    )


__all__ = [
    "CostEstimate",
    "ICostModel",
    "IntraCostModel",
    "PPCostModel",
    "CostModel",
    "estimate_cost",
]
