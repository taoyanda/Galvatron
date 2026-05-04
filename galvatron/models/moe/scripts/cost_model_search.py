"""Brute-force config search CLI.

Thin command-line wrapper around :class:`MoESearcher`. The actual
search logic — enumeration, scoring, filtering, ranking — lives in
:mod:`galvatron.models.moe.cost_model.search`. External callers
(planners, design-space probes, "compose results across partial
models" loops) should import :class:`MoESearcher` directly instead of
calling this script's CLI.

Usage::

    docker exec hetu python3 scripts/cost_model_search.py
    docker exec hetu python3 scripts/cost_model_search.py --num-gpus 8 --num-layers 32

Filters infeasible combinations:

  - ``dp × pp × tp × ep != num_gpus``
  - ``num_layers % pp != 0``
  - ``global_bsz % (dp × chunks) != 0``
  - FSEP-on requires ``tp × ep == per_stage_world`` AND ``ep | num_experts``
  - ``peak_memory > gpu_memory_mb`` (OOM filter; can be disabled by
    passing ``gpu_memory_mb=0``).

Sorts by ``iter_ms`` ascending; ties broken by lower peak memory.
Top-K is printed with the breakdown source (``time_source`` /
``memory_source``) so the caller can tell which estimates came from a
runtime-profile lookup vs. analytical fall-back.

Back-compat shims
-----------------

The historical module-level helpers ``search``, ``estimate_one``, and
``enumerate_configs`` are still importable from this module; they
delegate to :class:`MoESearcher` so older imports keep working. New
code should use :class:`MoESearcher` directly.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..")))

from galvatron.models.moe.cost_model import (  # noqa: E402
    MoESearcher,
    PPCostModel,
    RankedSearch,
    SearchResult,
    enumerate_configs as _enumerate_configs,
)


# ---------------------------------------------------------------------------
# Back-compat module-level API
# ---------------------------------------------------------------------------
# Older code (e.g. cost_model_search_compare.py) imports ``search`` from
# this module and consumes ``viable`` / ``infeasible`` lists of dicts.
# We delegate to MoESearcher and project the dataclasses back to dicts
# so those callers don't need to change. New code should import
# MoESearcher directly.


def enumerate_configs(
    num_gpus: int,
    num_layers: int,
    num_experts: int,
    dp_modes: List[str],
    fsep_modes: List[str],
):
    """Back-compat alias for :func:`cost_model.enumerate_configs`."""
    return _enumerate_configs(
        num_gpus=num_gpus, num_layers=num_layers,
        num_experts=num_experts,
        dp_modes=dp_modes, fsep_modes=fsep_modes,
    )


def _result_to_dict(result: SearchResult) -> Dict[str, Any]:
    """Project a :class:`SearchResult` to the legacy result-dict shape
    historically returned by ``estimate_one``."""
    query = result.query
    if query is None:
        # Errored result — only ``cfg`` and ``error`` are meaningful.
        return {"cfg": result.cfg, "error": result.error}
    return {
        "cfg": result.cfg,
        "query": query,
        "iter_ms": query.iter_ms,
        "peak_memory_mb": query.peak_memory_mb,
        "peak_mb": query.peak_memory_mb,  # legacy alias
        "max_stage_ms": query.max_stage_ms,
        "bottleneck_stage": query.bottleneck_stage,
        "memory_stage": query.memory_stage,
        "num_attention_layers": query.num_attention_layers,
        "num_expert_layers": query.num_expert_layers,
        "asymmetric": query.asymmetric,
        "params_mb": query.breakdown.get("parameters_mb", 0.0),
        "optim_mb": query.breakdown.get("optimizer_mb", 0.0),
        "act_mb": query.breakdown.get("activations_mb", 0.0),
        "time_source": query.time_source,
        "memory_source": query.memory_source,
        # Errors only set on infeasible projections.
        **({"error": result.error} if result.error else {}),
    }


def estimate_one(
    cost_model: PPCostModel,
    config: Dict[str, Any],
    *,
    num_layers: int,
    num_gpus: int,
    global_bsz: int,
    seq_len: int,
    recompute: bool,
    num_attention_layers: Optional[int] = None,
    num_expert_layers: Optional[int] = None,
    num_stages_behind: int = 0,
    micro_bsz: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Back-compat shim. Returns the legacy result-dict shape. New code
    should use :meth:`MoESearcher.score` directly, which returns a
    :class:`SearchResult` dataclass.
    """
    searcher = MoESearcher(cost_model=cost_model)
    result = searcher.score(
        config,
        num_layers=num_layers, num_gpus=num_gpus,
        global_bsz=global_bsz, seq_len=seq_len, recompute=recompute,
        num_attention_layers=num_attention_layers,
        num_expert_layers=num_expert_layers,
        num_stages_behind=num_stages_behind,
        micro_bsz=micro_bsz,
    )
    if result.error == "micro_bsz × dp != global_bsz":
        # Historical contract: the inner sweep returns ``None`` (rather
        # than an error dict) for this specific filter so the caller
        # can skip silently.
        return None
    return _result_to_dict(result)


def search(
    model_name: str,
    *,
    num_gpus: int,
    num_layers: int,
    global_bsz: int,
    seq_len: int = 4096,
    gpu_memory_mb: Optional[float] = None,
    num_experts: int = 8,
    dp_modes: Tuple[str, ...] = ("zero2sdp", "zero3"),
    fsep_modes: Tuple[str, ...] = ("on", "off"),
    recompute: bool = True,
    trust: str = "any",
    num_attention_layers: Optional[int] = None,
    num_expert_layers: Optional[int] = None,
    num_stages_behind: int = 0,
    micro_bsz: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Back-compat shim. Returns ``(viable, infeasible)`` lists of
    legacy-shape result dicts. New code should use
    :meth:`MoESearcher.rank`, which returns a :class:`RankedSearch`."""
    searcher = MoESearcher(model_name)
    ranked = searcher.rank(
        num_gpus=num_gpus, num_layers=num_layers,
        global_bsz=global_bsz, seq_len=seq_len,
        gpu_memory_mb=gpu_memory_mb, num_experts=num_experts,
        dp_modes=dp_modes, fsep_modes=fsep_modes,
        recompute=recompute, trust=trust,
        num_attention_layers=num_attention_layers,
        num_expert_layers=num_expert_layers,
        num_stages_behind=num_stages_behind,
        micro_bsz=micro_bsz,
    )
    return (
        [_result_to_dict(r) for r in ranked.viable],
        [_result_to_dict(r) for r in ranked.infeasible],
    )


# ---------------------------------------------------------------------------
# CLI presentation helpers
# ---------------------------------------------------------------------------


def _shorten_source(source: str) -> str:
    """Compact label for the time_source / memory_source fields."""
    compact = (
        source.replace("runtime_profile[", "rt[")
              .replace("alpha_beta_fit", "α/β")
              .replace("linear_extrapolation", "lin")
              .replace("computation_profile_forward_only", "fwd")
              .replace("analytical_stage_memory", "anal")
    )
    if "+" in compact:
        compact = compact.rsplit("+", 1)[1]
    return compact[:18]


def _print_top_k(ranked: RankedSearch, top_k: int) -> None:
    header = (
        f"{'rk':>2} {'pp':>2} {'dp':>2} {'tp':>2} {'ep':>2} "
        f"{'dp_mode':>9} {'fsep':>4} | "
        f"{'iter_ms':>9} {'max_stg':>9} {'peak_mb':>9} "
        f"{'bot':>6} {'mem':>6} | "
        f"{'params':>7} {'optim':>7} {'act':>7} | "
        f"{'time_src':>18} {'mem_src':>18}"
    )
    print(header)
    print("-" * len(header))
    for rank, result in enumerate(ranked.top(top_k), 1):
        cfg = result.cfg
        query = result.query
        breakdown = query.breakdown
        print(
            f"{rank:>2} {cfg['pp']:>2} {cfg['dp']:>2} "
            f"{cfg['tp']:>2} {cfg['ep']:>2} "
            f"{cfg['dp_mode']:>9} {cfg['fsep']:>4} | "
            f"{query.iter_ms:>9.0f} "
            f"{query.max_stage_ms:>9.0f} "
            f"{query.peak_memory_mb:>9.0f} "
            f"{query.bottleneck_stage[:6]:>6} "
            f"{query.memory_stage[:6]:>6} | "
            f"{breakdown.get('parameters_mb', 0.0):>7.0f} "
            f"{breakdown.get('optimizer_mb', 0.0):>7.0f} "
            f"{breakdown.get('activations_mb', 0.0):>7.0f} | "
            f"{_shorten_source(query.time_source):>18} "
            f"{_shorten_source(query.memory_source):>18}"
        )


def _run_asymmetry_sweep(
    searcher: MoESearcher, args: argparse.Namespace,
    budget: Optional[float],
) -> None:
    """Sweep ``n_expert ∈ [num_layers + LO .. num_layers + HI]`` at fixed
    ``n_attn = num_layers``; print one summary row per point with the
    best config's iter_ms / max_stage_ms / peak_memory_mb."""
    lo, hi = args.asymmetry_range
    n_attn = args.num_layers
    effective_micro_bsz = (
        args.micro_bsz if args.micro_bsz is not None else args.global_bsz
    )
    micro_label = (
        ""
        if effective_micro_bsz == args.global_bsz
        else f"  micro_bsz={effective_micro_bsz}"
    )
    print(
        f"# Asymmetric-layer sweep: {searcher.model_name}\n"
        f"#   num_gpus={args.num_gpus}  n_attn={n_attn} "
        f"n_expert ∈ [{n_attn + lo} .. {n_attn + hi}]  "
        f"global_bsz={args.global_bsz}{micro_label}  "
        f"seq_len={args.seq_len}  trust={args.trust_source}\n"
    )
    header = (
        f"{'n_attn':>6} {'n_exp':>6} | {'pp':>2} {'dp':>2} {'tp':>2} {'ep':>2} "
        f"{'dp_mode':>9} {'fsep':>4} | "
        f"{'iter_ms':>9} {'max_stg':>9} {'peak_mb':>9} | mem_src"
    )
    print(header)
    print("-" * len(header))
    for delta in range(lo, hi + 1):
        n_exp = n_attn + delta
        if n_exp < 0:
            continue
        ranked = searcher.rank(
            num_gpus=args.num_gpus, num_layers=args.num_layers,
            global_bsz=args.global_bsz, seq_len=args.seq_len,
            gpu_memory_mb=budget, num_experts=args.num_experts,
            trust=args.trust_source,
            num_attention_layers=n_attn,
            num_expert_layers=n_exp,
            num_stages_behind=args.num_stages_behind,
            micro_bsz=args.micro_bsz,
        )
        if not ranked.viable:
            print(f"{n_attn:>6} {n_exp:>6} | (no viable configs)")
            continue
        best = ranked.best
        cfg = best.cfg
        print(
            f"{n_attn:>6} {n_exp:>6} | "
            f"{cfg['pp']:>2} {cfg['dp']:>2} "
            f"{cfg['tp']:>2} {cfg['ep']:>2} "
            f"{cfg['dp_mode']:>9} {cfg['fsep']:>4} | "
            f"{best.iter_ms:>9.0f} {best.max_stage_ms:>9.0f} "
            f"{best.peak_memory_mb:>9.0f} | "
            f"{_shorten_source(best.query.memory_source)}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="mixtral-8x7b-e8k2")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--num-attention-layers", type=int, default=None,
        help="Override the attention sublayer count (default: --num-layers). "
             "Diverging from --num-expert-layers requires a per-component "
             "computation profile at the queried (tp, ep, micro_bsz, seq).",
    )
    parser.add_argument(
        "--num-expert-layers", type=int, default=None,
        help="Override the MoE sublayer count (default: --num-layers). "
             "FSEP overhead and EP all-to-all volume scale with this count "
             "under the uniform-FSEP rule.",
    )
    parser.add_argument(
        "--asymmetry-range", type=int, nargs=2, metavar=("LO", "HI"),
        default=None,
        help="Sweep n_expert ∈ [num_layers + LO .. num_layers + HI] at "
             "fixed n_attn = num_layers. Each value runs a full search "
             "and reports the best config for that expert-layer count.",
    )
    parser.add_argument("--global-bsz", type=int, default=4)
    parser.add_argument(
        "--micro-bsz", type=int, default=None,
        help="Global microbatch size — total samples per "
             "forward-backward step. Defaults to --global-bsz "
             "(num_microbatches=1, single-step physics matching the "
             "calibration sweep). Set < --global-bsz to model "
             "multi-microbatch pipelines (num_microbatches="
             "global_bsz/micro_bsz > 1), giving pp>1 configs a non-"
             "degenerate (n_micro + pp − 1) × stage critical path. "
             "Must divide --global-bsz; per-config configs where "
             "dp × ep > micro_bsz go to infeasible.",
    )
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--gpu-memory-mb", type=float, default=45000.0,
        help="OOM filter: drop configs with peak > this. Set to 0 to disable.",
    )
    parser.add_argument(
        "--num-stages-behind", type=int, default=0, metavar="INT",
        help="Hyperparameter (count): extra microbatches of activation "
             "memory each stage reserves on top of the standard 1F1B "
             "in-flight count. Pure-additive — every stage's effective "
             "in-flight becomes (pp − k + 1) + num_stages_behind, "
             "uniform across all stages including the last and at "
             "pp == 1. Default 0 (calibrated baseline). Useful for "
             "modelling framework buffer overhead or probing OOM-margin-"
             "sensitive optima.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--show-infeasible", action="store_true")
    parser.add_argument(
        "--trust-source", choices=["any", "calibrated", "sample"], default="any",
        help="Filter configs by the source of their cost estimate. "
             "`any` ranks all configs (analytical fall-backs included); "
             "`calibrated` keeps only configs whose time + memory both come "
             "from a runtime-profile lookup; `sample` is strictest — only "
             "exact-N profile samples.",
    )
    args = parser.parse_args()

    budget = args.gpu_memory_mb if args.gpu_memory_mb > 0 else None
    searcher = MoESearcher(args.model)

    # Asymmetry sweep: one ranking per n_expert point.
    if args.asymmetry_range is not None:
        _run_asymmetry_sweep(searcher, args, budget)
        return

    ranked = searcher.rank(
        num_gpus=args.num_gpus, num_layers=args.num_layers,
        global_bsz=args.global_bsz, seq_len=args.seq_len,
        gpu_memory_mb=budget, num_experts=args.num_experts,
        trust=args.trust_source,
        num_attention_layers=args.num_attention_layers,
        num_expert_layers=args.num_expert_layers,
        num_stages_behind=args.num_stages_behind,
        micro_bsz=args.micro_bsz,
    )

    n_attn = (args.num_attention_layers
              if args.num_attention_layers is not None else args.num_layers)
    n_exp = (args.num_expert_layers
             if args.num_expert_layers is not None else args.num_layers)
    asymmetric = n_attn != n_exp
    layer_label = (
        f"num_layers={args.num_layers}" if not asymmetric
        else f"num_attention_layers={n_attn}  num_expert_layers={n_exp}"
    )
    reserve_label = (
        "" if args.num_stages_behind == 0
        else f"  num_stages_behind={args.num_stages_behind}"
    )
    effective_micro_bsz = (
        args.micro_bsz if args.micro_bsz is not None else args.global_bsz
    )
    micro_label = (
        ""
        if effective_micro_bsz == args.global_bsz
        else f"  micro_bsz={effective_micro_bsz} "
             f"(num_microbatches={args.global_bsz // effective_micro_bsz})"
    )

    print(
        f"# Cost-model config search: {args.model}\n"
        f"#   num_gpus={args.num_gpus}  {layer_label}  "
        f"global_bsz={args.global_bsz}{micro_label}  seq_len={args.seq_len}  "
        f"gpu_budget={'disabled' if budget is None else f'{budget:.0f} MB'}  "
        f"trust={args.trust_source}{reserve_label}\n"
        f"# {len(ranked.viable)} viable / {len(ranked.infeasible)} infeasible\n"
    )

    if not ranked.viable:
        print("no viable configurations found.")
        if ranked.infeasible and args.show_infeasible:
            print("\n## infeasible (top 10):")
            for result in ranked.infeasible[:10]:
                print(f"  {result.cfg}  →  {result.error or '?'}")
        return

    _print_top_k(ranked, args.top_k)

    print()
    best = ranked.best
    cfg = best.cfg
    print(
        f"## Optimal: pp={cfg['pp']} dp={cfg['dp']} "
        f"tp={cfg['tp']} ep={cfg['ep']} "
        f"{cfg['dp_mode']} fsep={cfg['fsep']}"
    )
    print(
        f"   iter_ms={best.iter_ms:.0f}  "
        f"max_stage_ms={best.max_stage_ms:.0f}  "
        f"peak_memory_mb={best.peak_memory_mb:.0f}  "
        f"(bottleneck={best.query.bottleneck_stage}, "
        f"mem_stage={best.query.memory_stage}, "
        f"time_src={_shorten_source(best.query.time_source)}, "
        f"mem_src={_shorten_source(best.query.memory_source)})"
    )

    if args.show_infeasible and ranked.infeasible:
        print()
        print(f"## infeasible ({len(ranked.infeasible)} total, "
              f"showing first 10):")
        for result in ranked.infeasible[:10]:
            cfg = result.cfg
            err = (result.error or "?")[:60]
            print(
                f"  pp={cfg['pp']} dp={cfg['dp']} "
                f"tp={cfg['tp']} ep={cfg['ep']} "
                f"{cfg['dp_mode']} fsep={cfg['fsep']}: {err}"
            )


if __name__ == "__main__":
    main()
