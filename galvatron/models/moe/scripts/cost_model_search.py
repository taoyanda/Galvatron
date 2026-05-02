"""Brute-force config search: enumerate (pp, dp, tp, ep, dp_mode, fsep)
combinations, estimate each via :class:`PPCostModel`, sort, and report
the best.

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

Sorts by ``total_iter_ms`` ascending; ties broken by lower peak memory.
Top-K is printed with the breakdown source (``time_source`` /
``memory_source``) so the caller can tell which estimates came from a
runtime-profile lookup vs. analytical fall-back.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..")))

from galvatron.models.moe.cost_model import PPCostModel  # noqa: E402


def _divisors(value: int) -> List[int]:
    return [d for d in range(1, value + 1) if value % d == 0]


def enumerate_configs(
    num_gpus: int,
    num_layers: int,
    num_experts: int,
    dp_modes: List[str],
    fsep_modes: List[str],
) -> Iterator[Dict[str, Any]]:
    """Yield every (pp, dp, tp, ep, dp_mode, fsep) tuple that satisfies the
    structural constraints. Viability checks (memory budget, etc.) are
    applied by the caller."""
    for pp in _divisors(num_gpus):
        if num_layers % pp != 0:
            continue
        per_stage_world = num_gpus // pp
        for dp in _divisors(per_stage_world):
            for tp in _divisors(per_stage_world // dp):
                ep = per_stage_world // (dp * tp)
                if ep < 1 or ep > num_experts:
                    continue
                for dp_mode in dp_modes:
                    for fsep in fsep_modes:
                        if fsep == "on":
                            # FSEP requires tp × ep == per_stage_world
                            # AND ep | num_experts (so capacity_per_device
                            # is integer and ≥ 1, satisfying ep × cap ≥ E).
                            if tp * ep != per_stage_world:
                                continue
                            if num_experts % ep != 0:
                                continue
                        yield dict(pp=pp, dp=dp, tp=tp, ep=ep,
                                   dp_mode=dp_mode, fsep=fsep)


def estimate_one(
    cost_model: PPCostModel,
    config: Dict[str, Any],
    *,
    num_layers: int,
    num_gpus: int,
    global_bsz: int,
    seq_len: int,
    recompute: bool,
) -> Optional[Dict[str, Any]]:
    """Return the cost-model estimate for one config, or ``None`` if the
    micro-batch can't be evenly partitioned across the dp dimension. On
    estimator errors (missing profile, bad shape) returns a dict with an
    ``error`` key for the caller to report."""
    micro_bsz = max(1, global_bsz // config["dp"])
    if config["dp"] * micro_bsz != global_bsz:
        return None
    try:
        estimate = cost_model.estimate(
            num_layers=num_layers, num_gpus=num_gpus,
            dp=config["dp"], pp=config["pp"],
            tp=config["tp"], ep=config["ep"],
            micro_batch_size=micro_bsz, global_batch_size=global_bsz,
            seq_len=seq_len, sequence_parallel=True,
            zero_stage=2 if config["dp_mode"] == "zero2sdp" else 3,
            sdp=(config["dp_mode"] in ("zero2sdp", "zero3")),
            recompute=recompute, bwd_mult=2.0,
            fsep=(config["fsep"] == "on"),
        )
    except (ValueError, KeyError) as err:
        return {"cfg": config, "error": str(err)}
    breakdown = estimate.breakdown
    return {
        "cfg": config,
        "iter_ms": estimate.total_iter_ms,
        "peak_mb": estimate.peak_memory_mb,
        "params_mb": breakdown.get("parameters_mb", 0.0),
        "optim_mb": breakdown.get("optimizer_mb", 0.0),
        "act_mb": breakdown.get("activations_mb", 0.0),
        "time_source": breakdown.get("time_source", "?"),
        "memory_source": breakdown.get("memory_source", "?"),
    }


def _trust_keep(result: Dict[str, Any], trust: str) -> bool:
    """Return True if ``result`` passes the trust-source filter.

    - ``any``: no filter.
    - ``calibrated``: both time and memory must come from
      ``runtime_profile`` (rules out fully-analytical configs without a
      calibration anchor).
    - ``sample``: both must come from an exact-N profile sample
      (strictest — only configs with a real measurement at the requested
      ``num_layers`` survive).
    """
    if trust == "any":
        return True
    time_source = str(result.get("time_source", ""))
    memory_source = str(result.get("memory_source", ""))
    if trust == "calibrated":
        return (time_source.startswith("runtime_profile")
                and memory_source.startswith("runtime_profile"))
    if trust == "sample":
        return "+sample" in time_source and "+sample" in memory_source
    raise ValueError(f"unknown trust mode: {trust!r}")


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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(viable, infeasible)`` lists. ``viable`` is sorted by
    ``iter_ms`` ascending; ``infeasible`` collects everything that
    errored out, exceeded ``gpu_memory_mb``, or was filtered by
    ``trust``."""
    cost_model = PPCostModel(model_name)
    viable: List[Dict[str, Any]] = []
    infeasible: List[Dict[str, Any]] = []
    for config in enumerate_configs(
        num_gpus, num_layers, num_experts, list(dp_modes), list(fsep_modes)
    ):
        result = estimate_one(
            cost_model, config,
            num_layers=num_layers, num_gpus=num_gpus,
            global_bsz=global_bsz, seq_len=seq_len, recompute=recompute,
        )
        if result is None:
            continue
        if "error" in result:
            infeasible.append(result)
            continue
        if (gpu_memory_mb is not None
                and result["peak_mb"] > gpu_memory_mb):
            result["error"] = (
                f"OOM: peak_mb={result['peak_mb']:.0f} > budget {gpu_memory_mb:.0f}"
            )
            infeasible.append(result)
            continue
        if not _trust_keep(result, trust):
            result["error"] = f"filtered (trust={trust})"
            infeasible.append(result)
            continue
        viable.append(result)
    viable.sort(key=lambda row: (row["iter_ms"], row["peak_mb"]))
    return viable, infeasible


def _shorten_source(source: str) -> str:
    """Compact label for the time_source / memory_source breakdown field."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="mixtral-8x7b-e8k2")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--global-bsz", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--gpu-memory-mb", type=float, default=45000.0,
        help="OOM filter: drop configs with peak > this. Set to 0 to disable.",
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
    viable, infeasible = search(
        args.model, num_gpus=args.num_gpus, num_layers=args.num_layers,
        global_bsz=args.global_bsz, seq_len=args.seq_len,
        gpu_memory_mb=budget, num_experts=args.num_experts,
        trust=args.trust_source,
    )

    print(
        f"# Cost-model config search: {args.model}\n"
        f"#   num_gpus={args.num_gpus}  num_layers={args.num_layers}  "
        f"global_bsz={args.global_bsz}  seq_len={args.seq_len}  "
        f"gpu_budget={'disabled' if budget is None else f'{budget:.0f} MB'}  "
        f"trust={args.trust_source}\n"
        f"# {len(viable)} viable / {len(infeasible)} infeasible\n"
    )

    if not viable:
        print("no viable configurations found.")
        if infeasible and args.show_infeasible:
            print("\n## infeasible (top 10):")
            for result in infeasible[:10]:
                print(f"  {result['cfg']}  →  {result.get('error', '?')}")
        return

    header = (
        f"{'rk':>2} {'pp':>2} {'dp':>2} {'tp':>2} {'ep':>2} "
        f"{'dp_mode':>9} {'fsep':>4} | "
        f"{'iter_ms':>9} {'peak_mb':>9} | "
        f"{'params':>7} {'optim':>7} {'act':>7} | "
        f"{'time_src':>18} {'mem_src':>18}"
    )
    print(header)
    print("-" * len(header))
    for rank, result in enumerate(viable[:args.top_k], 1):
        config = result["cfg"]
        print(
            f"{rank:>2} {config['pp']:>2} {config['dp']:>2} "
            f"{config['tp']:>2} {config['ep']:>2} "
            f"{config['dp_mode']:>9} {config['fsep']:>4} | "
            f"{result['iter_ms']:>9.0f} {result['peak_mb']:>9.0f} | "
            f"{result['params_mb']:>7.0f} {result['optim_mb']:>7.0f} "
            f"{result['act_mb']:>7.0f} | "
            f"{_shorten_source(result['time_source']):>18} "
            f"{_shorten_source(result['memory_source']):>18}"
        )

    print()
    best = viable[0]
    best_config = best["cfg"]
    print(
        f"## Optimal: pp={best_config['pp']} dp={best_config['dp']} "
        f"tp={best_config['tp']} ep={best_config['ep']} "
        f"{best_config['dp_mode']} fsep={best_config['fsep']}"
    )
    print(
        f"   iter_ms={best['iter_ms']:.0f}  peak_mb={best['peak_mb']:.0f}  "
        f"(time_src={_shorten_source(best['time_source'])}, "
        f"mem_src={_shorten_source(best['memory_source'])})"
    )

    if args.show_infeasible and infeasible:
        print()
        print(f"## infeasible ({len(infeasible)} total, showing first 10):")
        for result in infeasible[:10]:
            config = result["cfg"]
            err = result.get("error", "?")[:60]
            print(
                f"  pp={config['pp']} dp={config['dp']} "
                f"tp={config['tp']} ep={config['ep']} "
                f"{config['dp_mode']} fsep={config['fsep']}: {err}"
            )


if __name__ == "__main__":
    main()
