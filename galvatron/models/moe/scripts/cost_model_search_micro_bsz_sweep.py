"""Compare the optimal parallelization config across a small set of
micro-batch sizes.

For a fixed ``global_bsz``, runs ``MoESearcher.rank`` independently at each
``micro_bsz ∈ {1, 2, 4}`` (the values our calibration matrix covered) and
prints the top config per micro_bsz. The point: see how the optimal layout
shifts as ``num_microbatches = global_bsz // micro_bsz`` changes —
small ``micro_bsz`` favours pipeline parallelism (more microbatches, less
per-rank memory); large ``micro_bsz`` favours pure DP (fewer microbatches,
no pipeline bubble).

Feasibility per micro_bsz follows the standard rule
``per_rank_micro_bsz = micro_bsz / (dp × ep) >= 1`` — configs failing it
are filtered to the infeasible bucket. The cost-model loader and
``MoESearcher.score(micro_bsz=…)`` already enforce this, so we don't
re-implement it here.

Run inside the container so galvatron imports work:
    docker exec hetu python3 \\
      /root/Galvatron/galvatron/models/moe/scripts/cost_model_search_micro_bsz_sweep.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from galvatron.models.moe.cost_model.search import MoESearcher  # noqa: E402

DEFAULTS = {
    "model": "qwen-30b-a3b-e128k8",
    "num_gpus": 4,
    "num_layers": 4,
    "global_bsz": 128,
    "seq_len": 4096,
    "num_experts": 128,
    "gpu_memory_mb": 72657.0,
    "trust": "any",
    "micro_bsz_set": (1, 2, 4),
}


def _print_top(searcher: MoESearcher, args: argparse.Namespace,
               micro_bsz: int) -> None:
    """Run rank() at one micro_bsz, print the top-1 viable config (and a
    short infeasible reason summary so the reader sees why bigger DP/EP
    layouts dropped out)."""
    print()
    print(f"--- micro_bsz={micro_bsz}  num_microbatches="
          f"{args.global_bsz // micro_bsz} ---")
    ranked = searcher.rank(
        num_gpus=args.num_gpus,
        num_layers=args.num_layers,
        global_bsz=args.global_bsz,
        seq_len=args.seq_len,
        gpu_memory_mb=args.gpu_memory_mb,
        num_experts=args.num_experts,
        trust=args.trust,
        micro_bsz=micro_bsz,
    )
    viable = ranked.top(args.top_k)
    if not viable:
        print(f"  no viable configs at micro_bsz={micro_bsz} "
              f"(all {len(ranked.infeasible)} configs infeasible)")
        # Surface the dominant infeasible reason.
        reasons: dict = {}
        for r in ranked.infeasible:
            reasons.setdefault((r.error or "?")[:60], 0)
            reasons[(r.error or "?")[:60]] += 1
        for reason, count in sorted(reasons.items(),
                                    key=lambda kv: -kv[1])[:3]:
            print(f"    ({count}x) {reason}")
        return
    print(f"  {len(viable)} viable / {len(ranked.infeasible)} infeasible")
    print(f"  {'rk':>2} {'pp':>2} {'dp':>2} {'tp':>2} {'ep':>2} "
          f"{'dp_mode':>9} {'fsep':>4} | "
          f"{'iter_ms':>9} {'peak_mb':>8} | "
          f"{'time_src':<28} {'mem_src':<28}")
    print("  " + "-" * 116)
    for i, r in enumerate(viable, 1):
        cfg = r.cfg
        q = r.query
        # Time/memory source come from the cost-model query; trim to
        # readable widths.
        ts = (q.time_source or "")[:28]
        ms = (q.memory_source or "")[:28]
        print(f"  {i:>2} {cfg['pp']:>2} {cfg['dp']:>2} {cfg['tp']:>2} "
              f"{cfg['ep']:>2} {cfg['dp_mode']:>9} {cfg['fsep']:>4} | "
              f"{q.iter_ms:>9.0f} {q.peak_memory_mb:>8.0f} | "
              f"{ts:<28} {ms:<28}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--num-gpus", type=int, default=DEFAULTS["num_gpus"])
    parser.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    parser.add_argument("--global-bsz", type=int, default=DEFAULTS["global_bsz"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--num-experts", type=int, default=DEFAULTS["num_experts"])
    parser.add_argument("--gpu-memory-mb", type=float,
                        default=DEFAULTS["gpu_memory_mb"])
    parser.add_argument("--trust", default=DEFAULTS["trust"],
                        choices=["any", "calibrated", "sample"])
    parser.add_argument("--micro-bsz", type=int, nargs="+",
                        default=list(DEFAULTS["micro_bsz_set"]),
                        help="Micro batch sizes to compare (default: 1 2 4).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-k configs to print per micro_bsz.")
    args = parser.parse_args()

    searcher = MoESearcher(args.model)
    print(f"# micro_bsz comparison sweep: {args.model}")
    print(f"#   num_gpus={args.num_gpus}  num_layers={args.num_layers}  "
          f"global_bsz={args.global_bsz}  seq_len={args.seq_len}  "
          f"gpu_budget={args.gpu_memory_mb:.0f} MB  trust={args.trust}")
    print(f"#   micro_bsz set: {args.micro_bsz}")
    for mb in args.micro_bsz:
        if args.global_bsz % mb != 0:
            print(f"\n--- micro_bsz={mb}: skip (global_bsz % micro_bsz != 0) ---")
            continue
        _print_top(searcher, args, mb)


if __name__ == "__main__":
    main()
