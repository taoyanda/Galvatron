"""Single-table drift analysis: cost-model estimate vs real measurement.

Reads the estimate + real JSONs, prints a concise ASCII table with
absolute and percent drift on iter-time and CUDA peak memory.
Aggregate row at the bottom (median, mean, max abs %) makes it easy to
spot regressions when re-running after cost-model changes.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))


Key = Tuple[int, int, str, int, str]


def _key(c: Dict) -> Key:
    return (c["tp"], c["ep"], c["dp_mode"], c["global_bsz"], c.get("fsep", "off"))


def drift_pct(est: Optional[float], real: Optional[float]) -> Optional[float]:
    if est is None or real is None or real == 0:
        return None
    return 100.0 * (est - real) / real


def fmt(v: Optional[float], spec: str = "{:.0f}") -> str:
    return spec.format(v) if v is not None else "—"


def main() -> None:
    est_path = os.path.join(CONFIGS_DIR, "cost_model_estimate_4gpu_4layer.json")
    real_path = os.path.join(CONFIGS_DIR, "cost_model_real_4gpu_4layer.json")
    est: Dict[Key, Dict] = {_key(c): c for c in json.load(open(est_path))["configs"]}
    real: Dict[Key, Dict] = {_key(c): c for c in json.load(open(real_path))["configs"]}

    rows: List[Dict] = []
    for k in sorted(real.keys()):
        e = est.get(k, {})
        r = real[k]
        e_it, r_it = e.get("iter_ms"), r.get("iter_ms")
        e_mem, r_mem = e.get("peak_memory_mb"), r.get("cuda_peak_mb")
        rows.append({
            "key": k,
            "real_iter": r_it, "est_iter": e_it,
            "iter_delta": (e_it - r_it) if (e_it is not None and r_it is not None) else None,
            "iter_pct": drift_pct(e_it, r_it),
            "real_mem": r_mem, "est_mem": e_mem,
            "mem_delta": (e_mem - r_mem) if (e_mem is not None and r_mem is not None) else None,
            "mem_pct": drift_pct(e_mem, r_mem),
            "src": (e.get("time_source") or "?").replace("runtime_profile", "rt").replace(
                "computation_profile_forward_only", "fwd"
            ),
        })

    # Header
    hdr = (
        f"{'tp':>2} {'ep':>2} {'dp_mode':>9} {'bsz':>3} {'fsep':>4} | "
        f"{'real_iter':>9} {'est_iter':>9} {'Δ_ms':>7} {'Δ%':>6} | "
        f"{'real_mem':>9} {'est_mem':>9} {'Δ_mb':>7} {'Δ%':>6} | src"
    )
    print(hdr)
    print("-" * len(hdr))

    iter_pcts: List[float] = []
    mem_pcts: List[float] = []
    for r in rows:
        tp, ep, dpm, bsz, fsep = r["key"]
        src = r["src"]
        # Truncate src to a short tag.
        if src.startswith("rt"):
            src_short = "rt"
        elif src.startswith("fwd"):
            src_short = "fwd"
        else:
            src_short = src[:6]
        print(
            f"{tp:>2} {ep:>2} {dpm:>9} {bsz:>3} {fsep:>4} | "
            f"{fmt(r['real_iter'], '{:>9.0f}')} "
            f"{fmt(r['est_iter'],  '{:>9.0f}')} "
            f"{fmt(r['iter_delta'],'{:>+7.0f}')} "
            f"{fmt(r['iter_pct'],  '{:>+6.1f}')} | "
            f"{fmt(r['real_mem'],  '{:>9.0f}')} "
            f"{fmt(r['est_mem'],   '{:>9.0f}')} "
            f"{fmt(r['mem_delta'], '{:>+7.0f}')} "
            f"{fmt(r['mem_pct'],   '{:>+6.1f}')} | {src_short}"
        )
        if r["iter_pct"] is not None:
            iter_pcts.append(r["iter_pct"])
        if r["mem_pct"] is not None:
            mem_pcts.append(r["mem_pct"])

    print("-" * len(hdr))

    def _agg(pcts: List[float], label: str) -> str:
        if not pcts:
            return f"{label:<18s} —"
        absp = [abs(p) for p in pcts]
        return (
            f"{label:<18s} "
            f"median_signed={statistics.median(pcts):+.1f}%  "
            f"mean_abs={statistics.mean(absp):.1f}%  "
            f"max_abs={max(absp):.1f}%  "
            f"n={len(pcts)}"
        )

    print()
    print(_agg(iter_pcts, "iter-time drift:"))
    print(_agg(mem_pcts,  "cuda-peak drift:"))


if __name__ == "__main__":
    sys.exit(main())
