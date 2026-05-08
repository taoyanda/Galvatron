"""Alpha-beta validation: peak_memory_mb(N) = alpha + beta × N.

Compares the cost model's linear-in-num_layers extrapolation against real
measurements at the chosen calibration shape, reads each
``cost_model_real_*_nl<N>.log`` produced by ``cost_model_real_test.sh``
when invoked with ``NUM_LAYERS=<N>``, fits an OLS line through the real
points, and tabulates predicted vs measured at every N.

Run after a sweep:
    for N in 2 6 8; do
        NUM_LAYERS=$N bash scripts/cost_model_real_test.sh 1 4 zero2sdp 4 off
    done
    docker exec hetu python3 scripts/cost_model_alpha_beta.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(_HERE, ".."))
LOG_DIR = os.path.join(MODEL_DIR, "logs")
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..")))

from galvatron.models.moe.cost_model import CostModel  # noqa: E402

# Validation shape: simplest one with a fully-calibrated runtime profile
# entry. Same (tp, ep, dp_mode, bsz, fsep) used in cost_model_real_test.sh.
SHAPE = dict(tp=1, ep=4, dp_mode="zero2sdp", global_bsz=4, fsep="off")
SEQ_LEN = 4096
NUM_GPUS = 4

_REAL_OPT_RE = re.compile(
    r"\[real_measure\] optimizer_mb=([\d.]+)\s+activation_peak_mb=([\d.]+)\s+cuda_peak_mb=([\d.]+)"
)


def _log_path(num_layers: int) -> str:
    s = SHAPE
    base = f"cost_model_real_tp{s['tp']}_ep{s['ep']}_{s['dp_mode']}_bsz{s['global_bsz']}_fsep{s['fsep']}"
    if num_layers == 4:
        return os.path.join(LOG_DIR, f"{base}.log")
    return os.path.join(LOG_DIR, f"{base}_nl{num_layers}.log")


def _parse_real(path: str) -> Optional[Dict[str, float]]:
    if not os.path.isfile(path):
        return None
    res: Dict[str, float] = {}
    with open(path) as f:
        for line in f:
            m = _REAL_OPT_RE.search(line)
            if m:
                res["optimizer_mb"] = float(m.group(1))
                res["activation_peak_mb"] = float(m.group(2))
                res["cuda_peak_mb"] = float(m.group(3))
                break
    return res or None


def _ols_fit(xs: List[int], ys: List[float]) -> Tuple[float, float]:
    """Fit y = a + b × x; return (a, b). Requires len(xs) >= 2."""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return ys[0], 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def main() -> None:
    s = SHAPE
    cm = CostModel("qwen-30b-a3b-e128k8")
    dp = NUM_GPUS // (s["tp"] * s["ep"])
    micro_bsz = s["global_bsz"] // dp

    # Discover available num_layers data points by globbing the log dir.
    candidates = sorted({4} | {
        int(m.group(1))
        for fn in os.listdir(LOG_DIR)
        for m in [re.match(
            rf"cost_model_real_tp{s['tp']}_ep{s['ep']}_{s['dp_mode']}"
            rf"_bsz{s['global_bsz']}_fsep{s['fsep']}_nl(\d+)\.log$",
            fn,
        )]
        if m
    })

    rows: List[Dict] = []
    for nl in candidates:
        real = _parse_real(_log_path(nl))
        if real is None:
            print(f"# skip num_layers={nl}: no real log at {_log_path(nl)}")
            continue
        est = cm.estimate(
            num_layers=nl, num_gpus=NUM_GPUS, dp=dp, pp=1,
            tp=s["tp"], ep=s["ep"], micro_batch_size=micro_bsz,
            global_batch_size=s["global_bsz"], seq_len=SEQ_LEN,
            sequence_parallel=True, pp_schedule="1f1b",
            zero_stage=2 if s["dp_mode"] == "zero2sdp" else 3,
            sdp=True, recompute=True, bwd_mult=2.0,
            fsep=(s["fsep"] == "on"),
        )
        rows.append({
            "num_layers": nl,
            "real_peak": real["cuda_peak_mb"],
            "est_peak": est.peak_memory_mb,
            "delta_pct": 100.0 * (est.peak_memory_mb - real["cuda_peak_mb"]) / real["cuda_peak_mb"],
            "memory_source": est.breakdown.get("memory_source", "?"),
        })

    # Always print cost-model alpha-beta predictions across a wide range of
    # N so the linear extrapolation is visible regardless of ground truth.
    print(f"## Cost-model alpha-beta predictions: peak_memory_mb(N) = α + β × N")
    print(f"shape: tp={s['tp']} ep={s['ep']} {s['dp_mode']} "
          f"bsz={s['global_bsz']} fsep={s['fsep']} seq={SEQ_LEN}")
    print()
    print(f"{'N':>3} | {'est_peak_mb':>12} | {'params':>7} {'optim':>7} {'act':>7} | source")
    print("-" * 75)
    pred_xs: List[int] = []; pred_ys: List[float] = []
    for nl in (2, 4, 6, 8, 12, 16, 24, 32):
        est = cm.estimate(
            num_layers=nl, num_gpus=NUM_GPUS, dp=dp, pp=1,
            tp=s["tp"], ep=s["ep"], micro_batch_size=micro_bsz,
            global_batch_size=s["global_bsz"], seq_len=SEQ_LEN,
            sequence_parallel=True, pp_schedule="1f1b",
            zero_stage=2 if s["dp_mode"] == "zero2sdp" else 3,
            sdp=True, recompute=True, bwd_mult=2.0,
            fsep=(s["fsep"] == "on"),
        )
        bd = est.breakdown
        src = bd.get("memory_source", "?")[:24]
        print(
            f"{nl:>3} | {est.peak_memory_mb:>12.0f} | "
            f"{bd['parameters_mb']:>7.0f} {bd['optimizer_mb']:>7.0f} "
            f"{bd['activations_mb']:>7.0f} | {src}"
        )
        pred_xs.append(nl); pred_ys.append(est.peak_memory_mb)
    alpha_cm, beta_cm = _ols_fit(pred_xs, pred_ys)
    print()
    print(f"cost-model α (constant)        : {alpha_cm:>8.0f} MB")
    print(f"cost-model β (per-layer)       : {beta_cm:>8.1f} MB/layer")
    print()

    if not rows:
        print(
            "## Empirical validation: NOT YET RUN\n"
            "to gather ground truth, run from the host (when GPU memory is free):\n"
            f"  for N in 2 6 8; do NUM_LAYERS=$N bash "
            f"{MODEL_DIR}/scripts/cost_model_real_test.sh "
            f"{s['tp']} {s['ep']} {s['dp_mode']} {s['global_bsz']} {s['fsep']}; done\n"
            "then re-run this script to compute the empirical α/β and the\n"
            "drift table below."
        )
        sys.exit(0)

    if len(rows) < 2:
        print("## Empirical validation: only one ground-truth point — fit skipped")
        print("Run additional `NUM_LAYERS=<N> bash cost_model_real_test.sh ...`")
        print("invocations at distinct N values to populate the empirical fit.")
        return

    # Empirical alpha-beta fit through real measurements.
    xs = [r["num_layers"] for r in rows]
    real_ys = [r["real_peak"] for r in rows]
    est_ys = [r["est_peak"] for r in rows]
    alpha_real, beta_real = _ols_fit(xs, real_ys)
    alpha_emp_cm, beta_emp_cm = _ols_fit(xs, est_ys)

    print("## Empirical validation: peak_memory_mb(N) = α + β × N")
    print()
    print(f"{'N':>3} | {'real_peak_mb':>12} | {'est_peak_mb':>12} | {'Δ_mb':>7} | {'Δ%':>6} | source")
    print("-" * 75)
    for r in rows:
        src = r["memory_source"][:24]
        print(
            f"{r['num_layers']:>3} | {r['real_peak']:>12.0f} | {r['est_peak']:>12.0f} | "
            f"{r['est_peak'] - r['real_peak']:>+7.0f} | {r['delta_pct']:>+6.1f} | {src}"
        )
    print()
    print(f"empirical fit (real)      : α = {alpha_real:>8.0f} MB  β = {beta_real:>7.1f} MB/layer")
    print(f"cost-model fit (estimates): α = {alpha_emp_cm:>8.0f} MB  β = {beta_emp_cm:>7.1f} MB/layer")
    if alpha_real != 0 and beta_real != 0:
        print(f"α drift: {(alpha_emp_cm - alpha_real):+.0f} MB "
              f"({100*(alpha_emp_cm-alpha_real)/alpha_real:+.1f}%)")
        print(f"β drift: {(beta_emp_cm - beta_real):+.1f} MB/layer "
              f"({100*(beta_emp_cm-beta_real)/beta_real:+.1f}%)")


if __name__ == "__main__":
    main()
