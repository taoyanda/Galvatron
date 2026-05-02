"""PP cost-model validation: sweep PP degree, compare estimates vs reality.

Reads ``cost_model_real_tp{TP}_ep{EP}_{DP}_bsz{B}_fsep{FSEP}[_nl{N}]_pp{P}.log``
(produced by ``cost_model_real_test.sh`` with ``PP={P}`` env var), pairs
each run with the corresponding ``PPCostModel`` prediction, and prints a
side-by-side drift table.

Run after a sweep:
    for PP in 1 2 4; do
        case $PP in
            1) layout="1 4 zero2sdp 4 off" ;;
            2) layout="1 2 zero2sdp 4 off" ;;
            4) layout="1 1 zero2sdp 4 off" ;;
        esac
        PP=$PP bash scripts/cost_model_real_test.sh $layout
    done
    docker exec hetu python3 scripts/cost_model_pp_drift.py
"""
from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(_HERE, ".."))
LOG_DIR = os.path.join(MODEL_DIR, "logs")
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..")))

from galvatron.models.moe.cost_model import PPCostModel  # noqa: E402

NUM_GPUS = 4
SEQ_LEN = 4096
NUM_LAYERS = 4

# (pp, tp, ep, dp_mode, bsz, fsep). For each PP degree the tp/ep/dp_mode
# product per stage = NUM_GPUS / pp. Same global bsz for an apples-to-apples
# critical-path comparison.
SWEEP: List[Tuple[int, int, int, str, int, str]] = [
    (1, 1, 4, "zero2sdp", 4, "off"),
    (2, 1, 2, "zero2sdp", 4, "off"),
    (4, 1, 1, "zero2sdp", 4, "off"),
]


_REAL_OPT_RE = re.compile(
    r"\[real_measure\] optimizer_mb=([\d.]+)\s+activation_peak_mb=([\d.]+)\s+cuda_peak_mb=([\d.]+)"
)
_STAGE_RE = re.compile(
    r"\[stage_time\] fwd_bwd_ms=([\d.]+)\s+opt_ms=([\d.]+)\s+window=\[(\d+),(\d+)\)"
)
_ITER_RE = re.compile(r"Average iteration time is:\s*([\d.]+)")


def _log_path(pp: int, tp: int, ep: int, dp_mode: str, bsz: int, fsep: str,
              num_layers: int) -> str:
    base = f"cost_model_real_tp{tp}_ep{ep}_{dp_mode}_bsz{bsz}_fsep{fsep}"
    if num_layers != 4:
        base += f"_nl{num_layers}"
    if pp != 1:
        base += f"_pp{pp}"
    return os.path.join(LOG_DIR, base + ".log")


def _parse_real(path: str) -> Optional[Dict[str, float]]:
    if not os.path.isfile(path):
        return None
    res: Dict[str, float] = {}
    iter_samples: List[float] = []
    with open(path) as f:
        for line in f:
            m = _REAL_OPT_RE.search(line)
            if m:
                res["optimizer_mb"] = float(m.group(1))
                res["activation_peak_mb"] = float(m.group(2))
                res["cuda_peak_mb"] = float(m.group(3))
                continue
            m = _STAGE_RE.search(line)
            if m:
                res["fwd_bwd_ms"] = float(m.group(1))
                res["opt_ms"] = float(m.group(2))
                continue
            m = _ITER_RE.search(line)
            if m:
                iter_samples.append(float(m.group(1)))
    if iter_samples:
        res["iter_ms"] = (sum(iter_samples) / len(iter_samples)) * 1000.0
    return res or None


def _drift_pct(est: Optional[float], real: Optional[float]) -> Optional[float]:
    if est is None or real is None or real == 0:
        return None
    return 100.0 * (est - real) / real


def _fmt(v: Optional[float], spec: str = "{:.0f}") -> str:
    return spec.format(v) if v is not None else "—"


def main() -> None:
    cm = PPCostModel("mixtral-8x7b-e8k2")

    print(
        "## PP cost-model validation: peak_iter_ms / peak_memory_mb vs real"
    )
    print(
        f"shape: tp/ep/dp vary with pp such that pp×dp×tp×ep={NUM_GPUS}; "
        f"num_layers={NUM_LAYERS} bsz=4 seq={SEQ_LEN} fsep=off"
    )
    print()
    hdr = (
        f"{'pp':>2} {'tp':>2} {'ep':>2} {'dp_mode':>9} {'bsz':>3} | "
        f"{'real_iter':>9} {'est_iter':>9} {'Δ_iter%':>8} | "
        f"{'real_fwd_bwd':>12} {'real_opt':>9} | "
        f"{'real_mem':>9} {'est_mem':>9} {'Δ_mem%':>7} | mem_src"
    )
    print(hdr)
    print("-" * len(hdr))

    iter_pcts: List[float] = []
    mem_pcts: List[float] = []
    for (pp, tp, ep, dp_mode, bsz, fsep) in SWEEP:
        dp = NUM_GPUS // (pp * tp * ep)
        micro_bsz = max(1, bsz // dp)
        path = _log_path(pp, tp, ep, dp_mode, bsz, fsep, NUM_LAYERS)
        real = _parse_real(path)

        try:
            est = cm.estimate(
                num_layers=NUM_LAYERS, num_gpus=NUM_GPUS,
                dp=dp, pp=pp, tp=tp, ep=ep,
                micro_batch_size=micro_bsz,
                global_batch_size=bsz,
                seq_len=SEQ_LEN,
                sequence_parallel=True,
                zero_stage=2 if dp_mode == "zero2sdp" else 3,
                sdp=(dp_mode in ("zero2sdp", "zero3")),
                recompute=True, bwd_mult=2.0,
                fsep=(fsep == "on"),
            )
        except Exception as e:
            print(f"{pp:>2} {tp:>2} {ep:>2} {dp_mode:>9} {bsz:>3} | "
                  f"estimate FAILED: {e}")
            continue

        bd = est.breakdown
        mem_src = (bd.get("memory_source", "?")
                   .replace("runtime_profile", "rt")
                   .replace("alpha_beta_fit", "α/β")
                   .replace("linear_extrapolation", "lin")
                   .replace("analytical_stage_memory", "analytical"))[:24]
        if real:
            r_iter = real.get("iter_ms")
            r_mem = real.get("cuda_peak_mb")
            r_fb = real.get("fwd_bwd_ms")
            r_opt = real.get("opt_ms")
            d_iter = _drift_pct(est.total_iter_ms, r_iter)
            d_mem = _drift_pct(est.peak_memory_mb, r_mem)
            if d_iter is not None: iter_pcts.append(d_iter)
            if d_mem is not None: mem_pcts.append(d_mem)
            print(
                f"{pp:>2} {tp:>2} {ep:>2} {dp_mode:>9} {bsz:>3} | "
                f"{_fmt(r_iter):>9} {est.total_iter_ms:>9.0f} "
                f"{_fmt(d_iter, '{:>+8.1f}')} | "
                f"{_fmt(r_fb):>12} {_fmt(r_opt):>9} | "
                f"{_fmt(r_mem):>9} {est.peak_memory_mb:>9.0f} "
                f"{_fmt(d_mem, '{:>+7.1f}')} | {mem_src}"
            )
        else:
            print(
                f"{pp:>2} {tp:>2} {ep:>2} {dp_mode:>9} {bsz:>3} | "
                f"{'—':>9} {est.total_iter_ms:>9.0f} {'—':>8} | "
                f"{'—':>12} {'—':>9} | "
                f"{'—':>9} {est.peak_memory_mb:>9.0f} {'—':>7} | {mem_src}"
            )

    print("-" * len(hdr))
    if iter_pcts:
        absp = [abs(p) for p in iter_pcts]
        print(
            f"iter-time drift:   median_signed={sorted(iter_pcts)[len(iter_pcts)//2]:+.1f}%"
            f"  mean_abs={sum(absp)/len(absp):.1f}%  max_abs={max(absp):.1f}%"
            f"  n={len(iter_pcts)}"
        )
    if mem_pcts:
        absp = [abs(p) for p in mem_pcts]
        print(
            f"cuda-peak drift:   median_signed={sorted(mem_pcts)[len(mem_pcts)//2]:+.1f}%"
            f"  mean_abs={sum(absp)/len(absp):.1f}%  max_abs={max(absp):.1f}%"
            f"  n={len(mem_pcts)}"
        )
    if not iter_pcts and not mem_pcts:
        print(
            "no real runs found; launch the sweep first:\n"
            "  PP=2 bash scripts/cost_model_real_test.sh 1 2 zero2sdp 4 off\n"
            "  PP=4 bash scripts/cost_model_real_test.sh 1 1 zero2sdp 4 off"
        )


if __name__ == "__main__":
    sys.exit(main())
