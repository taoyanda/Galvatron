"""Parse per-config logs from cost_model_real_test.sh and dump a single JSON.

For each tuple ``(tp, ep, dp_mode, bsz, fsep)`` the matching log file
``logs/cost_model_real_tp<TP>_ep<EP>_<DP_MODE>_bsz<BSZ>_fsep<MODE>.log`` is
parsed for:

  - [real_measure] params_mb=…
  - [real_measure] optimizer_mb=… activation_peak_mb=… cuda_peak_mb=…
  - [stage_time] fwd_bwd_ms=… opt_ms=… window=[a,b)
  - [mem_evo] post_construct_mb=… / pre_fwd_mb=… / post_fwd_bwd_mb=… /
    post_opt_step_mb=… / post_zero_grad_mb=…
  - Average iteration time is: X s

Writes ``configs/cost_model_real_4gpu_4layer.json``.
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
CONFIGS_DIR = os.path.join(MODEL_DIR, "configs")

# Tuple format must match cost_model_real_test.sh DEFAULT_CONFIGS:
#   (tp, ep, dp_mode, bsz, fsep)  with dp_mode ∈ {zero2sdp, zero3},
#   fsep ∈ {on, off}.
CONFIGS: List[Tuple[int, int, str, int, str]] = [
    # FSEP-on track (smart routing + LAER): tp == tp_of_ep, tp × ep == 4.
    (1, 4, "zero2sdp", 4, "on"),
    (1, 4, "zero3",    4, "on"),
    (2, 2, "zero2sdp", 4, "on"),
    (2, 2, "zero3",    4, "on"),
    # FSEP-off track for the same (tp, ep, dp_mode, bsz) shapes — naive MoE.
    (1, 4, "zero2sdp", 4, "off"),
    (1, 4, "zero3",    4, "off"),
    (2, 2, "zero2sdp", 4, "off"),
    (2, 2, "zero3",    4, "off"),
    # FSEP-off ep=1 — FSEP cannot satisfy its constraint here.
    (2, 1, "zero2sdp", 2, "off"), (2, 1, "zero2sdp", 4, "off"),
    (2, 1, "zero3",    2, "off"), (2, 1, "zero3",    4, "off"),
    (4, 1, "zero2sdp", 4, "off"),
    (4, 1, "zero3",    4, "off"),
]

_PARAMS_RE = re.compile(r"\[real_measure\] params_mb=([\d.]+)")
_OPT_PRED_RE = re.compile(r"\[real_measure\] optimizer_mb_pred=([\d.]+)")
_OPT_REAL_RE = re.compile(
    r"\[real_measure\] optimizer_mb=([\d.]+)\s+activation_peak_mb=([\d.]+)\s+cuda_peak_mb=([\d.]+)"
)
_ACT_ONLY_RE = re.compile(
    r"\[real_measure\] activation_peak_mb=([\d.]+)\s+cuda_peak_mb=([\d.]+)"
)
_ITER_RE = re.compile(r"Average iteration time is:\s*([\d.]+)")
_STAGE_RE = re.compile(
    r"\[stage_time\] fwd_bwd_ms=([\d.]+)\s+opt_ms=([\d.]+)\s+window=\[(\d+),(\d+)\)"
)
_MEM_EVO_RE = re.compile(r"\[mem_evo\] (\w+)_mb=([\d.]+)")

_MEM_EVO_KEYS = (
    "post_construct",
    "pre_fwd",
    "post_fwd_bwd",
    "post_opt_step",
    "post_zero_grad",
)


def parse_log(path: str) -> Dict[str, Optional[float]]:
    res: Dict[str, Optional[float]] = {
        "params_mb": None,
        "optimizer_mb_pred": None,
        "optimizer_mb": None,
        "activation_peak_mb": None,
        "cuda_peak_mb": None,
        "iter_ms": None,
        "fwd_bwd_ms": None,
        "opt_step_ms": None,
        "stage_window": None,
    }
    for k in _MEM_EVO_KEYS:
        res[f"mem_evo_{k}_mb"] = None
    if not os.path.isfile(path):
        return res
    iter_samples: List[float] = []
    with open(path) as f:
        for line in f:
            m = _PARAMS_RE.search(line)
            if m:
                res["params_mb"] = float(m.group(1))
                continue
            m = _OPT_PRED_RE.search(line)
            if m:
                res["optimizer_mb_pred"] = float(m.group(1))
                continue
            m = _OPT_REAL_RE.search(line)
            if m:
                res["optimizer_mb"] = float(m.group(1))
                res["activation_peak_mb"] = float(m.group(2))
                res["cuda_peak_mb"] = float(m.group(3))
                continue
            m = _ACT_ONLY_RE.search(line)
            if m:
                res["activation_peak_mb"] = float(m.group(1))
                res["cuda_peak_mb"] = float(m.group(2))
                continue
            m = _STAGE_RE.search(line)
            if m:
                res["fwd_bwd_ms"] = float(m.group(1))
                res["opt_step_ms"] = float(m.group(2))
                res["stage_window"] = [int(m.group(3)), int(m.group(4))]
                continue
            m = _MEM_EVO_RE.search(line)
            if m:
                key = m.group(1)
                if key in _MEM_EVO_KEYS:
                    res[f"mem_evo_{key}_mb"] = float(m.group(2))
                continue
            m = _ITER_RE.search(line)
            if m:
                iter_samples.append(float(m.group(1)))
    if iter_samples:
        res["iter_ms"] = (sum(iter_samples) / len(iter_samples)) * 1000.0
    return res


def _log_path_for(tp: int, ep: int, dp_mode: str, bsz: int, fsep: str) -> str:
    return os.path.join(
        LOG_DIR,
        f"cost_model_real_tp{tp}_ep{ep}_{dp_mode}_bsz{bsz}_fsep{fsep}.log",
    )


def main() -> None:
    out: Dict[str, object] = {"configs": []}
    print(
        f"{'tp':>2} {'ep':>2} {'dp_mode':>9} {'bsz':>3} {'fsep':>4} | "
        f"{'iter_ms':>9} {'fwd_bwd':>9} {'opt':>7} | "
        f"{'params':>8} {'opt':>8} {'act_peak':>9} {'cuda_peak':>10}"
    )
    print("-" * 110)
    for tp, ep, dp_mode, bsz, fsep in CONFIGS:
        log_path = _log_path_for(tp, ep, dp_mode, bsz, fsep)
        m = parse_log(log_path)
        row = {
            "tp": tp, "ep": ep, "dp_mode": dp_mode, "global_bsz": bsz, "fsep": fsep,
            "log_path": log_path, **m,
        }
        out["configs"].append(row)

        def _f(v: Optional[float], w: int = 8, fmt: str = "{:>8.0f}") -> str:
            return fmt.format(v) if v is not None else ("-" * 3).rjust(w)
        opt_real = m.get("optimizer_mb")
        opt_show = opt_real if opt_real else m.get("optimizer_mb_pred")
        print(
            f"{tp:>2} {ep:>2} {dp_mode:>9} {bsz:>3} {fsep:>4} | "
            f"{_f(m['iter_ms'], 9, '{:>9.1f}')} "
            f"{_f(m['fwd_bwd_ms'], 9, '{:>9.1f}')} "
            f"{_f(m['opt_step_ms'], 7, '{:>7.1f}')} | "
            f"{_f(m['params_mb'])} {_f(opt_show)} "
            f"{_f(m['activation_peak_mb'], 9, '{:>9.0f}')} "
            f"{_f(m['cuda_peak_mb'], 10, '{:>10.0f}')}"
        )
    out_path = os.path.join(CONFIGS_DIR, "cost_model_real_4gpu_4layer.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
