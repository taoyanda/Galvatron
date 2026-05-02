"""Aggregate three profile artifacts that the MoE cost model consumes,
all derived from the existing ``cost_model_real_test.sh`` log sweep:

  1. ``configs/optimizer_step_profiling_<prec>_<model>.json``
     Per-rank Adam step throughput (MB/ms). Adam is HBM-bandwidth-bound
     and largely shape-independent on a fixed cluster, so a single
     median throughput suffices. Also contains the empirical
     ``optimizer_to_params_ratio`` used to size optimizer state.

  2. ``configs/runtime_profiling_<prec>_<model>.json``
     Full-iteration ``fwd_bwd_ms`` / ``opt_ms`` / ``cuda_peak_mb``
     measured at each ``(tp, ep, micro_bsz, seq, fsep[, pp])`` shape.
     Replaces the analytical (forward-only × (1+bwd_mult)) formula for
     shapes we have profiled — most importantly captures the FSEP
     smart-routing kernel's backward cost, which the forward-only
     computation profile cannot. When multiple ``num_layers`` runs are
     available for the same shape this file also records an OLS
     ``alpha + beta × num_layers`` fit per memory/time component, used
     by the cost model to extrapolate to other ``num_layers`` values.

  3. ``configs/fsep_overhead_profiling_<prec>_<model>.json``
     Per-MoE-layer FSEP time + memory overhead, derived as the delta
     between matched ``fsep=on`` and ``fsep=off`` runs at the same
     ``(tp, ep, micro_bsz, seq[, pp])`` shape. Used by the cost model's
     analytical fall-back when the user requests ``fsep=True`` but no
     calibrated runtime entry is available.

Inputs::

    galvatron/models/moe/logs/cost_model_real_tp{TP}_ep{EP}_{DPMODE}
        _bsz{BSZ}_fsep{ON|OFF}[_nl{N}][_pp{P}].log

Why this lives in a profile script and not as constants in the cost
model: cluster-specific calibration must be regenerated per cluster,
so we keep the analytical model code free of magic numbers and make
the calibration visible (matches the existing pattern of
``computation_profiling_*.json`` / ``memory_profiling_*.json``).
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(_HERE, ".."))
LOG_DIR = os.path.join(MODEL_DIR, "logs")
CONFIGS_DIR = os.path.join(MODEL_DIR, "configs")

MODEL = "mixtral-8x7b-e8k2"
PRECISION = "bf16"
SEQ_LEN = 4096
NUM_MOE_LAYERS = 4

# Filename grammar: cost_model_real_tp{TP}_ep{EP}_{DPMODE}_bsz{BSZ}_
# fsep{on|off}[_nl{N}][_pp{P}].log
_LOG_NAME_RE = re.compile(
    r"cost_model_real_tp(\d+)_ep(\d+)_(zero2sdp|zero3)_bsz(\d+)_fsep(on|off)"
    r"(?:_nl(\d+))?(?:_pp(\d+))?\.log$"
)
DEFAULT_NUM_LAYERS = 4  # filenames without `_nl<N>` were taken at N=4
DEFAULT_PP = 1          # filenames without `_pp<P>` were taken at PP=1

# Per-rank instrumentation lines emitted by train_dist_random.py.
_PARAMS_RE = re.compile(r"\[real_measure\] params_mb=([\d.]+)")
_OPT_AND_PEAK_RE = re.compile(
    r"\[real_measure\] optimizer_mb=([\d.]+)\s+activation_peak_mb=([\d.]+)\s+"
    r"cuda_peak_mb=([\d.]+)"
)
# Per-stage time instrumentation: averaged fwd+bwd and opt-step ms over
# a profiler window [start, end).
_STAGE_TIME_RE = re.compile(
    r"\[stage_time\] fwd_bwd_ms=([\d.]+)\s+opt_ms=([\d.]+)\s+window=\[(\d+),(\d+)\)"
)
_AVG_ITER_RE = re.compile(r"Average iteration time is:\s*([\d.]+)")

NUM_GPUS_PER_NODE = 4


def parse_log(path: str) -> Dict[str, Optional[float]]:
    """Extract every cost-model-relevant scalar from one log file.

    Returns ``None`` for any field that wasn't found; callers decide
    whether the missing value is fatal (e.g. opt_ms is required for
    Adam-throughput aggregation; iter_ms is optional)."""
    parsed: Dict[str, Optional[float]] = {
        "params_mb": None, "optimizer_mb": None,
        "activation_peak_mb": None, "cuda_peak_mb": None,
        "fwd_bwd_ms": None, "opt_ms": None,
        "iter_ms": None,
    }
    iter_samples: List[float] = []
    with open(path) as log_file:
        for line in log_file:
            params_match = _PARAMS_RE.search(line)
            if params_match:
                parsed["params_mb"] = float(params_match.group(1))
                continue
            opt_match = _OPT_AND_PEAK_RE.search(line)
            if opt_match:
                parsed["optimizer_mb"] = float(opt_match.group(1))
                parsed["activation_peak_mb"] = float(opt_match.group(2))
                parsed["cuda_peak_mb"] = float(opt_match.group(3))
                continue
            stage_match = _STAGE_TIME_RE.search(line)
            if stage_match:
                parsed["fwd_bwd_ms"] = float(stage_match.group(1))
                parsed["opt_ms"] = float(stage_match.group(2))
                continue
            iter_match = _AVG_ITER_RE.search(line)
            if iter_match:
                iter_samples.append(float(iter_match.group(1)))
    if iter_samples:
        # Profiler emits seconds; cost model speaks ms.
        parsed["iter_ms"] = (sum(iter_samples) / len(iter_samples)) * 1000.0
    return parsed


def _micro_bsz(global_bsz: int, tp: int, ep: int,
               pp: int = 1, chunks: int = 1) -> int:
    """Per-DP-rank micro batch size = ``global_bsz / (dp × chunks)``.

    ``dp = NUM_GPUS_PER_NODE / (pp × tp × ep)``. The previous version
    assumed ``pp == 1`` and over-counted dp at pp>1, so the resulting
    runtime-profile key picked the wrong micro-bsz suffix.
    """
    dp = max(1, NUM_GPUS_PER_NODE // (pp * tp * ep))
    return max(1, global_bsz // dp // chunks)


def _shape_key(tp: int, ep: int, micro_bsz: int, fsep: str,
               pp: int = DEFAULT_PP) -> str:
    """Per-(shape, pp) key.

    For ``pp == 1`` the suffix is omitted so existing entries (and the
    already-released alpha-beta fit at pp=1) keep their current keys
    without churn. ``pp > 1`` adds an explicit ``_pp{P}`` suffix.
    """
    base = f"tp{tp}_ep{ep}_bsz{micro_bsz}_seq{SEQ_LEN}_fsep{fsep}"
    return base if pp == DEFAULT_PP else f"{base}_pp{pp}"


def _ols_fit(xs: List[int], ys: List[float]) -> Tuple[float, float]:
    """Return ``(alpha, beta)`` such that ``y ≈ alpha + beta × x`` (OLS).

    Falls back to ``(ys[0], 0)`` if the design matrix is rank-deficient
    (all xs identical). Requires ``len(xs) == len(ys) >= 2``.
    """
    n_points = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n_points * sum_xx - sum_x * sum_x
    if denom == 0:
        return ys[0], 0.0
    beta = (n_points * sum_xy - sum_x * sum_y) / denom
    alpha = (sum_y - beta * sum_x) / n_points
    return alpha, beta


def _enumerate_logs() -> List[Dict]:
    """Walk LOG_DIR, parse every cost_model_real_*.log, return one dict
    per log with the parsed fields plus the (tp, ep, ...) it was tagged
    with in the filename."""
    rows: List[Dict] = []
    for filename in sorted(os.listdir(LOG_DIR)):
        name_match = _LOG_NAME_RE.match(filename)
        if not name_match:
            continue
        tp = int(name_match.group(1))
        ep = int(name_match.group(2))
        dp_mode = name_match.group(3)
        global_bsz = int(name_match.group(4))
        fsep = name_match.group(5)
        num_layers = (int(name_match.group(6))
                      if name_match.group(6) else DEFAULT_NUM_LAYERS)
        pp = int(name_match.group(7)) if name_match.group(7) else DEFAULT_PP
        path = os.path.join(LOG_DIR, filename)
        parsed = parse_log(path)
        if parsed["opt_ms"] is None or parsed["optimizer_mb"] is None:
            print(f"# skip {filename}: missing instrumentation lines")
            continue
        rows.append({
            "tp": tp, "ep": ep, "dp_mode": dp_mode,
            "global_bsz": global_bsz, "fsep": fsep,
            "num_layers": num_layers, "pp": pp,
            **parsed,
        })
    return rows


def _build_optimizer_step_profile(rows: List[Dict]) -> Dict:
    """Aggregate Adam step throughput + optimizer-to-params ratio."""
    samples = []
    throughputs: List[float] = []
    optimizer_to_params_ratios: List[float] = []
    for row in rows:
        throughput = row["optimizer_mb"] / row["opt_ms"]
        samples.append({
            "tp": row["tp"], "ep": row["ep"], "dp_mode": row["dp_mode"],
            "bsz": row["global_bsz"], "fsep": row["fsep"],
            "optimizer_mb": row["optimizer_mb"], "opt_ms": row["opt_ms"],
            "throughput_mb_per_ms": throughput,
        })
        throughputs.append(throughput)
        if row.get("params_mb") and row["params_mb"] > 0:
            optimizer_to_params_ratios.append(
                row["optimizer_mb"] / row["params_mb"]
            )

    return {
        "model": MODEL, "precision": PRECISION, "seq_len": SEQ_LEN,
        "throughput_mb_per_ms_median": statistics.median(throughputs),
        "throughput_mb_per_ms_min": min(throughputs),
        "throughput_mb_per_ms_max": max(throughputs),
        "throughput_mb_per_ms_mean": statistics.mean(throughputs),
        # Empirical optimizer_mb / params_mb on this cluster + framework.
        # For bf16 + FSDP + Adam the typical observed value is ~2.0 (Adam
        # keeps fp32 m + fp32 v only; no fp32 master copy and grads are
        # released after the reduce-scatter). Cost model uses this in
        # place of the legacy ``MODEL_STATE_MULT = 4`` constant.
        "optimizer_to_params_ratio_median": (
            statistics.median(optimizer_to_params_ratios)
            if optimizer_to_params_ratios else 3.0
        ),
        "optimizer_to_params_ratio_n": len(optimizer_to_params_ratios),
        "n_samples": len(throughputs),
        "samples": samples,
    }


def _build_runtime_profile(rows: List[Dict]) -> Dict:
    """Group rows by (shape, pp, num_layers); average across dp_modes;
    fit ``alpha + beta × num_layers`` for shapes with ≥ 2 N samples."""
    # Aggregate by (shape_key, num_layers); multiple dp_mode rows at the
    # same (shape, pp, N) average together. We store one canonical
    # entry per (shape, N).
    by_shape_n: Dict[Tuple[str, int], Dict] = {}
    for row in rows:
        if row["fwd_bwd_ms"] is None:
            continue
        micro_bsz = _micro_bsz(row["global_bsz"], row["tp"],
                               row["ep"], row["pp"])
        shape_key = _shape_key(row["tp"], row["ep"], micro_bsz,
                               row["fsep"], row["pp"])
        index_key = (shape_key, row["num_layers"])
        prior = by_shape_n.get(index_key)
        if prior is None:
            by_shape_n[index_key] = {
                "tp": row["tp"], "ep": row["ep"],
                "global_bsz": row["global_bsz"], "micro_bsz": micro_bsz,
                "fsep": row["fsep"], "seq": SEQ_LEN,
                "pp": row["pp"], "num_layers": row["num_layers"],
                "fwd_bwd_ms": row["fwd_bwd_ms"],
                "opt_ms": row["opt_ms"],
                "iter_ms": row.get("iter_ms"),
                "params_mb": row.get("params_mb"),
                "optimizer_mb": row.get("optimizer_mb"),
                "activation_peak_mb": row.get("activation_peak_mb"),
                "cuda_peak_mb": row.get("cuda_peak_mb"),
                "n_dp_samples": 1,
                "dp_modes": [row["dp_mode"]],
            }
        else:
            count = prior["n_dp_samples"]
            for field_name in ("fwd_bwd_ms", "opt_ms", "iter_ms",
                               "params_mb", "optimizer_mb",
                               "activation_peak_mb", "cuda_peak_mb"):
                new_value = row.get(field_name)
                if new_value is None or prior.get(field_name) is None:
                    continue
                prior[field_name] = (
                    (prior[field_name] * count + new_value) / (count + 1)
                )
            prior["n_dp_samples"] = count + 1
            prior["dp_modes"].append(row["dp_mode"])

    # Build the by_shape view used by IntraCostModel:
    #   - if multiple N values exist for a shape, fit alpha + beta × N
    #     for each memory/time component via OLS;
    #   - otherwise expose the single-N sample directly (back-compat).
    samples_per_shape: Dict[str, List[Dict]] = {}
    for (shape_key, _num_layers), entry in by_shape_n.items():
        samples_per_shape.setdefault(shape_key, []).append(entry)

    by_shape: Dict[str, Dict] = {}
    for shape_key, samples in samples_per_shape.items():
        # Pick the sample whose num_layers matches DEFAULT_NUM_LAYERS as
        # the canonical entry; fall back to the smallest N otherwise.
        canonical = next(
            (s for s in samples if s["num_layers"] == DEFAULT_NUM_LAYERS),
            min(samples, key=lambda s: s["num_layers"]),
        )
        out = dict(canonical)  # back-compat fields (single-N view)
        out["samples_by_num_layers"] = sorted(
            samples, key=lambda s: s["num_layers"]
        )
        if len(samples) >= 2:
            xs = [int(s["num_layers"]) for s in samples]
            fits: Dict[str, Dict[str, float]] = {}
            # ``iter_ms`` is the full-iteration time including PP bubble
            # + comm — critical for validating PPCostModel under bubble
            # overhead.
            for field_name in ("params_mb", "optimizer_mb",
                               "activation_peak_mb", "cuda_peak_mb",
                               "fwd_bwd_ms", "opt_ms", "iter_ms"):
                ys = [
                    float(s[field_name])
                    for s in samples if s.get(field_name) is not None
                ]
                if len(ys) == len(xs) and len(ys) >= 2:
                    alpha, beta = _ols_fit(xs, ys)
                    fits[field_name] = {
                        "alpha": alpha, "beta": beta, "n_points": len(ys),
                    }
            out["alpha_beta_fit"] = fits
        by_shape[shape_key] = out

    return {
        "model": MODEL, "precision": PRECISION, "seq_len": SEQ_LEN,
        "num_layers_profiled": NUM_MOE_LAYERS,
        "by_shape": by_shape,
    }, by_shape_n


def _build_fsep_overhead_profile(by_shape_n: Dict) -> Dict:
    """Find ``(tp, ep, micro_bsz, pp, num_layers)`` tuples where both
    ``fsep=on`` and ``fsep=off`` were measured; the per-MoE-layer
    overhead is ``(on - off) / num_layers`` (fwd+bwd, with whatever
    recompute setting was active during calibration). Memory overhead
    is the same idea on ``cuda_peak_mb``. Used by IntraCostModel's
    analytical fall-back so FSEP-on configs without a calibrated
    runtime entry get a non-zero penalty."""
    pairs: Dict[Tuple[int, int, int, int, int], Dict[str, Dict]] = {}
    for (_shape_key, _num_layers), entry in by_shape_n.items():
        # Index by the underlying (tp, ep, micro_bsz, pp, num_layers)
        # rather than re-parsing the shape key.
        index_key = (
            entry["tp"], entry["ep"], entry["micro_bsz"],
            entry["pp"], entry["num_layers"],
        )
        pairs.setdefault(index_key, {})[entry["fsep"]] = entry

    samples: List[Dict] = []
    by_shape: Dict[str, Dict] = {}
    for (_index_key, both) in pairs.items():
        if "on" not in both or "off" not in both:
            continue
        on, off = both["on"], both["off"]
        num_layers = on["num_layers"]
        if num_layers <= 0:
            continue
        time_overhead_per_layer_ms = (
            (on["fwd_bwd_ms"] - off["fwd_bwd_ms"]) / num_layers
        )
        memory_overhead_per_layer_mb: Optional[float] = None
        if (on.get("cuda_peak_mb") is not None
                and off.get("cuda_peak_mb") is not None):
            memory_overhead_per_layer_mb = (
                (on["cuda_peak_mb"] - off["cuda_peak_mb"]) / num_layers
            )
        samples.append({
            "tp": on["tp"], "ep": on["ep"],
            "micro_bsz": on["micro_bsz"], "pp": on["pp"],
            "num_layers": num_layers,
            "fwd_bwd_off_ms": off["fwd_bwd_ms"],
            "fwd_bwd_on_ms": on["fwd_bwd_ms"],
            "time_overhead_per_layer_ms": time_overhead_per_layer_ms,
            "cuda_peak_off_mb": off.get("cuda_peak_mb"),
            "cuda_peak_on_mb": on.get("cuda_peak_mb"),
            "memory_overhead_per_layer_mb": memory_overhead_per_layer_mb,
        })
        # Per-shape key drops num_layers from the lookup key so callers
        # can request a different num_layers and we just scale by it.
        shape_key = (
            f"tp{on['tp']}_ep{on['ep']}_bsz{on['micro_bsz']}_seq{SEQ_LEN}"
            + ("" if on["pp"] == DEFAULT_PP else f"_pp{on['pp']}")
        )
        prior = by_shape.get(shape_key)
        if prior is None:
            by_shape[shape_key] = {
                "time_overhead_per_layer_ms": time_overhead_per_layer_ms,
                "memory_overhead_per_layer_mb": memory_overhead_per_layer_mb,
                "n_samples": 1,
            }
        else:
            count = prior["n_samples"]
            prior["time_overhead_per_layer_ms"] = (
                (prior["time_overhead_per_layer_ms"] * count
                 + time_overhead_per_layer_ms)
                / (count + 1)
            )
            if (memory_overhead_per_layer_mb is not None
                    and prior["memory_overhead_per_layer_mb"] is not None):
                prior["memory_overhead_per_layer_mb"] = (
                    (prior["memory_overhead_per_layer_mb"] * count
                     + memory_overhead_per_layer_mb)
                    / (count + 1)
                )
            prior["n_samples"] = count + 1

    if samples:
        time_default = statistics.median(
            s["time_overhead_per_layer_ms"] for s in samples
        )
        memory_values = [
            s["memory_overhead_per_layer_mb"] for s in samples
            if s["memory_overhead_per_layer_mb"] is not None
        ]
        memory_default = statistics.median(memory_values) if memory_values else 0.0
    else:
        time_default = 0.0
        memory_default = 0.0

    return {
        "model": MODEL, "precision": PRECISION, "seq_len": SEQ_LEN,
        "default_time_overhead_per_layer_ms": time_default,
        "default_memory_overhead_per_layer_mb": memory_default,
        "by_shape": by_shape,
        "samples": samples,
    }


def main() -> None:
    rows = _enumerate_logs()
    if not rows:
        print("no parseable logs found; run cost_model_real_test.sh first")
        sys.exit(1)

    optimizer_profile = _build_optimizer_step_profile(rows)
    optimizer_path = os.path.join(
        CONFIGS_DIR, f"optimizer_step_profiling_{PRECISION}_{MODEL}.json"
    )
    with open(optimizer_path, "w") as out_file:
        json.dump(optimizer_profile, out_file, indent=2)
    print(f"wrote {optimizer_path}")
    print(
        f"  Adam throughput: median={optimizer_profile['throughput_mb_per_ms_median']:.2f} "
        f"min={optimizer_profile['throughput_mb_per_ms_min']:.2f} "
        f"max={optimizer_profile['throughput_mb_per_ms_max']:.2f} MB/ms "
        f"({optimizer_profile['n_samples']} samples)"
    )
    print(
        f"  optimizer_to_params_ratio: median="
        f"{optimizer_profile['optimizer_to_params_ratio_median']:.2f} "
        f"({optimizer_profile['optimizer_to_params_ratio_n']} samples)"
    )

    runtime_profile, by_shape_n = _build_runtime_profile(rows)
    runtime_path = os.path.join(
        CONFIGS_DIR, f"runtime_profiling_{PRECISION}_{MODEL}.json"
    )
    with open(runtime_path, "w") as out_file:
        json.dump(runtime_profile, out_file, indent=2)
    print(f"wrote {runtime_path}")

    fsep_overhead_profile = _build_fsep_overhead_profile(by_shape_n)
    fsep_path = os.path.join(
        CONFIGS_DIR, f"fsep_overhead_profiling_{PRECISION}_{MODEL}.json"
    )
    with open(fsep_path, "w") as out_file:
        json.dump(fsep_overhead_profile, out_file, indent=2)
    print(f"wrote {fsep_path}")

    print(
        f"  FSEP overhead: time_default={fsep_overhead_profile['default_time_overhead_per_layer_ms']:.1f} ms/layer  "
        f"mem_default={fsep_overhead_profile['default_memory_overhead_per_layer_mb']:.1f} MB/layer  "
        f"({len(fsep_overhead_profile['samples'])} on/off pairs)"
    )
    for shape_key, entry in sorted(fsep_overhead_profile["by_shape"].items()):
        memory_overhead = entry.get("memory_overhead_per_layer_mb") or 0.0
        print(
            f"    {shape_key}: time={entry['time_overhead_per_layer_ms']:.1f} ms/layer  "
            f"mem={memory_overhead:.1f} MB/layer  (n={entry['n_samples']})"
        )

    print(f"  runtime samples: {len(runtime_profile['by_shape'])} shapes")
    for shape_key, entry in sorted(runtime_profile["by_shape"].items()):
        n_layer_points = len(entry.get("samples_by_num_layers", [entry]))
        cuda_fit = entry.get("alpha_beta_fit", {}).get("cuda_peak_mb")
        if cuda_fit:
            print(
                f"    {shape_key}: fwd_bwd={entry['fwd_bwd_ms']:.0f} ms "
                f"opt={entry['opt_ms']:.0f} ms  N_pts={n_layer_points}  "
                f"α_cuda={cuda_fit['alpha']:.0f} MB "
                f"β_cuda={cuda_fit['beta']:.1f} MB/layer"
            )
        else:
            print(
                f"    {shape_key}: fwd_bwd={entry['fwd_bwd_ms']:.0f} ms "
                f"opt={entry['opt_ms']:.0f} ms  N_pts={n_layer_points}"
            )


if __name__ == "__main__":
    main()
