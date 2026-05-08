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

MODEL = "qwen-30b-a3b-e128k8"
PRECISION = "bf16"
SEQ_LEN = 4096
NUM_MOE_LAYERS = 4

# Filename grammar: cost_model_real_tp{TP}_ep{EP}_{DPMODE}_bsz{BSZ}_
# fsep{on|off}[_nl{N}][_pp{P}][_chunks{C}][_unit{ALL|ATTENTION|MLP}].log
#
# - ``bsz{BSZ}`` is the trainer's ``--global_train_batch_size`` argument.
#   At chunks=1 this equals ``micro_bsz``; at chunks>1 the cost model
#   recovers ``micro_bsz = BSZ / chunks`` for shape-key construction.
# - ``_unit{X}``, ``_chunks{C}``, ``_pp{P}``, ``_nl{N}`` are all optional
#   for back-compat with logs captured before the corresponding axes
#   were introduced.
_LOG_NAME_RE = re.compile(
    r"cost_model_real_tp(\d+)_ep(\d+)_(zero2sdp|zero3)_bsz(\d+)_fsep(on|off)"
    r"(?:_nl(\d+))?(?:_pp(\d+))?(?:_chunks(\d+))?"
    r"(?:_unit(all|attention|mlp))?\.log$"
)
DEFAULT_NUM_LAYERS = 4  # filenames without `_nl<N>` were taken at N=4
DEFAULT_PP = 1          # filenames without `_pp<P>` were taken at PP=1
DEFAULT_CHUNKS = 1      # filenames without `_chunks<C>` were taken at chunks=1
DEFAULT_PROFILE_UNIT = "all"  # filenames without `_unit<X>` are full-iter "all"

# Per-rank instrumentation lines emitted by train_dist_random.py.
_PARAMS_RE = re.compile(r"\[real_measure\] params_mb=([\d.]+)")
_OPT_AND_PEAK_RE = re.compile(
    r"\[real_measure\] optimizer_mb=([\d.]+)\s+activation_peak_mb=([\d.]+)\s+"
    r"cuda_peak_mb=([\d.]+)"
    r"(?:\s+cuda_peak_reserved_mb=([\d.]+))?"
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
        # ``cuda_peak_reserved_mb`` is what the caching allocator holds at
        # peak — typically larger than the allocated peak (which only
        # counts live tensors) and matches what nvidia-smi sees. Captured
        # by the trainer's [real_measure] line; ``None`` for older logs.
        "cuda_peak_reserved_mb": None,
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
                if opt_match.group(4) is not None:
                    parsed["cuda_peak_reserved_mb"] = float(opt_match.group(4))
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


def _micro_bsz_at_calibration(global_bsz: int, chunks: int = 1) -> int:
    """The micro-batch size embedded in a calibration run.

    The cost-model side defines ``micro_bsz`` as "samples processed per
    fwd-bwd step at one PP stage" and ``num_microbatches = global_bsz //
    micro_bsz``. The profile sweep sets ``chunks=1``, so each calibration
    run has ``num_microbatches = 1`` and therefore ``micro_bsz =
    global_bsz``. The cost model looks up calibration entries by this
    micro_bsz directly, with ``DP*EP`` only enforced as a feasibility
    guard at query time (you can't split a sample fractionally).
    """
    return max(1, global_bsz // chunks)


def _shape_key(tp: int, ep: int, micro_bsz: int, fsep: str,
               pp: int = DEFAULT_PP) -> str:
    """Per-(shape, pp) key.

    Encodes ``(tp, ep, micro_bsz, fsep[, pp])`` — the axes the cost model
    actually queries. ``micro_bsz`` here is the per-stage compute batch
    size (= trainer's global_train_batch_size at chunks=1 calibration).
    DP and EP only enter as feasibility guards (per-rank ≥ 1 sample),
    not as shape-key dimensions.

    For ``pp == 1`` the suffix is omitted (PP=1 is the cost-model
    default). ``pp > 1`` adds an explicit ``_pp{P}`` suffix.
    """
    base = f"tp{tp}_ep{ep}_micro_bsz{micro_bsz}_seq{SEQ_LEN}_fsep{fsep}"
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
        # group(8) is the optional ``_chunks{C}`` suffix; legacy logs
        # without it were taken at chunks=1.
        chunks = int(name_match.group(8)) if name_match.group(8) else DEFAULT_CHUNKS
        # group(9) is the optional ``_unit{X}`` suffix introduced when step 6
        # was merged into step 8. Legacy logs have no suffix → "all".
        profile_unit = (name_match.group(9) or DEFAULT_PROFILE_UNIT)
        path = os.path.join(LOG_DIR, filename)
        parsed = parse_log(path)
        # ``opt_ms`` and ``optimizer_mb`` are only emitted by the full-iter
        # ``profile_unit=all`` and ``mlp`` runs. Attention-only runs skip
        # the optimizer step (no MoE params updated) so we keep them even
        # if those fields are absent — they still carry valid fwd_bwd_ms /
        # cuda_peak_mb for the per-component breakdown.
        if profile_unit == "all" and (
            parsed["opt_ms"] is None or parsed["optimizer_mb"] is None
        ):
            print(f"# skip {filename}: missing instrumentation lines")
            continue
        rows.append({
            "tp": tp, "ep": ep, "dp_mode": dp_mode,
            "global_bsz": global_bsz, "fsep": fsep,
            "num_layers": num_layers, "pp": pp,
            "chunks": chunks,
            "profile_unit": profile_unit,
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
        micro_bsz = _micro_bsz_at_calibration(
            row["global_bsz"], row.get("chunks", 1)
        )
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
        # Under 1:1 the calibration's num_layers equals the expert-layer
        # count, so this divisor is the right one for "per expert layer."
        # The new field name documents that explicitly; the legacy alias
        # `time_overhead_per_layer_ms` is kept for back-compat.
        samples.append({
            "tp": on["tp"], "ep": on["ep"],
            "micro_bsz": on["micro_bsz"], "pp": on["pp"],
            "num_layers": num_layers,
            "num_expert_layers": num_layers,  # 1:1 invariant
            "fwd_bwd_off_ms": off["fwd_bwd_ms"],
            "fwd_bwd_on_ms": on["fwd_bwd_ms"],
            "time_overhead_per_expert_layer_ms": time_overhead_per_layer_ms,
            "time_overhead_per_layer_ms": time_overhead_per_layer_ms,  # legacy alias
            "cuda_peak_off_mb": off.get("cuda_peak_mb"),
            "cuda_peak_on_mb": on.get("cuda_peak_mb"),
            "memory_overhead_per_expert_layer_mb": memory_overhead_per_layer_mb,
            "memory_overhead_per_layer_mb": memory_overhead_per_layer_mb,  # legacy alias
        })
        # Per-shape key drops num_layers from the lookup key so callers
        # can request a different num_layers and we just scale by it.
        shape_key = (
            f"tp{on['tp']}_ep{on['ep']}_micro_bsz{on['micro_bsz']}_seq{SEQ_LEN}"
            + ("" if on["pp"] == DEFAULT_PP else f"_pp{on['pp']}")
        )
        prior = by_shape.get(shape_key)
        if prior is None:
            by_shape[shape_key] = {
                "time_overhead_per_expert_layer_ms": time_overhead_per_layer_ms,
                "memory_overhead_per_expert_layer_mb": memory_overhead_per_layer_mb,
                # Legacy aliases for back-compat with callers that haven't
                # been updated yet.
                "time_overhead_per_layer_ms": time_overhead_per_layer_ms,
                "memory_overhead_per_layer_mb": memory_overhead_per_layer_mb,
                "n_samples": 1,
            }
        else:
            count = prior["n_samples"]
            new_time = (
                (prior["time_overhead_per_expert_layer_ms"] * count
                 + time_overhead_per_layer_ms)
                / (count + 1)
            )
            prior["time_overhead_per_expert_layer_ms"] = new_time
            prior["time_overhead_per_layer_ms"] = new_time
            if (memory_overhead_per_layer_mb is not None
                    and prior["memory_overhead_per_expert_layer_mb"] is not None):
                new_mem = (
                    (prior["memory_overhead_per_expert_layer_mb"] * count
                     + memory_overhead_per_layer_mb)
                    / (count + 1)
                )
                prior["memory_overhead_per_expert_layer_mb"] = new_mem
                prior["memory_overhead_per_layer_mb"] = new_mem
            prior["n_samples"] = count + 1

    if samples:
        time_default = statistics.median(
            s["time_overhead_per_expert_layer_ms"] for s in samples
        )
        memory_values = [
            s["memory_overhead_per_expert_layer_mb"] for s in samples
            if s["memory_overhead_per_expert_layer_mb"] is not None
        ]
        memory_default = statistics.median(memory_values) if memory_values else 0.0
    else:
        time_default = 0.0
        memory_default = 0.0

    return {
        "model": MODEL, "precision": PRECISION, "seq_len": SEQ_LEN,
        # New canonical names: clarify "per expert layer" vs the
        # historical "per layer" (which under 1:1 calibration was the
        # same thing). The cost model prefers the new names but falls
        # back to the legacy ones when a re-aggregation hasn't run.
        "default_time_overhead_per_expert_layer_ms": time_default,
        "default_memory_overhead_per_expert_layer_mb": memory_default,
        "default_time_overhead_per_layer_ms": time_default,    # legacy alias
        "default_memory_overhead_per_layer_mb": memory_default,  # legacy alias
        "num_expert_layers_in_calibration": NUM_MOE_LAYERS,
        "by_shape": by_shape,
        "samples": samples,
    }


def _build_unit_breakdown(unit_rows: List[Dict]) -> Dict:
    """Build per-shape per-component time breakdown from non-"all" rows.

    Returns ``{shape_key: {"per_dp_mode": {dp_mode: {component: ms, ...}}}}``.
    The cost model uses these to skip the analytical ``bwd_mult × forward``
    fall-back from step 6: end-to-end per-component fwd+bwd is now
    measured directly.

    DP modes are kept **isolated**, not averaged: zero3 and zero2sdp produce
    measurably different fwd+bwd times (zero3 re-shards parameters before
    backward and pays an extra all-gather), and the cost model picks the
    matching block based on the caller's ``(zero_stage, sdp)``. See
    ``feedback_dp_mode_isolated_in_unit_breakdown.md`` for the rationale.

    Shape key matches ``_build_runtime_profile``'s key (tp, ep, micro_bsz,
    seq, fsep[, pp]) so the breakdown can be merged into ``by_shape``.

    Notes:
      - attention rows are fsep-agnostic (we collapse them to fsep=off in
        the shell script). The same attention measurement is reused for
        both fsep=on and fsep=off shape keys *within the same dp_mode*.
      - mlp rows are fsep-aware: fsep=on and fsep=off live in the
        respective shape keys. Per-shape per-dp_mode on/off delta is the
        FSEP overhead.
      - duplicate rows at the same (shape, dp_mode) average together
        (re-runs / multiple ranks); cross-dp_mode samples never average.
      - all rows are handled by the legacy aggregator and not seen here.
    """
    breakdown: Dict[str, Dict] = {}

    # First pass: index attention measurements by
    # (tp, ep, micro_bsz, pp, dp_mode). The fsep dimension is dropped —
    # attention is FSEP-agnostic by design (the shell script skips
    # ``unit=attention, fsep=on`` as redundant). DP mode IS kept.
    attn_by_dp: Dict[Tuple, List[float]] = {}
    for row in unit_rows:
        if row.get("profile_unit") != "attention":
            continue
        if row.get("fwd_bwd_ms") is None:
            continue
        micro_bsz = _micro_bsz_at_calibration(
            row["global_bsz"], row.get("chunks", 1)
        )
        key = (row["tp"], row["ep"], micro_bsz, row["pp"], row["dp_mode"])
        attn_by_dp.setdefault(key, []).append(row["fwd_bwd_ms"])

    # Second pass: collect mlp samples per (shape, dp_mode) and mate with
    # the attention measurement from the SAME dp_mode.
    mlp_samples: Dict[Tuple[str, str], List[float]] = {}
    mlp_meta: Dict[Tuple[str, str], Dict] = {}
    for row in unit_rows:
        if row.get("profile_unit") != "mlp":
            continue
        if row.get("fwd_bwd_ms") is None:
            continue
        micro_bsz = _micro_bsz_at_calibration(
            row["global_bsz"], row.get("chunks", 1)
        )
        shape_key = _shape_key(
            row["tp"], row["ep"], micro_bsz, row["fsep"], row["pp"],
        )
        cell_key = (shape_key, row["dp_mode"])
        mlp_samples.setdefault(cell_key, []).append(row["fwd_bwd_ms"])
        meta = mlp_meta.setdefault(cell_key, {})
        meta.setdefault(
            "attn_key",
            (row["tp"], row["ep"], micro_bsz, row["pp"], row["dp_mode"]),
        )
        meta.setdefault("unit_num_layers", row["num_layers"])

    for (shape_key, dp_mode), samples in mlp_samples.items():
        shape_block = breakdown.setdefault(
            shape_key, {"per_dp_mode": {}}
        )
        per_dp = shape_block["per_dp_mode"].setdefault(dp_mode, {})
        per_dp["mlp_fwd_bwd_ms"] = sum(samples) / len(samples)
        per_dp["mlp_n_samples"] = len(samples)
        attn_samples = attn_by_dp.get(mlp_meta[(shape_key, dp_mode)]["attn_key"], [])
        if attn_samples:
            per_dp["attention_fwd_bwd_ms"] = (
                sum(attn_samples) / len(attn_samples)
            )
            per_dp["attention_n_samples"] = len(attn_samples)
        per_dp["unit_num_layers"] = (
            mlp_meta[(shape_key, dp_mode)]["unit_num_layers"]
        )

    # Convenience surface: list of dp_modes present per shape, sorted.
    for shape_block in breakdown.values():
        shape_block["dp_modes"] = sorted(shape_block["per_dp_mode"].keys())

    return breakdown


def _build_chunks_overhead_profile(
    chunks1_rows: List[Dict], chunks_n_rows: List[Dict]
) -> Dict:
    """Pair chunks=1 vs chunks>1 measurements per shape to derive the
    per-microbatch overhead the cost model misses.

    Calibration captures iter_ms and cuda_peak at chunks=1 (one microbatch
    per iteration), so the analytical Alpa formula
    ``T = bottleneck × (num_mb − 1) + Σ stages + opt`` predicts iter_ms
    correctly **as long as** the per-microbatch overhead is folded into
    the calibrated bottleneck. It isn't: synchronous grad reduce (forced
    at chunks > 1 under MoE), 32× more PP send/recv ops, and per-step
    scheduler overhead all add cost that scales linearly with
    num_microbatches but isn't measurable from a single-microbatch run.
    Validation at chunks=32 showed +59 ms / microbatch under-prediction.
    Memory has the same story: reserved-vs-allocated fragmentation grows
    with chunks (caching allocator churn), so a single offset per shape
    isn't enough — we need a slope.

    Method: for each shape that has BOTH a chunks=1 and a chunks>1
    measurement (same tp, ep, micro_bsz, pp, fsep, dp_mode, profile_unit),
    compute:
      time_per_extra_mb_ms       = Δ iter_ms / Δ num_microbatches
      reserved_per_extra_mb_mb   = Δ cuda_peak_reserved_mb / Δ num_microbatches
      allocated_per_extra_mb_mb  = Δ cuda_peak_mb / Δ num_microbatches
    then store per shape (across whichever dp_modes were profiled). Cost
    model adds these slopes × (num_microbatches − 1) on top of the
    chunks=1-anchored prediction.
    """
    def _index(rows: List[Dict]) -> Dict[Tuple, Dict]:
        out: Dict[Tuple, Dict] = {}
        for row in rows:
            if row.get("profile_unit", "all") != "all":
                continue
            if row.get("iter_ms") is None:
                continue
            micro_bsz = _micro_bsz_at_calibration(
                row["global_bsz"], row.get("chunks", 1)
            )
            key = (row["tp"], row["ep"], micro_bsz, row["fsep"], row["pp"],
                   row["dp_mode"])
            out[key] = row
        return out

    chunks1_by_key = _index(chunks1_rows)
    chunks_n_by_key = _index(chunks_n_rows)

    by_shape: Dict[str, Dict] = {}
    for key, row_n in chunks_n_by_key.items():
        row_1 = chunks1_by_key.get(key)
        if row_1 is None:
            continue
        # num_microbatches at calibration time = chunks (each chunk is
        # one microbatch under chunks=1 / chunks=N profiling at this
        # script's defaults).
        d_num_mb = row_n["chunks"] - row_1["chunks"]
        if d_num_mb <= 0:
            continue
        d_iter_ms = row_n["iter_ms"] - row_1["iter_ms"]
        time_slope = d_iter_ms / d_num_mb
        d_alloc_mb: Optional[float] = None
        d_reserved_mb: Optional[float] = None
        if (row_n.get("cuda_peak_mb") is not None
                and row_1.get("cuda_peak_mb") is not None):
            d_alloc_mb = row_n["cuda_peak_mb"] - row_1["cuda_peak_mb"]
        if (row_n.get("cuda_peak_reserved_mb") is not None
                and row_1.get("cuda_peak_reserved_mb") is not None):
            d_reserved_mb = (row_n["cuda_peak_reserved_mb"]
                             - row_1["cuda_peak_reserved_mb"])
        tp_v, ep_v, micro_bsz, fsep, pp_v, dp_mode = key
        shape_key = _shape_key(tp_v, ep_v, micro_bsz, fsep, pp_v)
        block = by_shape.setdefault(shape_key, {"per_dp_mode": {}})
        block["per_dp_mode"][dp_mode] = {
            "time_per_extra_microbatch_ms": time_slope,
            "alloc_per_extra_microbatch_mb": d_alloc_mb,
            "reserved_per_extra_microbatch_mb": d_reserved_mb,
            "chunks_calibrated": [row_1["chunks"], row_n["chunks"]],
            "iter_ms_at_chunks": [row_1["iter_ms"], row_n["iter_ms"]],
        }

    # Convenience: median time/memory slope across shapes (used by cost
    # model as a fallback when a query lands on a shape we didn't pair).
    time_slopes: List[float] = []
    reserved_slopes: List[float] = []
    alloc_slopes: List[float] = []
    for block in by_shape.values():
        for dp_block in block["per_dp_mode"].values():
            time_slopes.append(dp_block["time_per_extra_microbatch_ms"])
            if dp_block["reserved_per_extra_microbatch_mb"] is not None:
                reserved_slopes.append(
                    dp_block["reserved_per_extra_microbatch_mb"]
                )
            if dp_block["alloc_per_extra_microbatch_mb"] is not None:
                alloc_slopes.append(
                    dp_block["alloc_per_extra_microbatch_mb"]
                )

    return {
        "model": MODEL, "precision": PRECISION, "seq_len": SEQ_LEN,
        "default_time_per_extra_microbatch_ms": (
            statistics.median(time_slopes) if time_slopes else 0.0
        ),
        "default_reserved_per_extra_microbatch_mb": (
            statistics.median(reserved_slopes) if reserved_slopes else 0.0
        ),
        "default_alloc_per_extra_microbatch_mb": (
            statistics.median(alloc_slopes) if alloc_slopes else 0.0
        ),
        "n_shape_pairs": len(by_shape),
        "by_shape": by_shape,
    }


def main() -> None:
    rows = _enumerate_logs()
    if not rows:
        print("no parseable logs found; run cost_model_real_test.sh first")
        sys.exit(1)

    # Partition rows along two axes:
    #
    # 1. chunks=1 vs chunks>1 — legacy aggregators (optimizer / runtime /
    #    fsep_overhead / unit_breakdown) only consume chunks=1 rows, since
    #    iter_ms and cuda_peak differ between chunks regimes and shouldn't
    #    be averaged together. Chunks>1 rows feed the new chunks-overhead
    #    aggregator that pairs them against chunks=1 to derive per-
    #    microbatch time + reserved-memory deltas.
    # 2. profile_unit "all" vs attention/mlp — same as before.
    chunks1_rows = [r for r in rows if r.get("chunks", 1) == 1]
    chunks_n_rows = [r for r in rows if r.get("chunks", 1) != 1]
    all_rows = [r for r in chunks1_rows if r.get("profile_unit", "all") == "all"]
    unit_rows = [r for r in chunks1_rows if r.get("profile_unit", "all") != "all"]

    optimizer_profile = _build_optimizer_step_profile(all_rows)
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

    runtime_profile, by_shape_n = _build_runtime_profile(all_rows)

    # Merge per-component breakdown (attention / mlp) under each shape's
    # entry. Cost model reads ``unit_breakdown`` to skip the bwd_mult
    # estimate when an end-to-end fwd+bwd MLP measurement is available.
    unit_breakdown = _build_unit_breakdown(unit_rows)
    if unit_breakdown:
        merged = 0
        for shape_key, block in unit_breakdown.items():
            target = runtime_profile["by_shape"].get(shape_key)
            if target is None:
                # Per-component shape exists in unit_rows but no full-iter
                # "all" row was profiled for it. Surface it anyway under a
                # sparse entry so downstream tooling can still see the
                # measurements (e.g. for the FSEP-overhead derivation).
                runtime_profile["by_shape"][shape_key] = {
                    "shape_only_unit_breakdown": True,
                    "unit_breakdown": block,
                }
            else:
                target["unit_breakdown"] = block
                merged += 1
        print(
            f"  merged unit_breakdown into {merged}/{len(unit_breakdown)} shapes"
        )

    runtime_path = os.path.join(
        CONFIGS_DIR, f"runtime_profiling_{PRECISION}_{MODEL}.json"
    )
    with open(runtime_path, "w") as out_file:
        json.dump(runtime_profile, out_file, indent=2)
    print(f"wrote {runtime_path}")

    fsep_overhead_profile = _build_fsep_overhead_profile(by_shape_n)

    # Per-shape per-dp_mode FSEP overhead from per-component mlp on/off
    # measurements. DP modes are kept isolated (zero3 vs zero2sdp differ
    # measurably in mlp fwd+bwd, and so does their FSEP overhead). The
    # legacy scalar field becomes the mean across whichever dp_modes were
    # profiled, for back-compat with callers that don't yet route by mode.
    mlp_pairs: Dict[Tuple, Dict[str, Dict]] = {}
    for shape_key, block in unit_breakdown.items():
        per_dp = block.get("per_dp_mode") or {}
        # Reconstruct (tp, ep, micro_bsz, pp, fsep) from the shape_key.
        # Cheaper than re-walking unit_rows.
        try:
            tp_part, rest = shape_key.split("_ep", 1)
            ep_part, rest = rest.split("_micro_bsz", 1)
            bsz_part, rest = rest.split("_seq", 1)
            _seq_part, rest = rest.split("_fsep", 1)
            fsep_value, *pp_tail = rest.split("_pp", 1)
            tp_v = int(tp_part[2:])
            ep_v = int(ep_part)
            bsz_v = int(bsz_part)
            pp_v = int(pp_tail[0]) if pp_tail else DEFAULT_PP
        except (ValueError, IndexError):
            continue
        for dp_mode, dp_block in per_dp.items():
            if "mlp_fwd_bwd_ms" not in dp_block:
                continue
            index_key = (tp_v, ep_v, bsz_v, pp_v, dp_mode)
            mlp_pairs.setdefault(index_key, {})[fsep_value] = dp_block

    nl_for_overhead = NUM_MOE_LAYERS  # 1:1 layer mapping during calibration
    # Group per-dp_mode overhead values per (tp, ep, bsz, pp) shape so we
    # can emit both the per-mode dict and the legacy scalar mean.
    per_shape_overheads: Dict[str, Dict[str, float]] = {}
    per_shape_mlp_pairs: Dict[str, Dict[str, Dict]] = {}
    for (tp_v, ep_v, bsz_v, pp_v, dp_mode), pair in mlp_pairs.items():
        if "on" not in pair or "off" not in pair:
            continue
        time_delta = pair["on"]["mlp_fwd_bwd_ms"] - pair["off"]["mlp_fwd_bwd_ms"]
        per_layer_ms = time_delta / max(1, nl_for_overhead)
        shape_key = (
            f"tp{tp_v}_ep{ep_v}_micro_bsz{bsz_v}_seq{SEQ_LEN}"
            + ("" if pp_v == DEFAULT_PP else f"_pp{pp_v}")
        )
        per_shape_overheads.setdefault(shape_key, {})[dp_mode] = per_layer_ms
        per_shape_mlp_pairs.setdefault(shape_key, {})[dp_mode] = {
            "off_ms": pair["off"]["mlp_fwd_bwd_ms"],
            "on_ms": pair["on"]["mlp_fwd_bwd_ms"],
        }

    for shape_key, per_dp_overhead in per_shape_overheads.items():
        target = fsep_overhead_profile["by_shape"].setdefault(
            shape_key, {"n_samples": 0}
        )
        target["time_overhead_per_expert_layer_ms_from_mlp_per_dp_mode"] = (
            per_dp_overhead
        )
        target["mlp_fwd_bwd_per_dp_mode"] = per_shape_mlp_pairs[shape_key]
        # Legacy scalar = mean across DP modes (back-compat for callers that
        # don't yet route by dp_mode). When only one dp_mode is profiled
        # the mean is just that value.
        scalar_overhead = (
            sum(per_dp_overhead.values()) / len(per_dp_overhead)
        )
        target["time_overhead_per_expert_layer_ms_from_mlp"] = scalar_overhead
        if "time_overhead_per_layer_ms" not in target:
            target["time_overhead_per_layer_ms"] = scalar_overhead
            target["time_overhead_per_expert_layer_ms"] = scalar_overhead

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

    chunks_overhead_profile = _build_chunks_overhead_profile(
        chunks1_rows, chunks_n_rows
    )
    chunks_path = os.path.join(
        CONFIGS_DIR,
        f"chunks_overhead_profiling_{PRECISION}_{MODEL}.json",
    )
    with open(chunks_path, "w") as out_file:
        json.dump(chunks_overhead_profile, out_file, indent=2)
    print(f"wrote {chunks_path}")
    print(
        f"  chunks overhead: time_default="
        f"{chunks_overhead_profile['default_time_per_extra_microbatch_ms']:.2f} ms/microbatch  "
        f"reserved_default="
        f"{chunks_overhead_profile['default_reserved_per_extra_microbatch_mb']:.1f} MB/microbatch  "
        f"({chunks_overhead_profile['n_shape_pairs']} paired shapes)"
    )
    for shape_key, entry in sorted(chunks_overhead_profile["by_shape"].items()):
        for dp_mode, dp_block in sorted(entry["per_dp_mode"].items()):
            time_slope = dp_block["time_per_extra_microbatch_ms"]
            res = dp_block.get("reserved_per_extra_microbatch_mb")
            res_str = f"{res:.0f} MB" if res is not None else "n/a"
            print(
                f"    {shape_key}/{dp_mode}: "
                f"time={time_slope:.1f} ms/mb  reserved_Δ={res_str}"
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
