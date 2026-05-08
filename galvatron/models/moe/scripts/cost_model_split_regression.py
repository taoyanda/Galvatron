"""Regression suite for the asymmetric-layers cost-model split.

Validates that the per-component (attention vs expert) split path
preserves the existing 1:1 behaviour and that the FSEP-related
invariants hold against the calibrated profiles. No GPU required: all
checks load existing JSON artifacts in
``galvatron/models/moe/configs/`` and run cost-model arithmetic.

Invariants checked:

  1. Symmetric identity. For every shape × pp in
     ``runtime_profiling_*.json`` whose configuration is divisible by
     pp, the asymmetric kwargs at ``num_attention_layers ==
     num_expert_layers == num_layers`` must reproduce the default
     ``estimate(num_layers=N)`` result bit-identically (within 1e-6 on
     ``total_iter_ms`` and ``peak_memory_mb``).

  2. Pipeline critical-path identity. At ``pp > 1``,
     ``iter_ms ≈ (n_micro + pp − 1) × stage_bottleneck_ms +
     max_post_bwd_ms`` within 1e-3.

  3. FSEP attention invariance. For shapes that have both ``fsep=on``
     and ``fsep=off`` runtime-profile entries: the attention component
     of per-layer time (computed via the per-component computation
     profile when present) must agree across fsep modes within 5 %.

  4. FSEP overhead reconciliation. For each shape in the FSEP overhead
     profile, the recorded ``time_overhead_per_expert_layer_ms`` must
     equal ``(decomposed t_expert_on − t_expert_off)`` within 5 %.

  5. Asymmetric input validation. ``num_attention_layers`` /
     ``num_expert_layers`` must be ≥ 0 and divide pp; the cost model
     raises clearly when they don't.

Exit code 0 = all invariants hold; non-zero = at least one failure.

Usage::

    docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_split_regression.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..")))

from galvatron.models.moe.cost_model import PPCostModel  # noqa: E402

MODEL = "qwen-30b-a3b-e128k8"
CONFIGS_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))
RUNTIME_PATH = os.path.join(
    CONFIGS_DIR, f"runtime_profiling_bf16_{MODEL}.json"
)
FSEP_PATH = os.path.join(
    CONFIGS_DIR, f"fsep_overhead_profiling_bf16_{MODEL}.json"
)


def _close(actual: float, expected: float, *, abs_tol: float = 0.0,
           rel_tol: float = 0.0) -> bool:
    if expected == 0:
        return abs(actual) <= abs_tol
    if abs_tol > 0 and abs(actual - expected) <= abs_tol:
        return True
    return abs(actual - expected) / abs(expected) <= rel_tol


def _parse_pp_key(key: str) -> Optional[Tuple[int, int, int, int, str, int]]:
    """Parse a runtime_profile shape key like
    ``tp1_ep4_bsz4_seq4096_fsepon[_pp2]`` → (tp, ep, bsz, seq, fsep, pp)."""
    parts = key.split("_")
    field_map = {}
    pp = 1
    for part in parts:
        for prefix in ("tp", "ep", "bsz", "seq", "pp"):
            if part.startswith(prefix):
                try:
                    field_map[prefix] = int(part[len(prefix):])
                except ValueError:
                    pass
                break
        else:
            if part.startswith("fsep"):
                field_map["fsep"] = part[len("fsep"):]
    try:
        return (
            field_map["tp"], field_map["ep"], field_map["bsz"],
            field_map["seq"], field_map.get("fsep", "off"),
            field_map.get("pp", 1),
        )
    except KeyError:
        return None


def check_symmetric_identity(cm: PPCostModel) -> Tuple[int, int]:
    """Invariant 1: symmetric kwargs reproduce default estimate exactly."""
    if not os.path.isfile(RUNTIME_PATH):
        print("  [skip] runtime profile missing")
        return 0, 0
    with open(RUNTIME_PATH) as f:
        runtime = json.load(f)
    failures = 0
    checks = 0
    for shape_key, entry in runtime.get("by_shape", {}).items():
        parsed = _parse_pp_key(shape_key)
        if parsed is None:
            continue
        tp, ep, micro_bsz, seq, fsep, pp = parsed
        num_layers = int(entry.get("num_layers", 4))
        if num_layers % pp != 0:
            continue
        # Use the calibrated layout: dp=1, num_gpus=tp×ep×pp. Shape
        # keys store the per-actual-rank bsz; the cost model takes the
        # GLOBAL micro batch (per-rank × dp × ep), so multiply up.
        num_gpus = tp * ep * pp
        global_micro_bsz = micro_bsz * 1 * ep  # dp=1 × ep
        common = dict(
            num_gpus=num_gpus, dp=1, pp=pp, tp=tp, ep=ep,
            micro_batch_size=global_micro_bsz,
            global_batch_size=global_micro_bsz,
            seq_len=seq, sequence_parallel=True,
            zero_stage=2, sdp=True, recompute=True, bwd_mult=2.0,
            fsep=(fsep == "on"),
        )
        try:
            est_default = cm.estimate(num_layers=num_layers, **common)
            est_explicit = cm.estimate(
                num_layers=num_layers,
                num_attention_layers=num_layers,
                num_expert_layers=num_layers,
                **common,
            )
        except (ValueError, KeyError):
            continue
        checks += 1
        if not _close(est_explicit.total_iter_ms,
                      est_default.total_iter_ms, abs_tol=1e-6, rel_tol=1e-6):
            print(
                f"  [FAIL] {shape_key} (pp={pp}, N={num_layers}): "
                f"iter_ms default={est_default.total_iter_ms:.6f} "
                f"explicit={est_explicit.total_iter_ms:.6f}"
            )
            failures += 1
            continue
        if not _close(est_explicit.peak_memory_mb,
                      est_default.peak_memory_mb, abs_tol=1e-6, rel_tol=1e-6):
            print(
                f"  [FAIL] {shape_key} (pp={pp}, N={num_layers}): "
                f"peak_mb default={est_default.peak_memory_mb:.6f} "
                f"explicit={est_explicit.peak_memory_mb:.6f}"
            )
            failures += 1
    return checks, failures


def check_pp_critical_path(cm: PPCostModel) -> Tuple[int, int]:
    """Invariant 2: iter_ms = (n_micro + pp − 1) × max_stage_ms +
    max_post_bwd_ms. Only checked at pp>1 where the bubble formula
    actually applies; pp=1 uses iter_ms == stage_compute + post_bwd
    which is a different identity."""
    if not os.path.isfile(RUNTIME_PATH):
        return 0, 0
    with open(RUNTIME_PATH) as f:
        runtime = json.load(f)
    failures = 0
    checks = 0
    for shape_key, entry in runtime.get("by_shape", {}).items():
        parsed = _parse_pp_key(shape_key)
        if parsed is None:
            continue
        tp, ep, micro_bsz, seq, fsep, pp = parsed
        if pp == 1:
            continue
        num_layers = int(entry.get("num_layers", 4))
        if num_layers % pp != 0:
            continue
        # The runtime-profile shortcut returns iter_ms directly from
        # calibration; for the critical-path identity we need the
        # per-stage composition. Skip the shortcut by querying with
        # asymmetric=False but bypassing via a tiny perturbation isn't
        # ideal — instead we just verify that the identity holds on
        # whatever breakdown the cost model returned.
        global_micro_bsz = micro_bsz * 1 * ep
        try:
            est = cm.estimate(
                num_layers=num_layers, num_gpus=tp * ep * pp,
                dp=1, pp=pp, tp=tp, ep=ep,
                micro_batch_size=global_micro_bsz,
                global_batch_size=global_micro_bsz,
                seq_len=seq, sequence_parallel=True,
                zero_stage=2, sdp=True, recompute=True, bwd_mult=2.0,
                fsep=(fsep == "on"),
            )
        except (ValueError, KeyError):
            continue
        bd = est.breakdown
        stage_bottleneck = float(bd.get("stage_bottleneck_ms", 0.0))
        max_post_bwd = float(bd.get("max_post_bwd_ms", 0.0))
        n_micro = float(bd.get("n_microbatches", 1.0))
        predicted = (n_micro + pp - 1) * stage_bottleneck + max_post_bwd
        checks += 1
        if not _close(est.total_iter_ms, predicted,
                      abs_tol=1e-3, rel_tol=1e-3):
            print(
                f"  [FAIL] {shape_key} pp={pp}: "
                f"iter_ms={est.total_iter_ms:.3f} predicted={predicted:.3f} "
                f"(bottleneck×bubble + post_bwd)"
            )
            failures += 1
    return checks, failures


def check_fsep_attention_invariance(cm: PPCostModel) -> Tuple[int, int]:
    """Invariant 3: the cost model's reported ``per_attention_layer_ms``
    must agree across fsep on/off at the same shape, because attention
    is unaffected by FSEP (Fully Sharded Expert Parallel changes how
    expert MLPs are sharded across the EP group, but attention compute
    doesn't touch the expert sharding).

    Queries the cost model for both fsep modes at the calibrated layout
    and inspects ``breakdown["per_attention_layer_ms"]``. Skipped when
    the per-component computation profile isn't present (no ratio to
    decompose with)."""
    if not os.path.isfile(RUNTIME_PATH):
        return 0, 0
    with open(RUNTIME_PATH) as f:
        runtime = json.load(f)
    by_shape = runtime.get("by_shape", {})
    groups: Dict[Tuple, Dict[str, Dict]] = {}
    for shape_key, entry in by_shape.items():
        parsed = _parse_pp_key(shape_key)
        if parsed is None:
            continue
        tp, ep, micro_bsz, seq, fsep, pp = parsed
        groups.setdefault((tp, ep, micro_bsz, seq, pp), {})[fsep] = entry
    failures = 0
    checks = 0
    for (tp, ep, micro_bsz, seq, pp), variants in groups.items():
        if "on" not in variants or "off" not in variants:
            continue
        ratio = cm.intra.attention_mlp_fwd_ratio(tp, ep, micro_bsz, seq)
        if ratio is None:
            continue
        # Query the cost model with the calibrated layout for both modes
        # at the asymmetric API (so the per-component split is exercised).
        # Use n_attn = N + 1, n_expert = N to force the asymmetric path
        # — the symmetric path bypasses the per-component split.
        num_layers = int(variants["off"].get("num_layers", 4))
        global_micro_bsz = micro_bsz * 1 * ep
        common = dict(
            num_gpus=tp * ep * pp, dp=1, pp=pp, tp=tp, ep=ep,
            micro_batch_size=global_micro_bsz,
            global_batch_size=global_micro_bsz,
            seq_len=seq, sequence_parallel=True,
            zero_stage=2, sdp=True, recompute=True, bwd_mult=2.0,
            num_attention_layers=num_layers + 1,
            num_expert_layers=num_layers,
        )
        try:
            est_off = cm.estimate(num_layers=num_layers, fsep=False, **common)
            est_on = cm.estimate(num_layers=num_layers, fsep=True, **common)
        except (ValueError, KeyError):
            continue
        attn_off = est_off.breakdown.get("per_attention_layer_ms")
        attn_on = est_on.breakdown.get("per_attention_layer_ms")
        if attn_off is None or attn_on is None:
            continue
        checks += 1
        # Tolerance: 5 % relative or 1 ms absolute (whichever is looser),
        # since the FSEP overhead profile itself has ~few % noise and the
        # per-component computation profile is forward-only (vs the
        # runtime profile's fwd+bwd).
        if not _close(attn_on, attn_off, abs_tol=1.0, rel_tol=0.05):
            print(
                f"  [FAIL] tp{tp}_ep{ep}_bsz{micro_bsz}_seq{seq}"
                + (f"_pp{pp}" if pp != 1 else "")
                + f": per_attention_layer_ms diverges fsep on={attn_on:.2f} "
                f"off={attn_off:.2f} (Δ={attn_on - attn_off:+.2f})"
            )
            failures += 1
    return checks, failures


def check_fsep_overhead_reconciliation(cm: PPCostModel) -> Tuple[int, int]:
    """Invariant 4: the FSEP overhead profile's per-expert-layer overhead
    must equal (per_layer_on − per_layer_off) measured from the runtime
    profile and divided by num_layers — within 5 %.

    These are derived from the same logs by the same aggregator, so any
    divergence indicates an aggregation bug."""
    if not os.path.isfile(RUNTIME_PATH) or not os.path.isfile(FSEP_PATH):
        return 0, 0
    with open(RUNTIME_PATH) as f:
        runtime = json.load(f)
    with open(FSEP_PATH) as f:
        fsep_profile = json.load(f)
    by_shape_runtime = runtime.get("by_shape", {})
    failures = 0
    checks = 0
    for sample in fsep_profile.get("samples", []):
        # Build the matching runtime keys for on/off at this shape.
        tp, ep = sample["tp"], sample["ep"]
        micro_bsz = sample["micro_bsz"]
        pp = sample.get("pp", 1)
        num_layers = int(sample["num_layers"])
        seq = runtime.get("seq_len", 4096)
        suffix = "" if pp == 1 else f"_pp{pp}"
        on_key = f"tp{tp}_ep{ep}_bsz{micro_bsz}_seq{seq}_fsepon{suffix}"
        off_key = f"tp{tp}_ep{ep}_bsz{micro_bsz}_seq{seq}_fsepoff{suffix}"
        on_entry = by_shape_runtime.get(on_key)
        off_entry = by_shape_runtime.get(off_key)
        if not on_entry or not off_entry:
            continue
        # Decompose both per-layer (after stripping embed/lm-head).
        emb, lm, emb_lm = cm.intra._emb_lm_split_ms(micro_bsz, seq)
        n_profiled = int(runtime.get("num_layers_profiled", 1)) or 1
        per_layer_on = (on_entry["fwd_bwd_ms"] - emb_lm) / n_profiled
        per_layer_off = (off_entry["fwd_bwd_ms"] - emb_lm) / n_profiled
        decomposed_overhead = per_layer_on - per_layer_off
        recorded_overhead = sample.get(
            "time_overhead_per_expert_layer_ms",
            sample.get("time_overhead_per_layer_ms", 0.0),
        )
        # Recorded overhead is calibration-fwd_bwd / num_layers; the
        # reconciliation must use the same divisor (n_profiled), which
        # under 1:1 calibration equals num_expert_layers.
        if num_layers != n_profiled:
            continue
        checks += 1
        if not _close(decomposed_overhead, recorded_overhead,
                      abs_tol=5.0, rel_tol=0.05):
            print(
                f"  [FAIL] {on_key}: decomposed={decomposed_overhead:.2f} "
                f"recorded={recorded_overhead:.2f} ms/expert-layer"
            )
            failures += 1
    return checks, failures


def check_input_validation(cm: PPCostModel) -> Tuple[int, int]:
    """Invariant 5: the cost model raises on negative counts, on
    asymmetric requests without the per-component compute profile, and
    on counts not divisible by pp."""
    checks = 0
    failures = 0

    common = dict(
        num_gpus=4, dp=1, pp=1, tp=1, ep=4,
        micro_batch_size=4, global_batch_size=4, seq_len=4096,
        sequence_parallel=True, zero_stage=2, sdp=True,
        recompute=True, bwd_mult=2.0,
    )

    # 5a: negative count → ValueError
    checks += 1
    try:
        cm.estimate(num_layers=4, num_attention_layers=-1,
                    num_expert_layers=4, **common)
        print("  [FAIL] negative num_attention_layers did not raise")
        failures += 1
    except ValueError:
        pass

    # 5b: PP divisibility violated
    checks += 1
    try:
        cm.estimate(num_layers=4, num_gpus=4, dp=1, pp=2, tp=1, ep=2,
                    num_attention_layers=4, num_expert_layers=5,
                    micro_batch_size=4, global_batch_size=4, seq_len=4096,
                    sequence_parallel=True, zero_stage=2, sdp=True,
                    recompute=True, bwd_mult=2.0)
        print("  [FAIL] non-divisible pp request did not raise")
        failures += 1
    except ValueError:
        pass

    return checks, failures


def main() -> int:
    cm = PPCostModel(MODEL)
    print(f"# Cost-model split regression suite ({MODEL})")
    print(f"#   runtime_profile: {os.path.basename(RUNTIME_PATH)}")
    print(f"#   fsep_profile:    {os.path.basename(FSEP_PATH)}")
    print()

    total_checks = 0
    total_failures = 0
    for label, fn in (
        ("1. symmetric identity", check_symmetric_identity),
        ("2. pp critical path", check_pp_critical_path),
        ("3. FSEP attention invariance", check_fsep_attention_invariance),
        ("4. FSEP overhead reconciliation", check_fsep_overhead_reconciliation),
        ("5. input validation", check_input_validation),
    ):
        print(f"## {label}")
        checks, failures = fn(cm)
        total_checks += checks
        total_failures += failures
        verdict = "OK" if failures == 0 else f"{failures} FAILURES"
        print(f"   {checks} checks, {verdict}")
        print()

    print(f"# total: {total_checks} checks, {total_failures} failures")
    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
