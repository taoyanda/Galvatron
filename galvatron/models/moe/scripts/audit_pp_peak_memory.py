"""Phase 0a — audit cost-model peak memory vs PP depth.

Goal: characterize how ``cost_model.peak_memory_mb`` scales with ``pp`` for a
range of inputs, identify where the prediction falls back to analytical
extrapolation (no calibrated PP entry), and compare predicted vs measured
PP=2 peak on the matrix we just calibrated.

This is a no-GPU read-only audit — it loads the JSON profile and exercises
``CostModel.estimate`` with hypothetical (pp, num_microbatches) tuples.

Run:
    docker exec hetu python3 \\
      /root/Galvatron/galvatron/models/moe/scripts/audit_pp_peak_memory.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
CONFIGS_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))
sys.path.insert(0, REPO_ROOT)

from galvatron.models.moe.cost_model import CostModel  # noqa: E402

MODEL = "qwen-30b-a3b-e128k8"
PRECISION = "bf16"
SEQ_LEN = 4096
NUM_GPUS = 4

_RUNTIME_PATH = os.path.join(
    CONFIGS_DIR, f"runtime_profiling_{PRECISION}_{MODEL}.json"
)


def _calibrated_peak(profile: Dict, tp: int, ep: int, bsz: int,
                     fsep: bool, pp: int) -> Optional[float]:
    """Look up the calibrated cuda_peak_mb for one (tp, ep, bsz, fsep, pp).

    Shape key matches PPCostModel's ``pp_key`` format.
    """
    fsep_tag = "on" if fsep else "off"
    base = f"tp{tp}_ep{ep}_bsz{bsz}_seq{SEQ_LEN}_fsep{fsep_tag}"
    key = base if pp == 1 else f"{base}_pp{pp}"
    entry = profile.get("by_shape", {}).get(key)
    if entry is None:
        return None
    return entry.get("cuda_peak_mb")


def _section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def section_a_calibrated_pp1_vs_pp2(cm: CostModel, profile: Dict) -> None:
    """Section A: report measured PP=1 vs PP=2 peak across calibrated shapes.

    Establishes the empirical activation-amplification baseline. With
    chunks=1 + global_bsz tuned so num_microbatches=1, we expect the two
    to agree (no microbatch stacking). Anything else is a measurement
    inconsistency worth flagging."""
    _section("A. Calibrated PP=1 vs PP=2 peak (same per-rank workload)")
    print(f"{'shape (tp,ep,bsz,fsep)':<32} {'pp1_peak':>10} {'pp2_peak':>10} {'Δ MB':>8} {'Δ%':>6}")
    print("-" * 76)
    pairs = [
        # (tp, ep, fsep, pp1_bsz, pp2_bsz)  — bsz differs because PP=2 calibration
        # ran with global_bsz=4 / dp=1 / pp_stages=2 → per-rank micro-bsz halves.
        (1, 1, False, 1, 2),
        (1, 1, True,  1, 2),
        (1, 2, False, 1, 2),
        (1, 2, True,  1, 2),
    ]
    for tp, ep, fsep, b1, b2 in pairs:
        p1 = _calibrated_peak(profile, tp, ep, b1, fsep, 1)
        p2 = _calibrated_peak(profile, tp, ep, b2, fsep, 2)
        if p1 is None or p2 is None:
            continue
        d = p2 - p1
        pct = 100 * d / p1
        print(f"tp{tp},ep{ep},bsz{b1}->{b2},fsep{'on' if fsep else 'off':>3} "
              f" {p1:>10.0f} {p2:>10.0f} {d:>+8.1f} {pct:>+5.1f}%")


def section_b_pp_scaling_predicted(cm: CostModel) -> None:
    """Section B: cost-model peak prediction across hypothetical PP values.

    Holds (tp, ep, dp) fixed and varies pp via ``num_stages_behind``,
    isolating the activation-amplification reserve term. With NUM_GPUS=4
    we can't actually call ``estimate(pp=8)`` (insufficient devices), so
    we vary ``num_stages_behind`` directly to simulate "as if pp were
    larger."
    """
    _section("B. Cost-model peak vs num_stages_behind (simulated PP depth)")
    cfg = dict(
        num_layers=4, num_gpus=NUM_GPUS, dp=1, pp=1, tp=1, ep=4,
        micro_batch_size=4, global_batch_size=4, seq_len=SEQ_LEN,
        sequence_parallel=True, pp_schedule="1f1b", zero_stage=3,
        sdp=True, recompute=True, bwd_mult=2.0, fsep=False,
    )
    base = cm.estimate(**cfg, num_stages_behind=0)
    print(f"  base (num_stages_behind=0): peak={base.peak_memory_mb:.0f} MB  "
          f"act={base.breakdown['activations_mb']:.0f} MB  "
          f"src={base.breakdown.get('memory_source','?')}")
    print()
    print(f"{'n_behind':>9} {'peak_mb':>10} {'Δ vs n=0':>12} {'extra_reserve_mb':>18}")
    print("-" * 56)
    for n in (0, 1, 2, 3, 4, 7):
        est = cm.estimate(**cfg, num_stages_behind=n)
        delta = est.peak_memory_mb - base.peak_memory_mb
        extra = est.breakdown.get("num_stages_behind_extra_mb", 0.0)
        print(f"{n:>9} {est.peak_memory_mb:>10.0f} {delta:>+12.1f} {extra:>18.1f}")
    # Linearity check: extra_reserve should be exactly n × per_microbatch_act_mb.
    per_mb_est = (cm.estimate(**cfg, num_stages_behind=4).peak_memory_mb
                  - base.peak_memory_mb) / 4.0
    print(f"\n  per-microbatch reserve (analytical): {per_mb_est:.1f} MB")


def section_c_pp1_to_pp2_prediction(cm: CostModel, profile: Dict) -> None:
    """Section C: predict PP=2 peak from PP=1 calibration + analytical
    extra_reserve, compare to measured PP=2 peak.

    Plan-doc 0c: bound the analytical-formula error against ground truth
    on at least 4 shapes."""
    _section("C. Predicted PP=2 peak (PP=1 cal + analytical reserve) vs measured")
    print(f"{'tp,ep,fsep':<14} {'pp1_meas':>10} {'predicted':>10} "
          f"{'pp2_meas':>10} {'pred-meas':>10} {'pct':>7}")
    print("-" * 72)
    rows = []
    pairs = [
        (1, 1, False, 1, 2),
        (1, 1, True,  1, 2),
        (1, 2, False, 1, 2),
        (1, 2, True,  1, 2),
    ]
    for tp, ep, fsep, b1, b2 in pairs:
        meas_pp1 = _calibrated_peak(profile, tp, ep, b1, fsep, 1)
        meas_pp2 = _calibrated_peak(profile, tp, ep, b2, fsep, 2)
        if meas_pp1 is None or meas_pp2 is None:
            continue
        # Predict PP=2 peak: use PP=1 calibrated peak as base, add reserve
        # for n_behind = 1 (first stage of a PP=2 1F1B pipeline holds 1
        # extra microbatch beyond the natural 1-in-flight). Bsz scales:
        # per-rank halves at PP=2 in this calibration.
        per_mb_act = cm.intra.per_microbatch_activation_mb(
            num_layers=4 // 2,         # layers per stage at pp=2
            per_rank_micro_bsz=b2,     # per-stage bsz under pp=2
            seq_len=SEQ_LEN, tp=tp, recompute=True, sequence_parallel=True,
        )
        # Naive prediction: PP=1 peak (which embeds 4-layer state + 1
        # microbatch act @ b1) + 1 microbatch extra act @ pp=2 layers/bsz.
        # This is a *first-cut* — section B+the deeper Phase 0b refinement
        # will tighten what's stacked.
        predicted = meas_pp1 + 1 * per_mb_act
        diff = predicted - meas_pp2
        pct = 100 * diff / meas_pp2
        rows.append((tp, ep, fsep, meas_pp1, predicted, meas_pp2, diff, pct))
        tag = f"tp{tp},ep{ep},{'on' if fsep else 'off':>3}"
        print(f"{tag:<14} {meas_pp1:>10.0f} {predicted:>10.0f} "
              f"{meas_pp2:>10.0f} {diff:>+10.1f} {pct:>+6.1f}%")
    if rows:
        import statistics
        pcts = [r[7] for r in rows]
        print(f"\n  abs error: median={statistics.median(abs(p) for p in pcts):.2f}%  "
              f"max={max(abs(p) for p in pcts):.2f}%")


def section_d_microbatch_amplification(cm: CostModel) -> None:
    """Section D: with the Phase 0b fix, varying ``num_microbatches`` at
    fixed pp must auto-derive the natural microbatch stacking reserve
    (``natural_n_behind = min(pp, num_mb) − 1``) and surface it via
    ``num_stages_behind_extra_mb`` in the breakdown."""
    _section("D. Auto-derived microbatch stacking (Phase 0b fix)")
    # tp1, ep1, pp=2, dp=2, fsep=off. Calibrated entry exists at
    # tp1_ep1_bsz2_seq4096_fsepoff_pp2 (per-rank bsz=2 since dp*ep=2).
    base = dict(
        num_layers=4, num_gpus=NUM_GPUS, dp=2, pp=2, tp=1, ep=1,
        micro_batch_size=4,  # per-rank = 4/(dp*ep) = 4/2 = 2
        seq_len=SEQ_LEN, sequence_parallel=True, pp_schedule="1f1b",
        zero_stage=3, sdp=True, recompute=True, bwd_mult=2.0, fsep=False,
    )
    print("Config: pp=2, tp=1, ep=1, dp=2, micro_batch_size=4 (per-rank=2), "
          "recompute=True, fsep=off")
    print(f"{'global_bsz':>10} {'num_mb':>7} {'natural_n_behind':>17} "
          f"{'peak_mb':>9} {'extra':>7} {'mem_src':<60}")
    print("-" * 116)
    for global_bsz in (4, 8, 12, 16, 32):
        try:
            est = cm.estimate(**base, global_batch_size=global_bsz,
                              num_stages_behind=0)
        except (ValueError, KeyError) as exc:
            print(f"  global_bsz={global_bsz}: skip ({type(exc).__name__})")
            continue
        bd = est.breakdown
        num_mb = bd.get("num_microbatches", 0)
        nn = bd.get("natural_n_behind", 0)
        extra = bd.get("num_stages_behind_extra_mb", 0)
        ms = bd.get("memory_source", "")
        print(f"{global_bsz:>10} {num_mb:>7.0f} {nn:>17.0f} "
              f"{est.peak_memory_mb:>9.0f} {extra:>7.1f} {ms:<60}")


def main() -> None:
    print(f"Phase 0a audit — {MODEL} ({PRECISION}, seq={SEQ_LEN}, GPUs={NUM_GPUS})")
    if not os.path.isfile(_RUNTIME_PATH):
        print(f"ERROR: runtime profile missing at {_RUNTIME_PATH}")
        sys.exit(1)
    with open(_RUNTIME_PATH) as f:
        profile = json.load(f)
    cm = CostModel(MODEL)
    section_a_calibrated_pp1_vs_pp2(cm, profile)
    section_b_pp_scaling_predicted(cm)
    section_c_pp1_to_pp2_prediction(cm, profile)
    section_d_microbatch_amplification(cm)


if __name__ == "__main__":
    main()
