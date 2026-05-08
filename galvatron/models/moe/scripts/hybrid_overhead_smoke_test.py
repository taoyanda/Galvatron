"""Hybrid (analytical + empirical) chunks-overhead smoke test.

Hypothesis: the per-microbatch cost decomposes as
    slope = stage_compute_with_comm + chunks_OH

where ``stage_compute_with_comm`` is recoverable from chunks=1 calibration
(modulo a per_iter_OH bias) and ``chunks_OH`` is the chunks>1-specific
multi-microbatch overhead. The chunks>1 overhead has two sub-components:
  (a) Analytical: synchronous comm operations that only fire when
      ``num_microbatches > 1`` and that we can predict from message size
      and link bandwidth — for our setup this is dominated by the
      ``--no_async_grad_reduce`` path's reduce-scatter on attention/non-
      MoE params (MoE expert params and dp=1 cases have no reduce).
  (b) Empirical: per-microbatch scheduler/Python overhead that doesn't
      have a clean closed form — fit as a single regime constant per
      (PP, FSEP) combination.

For the calibrated shapes, this script reports:
  - empirical slope (from chunks=2 calibration)
  - analytical chunks>1-specific reduce-scatter cost
  - per-(PP, FSEP) regime residual = (slope − bottleneck − analytical)
  - hybrid predicted slope = bottleneck + analytical + regime_constant
  - drift = hybrid predicted − empirical, per shape

Smoke test pass criterion: hybrid drift should be smaller and more
uniform than pure analytical (Alpa-only) drift, ideally within ~10-15%
of the empirical slope on average. If so, the hybrid is a viable
backstop for shapes the chunks=2 sweep doesn't cover.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))

MODEL = "qwen-30b-a3b-e128k8"
PRECISION = "bf16"
SEQ_LEN = 4096

# Hardware (4×A100 PCIe, 2 NVLink islands of 2 GPUs each).
INTRA_ISLAND_BW_GBPS = 250.0
INTER_ISLAND_BW_GBPS = 21.0

HIDDEN = 2048
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 8
NUM_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 64
NUM_LAYERS = 4
BYTES_PER_BF16 = 2


def _parse_shape(key: str):
    rest = key
    tp_part, rest = rest.split("_ep", 1)
    ep_part, rest = rest.split("_micro_bsz", 1)
    micro_part, rest = rest.split("_seq", 1)
    _seq, rest = rest.split("_fsep", 1)
    fsep_part, *pp_tail = rest.split("_pp", 1)
    return {
        "tp": int(tp_part[2:]),
        "ep": int(ep_part),
        "micro_bsz": int(micro_part),
        "fsep": fsep_part,
        "pp": int(pp_tail[0]) if pp_tail else 1,
    }


def _attention_params_per_layer_bytes() -> float:
    """Attention block params: Q, K, V, O projections (bf16).

    GQA: Q has full num_heads × head_dim; K/V have num_kv_heads × head_dim.
    """
    q_params = HIDDEN * NUM_HEADS * HEAD_DIM
    kv_params = HIDDEN * NUM_KV_HEADS * HEAD_DIM
    o_params = NUM_HEADS * HEAD_DIM * HIDDEN
    return (q_params + 2 * kv_params + o_params) * BYTES_PER_BF16


def _chunks_specific_sync_reduce_ms(shape: dict, dp: int) -> float:
    """Analytical estimate of the chunks>1-specific sync reduce-scatter
    cost per microbatch. Under ``--no_async_grad_reduce`` (forced at
    chunks>1 with MoE), each microbatch's backward synchronously reduces
    attention/non-MoE gradients across the dp×ep replication group.

    MoE expert gradients are NOT reduced (each rank holds unique experts).
    Returns 0 when dp×ep == 1 (no replication, no reduce).
    """
    pp = shape["pp"]
    ep = shape["ep"]
    # Replication group size: attention/non-MoE params replicate across
    # dp×ep ranks. With ep>1 the body's TP=1 sees every rank as a
    # replica; the dp_of_ep group is dp×ep.
    replication_group = dp * ep
    if replication_group <= 1:
        return 0.0
    layers_per_stage = NUM_LAYERS // pp
    grad_bytes = layers_per_stage * _attention_params_per_layer_bytes()
    # Reduce-scatter sends ``(grp − 1) / grp`` of the data per rank in
    # the ring algorithm.
    bytes_per_rank = grad_bytes * (replication_group - 1) / replication_group
    # dp_of_ep group can span islands; pessimistically assume inter-
    # island when dp>1 (DP cross-island typical) and intra when ep>1
    # only. For dp=1 ep=2 it's intra-island. For dp=2 ep=1 it could be
    # either depending on layout — assume inter-island for safety.
    if dp > 1:
        bw = INTER_ISLAND_BW_GBPS
    else:
        bw = INTRA_ISLAND_BW_GBPS
    return bytes_per_rank / (bw * 1e9) * 1000.0


def main() -> None:
    chunks_path = os.path.join(
        CONFIGS_DIR, f"chunks_overhead_profiling_{PRECISION}_{MODEL}.json"
    )
    runtime_path = os.path.join(
        CONFIGS_DIR, f"runtime_profiling_{PRECISION}_{MODEL}.json"
    )
    with open(chunks_path) as f:
        chunks_profile = json.load(f)
    with open(runtime_path) as f:
        runtime_profile = json.load(f)

    print(f"# Hybrid chunks-overhead model smoke test")
    print(f"#   intra-island={INTRA_ISLAND_BW_GBPS} GB/s, "
          f"inter-island={INTER_ISLAND_BW_GBPS} GB/s")
    print()

    # Step 1: gather per-shape data.
    rows = []
    for shape_key, entry in sorted(chunks_profile["by_shape"].items()):
        shape = _parse_shape(shape_key)
        per_dp = entry.get("per_dp_mode", {})
        dp_block = per_dp.get("zero2sdp") or next(iter(per_dp.values()), None)
        if dp_block is None:
            continue
        rt_entry = runtime_profile["by_shape"].get(shape_key)
        if rt_entry is None or rt_entry.get("iter_ms") is None:
            continue
        slope = dp_block["time_per_extra_microbatch_ms"]
        iter_cal = rt_entry["iter_ms"]
        opt_cal = rt_entry.get("opt_ms") or 0.0
        bottleneck = max(0.0, iter_cal - opt_cal) / max(1, shape["pp"])
        # Recover dp from constraints: dp×pp×tp×ep = num_gpus = 4
        dp = max(1, 4 // (shape["pp"] * shape["tp"] * shape["ep"]))
        analytical_sync_ms = _chunks_specific_sync_reduce_ms(shape, dp)
        regime_key = (shape["pp"], shape["fsep"])
        rows.append({
            "shape_key": shape_key,
            "shape": shape,
            "dp": dp,
            "slope": slope,
            "bottleneck": bottleneck,
            "analytical_sync_ms": analytical_sync_ms,
            "regime_key": regime_key,
        })

    # Step 2: per-(PP, FSEP) regime constant fit, leave-one-out.
    # For each shape: hold it out, fit regime constant from the others
    # in the same regime, predict its slope, compute drift.
    print(f"{'shape':<48} {'pp':>2} {'fsep':>4} | "
          f"{'slope':>7} {'bottle':>7} {'a_sync':>7} {'regime_K':>9} | "
          f"{'pure_alpa':>10} {'pure_drift':>11} | "
          f"{'hybrid':>8} {'hyb_drift':>10}")
    print("-" * 134)

    pure_drifts = []
    hybrid_drifts = []
    for row in rows:
        # Leave-one-out regime constant.
        same_regime = [r for r in rows if r["regime_key"] == row["regime_key"]
                       and r is not row]
        if same_regime:
            constants = [
                r["slope"] - r["bottleneck"] - r["analytical_sync_ms"]
                for r in same_regime
            ]
            regime_const = statistics.median(constants)
        else:
            regime_const = 0.0  # no peers; no signal

        # Pure-Alpa prediction (Alpa baseline, no chunks_overhead).
        # Uses bottleneck as the steady-state per-microbatch cost.
        pure_alpa = row["bottleneck"]
        pure_drift = pure_alpa - row["slope"]
        pure_drifts.append(pure_drift)

        # Hybrid prediction.
        hybrid = row["bottleneck"] + row["analytical_sync_ms"] + regime_const
        hyb_drift = hybrid - row["slope"]
        hybrid_drifts.append(hyb_drift)

        print(
            f"{row['shape_key']:<48} {row['shape']['pp']:>2} "
            f"{row['shape']['fsep']:>4} | "
            f"{row['slope']:>7.1f} {row['bottleneck']:>7.1f} "
            f"{row['analytical_sync_ms']:>7.2f} {regime_const:>+9.1f} | "
            f"{pure_alpa:>10.1f} {pure_drift:>+11.1f} | "
            f"{hybrid:>8.1f} {hyb_drift:>+10.1f}"
        )

    print()
    if pure_drifts and hybrid_drifts:
        print(f"Pure-Alpa drift: median={statistics.median(pure_drifts):+.1f}  "
              f"abs_mean={statistics.mean(abs(d) for d in pure_drifts):.1f}  "
              f"abs_max={max(abs(d) for d in pure_drifts):.1f}")
        print(f"Hybrid    drift: median={statistics.median(hybrid_drifts):+.1f}  "
              f"abs_mean={statistics.mean(abs(d) for d in hybrid_drifts):.1f}  "
              f"abs_max={max(abs(d) for d in hybrid_drifts):.1f}")

    # Top-config smoke test: predict iter_ms at chunks=32 using each model.
    # Top config: tp1_ep2_micro_bsz4_fsepoff_pp2 / zero2sdp
    print()
    print("--- Top-config smoke test (chunks=32, measured iter_ms = 7507) ---")
    top_key = "tp1_ep2_micro_bsz4_seq4096_fsepoff_pp2"
    top_row = next((r for r in rows if r["shape_key"] == top_key), None)
    if top_row is not None:
        iter_cal = runtime_profile["by_shape"][top_key]["iter_ms"]
        num_mb = 32

        # Pure-empirical (current production): use slope directly.
        pred_emp = iter_cal + top_row["slope"] * (num_mb - 1)

        # Hybrid (held-out): use bottleneck + analytical + regime constant
        # fit from the other PP=2 fsep=off shapes.
        same_regime = [r for r in rows if r["regime_key"] == ("2", "off")
                       or r["regime_key"] == (2, "off")]
        same_regime = [r for r in same_regime if r is not top_row]
        if same_regime:
            consts = [r["slope"] - r["bottleneck"] - r["analytical_sync_ms"]
                      for r in same_regime]
            K = statistics.median(consts)
        else:
            K = 0.0
        hybrid_slope = top_row["bottleneck"] + top_row["analytical_sync_ms"] + K
        pred_hybrid = iter_cal + hybrid_slope * (num_mb - 1)

        # Pure-Alpa baseline: bottleneck only as the per-microbatch cost.
        pred_alpa = iter_cal + top_row["bottleneck"] * (num_mb - 1)

        for label, pred in [
            ("pure-Alpa (no chunks_OH)", pred_alpa),
            ("hybrid (analytical sync + held-out regime const)", pred_hybrid),
            ("pure-empirical (chunks=2 slope)", pred_emp),
        ]:
            drift = pred - 7507
            print(f"  {label:<55} pred={pred:>7.0f} ms  drift={drift:+6.0f} ms = {100*drift/7507:+.2f}%")


if __name__ == "__main__":
    main()
