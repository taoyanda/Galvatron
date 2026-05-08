"""Decompose the chunks_overhead residual: (slope − bottleneck).

Important: per-microbatch comm (PP send/recv, MoE all-to-all) happens in
*every* microbatch — at chunks=1 too — so it's already inside the
calibrated bottleneck. The slope minus bottleneck residual is therefore
NOT comm; it's the **chunks>1-specific overhead**:

  residual = chunks2_specific_overhead = sync grad reduce-scatter
                                       + multi-microbatch bookkeeping
                                       + cross-microbatch stream sync

The analytical comm columns below are diagnostic only — they show what
the per-microbatch comm cost *would* be (which is already in the
bottleneck), not what should explain the residual.

Bandwidths (measured on this 4×A100 PCIe box):
    intra-island NVLink: 250 GB/s
    inter-island PCIe (NODE): 21 GB/s

PP groups on this box at PP=2 with consecutive layout:
    stage 0 = ranks {0, 1} (NVLink island A)
    stage 1 = ranks {2, 3} (NVLink island B)
    → PP send/recv crosses islands (21 GB/s)
EP groups at ep=2 are typically intra-island (NVLink) per
``cost_model_real_test.sh``'s ``TP_CONSEC=0`` rule.

For each of the 16 paired shapes, this script:
1. Reads the chunks_overhead JSON's empirical slope.
2. Recovers the calibrated bottleneck from the runtime profile
   (``bottleneck ≈ (iter_ms − opt_ms) / pp``).
3. Computes residual = slope − bottleneck.
4. Computes the analytical estimate for that shape.
5. Reports both side-by-side. If they agree across shapes (within ~5-10
   ms of each other for the comm components, with a consistent
   ~scheduler_overhead residual), the analytical model is a viable
   backstop for shapes lacking chunks=2 calibration.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))

MODEL = "qwen-30b-a3b-e128k8"
PRECISION = "bf16"
SEQ_LEN = 4096

# Hardware (this 4-GPU PCIe-A100 box, two NVLink islands of 2 GPUs each).
INTRA_ISLAND_BW_GBPS = 250.0   # NVLink within island
INTER_ISLAND_BW_GBPS = 21.0    # PCIe NODE crossing islands

# Model arch constants (from cost_model_real_test.sh's CLI flags).
HIDDEN = 2048
INTERMEDIATE = 768  # MoE expert FF intermediate
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 8
NUM_LAYERS = 4
BYTES_PER_BF16 = 2


def _parse_shape_key(key: str):
    """Decode tp{T}_ep{E}_micro_bsz{M}_seq{S}_fsep{ON|OFF}[_pp{P}]."""
    rest = key
    tp_part, rest = rest.split("_ep", 1)
    ep_part, rest = rest.split("_micro_bsz", 1)
    micro_part, rest = rest.split("_seq", 1)
    _seq_part, rest = rest.split("_fsep", 1)
    fsep_part, *pp_tail = rest.split("_pp", 1)
    return {
        "tp": int(tp_part[2:]),
        "ep": int(ep_part),
        "micro_bsz": int(micro_part),
        "fsep": fsep_part,
        "pp": int(pp_tail[0]) if pp_tail else 1,
    }


def _pp_link_bw_gbps(pp: int) -> float:
    """PP send/recv bandwidth at the chosen layout. PP=1 has no PP comm.
    PP=2 with consecutive layout on this box crosses islands → 21 GB/s.
    """
    if pp == 1:
        return float("inf")  # no PP comm
    return INTER_ISLAND_BW_GBPS


def _ep_link_bw_gbps(ep: int) -> float:
    """MoE all-to-all bandwidth. EP=1 has no all-to-all. EP>1 on this
    box uses NVLink (cost_model_real_test.sh sets TP_CONSEC=0 to keep
    EP intra-island when possible)."""
    if ep == 1:
        return float("inf")
    return INTRA_ISLAND_BW_GBPS


def _analytical_per_microbatch_overhead_ms(shape: dict) -> dict:
    """Decompose the analytical per-microbatch overhead into components."""
    micro_bsz = shape["micro_bsz"]
    pp = shape["pp"]
    ep = shape["ep"]
    seq = SEQ_LEN

    # PP send/recv: per microbatch, one fwd activation send + one bwd
    # gradient send between adjacent PP stages. Activation tensor under
    # SBH layout = (seq, micro_bsz, hidden) × bf16. Two transfers per
    # microbatch (fwd-send + bwd-recv).
    if pp > 1:
        act_bytes = seq * micro_bsz * HIDDEN * BYTES_PER_BF16
        pp_p2p_ms = 2 * act_bytes / (_pp_link_bw_gbps(pp) * 1e9) * 1000.0
    else:
        pp_p2p_ms = 0.0

    # MoE all-to-all: per microbatch, per MoE layer, the dispatch
    # all-to-all sends tokens (×top_k expert assignments per token) to
    # the EP ranks holding the chosen experts. With uniform routing the
    # fraction landing remotely is (ep−1)/ep. Per-rank send volume:
    #
    #   per_a2a_bytes = seq × micro_bsz × top_k × hidden × bf16 × (ep−1)/ep
    #
    # The divisor is ``ep`` (EP-group size), NOT ``num_experts`` —
    # all-to-all groups by rank, not by individual expert.
    #
    # Per MoE layer per microbatch, 4 all-to-alls fire:
    #   fwd: dispatch + combine
    #   bwd: gradient scatter + gather (mirror of fwd)
    if ep > 1:
        per_a2a_bytes = (
            seq * micro_bsz * NUM_EXPERTS_PER_TOK * HIDDEN
            * BYTES_PER_BF16 * (ep - 1) / ep
        )
        layers_per_stage = NUM_LAYERS // pp
        a2a_ops_per_layer = 4  # 2 fwd (dispatch + combine) + 2 bwd
        moe_a2a_ms = (
            layers_per_stage * a2a_ops_per_layer * per_a2a_bytes
            / (_ep_link_bw_gbps(ep) * 1e9) * 1000.0
        )
    else:
        moe_a2a_ms = 0.0

    return {
        "pp_p2p_ms": pp_p2p_ms,
        "moe_a2a_ms": moe_a2a_ms,
        "comm_total_ms": pp_p2p_ms + moe_a2a_ms,
    }


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

    print(f"# Analytical per-microbatch overhead vs empirical residual")
    print(f"# Bandwidths: intra-island={INTRA_ISLAND_BW_GBPS} GB/s, "
          f"inter-island={INTER_ISLAND_BW_GBPS} GB/s")
    print()
    cols = (
        f"{'shape':<48} {'pp':>2} {'tp':>2} {'ep':>2} | "
        f"{'iter_cal':>9} {'opt_cal':>7} {'bottle':>7} | "
        f"{'slope':>7} {'chunks2_OH':>11} | "
        f"{'(diag) pp_p2p':>14} {'moe_a2a':>8} {'comm_in_bottleneck':>20}"
    )
    print(cols)
    print("-" * len(cols))

    sched_residuals = []
    for shape_key, entry in sorted(chunks_profile["by_shape"].items()):
        shape = _parse_shape_key(shape_key)
        per_dp = entry.get("per_dp_mode", {})
        dp_block = per_dp.get("zero2sdp") or next(iter(per_dp.values()), None)
        if dp_block is None:
            continue
        slope = dp_block["time_per_extra_microbatch_ms"]

        # Calibrated bottleneck recovered from chunks=1 runtime profile.
        rt_entry = runtime_profile["by_shape"].get(shape_key)
        if rt_entry is None:
            continue
        iter_cal = rt_entry.get("iter_ms")
        opt_cal = rt_entry.get("opt_ms") or 0.0
        if iter_cal is None:
            continue
        sum_stages = max(0.0, iter_cal - opt_cal)
        bottleneck = sum_stages / max(1, shape["pp"])

        residual = slope - bottleneck
        analytical = _analytical_per_microbatch_overhead_ms(shape)
        # Residual IS the chunks>1-specific overhead. Comm is already
        # in the bottleneck (per-microbatch at chunks=1 too) so we
        # don't subtract it. The analytical comm columns are diagnostic
        # only.
        sched_residuals.append(residual)

        print(
            f"{shape_key:<48} {shape['pp']:>2} {shape['tp']:>2} {shape['ep']:>2} | "
            f"{iter_cal:>9.1f} {opt_cal:>7.1f} {bottleneck:>7.1f} | "
            f"{slope:>7.1f} {residual:>+9.1f} | "
            f"{analytical['pp_p2p_ms']:>7.1f} {analytical['moe_a2a_ms']:>8.1f} {analytical['comm_total_ms']:>9.1f}"
        )

    if sched_residuals:
        import statistics
        print()
        print(
            f"scheduler-residual (residual − analytical_comm):  "
            f"median={statistics.median(sched_residuals):+.1f} ms  "
            f"mean={statistics.mean(sched_residuals):+.1f} ms  "
            f"stdev={statistics.stdev(sched_residuals) if len(sched_residuals)>1 else 0:+.1f} ms  "
            f"(n={len(sched_residuals)})"
        )
        print()
        print(
            "Interpretation: if stdev is small relative to the median, "
            "a single platform constant captures the residual cleanly →\n"
            "the analytical model is a viable backstop. Large stdev means "
            "shape-specific variance the analytical model misses; would\n"
            "need to either keep the empirical chunks_overhead per shape "
            "or refine the analytical decomposition."
        )


if __name__ == "__main__":
    main()
