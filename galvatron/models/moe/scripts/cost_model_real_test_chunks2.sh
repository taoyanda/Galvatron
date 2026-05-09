#!/usr/bin/env bash
# Chunks=2 calibration sweep — pairs with the existing chunks=1 main
# matrix to derive per-microbatch overhead (time + memory) per shape AND
# per component (attention vs mlp).
#
# Why per-shape: validation at chunks=32 showed iter_ms under-prediction
# of ~34% (5,613 ms predicted vs 7,507 ms measured) because the chunks=1
# calibration can't see the per-microbatch costs that scale with
# num_microbatches: forced sync grad reduce, 32× PP send/recv, scheduler
# overhead. Reserved-memory overhead has the same structure.
#
# Why per-component: in 1F1B, activations stack with chunks but grads
# accumulate in-place. So the chunks=2 vs chunks=1 cuda_peak delta per
# component (attention-only / mlp-only model) directly measures the
# per-microbatch ACTIVATION memory of that component, separated from
# the grad-bucket and optimizer-state contribution. This is what the
# IntraCostModel's PP ``extra_reserve_mb`` calculation wants — and it
# replaces the legacy Step 4 ``profile_memory.sh`` per-component data
# (which was unreliable: byte-identical across (tp, ep) variants).
#
# The matrix below mirrors the main matrix's currently-active rows
# (FSEP-on at gbsz ∈ {4, 2}, FSEP-off at gbsz=1). Profile-unit fan-out
# matches the main sweep ("all attention mlp") so per-component
# chunks=1/2 pairs land in the aggregator. ~12 base configs × 3 units
# = ~36 inner runs at ~100 s each ≈ 60-70 minutes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

# (pp tp ep dp_mode micro_bsz fsep) — mirror the main matrix's active
# rows. The main matrix is FSEP-on at gbsz ∈ {4, 2} and FSEP-off at
# gbsz=1 (FSEP-on is infeasible at micro_bsz=1 — pp×tp must equal
# world=4 for per-rank ≥ 1, but FSEP-on requires pp×tp < world).
# profile_unit fan-out (all/attention/mlp) is delegated to
# cost_model_real_test.sh.
GBSZ_BASE=$'1 1 1 zero2sdp 4 on
1 1 2 zero2sdp 4 on
1 1 4 zero2sdp 4 on
1 2 1 zero2sdp 4 on
1 2 2 zero2sdp 4 on
2 1 1 zero2sdp 4 on
2 1 2 zero2sdp 4 on
1 2 1 zero2sdp 2 on
2 1 2 zero2sdp 2 on
2 1 1 zero2sdp 2 on
1 4 1 zero2sdp 1 off
2 2 1 zero2sdp 1 off'

export CONFIGS_BASE_OVERRIDE="${GBSZ_BASE}"
export CHUNKS=2
# Per-microbatch overhead is layernum-invariant — it measures NCCL/Python
# scheduler cost per microbatch and per-microbatch activation, neither
# of which scales with num_layers in expectation. Run at the minimum
# layernum (2) instead of the main script's "2 4" sweep so each inner
# config finishes faster (~½ the compute per iter). The chunks=1 anchor
# against which this slope is fit is also at nl=2 (the main sweep's
# first NUM_LAYERS pass produces it). Pairing key in
# ``_build_chunks_overhead_profile`` includes num_layers, so the
# chunks=1↔chunks=2 match is exact.
export NUM_LAYERS_LIST="2"
exec bash "${MAIN_SCRIPT}"
