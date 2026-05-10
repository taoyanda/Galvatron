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
# The matrix below mirrors the main matrix's PP=1 rows post-multi-world
# refactor (PP=2 calibration is opt-in via cost_model_real_test_pp2.sh).
# Profile-unit fan-out follows the main script's default (``all`` only,
# unless overridden via DEFAULT_PROFILE_UNITS for the asymmetric path).
# 13 base configs × 1 unit (default) = 13 inner runs at ~80-100 s each
# ≈ 18-22 minutes for world=4. The world=2 / world=1 chunks=2 sisters
# (cost_model_real_test_chunks2_w2.sh / _w1.sh) cover those worlds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

# (pp tp ep dp_mode micro_bsz fsep) — mirror the main world=4 PP=1 matrix.
# Feasibility on world=4: dp×tp ≤ world AND tp×ep ≤ world (the FSDP+TP
# and TP×EP grids each fit). Plus the trainer's
# ``gbsz % (world/(pp×min(tp×ep, vocab_tp))) == 0`` assertion confines
# tp=1 entries to gbsz=4. (1,2,2) at gbsz=2 trips relocate_activations
# empirically, so it's excluded.
GBSZ_BASE=$'1 1 1 zero2sdp 4 on
1 1 2 zero2sdp 4 on
1 1 4 zero2sdp 4 on
1 2 1 zero2sdp 4 on
1 2 2 zero2sdp 4 on
1 4 1 zero2sdp 4 off
1 2 1 zero2sdp 2 on
1 4 1 zero2sdp 2 off
1 4 1 zero2sdp 1 off'

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
