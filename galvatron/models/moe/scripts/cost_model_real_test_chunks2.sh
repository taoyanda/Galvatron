#!/usr/bin/env bash
# Chunks=2 calibration sweep — pairs with the existing chunks=1 main
# matrix to derive per-microbatch overhead (time + reserved memory).
#
# Why: validation at chunks=32 showed iter_ms under-prediction of ~34%
# (5,613 ms predicted vs 7,507 ms measured) because the chunks=1
# calibration can't see the per-microbatch costs that scale with
# num_microbatches: forced sync grad reduce, 32× PP send/recv, scheduler
# overhead. Reserved-memory overhead has the same structure (5.5 GB
# fragmentation gap at chunks=32 not reflected in chunks=1 calibration).
#
# This sweep runs each shape from the main matrix at CHUNKS=2 (zero2sdp,
# profile_unit=all) so the aggregator can fit a per-shape per-microbatch
# slope from {chunks=1, chunks=2}.
#
# The matrix below mirrors the main matrix's (pp, tp, ep, fsep) tuples
# exactly — just under zero2sdp + profile_unit=all only. Running ~16
# configs at ~100 s each ≈ 25-30 minutes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

# (pp tp ep dp_mode micro_bsz fsep) — same axes as the main matrix's
# zero2sdp half. profile_unit defaults to "all" because chunks=2's
# purpose is the iter_ms / reserved-mem slope, not per-component time.
GBSZ4_BASE=$'1 1 1 zero2sdp 4 on
1 1 2 zero2sdp 4 on
1 1 4 zero2sdp 4 on
1 2 1 zero2sdp 4 on
1 2 2 zero2sdp 4 on
1 1 1 zero2sdp 4 off
1 1 2 zero2sdp 4 off
1 1 4 zero2sdp 4 off
1 2 1 zero2sdp 4 off
1 2 2 zero2sdp 4 off
1 4 1 zero2sdp 4 off
2 1 1 zero2sdp 4 on
2 1 2 zero2sdp 4 on
2 1 1 zero2sdp 4 off
2 1 2 zero2sdp 4 off
2 2 1 zero2sdp 4 off'

export CONFIGS_BASE_OVERRIDE="${GBSZ4_BASE}"
export CHUNKS=2
# chunks=2 calibration is for the iter_ms / reserved-mem slope only.
# Per-component (attention/mlp) profile units would triple the run count
# without contributing to the chunks_overhead aggregator. Force "all"
# only.
export DEFAULT_PROFILE_UNITS="all"
exec bash "${MAIN_SCRIPT}"
