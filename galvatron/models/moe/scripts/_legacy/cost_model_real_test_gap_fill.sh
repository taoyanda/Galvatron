#!/usr/bin/env bash
# Gap-fill calibration runs at smaller global_bsz values. The main
# ``cost_model_real_test.sh`` matrix uses gbsz=4 for every config, which
# means the runtime profile only covers ``micro_bsz=4`` (since
# chunks=1 + num_microbatches=1 → micro_bsz == global_bsz at calibration).
# The cost-model search at large global_bsz queries also enumerates
# micro_bsz ∈ {1, 2, 4} — the smaller values have no calibration data
# without this gap-fill.
#
# The matrix below is the smallest cover of (pp, tp, ep, fsep) tuples
# such that ``DP*EP ≤ micro_bsz`` (per-rank ≥ 1 sample) at each gbsz,
# following the same FSEP rules as the main matrix:
#   - FSEP-on requires tp*ep == per_stage_world AND ``ep | num_experts``
#     AND dp_of_ep_size > 1
#   - FSEP-off has no FSEP constraint
#
# Forwards everything to cost_model_real_test.sh via CONFIGS_BASE_OVERRIDE.
#
# Usage:
#   bash galvatron/models/moe/scripts/cost_model_real_test_gap_fill.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

# === gbsz=1 (micro_bsz=1) ===
# Feasibility: dp*ep ≤ 1 → DP=1, EP=1. PP*TP=4. Skip PP=4 (not in
# main matrix). FSEP-on excluded everywhere because dp_of_ep_size=1
# at every viable (pp, tp).
GBSZ1=$'1 4 1 zero3    1 off
1 4 1 zero2sdp 1 off
2 2 1 zero3    1 off
2 2 1 zero2sdp 1 off'

# === gbsz=2 (micro_bsz=2) ===
# Feasibility: dp*ep ≤ 2.
GBSZ2=$'1 2 1 zero3    2 off
1 2 1 zero2sdp 2 off
1 2 2 zero3    2 off
1 2 2 zero2sdp 2 off
1 2 2 zero3    2 on
1 2 2 zero2sdp 2 on
1 4 1 zero3    2 off
1 4 1 zero2sdp 2 off
2 1 1 zero3    2 off
2 1 1 zero2sdp 2 off
2 1 2 zero3    2 off
2 1 2 zero2sdp 2 off
2 1 2 zero3    2 on
2 1 2 zero2sdp 2 on
2 2 1 zero3    2 off
2 2 1 zero2sdp 2 off'

export CONFIGS_BASE_OVERRIDE="${GBSZ1}"$'\n'"${GBSZ2}"
exec bash "${MAIN_SCRIPT}"
