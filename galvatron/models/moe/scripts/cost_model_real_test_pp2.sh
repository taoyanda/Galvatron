#!/usr/bin/env bash
# Legacy PP=2 calibration sweep (opt-in).
#
# Why this is a separate script: the project's main calibration path now
# predicts PP>1 stage compute by looking up matching PP=1 calibrations at
# world = num_gpus / pp (see ``cost_model_real_test_w2.sh`` and
# ``cost_model/pp.py``). That flow doesn't need direct PP=2 measurements,
# so the main ``cost_model_real_test.sh`` ships PP=1 only. This sister
# script preserves the historical PP=2 calibration data path (FSEP-on at
# gbsz ∈ {4, 2}, FSEP-off at gbsz=1) for cross-checks against the new
# PP=2-via-PP=1 prediction or as fallback if the matching PP=1 sweep
# isn't available.
#
# Usage:
#   docker exec hetu bash -lc \
#     "cd /root/Galvatron && bash galvatron/models/moe/scripts/cost_model_real_test_pp2.sh"
#
# Same NUM_GPUS_PER_NODE as the main sweep (default 4). Logs land in the
# unchanged 4-GPU filename namespace (no ``_w`` tag because world=4 is
# the default sweep target).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

# (pp tp ep dp_mode micro_bsz fsep) — historical PP=2 entries removed
# from the main matrix in the multi-world refactor.
PP2_BASE=$'2 1 1 zero2sdp 4 on
2 1 2 zero2sdp 4 on
2 1 1 zero2sdp 2 on
2 1 2 zero2sdp 2 on
2 2 1 zero2sdp 1 off'

export CONFIGS_BASE_OVERRIDE="${PP2_BASE}"
exec bash "${MAIN_SCRIPT}"
