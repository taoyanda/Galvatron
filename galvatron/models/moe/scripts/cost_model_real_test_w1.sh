#!/usr/bin/env bash
# World=1 (single-GPU) PP=1 calibration sweep — FSEP-off only.
#
# Produces the calibration data that PP=4 stage compute prediction would
# consume via the matching-shape lookup (a PP=4 stage on a 4-GPU box
# has per-stage world = 4/4 = 1 GPU). FSEP-on is intrinsically
# infeasible at world=1: any PP=1 shape has dp×tp×ep=1, hence tp=1,
# but FSEP-on requires tp×pp < world (= 1 < 1 is impossible) AND
# dp_of_ep_size = world/(pp×tp) > 1 (= 1 > 1 is impossible). So this
# sweep is FSEP-off only at all gbsz values.
#
# Logs land with the ``_w1`` filename suffix.
#
# Matrix: 1 base shape × 3 mbsz values = 3 configs.
#
# Usage:
#   docker exec hetu bash -lc \
#     "cd /root/Galvatron && bash galvatron/models/moe/scripts/cost_model_real_test_w1.sh"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

# (pp tp ep dp_mode micro_bsz fsep) — only feasible PP=1 shape at
# world=1 is (tp=1, ep=1, dp=1) since dp×tp×ep=1. FSEP-off only.
W1_BASE=$'1 1 1 zero2sdp 4 off
1 1 1 zero2sdp 2 off
1 1 1 zero2sdp 1 off'

export CONFIGS_BASE_OVERRIDE="${W1_BASE}"
export NUM_GPUS_PER_NODE=1
exec bash "${MAIN_SCRIPT}"
