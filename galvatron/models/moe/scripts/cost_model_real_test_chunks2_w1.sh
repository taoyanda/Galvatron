#!/usr/bin/env bash
# Chunks=2 calibration sweep at world=1 — pairs with the chunks=1
# anchors produced by cost_model_real_test_w1.sh.
#
# Same role as the world=2 sister, but for the matching shape that a
# PP=4 stage on a 4-GPU box would see (per-stage world = 4/4 = 1 GPU).
# FSEP-off only.
#
# Logs land with ``_w1_chunks2`` filename suffix.
#
# Usage:
#   docker exec hetu bash -lc \
#     "cd /root/Galvatron && bash galvatron/models/moe/scripts/cost_model_real_test_chunks2_w1.sh"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

W1_BASE=$'1 1 1 zero2sdp 4 off
1 1 1 zero2sdp 2 off
1 1 1 zero2sdp 1 off'

export CONFIGS_BASE_OVERRIDE="${W1_BASE}"
export NUM_GPUS_PER_NODE=1
export CHUNKS=2
export NUM_LAYERS_LIST="2"
exec bash "${MAIN_SCRIPT}"
