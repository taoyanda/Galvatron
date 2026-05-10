#!/usr/bin/env bash
# Chunks=2 calibration sweep at world=2 — pairs with the chunks=1
# anchors produced by cost_model_real_test_w2.sh.
#
# Why this sister exists: per-microbatch overhead (NCCL/scheduler cost
# + per-microbatch activation) needs to be measured at the matching
# world that PP=2 stage compute prediction will look up. World=4
# chunks_overhead can't substitute because the per-rank shape and
# collective topology differ.
#
# Logs land with the ``_w2_chunks2`` filename suffix (the main script
# adds ``_w2`` for world ≠ 4, and ``_chunks2`` for chunks ≠ 1).
#
# Pairs with chunks=1 anchors at nl=2 (matches feedback_chunks_overhead
# _min_layernum convention).
#
# Usage:
#   docker exec hetu bash -lc \
#     "cd /root/Galvatron && bash galvatron/models/moe/scripts/cost_model_real_test_chunks2_w2.sh"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

# Same base matrix as cost_model_real_test_w2.sh.
W2_BASE=$'1 1 1 zero2sdp 4 on
1 1 2 zero2sdp 4 on
1 2 1 zero2sdp 4 off
1 1 1 zero2sdp 2 on
1 1 2 zero2sdp 2 on
1 2 1 zero2sdp 2 off
1 2 1 zero2sdp 1 off'

export CONFIGS_BASE_OVERRIDE="${W2_BASE}"
export NUM_GPUS_PER_NODE=2
export CHUNKS=2
export NUM_LAYERS_LIST="2"
exec bash "${MAIN_SCRIPT}"
