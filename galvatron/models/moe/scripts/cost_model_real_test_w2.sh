#!/usr/bin/env bash
# World=2 (2-GPU) PP=1 calibration sweep.
#
# Produces the calibration data that PP=2 stage compute prediction
# consumes via the matching-shape lookup (PP=2 on a 4-GPU box has
# per-stage world = 4/2 = 2 GPUs with the same per-rank (tp, ep, dp,
# mbsz) shape as a PP=1 run on 2 GPUs).
#
# Logs land with the ``_w2`` filename suffix (added automatically by
# ``cost_model_real_test.sh`` whenever world ≠ 4) so they don't collide
# with the 4-GPU baselines that share the same (tp, ep, mbsz, fsep)
# tuple but a different (dp, sdp) degree.
#
# Matrix: all feasible PP=1 shapes at world=2 (FSEP-on where
# tp×pp < world; FSEP-off where tp = world). 8 base configs.
#
# Usage:
#   docker exec hetu bash -lc \
#     "cd /root/Galvatron && bash galvatron/models/moe/scripts/cost_model_real_test_w2.sh"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/cost_model_real_test.sh"

# (pp tp ep dp_mode micro_bsz fsep)  — world=2 PP=1 feasibility:
#   dp × tp ≤ world AND tp × ep ≤ world (independent grid checks);
#   per_rank = mbsz / dp ≥ 1; FSEP-on iff tp < world (i.e., tp=1).
#   Plus the trainer's
#     gbsz % (world/(pp×min(tp×ep, vocab_tp))) == 0
#   assertion (vocab_tp = tp): tp=1 entries pinned to gbsz ∈ {2, 4}
#   on world=2 (gbsz=1 fails 1%2≠0); tp=2 entries can run any gbsz.
#
# === gbsz=4 (3 shapes) ===
#   (1,1,1) dp=2 ON, (1,1,2) dp=1 ON, (1,2,1) dp=1 OFF (tp=world)
# === gbsz=2 (3 shapes) ===
#   (1,1,1) dp=2 ON, (1,1,2) dp=1 ON, (1,2,1) dp=1 OFF
# === gbsz=1 (1 shape) — only tp=2 feasible per trainer ===
#   (1,2,1) dp=1 OFF
W2_BASE=$'1 1 1 zero2sdp 4 on
1 1 2 zero2sdp 4 on
1 2 1 zero2sdp 4 off
1 1 1 zero2sdp 2 on
1 1 2 zero2sdp 2 on
1 2 1 zero2sdp 2 off
1 2 1 zero2sdp 1 off'

export CONFIGS_BASE_OVERRIDE="${W2_BASE}"
export NUM_GPUS_PER_NODE=2
exec bash "${MAIN_SCRIPT}"
