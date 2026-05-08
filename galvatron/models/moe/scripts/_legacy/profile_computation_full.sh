#!/usr/bin/env bash
# Three-pass computation profiling for MoE models.
#
# Pass 1 (all):       full fwd+bwd time per layer — used by pipeline planner.
# Pass 2 (attention): attention-only slice — reference / breakdown.
# Pass 3 (mlp):       MoE MLP slice — used by LAER solver (solver.py:41
#                     reads layertype_0_bsz1_seq4096_mlp).
#
# All three passes write into the same computation_profiling_bf16_<model>.json
# produced by profiler.py. Keys are disambiguated by the profile_unit suffix
# appended in save_profiled_time and (after the patches above) in key_format
# and _process_computation_data.
#
# Requires the two model_profiler.py hunks from commit b7589ee to be restored.
# Without them, Pass 2/3 will clobber Pass 1's layertype_0_* keys.

set -euo pipefail

export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}

# Disable LAER online re-planning during profiling. At EP=1 (single GPU below)
# this only suppresses solver-worker overhead; at multi-GPU EP, disabling is
# debatable — see discussion in doc/LAER_MOE_NOTES.md.
export ENABLE_SOLVER=0

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_random.py"

MODEL_ARGS="
    --model_size mixtral-8x7b-e8k2 \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 4096"

# Layernum sweep: need two values so _process_computation_data can subtract
# and divide by (layernum_max - layernum_min). Matches the reference file.
LAYERNUM_MIN=2
LAYERNUM_MAX=4

# Batch-size sweep for the MLP pass (the one LAER consumes and the one that
# benefits most from a full curve for pp planning).
BSZ_MIN=1
BSZ_MAX=1
BSZ_STEP=1

COMMON_PROFILE_ARGS="
    --profile_mode batch \
    --profile_metric computation \
    --profile_min_batch_size ${BSZ_MIN} \
    --profile_max_batch_size ${BSZ_MAX} \
    --profile_batch_size_step ${BSZ_STEP} \
    --profile_seq_length_list 4096 \
    --layernum_min ${LAYERNUM_MIN} \
    --layernum_max ${LAYERNUM_MAX} \
    --mixed_precision bf16 \
    --use-flash-attn"

for UNIT in all attention mlp; do
    echo "========================================================"
    echo "  Computation profiling pass: profile_unit=${UNIT}"
    echo "========================================================"
    python3 profiler.py ${MODEL_ARGS} ${COMMON_PROFILE_ARGS} --profile_unit ${UNIT}
done
