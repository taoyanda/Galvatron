#!/usr/bin/env bash
# Per-layer memory profiling — three-pass loop (all / attention / mlp).
#
# Drives train_dist_frozen.py with --static_input + --laer_freeze_after_iter
# so per-component memory readings are reproducible across (re)runs.
#
# IMPORTANT — three-pass is FSEP-off only.
# ----------------------------------------
# Same constraint as profile_computation.sh: the attention / mlp passes
# strip the other layer type when constructing the model. Under FSEP
# (Fully Sharded Expert Parallel) the routing distribution depends on
# what feeds the router (raw static input in the mlp pass vs
# post-attention features in the all pass), and per-rank activation
# memory follows the routing imbalance. Stripping → biased
# per-component memory readings under FSEP-on.
#
# So this script enforces FSEP=off. To profile FSEP-on configs, use
# ``profile_memory_frozen.sh`` — that companion runs only the `all`
# pass and sweeps over (EP, capacity) tuples.
set -euo pipefail

export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}
export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export CUDA_HOME='/usr/local/cuda-12.1'
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# MPS bypass — see CLAUDE.md (project memory). SIGKILL'd ranks otherwise
# leave dirty contexts on the host's GPUs for tens of minutes.
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/no-such-mps}

# Disable inductor's compile-worker warm pool — see profile_computation.sh
# for the rationale.
export TORCHINDUCTOR_COMPILE_THREADS=1

# Disable LAER online re-planning during the FSEP-off baseline so
# memory readings are stable across iterations (no expert migration
# spikes). The FSEP-on companion script flips this to 1.
export ENABLE_SOLVER=${ENABLE_SOLVER:-0}

LAUNCHER="torchrun"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
# train_dist_frozen.py wires up --static_input + --laer_freeze_after_iter
# + the [real_measure] / [stage_time] instrumentation lines. The older
# train_dist_random.py is the minimal launcher and lacks the freeze /
# instrumentation surface the cost-model calibration needs.
export PROFILE_TRAINER="train_dist_frozen.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_INPUT_PATH="${STATIC_INPUT_PATH:-${SCRIPT_DIR}/../static_inputs/frozen_batch_mem.pt}"
mkdir -p "$(dirname "${STATIC_INPUT_PATH}")"
echo "Using static input tensor: ${STATIC_INPUT_PATH}"

MODEL_ARGS="
    --model_size mixtral-8x7b-e8k2 \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 4096"

BATCH_SIZE=2

# Three-pass loop over profile_unit (matching profile_computation.sh):
#   all       — full transformer block memory; consumed as the fall-back.
#   attention — attention-only memory contribution.
#   mlp       — MoE-only memory contribution.
# Per-component memory keys land in the same JSON; the cost model uses
# them to split activation memory across attention vs expert layers
# under asymmetric-layer queries. When only the `all` pass has run, the
# cost model falls back to splitting via the per-component time ratio
# (a coarser proxy, ±20 % vs measured).
#
# Refuse to run if MODEL_ARGS toggles FSEP on — see header comment.
# The three-pass attention/mlp memory readings would be biased by
# routing-distribution mismatch under FSEP.
if printf '%s' "${MODEL_ARGS}" | grep -q -- "--use_fsep"; then
    echo "[profile_memory] ERROR: --use_fsep is set in MODEL_ARGS." >&2
    echo "  The three-pass loop strips layer types when constructing" >&2
    echo "  the model, which under FSEP biases the routing distribution" >&2
    echo "  and therefore per-rank activation memory. Use" >&2
    echo "  profile_memory_frozen.sh for FSEP-on profiling — it runs" >&2
    echo "  only the 'all' pass." >&2
    exit 1
fi

# The outer loop on PROFILE_MODE switches between static and sequence
# sweep; sequence is currently disabled (left commented in the loop).
# for PROFILE_MODE in static sequence; do
for PROFILE_MODE in static; do
    echo "========================================================"
    echo "  Memory profiling pass: profile_mode=${PROFILE_MODE}"
    echo "========================================================"
    if [ "$PROFILE_MODE" = "static" ]; then
       PROFILE_ARGS="
        --profile_mode $PROFILE_MODE \
        --profile_metric memory \
        --profile_batch_size $BATCH_SIZE \
        --profile_seq_length_list 4096 \
        --layernum_min 1 \
        --layernum_max 2 \
        --max_tp_deg ${NUM_GPUS_PER_NODE} \
        --profile_dp_type zero3 \
        --mixed_precision bf16 \
        --sequence_parallel \
        --use-flash-attn \
        --static_input \
        --static_input_path ${STATIC_INPUT_PATH} \
        --laer_freeze_after_iter 5 \
        --dropout_prob 0"
    elif [ "$PROFILE_MODE" = "sequence" ]; then
        PROFILE_ARGS="
        --profile_mode $PROFILE_MODE \
        --profile_metric memory \
        --profile_batch_size $BATCH_SIZE \
        --profile_min_seq_length 2048 \
        --profile_max_seq_length 4096 \
        --layernum_min 1 \
        --layernum_max 2 \
        --max_tp_deg ${NUM_GPUS_PER_NODE} \
        --profile_dp_type zero3 \
        --mixed_precision bf16 \
        --sequence_parallel \
        --use-flash-attn \
        --static_input \
        --static_input_path ${STATIC_INPUT_PATH} \
        --laer_freeze_after_iter 5 \
        --dropout_prob 0"
    fi
    for UNIT in all attention mlp; do
        echo "[profile_memory] pass: profile_mode=${PROFILE_MODE} profile_unit=${UNIT}"
        python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS} --profile_unit ${UNIT}
    done

done
