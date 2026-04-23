export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}

# Disable LAER online re-planning during profiling so memory readings are
# stable across iterations (no expert migration spikes).
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

PROFILE_ARGS="
    --profile_mode sequence \
    --profile_type memory \
    --profile_batch_size 1 \
    --profile_min_seq_length 2048 \
    --profile_max_seq_length 4096 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg 8 \
    --profile_dp_type zero3 \
    --mixed_precision bf16 \
    --sequence_parallel \
    --use-flash-attn"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS}
