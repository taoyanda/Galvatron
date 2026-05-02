export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

export CUDA_HOME='/usr/local/cuda-12.1'
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Disable LAER online re-planning during profiling so memory readings are
# stable across iterations (no expert migration spikes).
export ENABLE_SOLVER=0

# LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="torchrun"
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
    --profile_mode static \
    --profile_metric memory \
    --profile_batch_size 4 \
    --profile_min_seq_length 2048 \
    --profile_max_seq_length 4096 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg ${NUM_GPUS_PER_NODE} \
    --profile_dp_type zero3 \
    --mixed_precision bf16 \
    --sequence_parallel \
    --use-flash-attn"

BATCH_SIZE=2

# for UNIT in static sequence; do
for UNIT in static; do
    echo "========================================================"
    echo "  Memory profiling pass: profile_unit=${UNIT}"
    echo "========================================================"
    if [ "$UNIT" = "static" ]; then
       PROFILE_ARGS="
        --profile_mode $UNIT \
        --profile_metric memory \
        --profile_batch_size $BATCH_SIZE \
        --profile_seq_length_list 4096 \
        --layernum_min 1 \
        --layernum_max 2 \
        --max_tp_deg ${NUM_GPUS_PER_NODE} \
        --profile_dp_type zero3 \
        --mixed_precision bf16 \
        --sequence_parallel \
        --use-flash-attn"
    elif [ "$UNIT" = "sequence" ]; then
        PROFILE_ARGS="
        --profile_mode $UNIT \
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
        --use-flash-attn"
    fi
    python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS}

done


# python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS}
