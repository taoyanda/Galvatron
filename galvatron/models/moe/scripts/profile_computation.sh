export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

export CUDA_HOME='/usr/local/cuda-12.1'


# Disable LAER online re-planning during profiling so per-layer time stays
# stationary (linear-fit assumption).
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
    --profile_mode batch \
    --profile_metric computation \
    --profile_min_batch_size 1 \
    --profile_max_batch_size 4 \
    --profile_batch_size_step 1 \
    --profile_seq_length_list 4096 \
    --layernum_min 1 \
    --layernum_max 2 \
    --mixed_precision bf16 \
    --use-flash-attn"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS}
