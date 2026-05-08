#!/usr/bin/env bash
# Validate the cost model's iter_ms prediction against a real training
# run at the cost-model search's TOP configuration:
#
#   global_bsz=128, micro_bsz=4 (chunks=32), PP=2, DP=1, TP=1, EP=2,
#   zero2+sdp, FSEP=off
#
# Predicted by cost model (Phase-0e iter_ms scaling): 5,613 ms/iter
# Measured: read from "Average iteration time is: X" in the trainer log.
#
# This validates that the calibration's iter_ms (captured at chunks=1,
# num_microbatches=1) extrapolates correctly to chunks=32 via the Alpa
# 1F1B critical-path formula.
set -euo pipefail

export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}

export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_HOME='/usr/local/cuda-12.1'
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/no-such-mps-validate-$$}
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ENABLE_SOLVER=${ENABLE_SOLVER:-1}

# Cross-NUMA NCCL P2P workaround — same as cost_model_real_test.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_world=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
_island=$(python3 "${SCRIPT_DIR}/detect_p2p_island_size.py" 2>/dev/null || echo 0)
NCCL_ENV=()
PP=2
raw_dp=$(( _world / PP / 1 ))   # tp=1
if [ "${_island}" -gt 0 ] && [ "${_island}" -lt "${_world}" ] && \
   [ "${raw_dp}" -gt "${_island}" ]; then
    NCCL_ENV=("NCCL_P2P_DISABLE=1")
fi

MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MODEL_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/validate_top_config_gbsz128_micro4.log"

# Top-config workload params.
GLOBAL_BSZ=128
CHUNKS=32              # → num_microbatches = 128/(dp*ep*chunks_per_dp_step? — galvatron uses chunks for num_microbatches)
TP=1
EP=2
NUM_GLOBAL_EXPERTS=128
CAP=$(( NUM_GLOBAL_EXPERTS / EP ))   # 64

# zero2+sdp: --default_dp_type zero2 --sdp 1; FSEP=off → tp_of_ep=1.
DP_TYPE_FLAG="zero2"
SDP_FLAG=1
SP_FLAG="--sequence-parallel"
TP_CONSEC=1            # tp=1 OR ep=1 → consec doesn't matter
FSEP_FLAG=""
TP_OF_EP=1
# At chunks=1 (calibration), FSDP grads reduce per-microbatch and the
# default async path is fine. At chunks=32 the default async path trips
# `_saved_grad_shard` assertion in fsdp_reduce_gradients because grads
# accumulate across microbatches before the reduce. ``--no_async_grad_reduce``
# forces synchronous grad reduction at backward, sidestepping the issue.
NO_ASYNC_FLAG="--no_async_grad_reduce"
PIPELINE_TYPE="pipedream_flush"
NUM_LAYERS=4
SEQ_LEN=4096
EPOCHS=20

STATIC_INPUT_PATH="${MODEL_DIR}/static_inputs/qwen-30b-a3b-e128k8_bs2_bf16.pt"
LAUNCHER="torchrun --nnodes ${NUM_NODES} --nproc_per_node ${NUM_GPUS_PER_NODE} --master_port ${MASTER_PORT}"

echo "[validate] launching top-config training"
echo "[validate]   pp=${PP} tp=${TP} ep=${EP} dp_mode=zero2sdp fsep=off"
echo "[validate]   global_bsz=${GLOBAL_BSZ} chunks=${CHUNKS} → num_microbatches=${CHUNKS}"
echo "[validate]   predicted iter_ms = 5613 (cost model, Alpa-style)"
echo "[validate]   log: ${LOG_PATH}"

cd "${MODEL_DIR}"
rc=0
env "${NCCL_ENV[@]}" timeout --kill-after=120 1800 \
    ${LAUNCHER} train_dist_frozen.py \
        --profile_mode batch --shape_order SBH --dropout_prob 0.0 \
        ${FSEP_FLAG} \
        --global_ep_deg ${EP} \
        --global_tp_of_ep_deg ${TP_OF_EP} \
        --expert_capacity_per_device ${CAP} \
        --profile_unit all \
        --set_experts_manually 0 \
        --model_size qwen-30b-a3b-e128k8 \
        --hidden_size 2048 --intermediate_size 768 --head_dim 64 \
        --num_attention_heads 32 --num_experts_per_tok 8 \
        --num_key_value_heads 4 --num_local_experts ${NUM_GLOBAL_EXPERTS} \
        --vocab_size 151936 --rms_norm_eps 1e-06 --rope_theta 10000000.0 \
        --router_aux_loss_coef 0.001 --is_moe_model \
        --set_model_config_manually 0 --set_layernum_manually 1 --set_seqlen_manually 1 \
        --global_train_batch_size ${GLOBAL_BSZ} \
        --epochs ${EPOCHS} --lr 0.0001 --adam_weight_decay 0.01 \
        --check_loss 0 --profile 1 --save_profiled_memory 0 \
        --profile_forward 0 --initialize_on_meta 1 \
        ${NO_ASYNC_FLAG} \
        --global_tp_consec ${TP_CONSEC} --sdp ${SDP_FLAG} --chunks ${CHUNKS} \
        --pipeline_type ${PIPELINE_TYPE} --default_dp_type ${DP_TYPE_FLAG} \
        --mixed_precision bf16 \
        ${SP_FLAG} \
        --use-flash-attn \
        --static_input --static_input_path "${STATIC_INPUT_PATH}" \
        --laer_freeze_after_iter 5 \
        --num_hidden_layers ${NUM_LAYERS} \
        --pp_deg ${PP} --global_checkpoint 1 \
        --vocab_tp ${TP} --global_tp_deg ${TP} \
        --seq_length ${SEQ_LEN} \
        > "${LOG_PATH}" 2>&1 || rc=$?

if [ "${rc}" -ne 0 ]; then
    echo "[validate] FAILED rc=${rc} — check ${LOG_PATH}"
    pkill -KILL -f "train_dist_frozen.py" 2>/dev/null || true
    pkill -KILL -f "torchrun" 2>/dev/null || true
    exit ${rc}
fi

echo "[validate] training run complete; extracting measurements..."
echo
grep -E "Average iteration time|stage_time|real_measure" "${LOG_PATH}" | tail -5
