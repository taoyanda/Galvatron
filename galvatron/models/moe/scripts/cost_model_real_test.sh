#!/usr/bin/env bash
# Real-measurement sweep for cost-model validation.
# Runs train_dist_random.py for each of the 6 valid (dp, tp, ep) configs on a
# 4-GPU node with 4 layers, capturing per-config logs that contain:
#   - [real_measure] params_mb=…
#   - [real_measure] optimizer_mb=… activation_peak_mb=… cuda_peak_mb=…
#   - Average iteration time is: X s
# Logs land in galvatron/models/moe/logs/cost_model_real_<dp>_<tp>_<ep>.log.
#
# Same NCCL/MPS env recipe as profile_computation_frozen.sh — see
# doc/laer_fsep_sweep_resolution.md.
#
# Usage:
#   bash galvatron/models/moe/scripts/cost_model_real_test.sh           # all 6
#   bash galvatron/models/moe/scripts/cost_model_real_test.sh dp tp ep  # one
set -euo pipefail

export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}

export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export CUDA_HOME='/usr/local/cuda-12.1'
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/no-such-mps}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TORCHINDUCTOR_COMPILE_THREADS=1

# Match production training's FSEP-required envelope (see train.sh):
#   - TORCH_NCCL_AVOID_RECORD_STREAMS=1: FSEP overrides manage tensor
#     lifetimes via custom events; PyTorch's auto record-stream double-
#     accounting otherwise frees activation storage at the wrong moment
#     and triggers an illegal-memory-access in silu/swiglu.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: matches train.sh.
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ENABLE_SOLVER=${ENABLE_SOLVER:-1}

NUM_LAYERS=${NUM_LAYERS:-4}
SEQ_LEN=4096
# Pipeline parallelism degree. With ``PP=1`` the layout matches the
# original sweep; for ``PP>1`` the per-stage world size becomes
# tp×ep×dp = NUM_GPUS_PER_NODE/PP, so the caller must pick (TP, EP, DP_MODE)
# such that the per-stage product fits.
PP=${PP:-1}
EPOCHS=20  # need ≥20 iters: profiler averages [10, 20), and we sample the
            # memory-evolution snapshot at iter 10.

# FSEP-viable parallelization envelope on 4 GPUs:
#     tp * ep == world_size  AND  tp == tp_of_ep_deg
# This collapses the 6-tuple matrix to two configs:
#     (tp=1, ep=4): one EP shard per rank, no TP within experts
#     (tp=2, ep=2): two EP shards × two TP-within-EP ranks per shard
# DP comes from --sdp 1 + --sequence-parallel (sequence-data-parallel) when
# --default_dp_type zero2 is used; under zero3 it comes from FSDP itself
# sharding params across all ranks.
#
# We sweep both DP modes for each FSEP config, plus bsz ∈ {2, 4}. Tuple
# format: "tp ep dp_mode bsz"  with dp_mode ∈ {zero2sdp, zero3}.
DEFAULT_CONFIGS=(
    # "tp ep dp_mode bsz fsep"  (fsep ∈ {on, off})
    #
    # FSEP-on track: smart routing + LAER solver, requires tp × ep ==
    # world_size AND tp == tp_of_ep_deg. On 4 GPUs that means (tp=1, ep=4)
    # and (tp=2, ep=2). Both DP modes the user supports for FSDP:
    # zero2+sdp and zero3.
    "1 4 zero2sdp 4 on"  "1 4 zero2sdp 8 on"
    "1 4 zero3    4 on"  "1 4 zero3    8 on"
    "2 2 zero2sdp 4 on"  "2 2 zero2sdp 8 on"
    "2 2 zero3    4 on"  "2 2 zero3    8 on"
    # FSEP-off track for the SAME (tp, ep) shapes — naive Megatron MoE
    # all-to-all + plain FSDP (no LAER, no smart routing). Run head-to-head
    # against the FSEP-on rows above so we can quote a real speedup.
    "1 4 zero2sdp 4 off"  "1 4 zero2sdp 8 off"
    "1 4 zero3    4 off"  "1 4 zero3    8 off"
    "2 2 zero2sdp 4 off"  "2 2 zero2sdp 8 off"
    "2 2 zero3    4 off"  "2 2 zero3    8 off"
    # FSEP-off track for ep=1 (FSEP can't satisfy its constraint here).
    # Lets us cover bsz=2 / smaller tp combos that bsz=4 doesn't.
    "2 1 zero2sdp 2 off"  "2 1 zero2sdp 4 off"
    "2 1 zero3    2 off"  "2 1 zero3    4 off"
    "4 1 zero2sdp 4 off"
    "4 1 zero3    4 off"
)

if [ "$#" -eq 5 ]; then
    CONFIGS=("$1 $2 $3 $4 $5")
elif [ "$#" -eq 4 ]; then
    CONFIGS=("$1 $2 $3 $4 on")
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MODEL_DIR}/logs"
STATIC_INPUT_PATH="${MODEL_DIR}/static_inputs/frozen_batch_comp.pt"
mkdir -p "${LOG_DIR}"

LAUNCHER="torchrun --nnodes ${NUM_NODES} --nproc_per_node ${NUM_GPUS_PER_NODE} --master_port ${MASTER_PORT}"

echo "[env] CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset>}"
echo "[env] NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-<unset>}"
echo "[env] NCCL_DEBUG=${NCCL_DEBUG}"
echo "[env] TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}"
echo "[env] ENABLE_SOLVER=${ENABLE_SOLVER}"

cd "${MODEL_DIR}"

for tuple in "${CONFIGS[@]}"; do
    read -r TP EP DP_MODE GLOBAL_BSZ FSEP_MODE <<< "${tuple}"
    FSEP_MODE=${FSEP_MODE:-on}
    CAP=$((8 / EP))  # global_experts=8 for mixtral-8x7b-e8k2
    if [ "${DP_MODE}" = "zero2sdp" ]; then
        DP_TYPE_FLAG="zero2"
        SDP_FLAG=1
        SP_FLAG="--sequence-parallel"
    else
        DP_TYPE_FLAG="zero3"
        SDP_FLAG=0
        SP_FLAG="--sequence-parallel"  # zero3 still benefits from SP for memory
    fi
    # tp_consec layout choice per (tp, ep) on this 4-GPU box (NVLink pairs
    # (0,1) and (2,3); cross-pair PCIe NODE):
    #   tp=1 ep=4 → tp_consec irrelevant (TP groups are size 1)
    #   tp=2 ep=2 → tp_consec=0 puts EP groups on NVLink, TP on PCIe NODE.
    #     EP all-to-all is the MoE hot path per layer, so prefer NVLink for
    #     EP. (See doc/laer_fsep_sweep_resolution.md for the topology.)
    if [ "${TP}" -gt 1 ] && [ "${EP}" -gt 1 ]; then
        TP_CONSEC=0
    else
        TP_CONSEC=1
    fi
    # FSEP-on requires tp == tp_of_ep_deg AND tp×ep == world_size, with all
    # the train.sh-style envelope (no_async_grad_reduce). FSEP-off skips the
    # smart-routing kernel and runs plain Megatron MoE so smaller (tp, ep)
    # at bsz=2 become viable. Both branches use ``pipedream_flush`` (1F1B)
    # to match train.sh and the cost model's default schedule — gpipe was
    # used previously for FSEP-off but with chunks=1 the two are identical.
    PIPELINE_TYPE="pipedream_flush"
    if [ "${FSEP_MODE}" = "on" ]; then
        FSEP_FLAG="--use_fsep"
        TP_OF_EP=${TP}
        NO_ASYNC_FLAG="--no_async_grad_reduce"
    else
        FSEP_FLAG=""
        TP_OF_EP=1
        NO_ASYNC_FLAG=""
    fi
    # Default log path matches the original calibration sweep
    # (NUM_LAYERS=4, PP=1). Non-default values are tagged in the filename
    # so each variant has its own log.
    LOG_PATH="${LOG_DIR}/cost_model_real_tp${TP}_ep${EP}_${DP_MODE}_bsz${GLOBAL_BSZ}_fsep${FSEP_MODE}"
    if [ "${NUM_LAYERS}" != "4" ]; then
        LOG_PATH="${LOG_PATH}_nl${NUM_LAYERS}"
    fi
    if [ "${PP}" != "1" ]; then
        LOG_PATH="${LOG_PATH}_pp${PP}"
    fi
    LOG_PATH="${LOG_PATH}.log"

    # Per-config throw-away MPS pipe dir to ensure no MPS control-socket
    # state can leak between configs.
    PER_RUN_MPS_DIR="/tmp/no-such-mps-${TP}-${EP}-${DP_MODE}-${GLOBAL_BSZ}-$$"
    rm -rf "${PER_RUN_MPS_DIR}" 2>/dev/null || true
    export CUDA_MPS_PIPE_DIRECTORY="${PER_RUN_MPS_DIR}"

    echo "========================================================"
    echo "  cost_model_real: pp=${PP} tp=${TP} ep=${EP} dp_mode=${DP_MODE} bsz=${GLOBAL_BSZ} fsep=${FSEP_MODE} nl=${NUM_LAYERS}"
    echo "  log: ${LOG_PATH}"
    echo "========================================================"

    rc=0
    timeout --kill-after=120 1200 \
        ${LAUNCHER} train_dist_random.py \
            --profile_mode batch --shape_order SBH --dropout_prob 0.0 \
            ${FSEP_FLAG} \
            --global_ep_deg ${EP} \
            --global_tp_of_ep_deg ${TP_OF_EP} \
            --expert_capacity_per_device ${CAP} \
            --profile_unit all \
            --set_experts_manually 0 \
            --model_size mixtral-8x7b-e8k2 \
            --hidden_size 4096 --intermediate_size 14336 --head_dim 128 \
            --num_attention_heads 32 --num_experts_per_tok 2 \
            --num_key_value_heads 8 --num_local_experts 8 \
            --vocab_size 32000 --rms_norm_eps 1e-05 --rope_theta 1000000.0 \
            --router_aux_loss_coef 0.0 --is_moe_model \
            --set_model_config_manually 0 --set_layernum_manually 1 --set_seqlen_manually 1 \
            --global_train_batch_size ${GLOBAL_BSZ} \
            --epochs ${EPOCHS} --lr 0.0001 --adam_weight_decay 0.01 \
            --check_loss 0 --profile 1 --save_profiled_memory 0 \
            --profile_forward 0 --initialize_on_meta 1 \
            ${NO_ASYNC_FLAG} \
            --global_tp_consec ${TP_CONSEC} --sdp ${SDP_FLAG} --chunks 1 \
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
        echo "[cost_model_real] tp=${TP} ep=${EP} dp_mode=${DP_MODE} bsz=${GLOBAL_BSZ} fsep=${FSEP_MODE} FAILED rc=${rc} — check ${LOG_PATH}"
        pkill -KILL -f "train_dist_random.py" 2>/dev/null || true
        pkill -KILL -f "torchrun" 2>/dev/null || true
        sleep 10  # let driver state quiesce before next config
        # don't abort the whole sweep — record per-config failure and proceed
    else
        echo "[cost_model_real] tp=${TP} ep=${EP} dp_mode=${DP_MODE} bsz=${GLOBAL_BSZ} fsep=${FSEP_MODE} OK"
        sleep 2
    fi
    # Clean the per-run MPS pipe dir whether the run succeeded or failed.
    # This removes any control-socket state that the CUDA driver tried to
    # create (or any state left by a SIGKILL'd rank), so the next config
    # inherits an empty path — preserving the bypass.
    rm -rf "${PER_RUN_MPS_DIR}" 2>/dev/null || true
done
echo "all configs done; logs in ${LOG_DIR}"
