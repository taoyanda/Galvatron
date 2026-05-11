#!/usr/bin/env bash
# End-to-end benchmark for a galvatron JSON config produced by cost_model_search.
#
# Drives train_dist_frozen.py with --galvatron_config_path <json>, so all
# parallel layout (pp_deg, tp_sizes_enc, tp_consecutive_flags, dp_types_enc,
# use_sp, vocab_tp, vocab_sp, global_bsz, chunks, ep_sizes_enc,
# tp_of_ep_sizes_enc, pp_division) comes from the JSON. Runs with --profile 0
# so the runtime profiler is dormant — no per-iter memory snapshots, no
# save_profiled_time writes, no instrumentation overhead. Iter timing comes
# from the ad-hoc [stage_time] CUDA-event block in train_dist_frozen.py, which
# wraps forward_backward and optimizer.step in cuda events and averages over
# iters [10, 20). Other calibration diagnostics ([real_measure] / [mem_evo] /
# [fsep_verify] / [static_input]) are suppressed via --quiet — only the
# [stage_time] line survives.
#
# As of the JSON-mode MoE patch in hybrid_parallel_config.py, the JSON carries
# per-layer ep_sizes_enc / tp_of_ep_sizes_enc. This script derives EP, TP_OF_EP,
# and NUM_LAYERS from the JSON directly so the CLI shrinks to <json> [fsep].
#
# Things the JSON still does NOT carry (must be supplied via env or CLI):
#   - FSEP toggle             -- positional arg, default "on"
#   - expert_capacity_per_device -- derived from NUM_GLOBAL_EXPERTS / EP
#   - model architecture      -- hidden_size, heads, etc. (hardcoded for qwen-30b-a3b)
#
# Back-compat fallback: if the JSON lacks ep_sizes_enc, the script reads EP
# from the env var ``EP_OVERRIDE`` and broadcasts. Same for tp_of_ep — falls
# back to tp_sizes_enc[0] under FSEP-on, 1 under FSEP-off.
#
# Usage:
#   bash galvatron/models/moe/scripts/bench_galvatron_config.sh <json> [fsep=on|off]
#
#   bash galvatron/models/moe/scripts/bench_galvatron_config.sh \
#     configs/galvatron_config_qwen30b_pp2_tp1_ep2.json off
#
# Overridable env:
#   NUM_NODES (1), NUM_GPUS_PER_NODE (4), MASTER_ADDR/PORT, NODE_RANK
#   MODEL_SIZE (qwen-30b-a3b-e128k8)
#   NUM_GLOBAL_EXPERTS (128), SEQ_LEN (4096), EPOCHS (20)
#   ENABLE_SOLVER (1), TIMEOUT_SEC (1800)
#   STATIC_INPUT (1 -> on, 0 -> off; default 1)
#   EP_OVERRIDE                 -- only consulted when JSON lacks ep_sizes_enc
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <galvatron_config.json> [fsep=on|off]" >&2
    exit 2
fi

GALVATRON_CONFIG="$1"
FSEP_MODE="${2:-on}"
CAP="${3:-1}"


if [ ! -f "${GALVATRON_CONFIG}" ]; then
    echo "[error] galvatron config not found: ${GALVATRON_CONFIG}" >&2
    exit 2
fi
if [ "${FSEP_MODE}" != "on" ] && [ "${FSEP_MODE}" != "off" ]; then
    echo "[error] fsep must be 'on' or 'off' (got '${FSEP_MODE}')" >&2
    exit 2
fi

# ─── derive EP / TP_OF_EP / NUM_LAYERS from the JSON ──────────────────────────
# Honors per-layer arrays where present (per the hybrid_parallel_config.py
# JSON-mode MoE patch) and falls back to legacy single-value semantics for
# pre-patch configs. Uses ep_sizes_enc[0] / tp_of_ep_sizes_enc[0] as the
# scalar exposed via --global_ep_deg / --global_tp_of_ep_deg (only consumed
# by the [fsep_verify] print under JSON mode; the runtime uses the per-layer
# arrays from the JSON itself).
parse_out=$(GALVATRON_CONFIG="${GALVATRON_CONFIG}" \
            FSEP_MODE="${FSEP_MODE}" \
            EP_OVERRIDE="${EP_OVERRIDE:-}" \
            python3 <<'PY'
import json, os, sys

with open(os.environ["GALVATRON_CONFIG"]) as f:
    cfg = json.load(f)

def parse_arr(val):
    if isinstance(val, str):
        return [int(x) for x in val.split(",")]
    return [int(x) for x in val]

if "tp_sizes_enc" not in cfg:
    print("ERR: JSON lacks tp_sizes_enc", file=sys.stderr); sys.exit(2)
tp = parse_arr(cfg["tp_sizes_enc"])
num_layers = len(tp)

if "ep_sizes_enc" in cfg:
    ep = parse_arr(cfg["ep_sizes_enc"])
    if len(ep) != num_layers:
        print(f"ERR: ep_sizes_enc length {len(ep)} != num_layers {num_layers}",
              file=sys.stderr); sys.exit(2)
elif os.environ.get("EP_OVERRIDE"):
    ep = [int(os.environ["EP_OVERRIDE"])] * num_layers
else:
    print("ERR: JSON lacks ep_sizes_enc and EP_OVERRIDE env var not set",
          file=sys.stderr); sys.exit(2)

if "tp_of_ep_sizes_enc" in cfg:
    tpoe = parse_arr(cfg["tp_of_ep_sizes_enc"])
    if len(tpoe) != num_layers:
        print(f"ERR: tp_of_ep_sizes_enc length {len(tpoe)} != num_layers {num_layers}",
              file=sys.stderr); sys.exit(2)
elif os.environ.get("FSEP_MODE") == "on":
    tpoe = tp[:]   # FSEP-on requires tp_of_ep == tp
else:
    tpoe = [1] * num_layers

# Under FSEP-on, the runtime requires tp_of_ep == tp per layer. Catch a
# misconfigured JSON early rather than letting the FSEP all-to-all blow up
# on backward.
if os.environ.get("FSEP_MODE") == "on" and tpoe != tp:
    print(f"ERR: FSEP-on requires tp_of_ep == tp per layer; "
          f"got tp={tp} tp_of_ep={tpoe}", file=sys.stderr); sys.exit(2)

# Output one value per line — bash readarray picks them up below.
print(ep[0])
print(tpoe[0])
print(num_layers)
PY
) || {
    echo "[error] failed to parse ${GALVATRON_CONFIG}" >&2
    exit 2
}

readarray -t _parts <<< "${parse_out}"
EP="${_parts[0]}"
TP_OF_EP="${_parts[1]}"
NUM_LAYERS="${_parts[2]}"
unset _parts parse_out

# ─── topology / NCCL / MPS — same envelope as cost_model_real_test.sh ─────────
export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-4}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${NODE_RANK:-0}

export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_HOME='/usr/local/cuda-12.1'
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/no-such-mps-bench-$$}
# export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ENABLE_SOLVER=${ENABLE_SOLVER:-1}
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="mlx5_0"

# Cross-NUMA NCCL P2P workaround — re-use the island detection script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_world=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
_island=$(python3 "${SCRIPT_DIR}/detect_p2p_island_size.py" 2>/dev/null || echo 0)
if [ "${_island}" -gt 0 ] && [ "${_island}" -lt "${_world}" ]; then
    export GALVATRON_P2P_ISLAND_SIZE=${_island}
    echo "[p2p] island_size=${_island} world=${_world}"
fi

# ─── workload constants ──────────────────────────────────────────────────────
MODEL_SIZE=${MODEL_SIZE:-qwen-30b-a3b-e128k8}
NUM_GLOBAL_EXPERTS=${NUM_GLOBAL_EXPERTS:-128}

SEQ_LEN=${SEQ_LEN:-4096}
EPOCHS=${EPOCHS:-20}
TIMEOUT_SEC=${TIMEOUT_SEC:-1800}
STATIC_INPUT_ON=${STATIC_INPUT:-1}

if [ "${FSEP_MODE}" = "on" ]; then
    FSEP_FLAG="--use_fsep"
    NO_ASYNC_FLAG="--no_async_grad_reduce"
else
    FSEP_FLAG=""
    # FSEP-off + chunks>1 still needs synchronous grad reduce under MoE top-k
    # (see cost_model_real_test.sh:330-340). chunks comes from the JSON, so we
    # play safe and always pass --no_async_grad_reduce; the runtime ignores
    # it when chunks=1 + no-FSEP would have been fine without it.
    NO_ASYNC_FLAG="--no_async_grad_reduce"
fi

# Static-input toggle. Default on for reproducibility (matches validate_top_config.sh);
# set STATIC_INPUT=0 if you want fresh routing each iter (closer to real training,
# noisier iter_ms).
if [ "${STATIC_INPUT_ON}" = "1" ]; then
    STATIC_INPUT_FLAGS=(--static_input --laer_freeze_after_iter 5)
else
    STATIC_INPUT_FLAGS=()
fi

# ─── log path ────────────────────────────────────────────────────────────────
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MODEL_DIR}/logs/bench_galvatron_config"
mkdir -p "${LOG_DIR}"
_cfg_base="$(basename "${GALVATRON_CONFIG}" .json)"
LOG_PATH="${LOG_DIR}/${_cfg_base}_ep${EP}_fsep${FSEP_MODE}_nl${NUM_LAYERS}_w${_world}.log"

cd "${MODEL_DIR}"
LAUNCHER="torchrun --nnodes ${NUM_NODES} --nproc_per_node ${NUM_GPUS_PER_NODE} \
  --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"

echo "========================================================"
echo "  e2e bench: ${GALVATRON_CONFIG}"
echo "  ep=${EP} tp_of_ep=${TP_OF_EP} fsep=${FSEP_MODE} cap=${CAP} nl=${NUM_LAYERS}"
echo "  static_input=${STATIC_INPUT_ON} world=${_world} timeout=${TIMEOUT_SEC}s"
echo "  log: ${LOG_PATH}"
echo "========================================================"

rc=0
# --global_ep_deg ${EP} \
# --global_tp_of_ep_deg ${TP_OF_EP} \

timeout --kill-after=120 "${TIMEOUT_SEC}" \
    ${LAUNCHER} train_dist_frozen.py \
        --galvatron_config_path "${GALVATRON_CONFIG}" \
        --quiet \
        --shape_order SBH --dropout_prob 0.0 \
        ${FSEP_FLAG} \
        --expert_capacity_per_device ${CAP} \
        --chunks ${CHUNKS} \
        --set_experts_manually 0 \
        --model_size ${MODEL_SIZE} \
        --hidden_size 2048 --intermediate_size 768 --head_dim 64 \
        --num_attention_heads 32 --num_experts_per_tok 8 \
        --num_key_value_heads 4 --num_local_experts ${NUM_GLOBAL_EXPERTS} \
        --vocab_size 151936 --rms_norm_eps 1e-06 --rope_theta 10000000.0 \
        --router_aux_loss_coef 0.001 --is_moe_model \
        --set_model_config_manually 0 --set_layernum_manually 1 --set_seqlen_manually 1 \
        --epochs ${EPOCHS} --lr 0.0001 --adam_weight_decay 0.01 \
        --check_loss 0 --profile 0 --initialize_on_meta 1 \
        ${NO_ASYNC_FLAG} \
        --pipeline_type pipedream_flush \
        --default_dp_type zero2 \
        --mixed_precision bf16 \
        --sequence-parallel \
        --use-flash-attn \
        "${STATIC_INPUT_FLAGS[@]}" \
        --num_hidden_layers ${NUM_LAYERS} \
        --global_checkpoint 1 \
        --seq_length ${SEQ_LEN} \
        > "${LOG_PATH}" 2>&1 || rc=$?

if [ "${rc}" -ne 0 ]; then
    echo "[bench] FAILED rc=${rc} — see ${LOG_PATH}"
    pkill -KILL -f "train_dist_frozen.py" 2>/dev/null || true
    pkill -KILL -f "torchrun" 2>/dev/null || true
    exit "${rc}"
fi

echo ""
echo "===== iter time ====="
if grep -E "^\[stage_time\]" "${LOG_PATH}"; then
    :
else
    echo "[warn] no '[stage_time]' line found — config may have died before iter ${EPOCHS}"
    echo "       tail of log:"
    tail -20 "${LOG_PATH}"
    exit 1
fi
