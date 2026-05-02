#!/usr/bin/env bash
# Memory profiling with LAER solver enabled on a FROZEN synthetic input.
#
# Mirrors profile_memory.sh but enables ENABLE_SOLVER=1 plus a frozen batch
# so peak-memory readings reflect the solver's steady-state expert layout
# rather than a transient migration spike.
#
#   - --laer_freeze_after_iter 5: stop layout migrations before memory is
#     sampled (memory profiler captures allocated/reserved peaks during the
#     training loop, post-warmup).
#   - --dropout_prob 0: router softmax bit-identical; keeps layout stable.
#
# Hardening transplanted from profile_computation_frozen.sh: NCCL P2P fix,
# MPS bypass, inductor warm-pool guard, per-config + outer-loop timeouts,
# and cascade-prevention on inner-config failure. See
# doc/profile_computation_frozen_fixes.md for the full history.

set -euo pipefail

export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}
export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# NCCL workaround for asymmetric DP-group hang on this host's two 2-rank DP
# groups {0,2} and {1,3} (identical NODE topology yet only one wedges).
# Forces SHM fallback for both pairs. See profile_computation_frozen.sh and
# Fix 10 in doc/profile_computation_frozen_fixes.md.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

export CUDA_HOME='/usr/local/cuda-12.1'

# Bypass nvidia-cuda-mps-server on the host. SIGKILL'd ranks otherwise leave
# stale MPS contexts that reject new clients on the same physical GPUs for
# tens of minutes — even across `docker restart hetu`. Pointing the pipe
# directory at a non-existent path forces direct per-process CUDA contexts.
# See Fix 8 in doc/profile_computation_frozen_fixes.md.
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/no-such-mps}

export ENABLE_SOLVER=1

# Disable torch._inductor's compile-worker warm pool (Fix 2): the pool forks
# /dev/nvidia* FDs into worker processes and leaves the host driver in a
# busy state that fails subsequent set_device() in the inner torchrun.
export TORCHINDUCTOR_COMPILE_THREADS=1

# Per-inner-config timeout (seconds): each (seq, tp) torchrun launch the
# profiler issues via os.system() is wrapped in `timeout --kill-after=...`
# inside model_profiler.py so a hung NCCL collective cannot strand the whole
# sweep. Memory profiling runs full forward+backward+optimizer iters under
# zero3, which on this hardware sometimes wedges; the inner timeout bounds
# the damage.
export GALVATRON_PROFILE_INNER_TIMEOUT=${GALVATRON_PROFILE_INNER_TIMEOUT:-900}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_INPUT_PATH="${STATIC_INPUT_PATH:-${SCRIPT_DIR}/../static_inputs/frozen_batch_mem.pt}"
mkdir -p "$(dirname "${STATIC_INPUT_PATH}")"
echo "Using static input tensor: ${STATIC_INPUT_PATH}"

# torchrun matches profile_computation_frozen.sh; the deprecated
# `python3 -m torch.distributed.launch` was the old default.
LAUNCHER="torchrun"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="profile_dist_static.py"

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
    --profile_seq_length_list 4096 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg ${NUM_GPUS_PER_NODE} \
    --profile_dp_type zero3 \
    --mixed_precision bf16 \
    --sequence_parallel \
    --use-flash-attn \
    --use_fsep \
    --static_input \
    --static_input_path ${STATIC_INPUT_PATH} \
    --laer_freeze_after_iter 5 \
    --dropout_prob 0"

# FSEP tuples over (ep, cap) with cap = num_global_experts / ep (=8/ep).
EP_CAP_TUPLES_DEFAULT=(
    "1 8"
    "2 4"
    "4 2"
)
# Allow env override (e.g. EP_CAP_TUPLES_OVERRIDE="1 8" for a focused run).
if [ -n "${EP_CAP_TUPLES_OVERRIDE:-}" ]; then
    read -r -a EP_CAP_TUPLES <<< "${EP_CAP_TUPLES_OVERRIDE}"
    _packed=()
    for ((i=0; i<${#EP_CAP_TUPLES[@]}; i+=2)); do
        _packed+=("${EP_CAP_TUPLES[i]} ${EP_CAP_TUPLES[i+1]}")
    done
    EP_CAP_TUPLES=("${_packed[@]}")
else
    EP_CAP_TUPLES=("${EP_CAP_TUPLES_DEFAULT[@]}")
fi

# Per-(EP,CAP) outer timeout: bound the total wall time spent on one outer
# sweep so a failure in one config can't hang the whole script.
OUTER_TIMEOUT=${OUTER_TIMEOUT:-5400}

for tuple in "${EP_CAP_TUPLES[@]}"; do
    read -r EP CAP <<< "${tuple}"
    echo "========================================================"
    echo "  Frozen memory pass: ep=${EP} cap=${CAP}"
    echo "========================================================"
    rc=0
    timeout --kill-after=${OUTER_KILL_AFTER:-120} "${OUTER_TIMEOUT}" \
        python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS} \
            --global_ep_deg ${EP} \
            --expert_capacity_per_device ${CAP} || rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "[outer rc=${rc}] ep=${EP} cap=${CAP} did not finish cleanly"
        # 124: TERM after OUTER_TIMEOUT, 137: KILL after kill-after grace.
        # Other non-zero (e.g. SystemExit(2) from
        # model_profiler._report_inner_rc): an inner config hung. Either
        # way we must NOT advance — host driver state is dirty for ~10 min
        # after a SIGKILL, so the next (EP, CAP) would inherit the
        # corruption and cascade.
        echo "[outer abort] cleaning up stragglers from ep=${EP} cap=${CAP}"
        pkill -KILL -f "profile_dist_static.py" 2>/dev/null || true
        pkill -KILL -f "torchrun" 2>/dev/null || true
        sleep 3
        echo "[outer abort] sweep stopped — restart hetu, wait 10 min, then retry the failing config in isolation"
        exit "${rc}"
    fi
done
