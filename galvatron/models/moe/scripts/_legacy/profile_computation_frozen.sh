#!/usr/bin/env bash
# Computation profiling with LAER solver enabled on a FROZEN synthetic input.
#
# Differences vs profile_computation_full.sh:
#   - ENABLE_SOLVER=1: LAER solver is active, so profiled times reflect the
#     post-solve steady-state expert layout (what real training will see once
#     the solver converges).
#   - --static_input + --static_input_path: every iteration reuses the same
#     synthetic batch. On the first run the batch is built from --seed +
#     --vocab_size and saved to the file; subsequent runs load it. Reproduces
#     identical routing decisions across invocations.
#   - --laer_freeze_after_iter 5: freeze the layout before the time profiler's
#     averaging window (iters [10, 20) in runtime_profiler). Must be < 10.
#   - --dropout_prob 0: keep router softmax bit-identical across iters.

set -euo pipefail

export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}
export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Leave NCCL_P2P_LEVEL unset so NCCL auto-discovers the per-pair
# transport at init. On the 2×2-island PCIe-A100 topology this lets
# NCCL keep NVLink P2P within each island ({0,1}, {2,3}) and fall
# back to SHM across islands without us hard-coding a level.
#
# Cross-island ring workaround: NCCL builds 2 channels for large
# all-reduces. On this 2×2-island fabric, channel 1's ring layout
# cannot close (no cross-island P2P route), so any DP / EP / TP group
# whose rank count exceeds an island wedges with "ring N does not
# contain rank 0". We auto-detect island size at script start; the
# inner profiler launcher reads ``GALVATRON_P2P_ISLAND_SIZE`` and
# prepends ``NCCL_P2P_DISABLE=1`` to the inner CMD ONLY for inner
# configs whose raw_dp = world/(pp×tp) > island_size — preserving
# multi-channel + intra-island NVLink P2P for everything that fits and
# fixing only the ring-spanning shapes. See
# ``doc/cross_numa_nccl_postmortem.md`` for why P2P_DISABLE was chosen
# over NCCL_MAX_NCHANNELS=1 / NCCL_IGNORE_DISABLED_P2P=1.
SCRIPT_DIR_FOR_DETECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_world=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
_island=$(python3 "${SCRIPT_DIR_FOR_DETECT}/detect_p2p_island_size.py" 2>/dev/null || echo 0)
if [ "${_island}" -gt 0 ] && [ "${_island}" -lt "${_world}" ]; then
    export GALVATRON_P2P_ISLAND_SIZE=${_island}
    echo "[p2p] detected island_size=${_island} world=${_world}; will prepend NCCL_P2P_DISABLE=1 for raw_dp>island launches"
fi
unset _world _island SCRIPT_DIR_FOR_DETECT

export CUDA_HOME='/usr/local/cuda-12.1'

# Bypass nvidia-cuda-mps-server. The host runs MPS (pid 9404 via
# /tmp/nvidia-mps/control), which any CUDA process in this container connects
# to by default. When SIGKILL'd ranks die mid-collective, MPS retains their
# stale contexts on the GPUs they touched (we've observed GPUs 0 and 2
# reporting "device busy" / "illegal memory access" for tens of minutes
# after a kill, even across `docker restart hetu`). Pointing the pipe dir
# at a non-existent path forces the CUDA driver to fall back to direct
# per-process contexts, eliminating the cascade entirely. See Fix 8 in
# doc/profile_computation_frozen_fixes.md.
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/no-such-mps}

export ENABLE_SOLVER=1

# torch._inductor.codecache.AsyncCompile.warm_pool() forks one worker per CPU at
# import time (triggered by `@torch.compile def gelu_impl` in megatron). Those
# forked workers inherit /dev/nvidia* FDs from the parent profiler.py process,
# which causes the CUDA driver to report "device busy" / "illegal memory access"
# when the subsequently-launched torchrun ranks try to set_device. Set this to 1
# to disable the warm pool — profiler.py never uses torch.compile in a hot path
# so there is no perf cost. See doc/profile_computation_frozen_fixes.md.
export TORCHINDUCTOR_COMPILE_THREADS=1

# Per-inner-config timeout (seconds): each (bsz, tp) torchrun launch the
# profiler issues via os.system() is wrapped in `timeout --kill-after=...`
# inside model_profiler.py so a hung NCCL collective cannot strand the whole
# sweep. Override here if needed.
export GALVATRON_PROFILE_INNER_TIMEOUT=${GALVATRON_PROFILE_INNER_TIMEOUT:-900}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_INPUT_PATH="${STATIC_INPUT_PATH:-${SCRIPT_DIR}/../static_inputs/frozen_batch_comp.pt}"
mkdir -p "$(dirname "${STATIC_INPUT_PATH}")"
echo "Using static input tensor: ${STATIC_INPUT_PATH}"

LAUNCHER="torchrun"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_frozen.py"

MODEL_ARGS="
    --model_size qwen-30b-a3b-e128k8 \
    --set_model_config_manually 0 \
    --vocab_size 151936 \
    --hidden_size 2048 \
    --num_attention_heads 32 \
    --num_key_value_heads 4 \
    --intermediate_size 768 \
    --num_local_experts 128 \
    --num_experts_per_tok 8 \
    --seq_length 4096"

LAYERNUM_MIN=2
LAYERNUM_MAX=4

BSZ_MIN=${BSZ_MIN:-1}
BSZ_MAX=${BSZ_MAX:-4}
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
    --max_tp_deg ${NUM_GPUS_PER_NODE} \
    --mixed_precision bf16 \
    --use_fsep \
    --use-flash-attn \
    --static_input \
    --static_input_path ${STATIC_INPUT_PATH} \
    --laer_freeze_after_iter 5 \
    --dropout_prob 0"

# FSEP tuples over (ep, cap) with cap = num_global_experts / ep (=8/ep).
# TP == TP_of_EP is swept internally by the profiler via --max_tp_deg, and
# when --use_fsep is set the profiler forces global_tp_of_ep_deg = global_tp_deg
# for every point in the TP sweep. Outer loop only needs EP/cap.
EP_CAP_TUPLES_DEFAULT=(
    "1 128"
    "2 64"
    "4 32"
)
# Allow the env to override the sweep, e.g. EP_CAP_TUPLES_OVERRIDE="1 8" for
# a focused diagnostic run.
if [ -n "${EP_CAP_TUPLES_OVERRIDE:-}" ]; then
    read -r -a EP_CAP_TUPLES <<< "${EP_CAP_TUPLES_OVERRIDE}"
    # Pack into "ep cap" tuples assumed two-at-a-time
    _packed=()
    for ((i=0; i<${#EP_CAP_TUPLES[@]}; i+=2)); do
        _packed+=("${EP_CAP_TUPLES[i]} ${EP_CAP_TUPLES[i+1]}")
    done
    EP_CAP_TUPLES=("${_packed[@]}")
else
    EP_CAP_TUPLES=("${EP_CAP_TUPLES_DEFAULT[@]}")
fi

# Per-(EP,CAP) outer timeout: bound the total wall time spent on one outer
# sweep, so even if every inner config hits its own timeout the script still
# advances to the next (EP,CAP) tuple. Roughly INNER * (#bsz * #tp) + slack.
OUTER_TIMEOUT=${OUTER_TIMEOUT:-5400}

# for UNIT in all attention mlp; do
for UNIT in all; do
    for tuple in "${EP_CAP_TUPLES[@]}"; do
        read -r EP CAP <<< "${tuple}"
        echo "========================================================"
        echo "  Frozen compute pass: unit=${UNIT} ep=${EP} cap=${CAP}"
        echo "========================================================"
        # `|| rc=$?` so a non-zero exit from `timeout` (124 on TERM, 137 on
        # KILL) doesn't trigger `set -e`. We log the rc and ABORT the sweep
        # (cascade-prevention; see feedback_hang_stop_restart memory).
        rc=0
        # Extended kill-after grace (was 30 s) so the per-rank atexit/SIGTERM
        # handler in train_dist_random.py has time to run
        # destroy_process_group() + destroy_nccl_comm() before we escalate
        # to SIGKILL. Without that grace, abrupt SIGKILL leaves the host
        # NVIDIA driver holding dirty CUDA contexts on GPUs 0/2 for ~10 min.
        timeout --kill-after=${OUTER_KILL_AFTER:-120} "${OUTER_TIMEOUT}" \
            python3 profiler.py ${MODEL_ARGS} ${COMMON_PROFILE_ARGS} \
                --profile_unit ${UNIT} \
                --global_ep_deg ${EP} \
                --expert_capacity_per_device ${CAP} || rc=$?
        if [ "${rc}" -ne 0 ]; then
            echo "[outer rc=${rc}] ep=${EP} cap=${CAP} unit=${UNIT} did not finish cleanly"
            # 124: TERM after OUTER_TIMEOUT, 137: KILL after kill-after grace.
            # Other non-zero (e.g. SystemExit(2) from model_profiler._report_inner_rc):
            # an inner config hung. Either way we must NOT advance — host NVIDIA
            # driver state is dirty for ~10 min after a SIGKILL, so the next
            # (EP, CAP) would inherit the corruption and cascade.
            echo "[outer abort] cleaning up stragglers from ep=${EP} cap=${CAP}"
            pkill -KILL -f "train_dist_frozen.py" 2>/dev/null || true
            pkill -KILL -f "torchrun" 2>/dev/null || true
            sleep 3
            echo "[outer abort] sweep stopped — restart hetu, wait 10 min, then retry the failing config in isolation"
            exit "${rc}"
        fi
    done
done
