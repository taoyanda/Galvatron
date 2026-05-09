#!/usr/bin/env bash
# Real-measurement sweep for cost-model validation on Qwen3-30B-A3B
# (128 experts, top-k=8) on 4 GPUs. Runs train_dist_frozen.py (which
# handles PP correctly under --static_input by auto-loading the per-bsz
# deterministic batch from static_inputs/{model_size}_bs{N}_{precision}.pt)
# for each valid (pp, tp, ep, dp_mode, bsz, fsep) config with NUM_LAYERS layers,
# capturing per-config logs that contain:
#   - [real_measure] params_mb=…
#   - [real_measure] optimizer_mb=… activation_peak_mb=… cuda_peak_mb=…
#   - Average iteration time is: X s
# Logs land in galvatron/models/moe/logs/cost_model_real_*.log.
#
# Trainer choice: train_dist_frozen.py (NOT train_dist_random.py).
# train_dist_random.py caches the first random batch in-memory, which
# means tokens shift between runs and FSEP-on vs FSEP-off comparisons
# aren't apples-to-apples; train_dist_frozen.py pulls deterministic
# tokens from the per-bsz files, so the speedup we quote is reproducible.
#
# Same NCCL/MPS env recipe as profile_computation_frozen.sh — see
# doc/laer_fsep_sweep_resolution.md.
#
# Usage:
#   bash galvatron/models/moe/scripts/cost_model_real_test.sh                   # full sweep
#   bash galvatron/models/moe/scripts/cost_model_real_test.sh pp tp ep dp bsz fsep  # one config
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
# Leave NCCL_P2P_LEVEL unset so NCCL auto-discovers the per-pair
# transport at init (NVLink within islands, SHM across islands on the
# 2×2-island PCIe-A100 topology). Forcing a level caused multi-channel
# ring construction failures for full-world DP groups.
export TORCHINDUCTOR_COMPILE_THREADS=1

# Cross-NUMA NCCL ring-construction workaround. On the 2×2-island PCIe-A100
# fabric, NCCL's multi-channel ring builder produces an inconsistent layout
# whenever a collective spans both islands (raw_dp > island_size), wedging
# the rank with a "ring 1 does not loop back to start" error. Detect the
# island size once here; later, per-config, we prepend NCCL_P2P_DISABLE=1
# to the inner torchrun whenever raw_dp > island_size. Configs that fit in
# one island stay on default NCCL transports (intra-island NVLink intact).
# See doc/cross_numa_nccl_postmortem.md for the full investigation.
SCRIPT_DIR_FOR_DETECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_world=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
_island=$(python3 "${SCRIPT_DIR_FOR_DETECT}/detect_p2p_island_size.py" 2>/dev/null || echo 0)
if [ "${_island}" -gt 0 ] && [ "${_island}" -lt "${_world}" ]; then
    export GALVATRON_P2P_ISLAND_SIZE=${_island}
    echo "[p2p] detected island_size=${_island} world=${_world}; will prepend NCCL_P2P_DISABLE=1 for raw_dp>island launches"
fi
unset _world _island SCRIPT_DIR_FOR_DETECT

# Match production training's FSEP-required envelope (see train.sh):
#   - TORCH_NCCL_AVOID_RECORD_STREAMS=1: FSEP overrides manage tensor
#     lifetimes via custom events; PyTorch's auto record-stream double-
#     accounting otherwise frees activation storage at the wrong moment
#     and triggers an illegal-memory-access in silu/swiglu.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: matches train.sh.
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export ENABLE_SOLVER=${ENABLE_SOLVER:-1}

# NUM_LAYERS_LIST: space-separated layernums profiled per shape. Default
# "2 4" gives the aggregator two N points to fit `iter_ms = α + β · N`
# per-shape (and same for params/optimizer/activation/peak), which the
# cost model uses to extrapolate to production num_layers (48 on Qwen3-
# A3B). Set NUM_LAYERS_LIST="4" to revert to the legacy single-point
# sweep. Single-value NUM_LAYERS env var is honored as a back-compat
# alias when NUM_LAYERS_LIST is unset.
NUM_LAYERS_LIST=${NUM_LAYERS_LIST:-${NUM_LAYERS:-"2 4"}}
SEQ_LEN=4096
EPOCHS=20  # need ≥20 iters: profiler averages [10, 20), and we sample the
            # memory-evolution snapshot at iter 10.
# CHUNKS controls the number of microbatches per training iteration. The
# matrix tuple's "bsz" column is the per-stage compute batch size
# (micro_bsz); the trainer's global_train_batch_size is then
# ``micro_bsz × CHUNKS`` so each microbatch matches the calibration
# anchor at chunks=1. CHUNKS>1 also forces ``--no_async_grad_reduce``
# under FSEP-off (FSEP-on already requires it) — without it FSDP trips
# the ``_saved_grad_shard`` assertion at end-of-iteration grad finalize.
CHUNKS=${CHUNKS:-1}

# Qwen3-30B-A3B (128 experts, top-k=8) on 4 GPUs. Cost-model calibration
# sweeps over (pp, tp, ep, dp_mode, bsz, fsep). Only PP ∈ {1, 2} per
# user directive (PP=4 would put one layer per stage and overshoot the
# cost-model's PP-aware aggregator).
NUM_GLOBAL_EXPERTS=128

# Validity rules on world=4:
#   - per-stage world = world / pp must accommodate tp × ep × dp_per_stage
#   - FSEP-on requires tp == tp_of_ep AND dp_of_ep_size > 1
#     (= world/(pp*tp) > 1, i.e. pp*tp < world). When pp*tp == world the
#     FSDP-EP group degenerates to a single rank and the FSEP all-to-all
#     kernel returns NULL on backward — those tuples are excluded from
#     the FSEP-on track.
#   - FSEP-off has no FSEP-shape constraint. Standard Megatron MoE alltoall
#     handles ep > 1 fine, and train_dist_frozen.py keeps the rest of the
#     pipeline + FSDP collective sequence well-behaved (verified by step 4
#     successfully running pp=2 and tp=4 configs with zero3+SP+sdp). So
#     for every (pp, tp, ep) tuple, FSEP-off rows sweep both zero3 and
#     zero2sdp.
# A sister script can override the base matrix by setting
# ``CONFIGS_BASE_OVERRIDE`` to a newline-separated string of 6-tuples
# before sourcing this file — used by ``cost_model_real_test_chunks2.sh``
# (chunks=2 sweep) and historically by the legacy
# ``_legacy/cost_model_real_test_gap_fill.sh``, whose gbsz=1/2 entries
# are now folded into ``DEFAULT_CONFIGS_BASE`` directly.
if [ -n "${CONFIGS_BASE_OVERRIDE:-}" ]; then
    # ``mapfile -t`` reads one line per array element, stripping the
    # trailing newline. Tuples must already be space-separated internally
    # (same format as the default array entries below).
    mapfile -t DEFAULT_CONFIGS_BASE <<< "${CONFIGS_BASE_OVERRIDE}"
    echo "[matrix] using CONFIGS_BASE_OVERRIDE (${#DEFAULT_CONFIGS_BASE[@]} configs)"
else
DEFAULT_CONFIGS_BASE=(
    # "pp tp ep dp_mode micro_bsz fsep"  (dp_mode = zero2sdp, fsep ∈ {on, off})
    #
    # zero3 was dropped from the default matrix per project decision —
    # empirically it was uniformly slower than zero2sdp (4-15% on
    # full-iter, more on MLP-only) at identical memory peaks, with no
    # case where it was preferable. Existing zero3 calibration entries
    # in the runtime profile remain valid; we just don't extend them.
    #
    # The "micro_bsz" column is the per-stage compute batch size (=
    # trainer's global_train_batch_size when CHUNKS=1; CHUNKS>1 doubles
    # gbsz to keep the per-microbatch shape identical to the chunks=1
    # anchor).
    #
    # The matrix covers gbsz ∈ {4, 2, 1} to match the search's
    # micro_bsz ∈ {1, 2, 4} enumeration. gbsz=4 is the primary
    # calibration anchor; gbsz=2 entries fill the (tp, ep) feasible
    # subset at micro_bsz=2 (DP × EP ≤ 2 per per-rank ≥ 1 sample);
    # gbsz=1 entries cover the FSEP-off-only feasible shapes at
    # micro_bsz=1 (FSEP-on is infeasible — see gbsz=1 block below).
    #
    # FSEP-on / FSEP-off split: this iteration ships FSEP-on for
    # gbsz ∈ {4, 2} (per directive). gbsz=1 has no FSEP-on feasible
    # shapes at all (pp×tp must equal world=4 for per-rank ≥ 1, but
    # FSEP-on requires pp×tp < world), so the gbsz=1 rows are FSEP-off.
    #
    # === gbsz=4 PP=1 FSEP-on (dp_of_ep > 1) ===
    "1 1 1 zero2sdp 4 on"
    "1 1 2 zero2sdp 4 on"
    "1 1 4 zero2sdp 4 on"
    "1 2 1 zero2sdp 4 on"
    "1 2 2 zero2sdp 4 on"
    # (1 4 1 ...) FSEP-on excluded — dp_of_ep_size=1 (degenerate)
    #
    # === gbsz=4 PP=1 FSEP-off (full (tp, ep) matrix) ===
    # "1 1 1 zero2sdp 4 off"
    # "1 1 2 zero2sdp 4 off"
    # "1 1 4 zero2sdp 4 off"
    # "1 2 1 zero2sdp 4 off"
    # "1 2 2 zero2sdp 4 off"
    # "1 4 1 zero2sdp 4 off"
    #
    # === gbsz=4 PP=2 FSEP-on (per-stage world=2; dp_of_ep > 1 → tp=1) ===
    "2 1 1 zero2sdp 4 on"
    "2 1 2 zero2sdp 4 on"
    # (2 2 1 ...) FSEP-on excluded — dp_of_ep = 4/(2*2) = 1 (degenerate)
    #
    # === gbsz=4 PP=2 FSEP-off (full (tp, ep) matrix that fits per-stage=2) ===
    # "2 1 1 zero2sdp 4 off"
    # "2 1 2 zero2sdp 4 off"
    # "2 2 1 zero2sdp 4 off"
    #
    # === gbsz=2 (micro_bsz=2; feasibility: DP*EP ≤ 2) ===
    # "1 2 1 zero2sdp 2 off"
    # "1 2 2 zero2sdp 2 off"
    # (1 2 2 ... 2 on) excluded: TP>1 AND EP>1 with per_rank=1
    # trips relocate_activations' batch-dim TP shard
    # (1 % TP=2 != 0). MoESearcher.score() rejects the same
    # shape upfront, so calibration here would be unused.
    "1 2 1 zero2sdp 2 on"
    # "1 4 1 zero2sdp 2 off"
    # "2 1 1 zero2sdp 2 off"
    # "2 1 2 zero2sdp 2 off"
    "2 1 2 zero2sdp 2 on"
    "2 1 1 zero2sdp 2 on"
    # "2 2 1 zero2sdp 2 off"
    #
    # === gbsz=1 (micro_bsz=1; feasibility: DP=1, EP=1; PP*TP=4) ===
    # FSEP-on infeasible at micro_bsz=1: pp*tp must equal world=4 for
    # per-rank ≥ 1, but FSEP-on requires pp*tp < world. So gbsz=1 is
    # the one place the otherwise FSEP-on-only matrix ships FSEP-off
    # rows — without them, search queries at micro_bsz=1 hit the
    # analytical fallback (search.py enumerates micro_bsz ∈ {1, 2, 4}).
    # The aggregator's unit_breakdown indexes attention fsep-agnostically,
    # so the attention measurements from these rows still inform any
    # FSEP-on shape predictions at micro_bsz=1 if such predictions are
    # ever derived analytically.
    "1 4 1 zero2sdp 1 off"
    "2 2 1 zero2sdp 1 off"
)
fi

# Profile-unit expansion: each base 6-tuple gets fanned out into per-component
# rows so the cost-model gets per-shape attention/MLP/full breakdown end-to-end
# (forward + backward + optimizer). This obsoletes step 6's forward-only
# per-block compute profile and removes the `bwd_mult` coefficient from
# downstream predictions — the backward time is measured directly.
#
# All three units (all, attention, mlp) run for every (fsep ∈ {on, off})
# tuple. Historically (under a mixed FSEP-on/off matrix) we skipped
# ``attention+fsep=on`` as byte-identical to ``attention+fsep=off`` — but
# under the FSEP-on-only matrix that skip orphans attention entirely (no
# fsep=off entries to fall back to). The attention-only model has no
# MoE/router/dispatcher, so the FSEP flag is a no-op for that pass; the
# aggregator's ``unit_breakdown`` indexes attention by
# (tp, ep, micro_bsz, pp, dp_mode) (fsep-agnostic), so duplicate samples
# at fsep=on / fsep=off are averaged transparently.
#
# Set DEFAULT_PROFILE_UNITS to override (e.g. "all" for legacy single-pass).
DEFAULT_PROFILE_UNITS=${DEFAULT_PROFILE_UNITS:-"all attention mlp"}

DEFAULT_CONFIGS=()
for tuple in "${DEFAULT_CONFIGS_BASE[@]}"; do
    read -r _pp _tp _ep _dp _bsz _fsep <<< "${tuple}"
    for unit in ${DEFAULT_PROFILE_UNITS}; do
        DEFAULT_CONFIGS+=("${_pp} ${_tp} ${_ep} ${_dp} ${_bsz} ${_fsep} ${unit}")
    done
done
unset _pp _tp _ep _dp _bsz _fsep

# Per-config arg parsing.
# 7-tuple: (pp, tp, ep, dp_mode, bsz, fsep, profile_unit) — full control.
# 6-tuple: (pp, tp, ep, dp_mode, bsz, fsep) → profile_unit defaulted to "all".
# 5-tuple legacy: (tp, ep, dp_mode, bsz, fsep) → pp=1, profile_unit="all".
# 4-tuple legacy: (tp, ep, dp_mode, bsz) → pp=1, fsep=on, profile_unit="all".
if [ "$#" -eq 7 ]; then
    CONFIGS=("$1 $2 $3 $4 $5 $6 $7")
elif [ "$#" -eq 6 ]; then
    CONFIGS=("$1 $2 $3 $4 $5 $6 all")
elif [ "$#" -eq 5 ]; then
    CONFIGS=("1 $1 $2 $3 $4 $5 all")
elif [ "$#" -eq 4 ]; then
    CONFIGS=("1 $1 $2 $3 $4 on all")
else
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MODEL_DIR}/logs"
# Dead arg — train_dist_frozen.py auto-resolves the deterministic batch
# via `static_inputs/{model_size}_bs{N}_{precision}.pt` per per-rank bsz;
# `--static_input_path` is never consulted. Default to the bs1 file just
# so the path argparse sees is real (and so we don't lie about which
# file is "in use" — the trainer auto-resolver still wins).
STATIC_INPUT_PATH="${MODEL_DIR}/static_inputs/qwen-30b-a3b-e128k8_bs1_bf16.pt"
mkdir -p "${LOG_DIR}"

LAUNCHER="torchrun --nnodes ${NUM_NODES} --nproc_per_node ${NUM_GPUS_PER_NODE} --master_port ${MASTER_PORT}"

echo "[env] CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset>}"
echo "[env] NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-<unset>}"
echo "[env] NCCL_DEBUG=${NCCL_DEBUG}"
echo "[env] TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}"
echo "[env] ENABLE_SOLVER=${ENABLE_SOLVER}"

cd "${MODEL_DIR}"

echo "[matrix] NUM_LAYERS_LIST=${NUM_LAYERS_LIST}"
for NUM_LAYERS in ${NUM_LAYERS_LIST}; do
echo "========================================================"
echo "  cost_model_real: starting layernum sweep nl=${NUM_LAYERS}"
echo "========================================================"
for tuple in "${CONFIGS[@]}"; do
    # The matrix tuple's "bsz" column is now the per-stage micro_bsz
    # (= trainer's gbsz at chunks=1). At CHUNKS>1 we double gbsz to keep
    # micro_bsz fixed. Read into MICRO_BSZ for clarity.
    read -r PP TP EP DP_MODE MICRO_BSZ FSEP_MODE PROFILE_UNIT <<< "${tuple}"
    FSEP_MODE=${FSEP_MODE:-on}
    PROFILE_UNIT=${PROFILE_UNIT:-all}
    GLOBAL_BSZ=$(( MICRO_BSZ * CHUNKS ))
    CAP=$((NUM_GLOBAL_EXPERTS / EP))  # 128 / EP for Qwen3-30B-A3B
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
        # FSEP-off async grad reduce is fine at chunks=1 (one microbatch
        # per iter → no cross-microbatch grad accumulation). At chunks>1,
        # the async path trips an FSDP `_saved_grad_shard` assertion at
        # end-of-iter under MoE top-k sparse expert activation: experts
        # that didn't receive gradients in this iteration leave the
        # manual ``fsdp_reduce_gradients`` walker without a sharded grad
        # to finalize. Forcing sync grad reduce sidesteps that path
        # (real training under chunks>1 + MoE has to do the same).
        if [ "${CHUNKS}" -gt 1 ]; then
            NO_ASYNC_FLAG="--no_async_grad_reduce"
        else
            NO_ASYNC_FLAG=""
        fi
    fi
    # Default log path matches the original calibration sweep
    # (NUM_LAYERS=4, PP=1, profile_unit=all). Non-default values are tagged
    # in the filename so each variant has its own log. Legacy logs without
    # ``_unit{X}`` are read by the aggregator as ``profile_unit="all"`` for
    # backwards compatibility.
    # Filename uses the trainer's gbsz (= MICRO_BSZ × CHUNKS), matching the
    # ``--global_train_batch_size`` argument actually passed in. The
    # aggregator recovers ``micro_bsz = gbsz / chunks`` for shape-key
    # construction. Legacy logs without ``_chunks{N}`` are interpreted as
    # chunks=1 by the aggregator (back-compat).
    LOG_PATH="${LOG_DIR}/cost_model_real_tp${TP}_ep${EP}_${DP_MODE}_bsz${GLOBAL_BSZ}_fsep${FSEP_MODE}"
    if [ "${NUM_LAYERS}" != "4" ]; then
        LOG_PATH="${LOG_PATH}_nl${NUM_LAYERS}"
    fi
    if [ "${PP}" != "1" ]; then
        LOG_PATH="${LOG_PATH}_pp${PP}"
    fi
    if [ "${CHUNKS}" != "1" ]; then
        LOG_PATH="${LOG_PATH}_chunks${CHUNKS}"
    fi
    if [ "${PROFILE_UNIT}" != "all" ]; then
        LOG_PATH="${LOG_PATH}_unit${PROFILE_UNIT}"
    fi
    LOG_PATH="${LOG_PATH}.log"

    # Per-config throw-away MPS pipe dir to ensure no MPS control-socket
    # state can leak between configs.
    PER_RUN_MPS_DIR="/tmp/no-such-mps-${TP}-${EP}-${DP_MODE}-${GLOBAL_BSZ}-${PROFILE_UNIT}-$$"
    rm -rf "${PER_RUN_MPS_DIR}" 2>/dev/null || true
    export CUDA_MPS_PIPE_DIRECTORY="${PER_RUN_MPS_DIR}"

    # Per-config NCCL P2P workaround: if this config's raw_dp (= world /
    # (pp * tp)) exceeds the detected island size, set NCCL_P2P_DISABLE=1
    # for the inner torchrun. We use a NCCL_ENV array passed to ``env``
    # because bash only honours the literal `VAR=value cmd` env-prefix
    # syntax in source text — after parameter expansion it'd be parsed as
    # a command name (rc=127). See doc/cross_numa_nccl_postmortem.md.
    NCCL_ENV=()
    raw_dp=$(( NUM_GPUS_PER_NODE * NUM_NODES / (PP * TP) ))
    if [ "${GALVATRON_P2P_ISLAND_SIZE:-0}" -gt 0 ] && [ "${raw_dp}" -gt "${GALVATRON_P2P_ISLAND_SIZE}" ]; then
        NCCL_ENV=("NCCL_P2P_DISABLE=1")
    fi

    echo "========================================================"
    echo "  cost_model_real: pp=${PP} tp=${TP} ep=${EP} dp_mode=${DP_MODE} bsz=${GLOBAL_BSZ} chunks=${CHUNKS} fsep=${FSEP_MODE} unit=${PROFILE_UNIT} nl=${NUM_LAYERS} ${NCCL_ENV[*]:+(P2P_DISABLE)}"
    echo "  log: ${LOG_PATH}"
    echo "========================================================"

    rc=0
    env "${NCCL_ENV[@]}" timeout --kill-after=120 1200 \
        ${LAUNCHER} train_dist_frozen.py \
            --profile_mode batch --shape_order SBH --dropout_prob 0.0 \
            ${FSEP_FLAG} \
            --global_ep_deg ${EP} \
            --global_tp_of_ep_deg ${TP_OF_EP} \
            --expert_capacity_per_device ${CAP} \
            --profile_unit ${PROFILE_UNIT} \
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
        echo "[cost_model_real] pp=${PP} tp=${TP} ep=${EP} dp_mode=${DP_MODE} bsz=${GLOBAL_BSZ} fsep=${FSEP_MODE} unit=${PROFILE_UNIT} FAILED rc=${rc} — check ${LOG_PATH}"
        pkill -KILL -f "train_dist_frozen.py" 2>/dev/null || true
        pkill -KILL -f "torchrun" 2>/dev/null || true
        sleep 10  # let driver state quiesce before next config
        # don't abort the whole sweep — record per-config failure and proceed
    else
        echo "[cost_model_real] pp=${PP} tp=${TP} ep=${EP} dp_mode=${DP_MODE} bsz=${GLOBAL_BSZ} fsep=${FSEP_MODE} unit=${PROFILE_UNIT} OK"
        sleep 2
    fi
    # Clean the per-run MPS pipe dir whether the run succeeded or failed.
    # This removes any control-socket state that the CUDA driver tried to
    # create (or any state left by a SIGKILL'd rank), so the next config
    # inherits an empty path — preserving the bypass.
    rm -rf "${PER_RUN_MPS_DIR}" 2>/dev/null || true
done
done  # NUM_LAYERS loop
echo "all configs done; logs in ${LOG_DIR}"
