#!/usr/bin/env bash
# Per-layer computation profiling — three-pass loop (all / attention / mlp).
#
# Drives train_dist_frozen.py with --static_input + --laer_freeze_after_iter
# so per-component slopes are reproducible across (re)runs.
#
# IMPORTANT — three-pass is FSEP-off only.
# ----------------------------------------
# The attention / mlp passes ask the profiler to construct a model
# containing **only that layer type** (the other type is dropped
# entirely). Under standard MoE (FSEP off, all experts local) this is
# fine — total MoE compute is fixed by the activated-parameter count,
# so which experts route gets picked doesn't affect wall time.
#
# Under FSEP (Fully Sharded Expert Parallel) it is NOT fine: tokens
# are routed across the EP group via all-to-all, and per-rank
# straggler time depends on the routing distribution. The mlp pass
# routes on the **raw static input** because no attention layer
# preceded it; the all pass routes on **post-attention features**.
# These are different distributions → different per-rank load →
# different per-expert-layer wall time. The mlp slope under FSEP-on
# would be biased and would not match what the full block sees in
# real training.
#
# So this script enforces FSEP=off. The legacy companion
# ``_legacy/profile_computation_frozen.sh`` ran the FSEP-on `all` pass
# over (EP, capacity) tuples; it's no longer part of the workflow because
# the runtime calibration sweep (cost_model_real_test.sh) now captures
# FSEP-on full-iter measurements at every (tp, ep, micro_bsz, fsep=on,
# pp) shape, and `fsep_overhead_profile` is built from the FSEP-on/off
# pairs in runtime_profiling — not from FSEP-on compute fwd-only data.
# The cost model's FSEP-on path uses the FSEP-off per-component ratio
# (via the attention-invariance rule) to avoid relying on biased FSEP-on
# per-component data.
set -euo pipefail

export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}

export OMP_NUM_THREADS=8
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

export CUDA_HOME='/usr/local/cuda-12.1'

# MPS bypass — see CLAUDE.md (project memory). SIGKILL'd ranks otherwise
# leave dirty contexts on the host's GPUs for tens of minutes.
# export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/no-such-mps}

# Disable LAER online re-planning during the FSEP-off baseline so
# per-layer time stays stationary (linear-fit assumption). The
# legacy FSEP-on companion (_legacy/profile_computation_frozen.sh)
# flipped this to 1 so the solver ran and converged before the freeze.
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
STATIC_INPUT_PATH="${STATIC_INPUT_PATH:-${SCRIPT_DIR}/../static_inputs/frozen_batch_comp.pt}"
mkdir -p "$(dirname "${STATIC_INPUT_PATH}")"
echo "Using static input tensor: ${STATIC_INPUT_PATH}"

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

PROFILE_ARGS="
    --profile_mode batch \
    --profile_metric computation \
    --profile_min_batch_size 1 \
    --profile_max_batch_size 4 \
    --profile_batch_size_step 1 \
    --profile_seq_length_list 4096 \
    --mlp_profile_mode prof_mlp \
    --layernum_min 2 \
    --layernum_max 4 \
    --mixed_precision bf16 \
    --use-flash-attn \
    --static_input \
    --static_input_path ${STATIC_INPUT_PATH} \
    --laer_freeze_after_iter 5 \
    --dropout_prob 0"

# Refuse to run if anything in MODEL_ARGS / PROFILE_ARGS toggled FSEP
# on — see header comment. The three-pass attention/mlp slopes would
# be biased by routing-distribution mismatch.
if printf '%s %s' "${MODEL_ARGS}" "${PROFILE_ARGS}" | grep -q -- "--use_fsep"; then
    echo "[profile_computation] ERROR: --use_fsep is set in MODEL_ARGS / PROFILE_ARGS." >&2
    echo "  The three-pass loop strips layer types when constructing the" >&2
    echo "  model, which under FSEP biases the routing distribution and" >&2
    echo "  therefore per-expert wall time. Use profile_computation_frozen.sh" >&2
    echo "  for FSEP-on profiling — it runs only the 'all' pass." >&2
    exit 1
fi

# Three-pass loop (FSEP-off only):
#   all       — full transformer block (attention + MoE) per-layer time;
#               consumed as the fall-back when components aren't split.
#   attention — attention-only fwd time per layer; used to compute the
#               attn/MoE split ratio for asymmetric-layer cost queries.
#   mlp       — MoE-only fwd time per layer; same purpose.
#
# All three write into the same JSON; the unit suffix disambiguates keys
# (layertype_0_bsz<B>_seq<S>{,_attention,_mlp}). The cost model reads the
# split slopes when present and falls back to the full-block slope when
# only the `all` pass has run.
# for UNIT in all attention mlp; do
for UNIT in mlp; do
    echo "[profile_computation] pass: profile_unit=${UNIT}"
    python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS} --profile_unit ${UNIT}
done
