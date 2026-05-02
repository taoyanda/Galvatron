#!/usr/bin/env bash
# Generate a deterministic synthetic batch file consumable by the MoE
# dataloader (--static_input --static_input_path <output>).
#
# Override any of: MODEL_SIZE, BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE, PRECISION,
# SEED, OUTPUT_PATH, USE_FLASH_ATTN (set to 0 to emit attention_mask).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_SIZE="${MODEL_SIZE:-mixtral-8x7b-e8k2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-1234}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/../static_inputs/${MODEL_SIZE}_bs${BATCH_SIZE}_${PRECISION}.pt}"

EXTRA_ARGS=""
[[ -n "${SEQ_LENGTH:-}" ]] && EXTRA_ARGS+=" --seq_length ${SEQ_LENGTH}"
[[ -n "${VOCAB_SIZE:-}" ]] && EXTRA_ARGS+=" --vocab_size ${VOCAB_SIZE}"
[[ "${USE_FLASH_ATTN}" == "1" ]] && EXTRA_ARGS+=" --use_flash_attn"

python3 "${SCRIPT_DIR}/../tools/generate_static_input.py" \
    --output_path "${OUTPUT_PATH}" \
    --model_size "${MODEL_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --precision "${PRECISION}" \
    --seed "${SEED}" \
    ${EXTRA_ARGS}
