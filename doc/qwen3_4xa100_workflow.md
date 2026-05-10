# Workflow: Cost-model calibration + search for Qwen3-30B-A3B on 4×A100 80GB

Practical playbook for a fresh 4×A100 80GB cloud instance: build the
package, profile the model, calibrate the cost model, verify drift, run
the search, validate. Target model is **Qwen3-30B-A3B** registered as
`qwen-30b-a3b-e128k8` in `galvatron/models/moe/meta_configs/`.

**Wall-clock estimate**: ~2.5 hours from blank instance to validated
search output. Most of it is the calibration sweeps (~50 min Step 8a
main + ~25 min Step 8b chunks=2) plus the per-component profiles
(Steps 3–5, ~1.5 h combined) and a ~3 min end-to-end validation.
Setup + drift checks are minutes. (Steps 6 and 7 — the legacy
FSEP-on per-block compute / memory profiles — are excluded from
the workflow; their data is captured by Step 8a's α + β × N fit.)

**Skim time**: read top-to-bottom in 15 min before running anything.
Each step has explicit expected outputs you should verify.

This doc is the maintained 4-GPU replacement for `qwen3_8xa100_workflow.md`
(now in `_legacy/`). Multi-node deltas are out of scope here — see the
legacy doc if you scale beyond one node.

---

## 0. Pre-flight

### 0.1 Hardware

- **4× NVIDIA A100 80GB SXM4** (single node)
- ≥ 256 GB host RAM
- ≥ 500 GB local disk (calibration logs + profile JSONs accumulate)

This box has a **2×2 NVLink-island topology**: GPUs (0, 1) on island A
and (2, 3) on island B share NVLink at ~250 GB/s; cross-island traffic
crosses PCIe NODE at ~21 GB/s. The calibration sweep auto-detects this
and conditionally sets `NCCL_P2P_DISABLE=1` for collectives that span
both islands (otherwise NCCL's multi-channel ring builder hangs with
"ring 1 does not loop back to start" — see
`doc/cross_numa_nccl_postmortem.md`).

Verify cleanly:
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader
# Expected: 4 lines, each ~0 MB / 81920 MiB, 0 % util
```

If any GPU is held, **abort or wait**. The calibration sweep will fail
unpredictably if competing workloads are present.

### 0.2 Software

| Component | Version |
| --- | --- |
| CUDA | 12.1 (12.0/12.4 work with minor pinning adjustments) |
| Python | 3.9.2 (or 3.12) |
| PyTorch | 2.1.0 + cu121 |
| flash-attn | 2.5.8 |
| apex | commit 312acb4 |
| TransformerEngine | commit 7f2afaa |

The reference container is `hetu` (galvatron-image) — most commands
below assume `docker exec hetu …` for repeatability. Drop the prefix if
you're not in a container.

### 0.3 Memory budget

Under FSDP + zero2sdp + bf16 + Adam at the top calibrated config (PP=2,
EP=2, micro_bsz=4):

| Term | Per-rank |
| --- | ---: |
| bf16 params (per shard) | 3.1 GB |
| Adam optimizer state | 6.2 GB |
| Activations at peak (steady-state, 1 microbatch) | 6.2 GB |
| **`cuda_peak_mb` (allocated)** | **~15.4 GB** |
| **`cuda_peak_reserved_mb` (matches nvidia-smi)** | **~21 GB at chunks=32** |

Plenty of headroom on 80 GB. Activation recomputation
(`--global_checkpoint 1`) is on by default in the calibration scripts.

Note the ~5.5 GB allocated-vs-reserved gap at chunks > 1: PyTorch's
caching allocator fragments under multi-microbatch workloads. If you
budget against `nvidia-smi`, use ~72 GB safe; if against the
`cuda_peak_mb` reported by the cost model, use ~45 GB.

### 0.4 Why this much profiling?

The cost model needs four layers of inputs to be accurate:

1. **Per-component computation slopes** (Step 3, FSEP-off three-pass) —
   feed the FSEP smart-routing solver and asymmetric attention/expert
   split for the analytical fall-back.
2. **Per-block calibration anchors** (Step 8 main) — pin the absolute
   time/memory at each `(tp, ep, micro_bsz, seq, fsep, pp)` shape; the
   sweep runs at `NUM_LAYERS_LIST="2 4"` by default, so the aggregator
   fits α + β × N per shape and the cost model can extrapolate to
   production `num_layers=48`. Single-N sweeps cannot separate per-iter
   bias (α) from per-layer cost (β).
3. **FSEP overhead profile** (derived from on/off pairs in the same
   sweep) — captures FSEP-on dispatch cost per expert layer; used by
   the analytical fall-back for FSEP-on shapes without a calibration
   anchor.
4. **Per-microbatch overhead profile** (Step 8b, chunks=2 calibration)
   — derives the per-shape `time_per_extra_microbatch_ms` slope. Used
   by the cost-model PP shortcut path to predict iter_ms at any
   `num_microbatches`. Closes a structural ~30 % gap between the
   chunks=1-only Alpa-analytical prediction and reality at chunks ≫ 1.

Skip any of these and the cost model degrades — typically ±20-35 %
drift instead of the ±2-5 % we see at fully-calibrated rows.

### 0.5 Files you'll edit (reference)

| File | What you'll change |
| --- | --- |
| `galvatron/models/moe/scripts/profile_computation.sh` | model dims for Qwen3 |
| `galvatron/models/moe/scripts/profile_memory.sh` | model dims for Qwen3 |
| ~~`galvatron/models/moe/scripts/profile_computation_frozen.sh`~~ | ~~model dims, EP/cap tuples~~ — **moved to `_legacy/`**; not part of the workflow |
| ~~`galvatron/models/moe/scripts/profile_memory_frozen.sh`~~ | ~~model dims, EP/cap tuples~~ — **moved to `_legacy/`**; cost model never read its `_fsep`-suffixed output |
| `galvatron/models/moe/scripts/cost_model_real_test.sh` | `NUM_GLOBAL_EXPERTS`, model dims (`DEFAULT_CONFIGS_BASE` matrix is 4-GPU-shaped already) |
| `galvatron/models/moe/scripts/profile_embedding_lmhead.py` | top-of-file `MODEL = ...` |
| `galvatron/models/moe/scripts/profile_cost_model_terms.py` | `MODEL = ...`, `NUM_MOE_LAYERS = ...` |

The cost-model package itself (`galvatron/models/moe/cost_model/`)
needs no changes — it's model-agnostic; everything is read from
`meta_configs/<model>.json` and the per-shape profile JSONs.

---

## 1. Install Galvatron + build kernels (~30 min)

```bash
git clone <repo-url> /workspace/Galvatron
cd /workspace/Galvatron

pip install -r requirements.txt
pip install -e . --no-build-isolation
```

The `-e .` install builds two CUDA / C++ extensions:

- `greedy_balancer` (csrc/greedy_balancer.cpp) — LAER planner
- `moe_all_to_all_kernels` (csrc/moe_all_to_all_*.{cpp,cu}) — fused NCCL MoE all-to-all

Verify both compiled:
```bash
python3 -c "import moe_all_to_all_kernels, greedy_balancer; print('OK')"
```

Failure means runtime training won't work — but the **cost model will**,
because `galvatron/models/moe/__init__.py` import-guards the runtime
stack. You can do all the cost-model work on a CPU-only machine if
needed; only the calibration sweep + per-component profiles require
GPUs.

### 1.1 Set environment

```bash
cat > /workspace/Galvatron/setup-env.sh <<'EOF'
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NODE_RANK=0
export OMP_NUM_THREADS=8

export NCCL_DEBUG=WARN
export TORCH_NCCL_AVOID_RECORD_STREAMS=1                 # required for FSEP correctness
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-such-mps
export TORCHINDUCTOR_COMPILE_THREADS=1
export ENABLE_SOLVER=0   # off by default; enabled per-script when FSEP needs it
EOF

source /workspace/Galvatron/setup-env.sh
```

### 1.2 Verify the meta-config exists

```bash
ls /workspace/Galvatron/galvatron/models/moe/meta_configs/qwen-30b-a3b-e128k8.json
```

Contents:
```json
{
    "hidden_size": 2048,
    "intermediate_size": 768,
    "max_position_embeddings": 4096,
    "num_attention_heads": 32,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "num_local_experts": 128,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000000.0,
    "router_aux_loss_coef": 0.001
}
```

---

## 2. Hardware bandwidth profile (~5 min)

Produces per-cluster bandwidth coefficients the cost model uses for DP
all-reduce and EP all-to-all timing.

```bash
cd /workspace/Galvatron/galvatron/profile_hardware
bash scripts/profile_hardware.sh
```

**Output**: `hardware_configs/{allreduce_bandwidth,p2p_bandwidth,...}_1nodes_4gpus_per_node.json`

Then propagate the bandwidth values into the cost-model's
`network_config.json`. **Measured on this 2×2-island PCIe-A100 box**:
intra-island NVLink ≈ 250 GB/s, inter-island PCIe NODE ≈ 21 GB/s.

```bash
cat > /workspace/Galvatron/galvatron/models/moe/configs/network_config.json <<'EOF'
{
    "intra_node_nvlink": 250.0,
    "inter_node_pcie": 21.0,
    "intra_node": 250.0,
    "inter_node": 25.0
}
EOF
```

(Legacy fields `intra_node` and `inter_node` are kept for back-compat
with parts of the cost model that haven't been updated to the
island-aware naming yet.)

---

## 3. Per-component computation profile, FSEP-off (~30 min, 1 GPU)

Three-pass loop over `--profile_unit ∈ {all, attention, mlp}`. Produces
per-layer fwd-only time slopes the cost model uses for the
asymmetric-attention/expert split when no per-component runtime sample
is available.

### 3.1 Edit `profile_computation.sh`

`MODEL_ARGS` block:
```bash
MODEL_ARGS="
    --model_size qwen-30b-a3b-e128k8 \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --vocab_size 151936 \
    --hidden_size 2048 \
    --num_attention_heads 32 \
    --num_key_value_heads 4 \
    --intermediate_size 768 \
    --num_local_experts 128 \
    --num_experts_per_tok 8 \
    --seq_length 4096"
```

### 3.2 Run

```bash
docker exec hetu bash -lc \
  "cd /root/Galvatron/galvatron/models/moe && bash scripts/profile_computation.sh"
```

### 3.3 Verify

```bash
ls galvatron/models/moe/configs/computation_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096.json
```

---

## 4. Per-component memory profile, FSEP-off (~60 min, 2 GPUs)

```bash
# Edit MODEL_ARGS in profile_memory.sh (same as Step 3.1)
# Set NUM_GPUS_PER_NODE=2 to sweep tp ∈ {1, 2}
docker exec hetu bash -lc \
  "cd /root/Galvatron/galvatron/models/moe && NUM_GPUS_PER_NODE=2 bash scripts/profile_memory.sh"
```

### 4.1 Verify

```bash
ls galvatron/models/moe/configs/memory_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096*.json
ls galvatron/models/moe/configs/non-solver/memory_profiling_bf16_qwen-30b-a3b-e128k8.json
```

The cost model reads the `non-solver/` processed file. If only the
raw per-(tp, ep) files exist and `non-solver/` is empty, the
processing step in `_process_memory_data` didn't fire — check the
log for "Already written processed memory" lines.

---

## 5. Embedding + LM-head standalone profile (~5 min, 1 GPU)

```python
# Edit galvatron/models/moe/scripts/profile_embedding_lmhead.py:
MODEL = "qwen-30b-a3b-e128k8"
PRECISION = "bf16"
DTYPE = torch.bfloat16
```

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/profile_embedding_lmhead.py
```

**Output**: `configs/embedding_lmhead_profiling_bf16_qwen-30b-a3b-e128k8.json`

LM-head fwd_bwd should dominate (it's a hidden→vocab GEMM of ~310 M
params: 2048 × 151936). Embed is just an indexed gather → backward
sparse-scatter; much smaller.

---

## 6. ~~FSEP-on per-block computation profile~~ — REMOVED FROM WORKFLOW

Step 6 used to run `profile_computation_frozen.sh` to produce FSEP-on
per-block fwd-only compute data. **It's been moved to
`scripts/_legacy/`** because:

1. The runtime calibration sweep (Step 8) now captures FSEP-on full-iter
   measurements at every (tp, ep, micro_bsz, fsep=on, pp) shape, with
   an `α + β × N` fit covering layer-count extrapolation. The cost
   model's PP shortcut path looks up runtime_profile directly.
2. `fsep_overhead_profile` is built from FSEP-on/off pairs in
   `runtime_profiling` — not from `computation_profiling_*_tp{T}_ep{E}.json`.
3. `v_comp` (greedy balancer) and the FSEP smart-routing solver read
   the FSEP-**off** computation profile (single-MLP per-token cost is
   FSEP-invariant).

The `_tp{T}_ep{E}`-suffixed compute-profile lookup path remains in
`cost_model/intra.py` as a tertiary fall-back, so the legacy script
can be re-run if the search ever queries shapes outside the
calibration matrix. Otherwise: skip this step.

---

## 7. ~~FSEP-on memory profile~~ — REMOVED FROM WORKFLOW

Step 7 used to run `profile_memory_frozen.sh` to produce FSEP-on per-(tp,
ep) memory profiles (`memory_profiling_*_tp{T}_ep{E}_fsep.json`).
**It's been moved to `scripts/_legacy/`** because:

1. The cost model has zero code paths that load `_fsep`-suffixed memory
   JSONs. Both `memory_profile` lookup paths in `intra.py` (the main
   loader at lines 85-102 and `_raw_memory_path` at 521-530) build
   non-fsep filenames; grep for `_fsep.json` in `cost_model/` returns no
   matches. The output of this script was produced but never read.
2. The runtime calibration sweep (Step 8) captures FSEP-on `cuda_peak_mb`,
   `params_mb`, `optimizer_mb`, `activation_peak_mb` at every shape with
   `α + β × N` fitting, covering production num_layers extrapolation.
3. `fsep_overhead_profile` (built from runtime FSEP on/off pairs in
   Step 9) supplies the per-MoE-layer memory delta used by the
   analytical fall-back in `intra.py:_fsep_memory_overhead_per_expert_layer_mb`.

Pre-existing `_fsep`-suffixed JSONs in `configs/` are harmless — leave
them in place or `git rm` separately.

---

## 8. Calibration sweep (~50 min main + ~25 min chunks=2)

This is what actually anchors the cost model's predictions on real
measurements. Each per-config run produces one log in `logs/` with
`[real_measure]`, `[stage_time]`, and `Average iteration time is:`
lines that `profile_cost_model_terms.py` aggregates in Step 9.

The sweep is split into two parts run sequentially:

- **8a. Main matrix** (~50 min): the production calibration covering all
  feasible `(pp, tp, ep, dp_mode, fsep)` tuples at micro_bsz ∈ {4, 2, 1}
  with per-component fan-out (`profile_unit ∈ {all, attention, mlp}`).
  Runs each shape at **two layernums** (`NUM_LAYERS_LIST="2 4"`) so the
  aggregator can fit `α + β · N` per shape — used to extrapolate to
  production `num_layers=48`. Without two N points, the cost model
  falls back to scaling-by-N which assumes `α = 0` (all overhead is
  per-layer, which it isn't — embedding/lm-head/PP-init costs are
  per-iter). gbsz ∈ {4, 2, 1} are all native to the matrix so a single
  sweep covers every shape the search will query (search enumerates
  micro_bsz ∈ {1, 2, 4}).
- **8b. Per-microbatch overhead** (~25 min): same matrix at chunks=2
  with full `{all, attention, mlp}` profile-unit fan-out, to derive
  both:
  - `time_per_extra_microbatch_ms` slopes per shape (Alpa-formula
    correction for forced-sync grad-reduce, PP send/recv churn,
    scheduler overhead).
  - **Per-component activation slopes** per shape (`attention_alloc_per_extra_microbatch_mb`
    and `mlp_alloc_per_extra_microbatch_mb`): in 1F1B, activations stack
    with chunks but grads accumulate in-place, so the chunks=2−chunks=1
    cuda_peak delta on a `unit=attention` / `unit=mlp` model directly
    measures per-microbatch activation memory of that component,
    separated from grad-bucket and optimizer-state contributions. The
    IntraCostModel consumes these for its 1F1B PP `extra_reserve_mb`
    calculation under asymmetric-layer queries (Phase 1b of
    asymmetric search plan).

  Pinned to `NUM_LAYERS_LIST="2"` only — per-microbatch overhead and
  per-microbatch activation are both layernum-invariant in expectation
  (the chunks delta isolates the per-microbatch axis), and smaller nl
  makes each inner config faster. The aggregator pairs chunks=1 ↔
  chunks=2 within the same `num_layers`, so the chunks=1 anchor at
  nl=2 from Step 8a is what's used.

### 8.1 Customize `cost_model_real_test.sh`

(a) **Top-of-file constants** (already 4-GPU-shaped):
```bash
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-4}
NUM_GLOBAL_EXPERTS=128
```
Both `NUM_NODES` and `NUM_GPUS_PER_NODE` honor env-var overrides so a
non-default world size can be requested ad-hoc (e.g.
`NUM_GPUS_PER_NODE=2 bash scripts/cost_model_real_test.sh 1 1 2 zero2sdp 4 on all`
to profile the matching shape that a PP=2 stage on 4 GPUs would see).
Whenever `world ≠ 4`, the log filename is suffixed with `_w{world}`
to avoid colliding with the default 4-GPU calibration log path —
critical because `logs/` is a symlink to the untracked `profile_logs/`,
so a same-name overwrite is unrecoverable from git.

(b) **Model launch args** (in the trainer invocation): match Qwen3's
hidden=2048, intermediate=768, num_local_experts=128, etc. The script
ships with these defaults.

(c) **Static-input convention**: the script auto-loads
`static_inputs/qwen-30b-a3b-e128k8_bs{N}_{precision}.pt` per per-rank
micro_bsz. Generate them once if missing:
```bash
docker exec hetu bash -lc \
  "cd /root/Galvatron/galvatron/models/moe && bash scripts/generate_static_input.sh"
```

(d) **Default matrix**: ships with FSEP-on-only zero2sdp entries at
gbsz ∈ {4, 2} (zero3 dropped — empirically slower at identical
memory peaks; FSEP-off dropped per current iteration directive),
plus two FSEP-off zero2sdp entries at gbsz=1 (FSEP-on is infeasible
at micro_bsz=1 because it requires `pp*tp < world` while the only
viable per-rank≥1 layouts have `pp*tp = world=4`). 12 base configs
(7 at gbsz=4 + 3 at gbsz=2 + 2 at gbsz=1) × {all, attention, mlp}
fan-out × {nl=2, nl=4} ≈ 72 runs. Attention is profiled on every
shape (the historical ``attention+fsep=on`` skip orphaned attention
entirely under FSEP-on-only matrices and was removed); the
aggregator's ``unit_breakdown`` indexes attention fsep-agnostically,
so the same measurement applies to both fsep on/off shape keys.

The `(PP=1, TP=2, EP=2, gbsz=2)` row is excluded from gbsz=2
(per_rank=1 + TP=2 + EP>1 trips Galvatron's `relocate_activations`
batch-dim shard; `MoESearcher.score()` rejects the same shape
upfront, so calibration here would be unused).

(e) **NCCL P2P workaround**: the script auto-detects island size via
`detect_p2p_island_size.py` and prepends `NCCL_P2P_DISABLE=1` for
collectives that span both islands. No manual config needed.

(f) **Aggregator nl handling** (`profile_cost_model_terms.py`):
- `runtime_profile`: keeps **both** N samples per shape and fits
  `α + β · N` per (params_mb, optimizer_mb, activation_peak_mb,
  cuda_peak_mb, fwd_bwd_ms, opt_ms, iter_ms). Cost model picks the
  exact-N sample first; falls back to the fit when the queried N
  isn't in the calibration set (e.g., production N=48).
- `fsep_overhead_profile`: pairs FSEP-on/off **within** the same N,
  then averages per-layer overhead across N pairs. Multi-N gives 2×
  more pair samples per shape → more robust slope.
- `chunks_overhead_profile`: pairs chunks=1/chunks=2 within the same
  N. Step 8b is pinned to nl=2 only, so the pairing matches the
  nl=2 chunks=1 anchor produced by Step 8a.
- `unit_breakdown` (per-component attention vs MLP split): **pinned
  to a single canonical N per aggregator run** (DEFAULT=4, else max
  N present). The cost model derives per-layer cost via
  `attention_fwd_bwd_ms / unit_num_layers`, which expects one N per
  shape; mixing N values silently halves per-layer cost. The
  aggregator prints `# unit_breakdown: multiple num_layers present
  [...], pinning to nl=4` when it filters. The nl=2 unit_breakdown
  measurements are discarded — α + β fitting per component is
  future work (see `doc/asymmetric_search_plan.md` Phase 1b).

### 8a. Main calibration sweep — multi-world PP=1 matrix

Multi-world PP=1 calibration now replaces direct PP=k measurement: for
PP > 1 prediction, the cost model looks up the matching shrunk-world
calibration (a PP=k stage on a 4-GPU box has the same per-rank compute
as a PP=1 run on a 4/k-GPU world with the same `(tp, ep, mbsz)`).
This captures the saturation regime that direct PP=k calibration at
small `num_layers/stage` misses; validation drops PP=2 nl=12 drift
from +13 % to ±4 %.

Run the three world sizes in sequence:
```bash
docker exec hetu bash -lc \
  "cd /root/Galvatron && \
   bash galvatron/models/moe/scripts/cost_model_real_test.sh    > /tmp/step8a_w4.log 2>&1 && \
   bash galvatron/models/moe/scripts/cost_model_real_test_w2.sh > /tmp/step8a_w2.log 2>&1 && \
   bash galvatron/models/moe/scripts/cost_model_real_test_w1.sh > /tmp/step8a_w1.log 2>&1"
```

Expected counts (default `DEFAULT_PROFILE_UNITS="all"`):
- `cost_model_real_test.sh` (world=4): **9 base × 2 nl = 18 runs, ~7 min**
- `cost_model_real_test_w2.sh` (world=2): **7 base × 2 nl = 14 runs, ~5 min**
- `cost_model_real_test_w1.sh` (world=1): **3 base × 2 nl = 6 runs, ~2 min**

Set `DEFAULT_PROFILE_UNITS="all attention mlp"` to enable the per-
component fan-out (×3 runs) — required only for the asymmetric search
path.

The legacy PP=2 calibration entries are now opt-in via
`cost_model_real_test_pp2.sh` (preserves the historical data path for
cross-checks; not part of the minimum-working profiling).

Logs from world ≠ 4 land with the `_w{world}` filename suffix to keep
them disjoint from the 4-GPU baselines (`logs/` is a symlink to the
untracked `profile_logs/`, so accidental overwrites are unrecoverable
from git — see `feedback_logs_symlink_trap` memory).

### 8b. Per-microbatch overhead calibration (chunks=2)

Three sister scripts mirror the multi-world chunks=1 matrix at
`CHUNKS=2`:
```bash
docker exec hetu bash -lc \
  "cd /root/Galvatron && \
   bash galvatron/models/moe/scripts/cost_model_real_test_chunks2.sh    > /tmp/step8b_w4.log 2>&1 && \
   bash galvatron/models/moe/scripts/cost_model_real_test_chunks2_w2.sh > /tmp/step8b_w2.log 2>&1 && \
   bash galvatron/models/moe/scripts/cost_model_real_test_chunks2_w1.sh > /tmp/step8b_w1.log 2>&1"
```

Expected: 9 + 7 + 3 = **19 runs at nl=2 each, ~7 min total**.
`NUM_LAYERS_LIST="2"` is pinned at the smallest nl per
`feedback_chunks_overhead_min_layernum` (chunks_overhead is layernum-
invariant in expectation; smaller nl = faster). Pairs with the
chunks=1 anchor at the same world from step 8a to derive the
per-microbatch time and activation slopes.

This step is essential for two reasons:

1. **iter_ms prediction at chunks > 1**: validation at chunks=32 drops
   the cost-model drift from +34 % to −2.3 % once the per-shape
   `time_per_extra_microbatch_ms` slope is in place.
2. **Per-component activation memory** (attention vs MLP per
   microbatch): in 1F1B activations stack with chunks but grads
   accumulate in-place, so the chunks=2 − chunks=1 cuda_peak delta on
   `unit=attention` / `unit=mlp` models directly measures
   per-microbatch activation per component, separated from
   grad-bucket / optimizer-state contributions. The IntraCostModel's
   PP `extra_reserve_mb` calculation under asymmetric layer-split
   queries reads `attention_alloc_per_extra_microbatch_mb` and
   `mlp_alloc_per_extra_microbatch_mb` from
   `chunks_overhead_profiling_*.json`. This replaces the legacy
   Step 4 `profile_memory.sh` per-component data path (which had
   reliability issues — values were byte-identical across (tp, ep)
   variants under SP, violating physical expectation).

### 8.4 Verify the calibration logs

```bash
ls galvatron/models/moe/logs/cost_model_real_*.log | wc -l
# expected: 72 (Step 8a main: 12 × 3 × 2) + 36 (Step 8b chunks=2: 12 × 3 × 1)
# = 108 logs total

# Sanity-check one log carries the [real_measure] + [stage_time] +
# Average iteration time lines:
docker exec hetu grep -E "real_measure|stage_time|Average iter" \
  galvatron/models/moe/logs/cost_model_real_tp1_ep4_zero2sdp_bsz4_fsepon.log
```

---

## 9. Aggregate calibration logs into JSON profiles (~10 sec, CPU)

### 9.1 Edit `profile_cost_model_terms.py`

Top of `galvatron/models/moe/scripts/profile_cost_model_terms.py`:
```python
MODEL = "qwen-30b-a3b-e128k8"
PRECISION = "bf16"
SEQ_LEN = 4096
NUM_MOE_LAYERS = 4   # what the calibration ran with (--num_hidden_layers)
NUM_GPUS_PER_NODE = 4
```

### 9.2 Run

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/profile_cost_model_terms.py
```

### 9.3 Outputs

| File | Contents |
| --- | --- |
| `optimizer_step_profiling_bf16_qwen-30b-a3b-e128k8.json` | Adam throughput (MB/ms) + optimizer_to_params_ratio |
| `runtime_profiling_bf16_qwen-30b-a3b-e128k8.json` | Per-shape calibrated `iter_ms`, `cuda_peak_mb`, `activation_peak_mb`, `params_mb`, `optimizer_mb`, `fwd_bwd_ms`, `opt_ms`. Plus `unit_breakdown.per_dp_mode` from per-component runs. |
| `fsep_overhead_profiling_bf16_qwen-30b-a3b-e128k8.json` | Per-shape FSEP overhead (`time_overhead_per_expert_layer_ms`, `memory_overhead_per_expert_layer_mb`); also `_from_mlp` derived from chunks=1 mlp on/off pairs |
| `chunks_overhead_profiling_bf16_qwen-30b-a3b-e128k8.json` | Per-shape per-dp_mode `time_per_extra_microbatch_ms` (the chunks=2 derived slope) |

### 9.4 Schema notes

Shape keys are `tp{T}_ep{E}_micro_bsz{M}_seq{S}_fsep{ON|OFF}[_pp{P}]`.
The `micro_bsz` field is the **per-stage compute batch size** (=
trainer's `--global_train_batch_size` at chunks=1; DP and EP only enter
as feasibility guards, not as shape-key dimensions). This was renamed
from a misleading `bsz` field that previously encoded per-rank micro
batch — the cost model and aggregator are consistent on `micro_bsz` end
to end.

---

## 10. Drift verification (no GPU, ~30 sec)

### 10.1 Regression suite (24 invariants)

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/cost_model_split_regression.py
```

Expected: all 24 checks PASS. Catches symmetric identity violations,
α/β fit residuals, FSEP-overhead inconsistencies, etc.

### 10.2 Per-shape drift table

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/cost_model_drift.py
```

Expected: ≤ 5 % drift at calibrated rows (including the chunks-overhead
correction at multi-microbatch queries).

### 10.3 PP critical-path drift

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/cost_model_pp_drift.py
```

### 10.4 α + β × N extrapolation drift

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/cost_model_alpha_beta.py
```

### 10.5 Multi-world matching-shape drift (PP > 1)

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/validate_unseen_drift.py
```

Expected after Step 8 multi-world sweep (Qwen3-30B-A3B, nl=12):

| Config | Δt | Δm |
|---|---:|---:|
| PP=2 mbsz=4 ch=1 | +0.6 % | −1.5 % |
| PP=2 mbsz=4 ch=2 | −1.8 % | −1.0 % |
| PP=2 mbsz=4 ch=4 | −0.6 % | −1.0 % |
| PP=2 mbsz=2 ch=1 | −2.7 % | −0.8 % |
| PP=2 mbsz=2 ch=2 | −5.1 % | −0.5 % |
| PP=1 mbsz=4 ch=1 | −7.7 % | −7.0 % |

PP > 1 max |Δt| ≤ 5.1 % (down from +13.3 % pre-multi-world). Overall
bounded by the unchanged PP=1 baseline.

---

## 11. Search (~1 sec, CPU)

### 11.1 Main search

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
    --model qwen-30b-a3b-e128k8 \
    --num-gpus 4 --num-layers 48 \
    --global-bsz 4 \
    --top-k 12 \
    --gpu-memory-mb 72657
```

Top output for the calibrated regime (gbsz=4):
```
1  2  1  1  2  zero2sdp  off |  iter_ms=379  peak_mb=15494  ...
```

I.e. PP=2, EP=2, zero2sdp, fsep=off — at 379 ms/iter on calibrated data.

`--gpu-memory-mb 72657` matches the safe budget for ~80 GB A100s (leaves
~7 GB for fragmentation; raise to 78000+ on H100 or if you've measured
fragmentation as smaller for your workload).

### 11.2 Micro_bsz comparison sweep

For larger global_bsz where `chunks > 1` matters:

```bash
docker exec hetu python3 \
  /root/Galvatron/galvatron/models/moe/scripts/cost_model_search_micro_bsz_sweep.py \
    --model qwen-30b-a3b-e128k8 \
    --num-gpus 4 --num-layers 48 \
    --global-bsz 128 \
    --micro-bsz 1 2 4 \
    --trust calibrated
```

Reports the optimal config separately at each `micro_bsz` value, with
calibrated-only filtering. At gbsz=128 the optimum is the same shape as
gbsz=4 (`PP=2 EP=2 zero2sdp fsep=off, micro_bsz=4`) at ~5,613 ms/iter.

#### gbsz=128 stress test (24 layers)

Cost-model query at `nl=24, gbsz=128` across PP ∈ {1, 2, 4} (peak ≤ 72.6 GB):

| pp tp ep mbsz chunks | iter_ms | peak_mb | feasible |
|---|---:|---:|:---|
| 1, 1, 4, 4, 32 | 80715 | 69828 | ✅ |
| 2, 2, 1, 4, 32 | **47195** | **49364** | ✅ best |
| 4, 1, 1, 4, 32 | 123401 | 53299 | ✅ |
| 2, 1, 2, 4, 32 | 31906 | 76925 | ❌ OOM (+4 GB) |

#### Coverage gaps

- **Per-component (attention/mlp) profile not populated by default.**
  Search reports `Per-component attention time slope missing` for
  shapes outside the calibrated `(tp, ep, mbsz)` domain. Re-run the
  Step 8 sweeps with `DEFAULT_PROFILE_UNITS="all attention mlp"` to
  unlock those paths (~+50 min total across all three worlds).
- **48 layers at PP=1 dp=1 is genuinely OOM** (~140 GB peak vs 72.6 GB
  budget). Need DP > 1 or PP > 1 to fit.

### 11.3 Top-config validation (smoke test)

End-to-end validation: actually run the trainer at the search's top
config and compare measured `iter_ms` to predicted.

```bash
docker exec hetu bash -lc \
  "cd /root/Galvatron && bash galvatron/models/moe/scripts/validate_top_config.sh"
```

Edit the script first if the top config differs (it ships with the
known-good `gbsz=128, chunks=32, PP=2, EP=2, zero2sdp, fsep=off,
micro_bsz=4`).

Expected: `Average iteration time` lands within ~3 % of the search's
predicted iter_ms. If it doesn't, either:
- The chunks_overhead profile is missing (re-run Step 8b).
- A regime-specific overhead has changed (e.g., new PyTorch version with
  different FSDP internals) — re-run Step 8b to refit slopes.

---

## 12. Cost-model architecture notes (Phase 0 fixes baked in)

The cost-model package (`galvatron/models/moe/cost_model/`) carries
several fixes that aren't visible from the workflow steps but matter
for understanding what the predictions mean:

- **Alpa-style 1F1B critical path** in `pp.py` analytical and shortcut
  paths: `T_pipeline = bottleneck × (num_mb − 1) + Σ stage_compute +
  opt`. Per-stage estimates are computed individually (not via
  first/middle/last representatives), so the formula extends correctly
  to asymmetric per-stage layer counts.
- **PP-depth-aware peak memory**: when num_microbatches > 1, the
  cost-model auto-derives `natural_n_behind = min(pp, num_mb) − 1` and
  reserves `n × per_microbatch_act_mb` of additional activation memory
  on top of the calibrated chunks=1 peak. No caller knob required.
- **Per-microbatch overhead**: `pp.py` shortcut path consumes
  `chunks_overhead_profile` when present and uses
  `iter_ms_at_chunks1 + slope × (num_mb − 1)` instead of the analytical
  Alpa term. Falls back to analytical if no chunks_overhead anchor
  exists for the shape.
- **DP-mode isolation in unit_breakdown**: per-component (attention/mlp)
  fwd+bwd time is stored per dp_mode (`zero2sdp` vs `zero3`) — the cost
  model picks the matching block based on the caller's
  `(zero_stage, sdp)`.
- **Search feasibility guards**: `MoESearcher.score()` rejects configs
  where (a) `dp × ep > micro_bsz` (per-rank < 1 sample), (b)
  `EP > 1 AND TP > 1 AND per_rank_micro_bsz % TP != 0` (Galvatron's
  `relocate_activations` batch-dim shard assertion at TP-layout
  transition boundaries — see Troubleshooting).

See `doc/asymmetric_search_plan.md` for the work that landed these
fixes (Phase 0a–f).

---

## 13. Troubleshooting

### NCCL hang during calibration

```
ring 1 does not loop back to start
```

Cross-island NCCL with multi-channel ring builder fails on the 2×2
NVLink-island PCIe-A100 fabric. Already mitigated by the
auto-`NCCL_P2P_DISABLE=1` per-config in `cost_model_real_test.sh`. If
it still fires:

- Verify `detect_p2p_island_size.py` returned a sensible island size:
  `python3 galvatron/models/moe/scripts/detect_p2p_island_size.py`
  (expected: `2` on this hardware).
- Check the per-config `(P2P_DISABLE)` tag in the sweep banner.
- Manually force `NCCL_P2P_DISABLE=1` for the whole sweep if needed.

See `doc/cross_numa_nccl_postmortem.md` for the full investigation.

### `_saved_grad_shard` assertion at chunks > 1

```
AssertionError: All sharded parameters that received a gradient in the
post-backward should use `_saved_grad_shard`
```

Galvatron's `fsdp_reduce_gradients` doesn't initialize `_saved_grad_shard`
on FSDP-wrapped MoE expert params that didn't receive gradients in a
microbatch (top-k=8 routing leaves most experts inactive). The
calibration scripts (`cost_model_real_test_chunks2.sh`) auto-add
`--no_async_grad_reduce` when `CHUNKS > 1`, which sidesteps this path.
Real training under `chunks > 1 + MoE` must do the same.

### `First dimension of the tensor should be divisible by tensor parallel size`

Galvatron's `relocate_activations` path
(`galvatron/core/runtime/redistribute.py:_split_along_first_dim_with_sequence_parallel`)
batch-dim shards under SBH + sequence_parallel=True at TP-layout
transition boundaries. Under MoE the boundary fires when EP > 1
(tp_of_ep ≠ body TP). Constraint: `per_rank_micro_bsz % TP == 0` when
`EP > 1 AND TP > 1`.

The cost-model search now rejects these configs upfront. Real
training: pick a larger `per_rank_micro_bsz` or set TP=1.

### iter_ms prediction off by ~30 % at chunks > 1

You're missing the `chunks_overhead_profiling_*.json` anchor. Re-run
Step 8b. If the gap remains, the chunks_overhead loader in
`pp.py` shortcut path may not be finding the file — check
`docker exec hetu ls galvatron/models/moe/configs/chunks_overhead_*.json`.

### OOM mid-iteration

- Verify `--global_checkpoint 1` is in the trainer flags
  (calibration scripts set this by default).
- Check `cuda_peak_reserved_mb` not just `cuda_peak_mb` —
  fragmentation under chunks > 1 + sync grad reduce can push reserved
  ~5 GB above allocated.
- Drop the GPU-memory budget when running search:
  `--gpu-memory-mb 65000` rather than 72657.

### "No computation profile found for tp=X, ep=Y" in search

Step 3 / Step 4 didn't produce a profile at that (tp, ep), and the
runtime calibration matrix doesn't cover the queried shape either.
Either calibrate it (re-run the relevant profile script with that
(tp, ep) added to the matrix; or for FSEP-on, the legacy
`_legacy/profile_computation_frozen.sh` produces per-(tp, ep) compute
files that the loader reads as a tertiary fall-back) or use
`--trust calibrated` to filter the search to runtime-profile-anchored
configs only.

---

## 14. Quick-reference command summary

```bash
# Full pipeline on a fresh 4×A100 instance:

# Setup
source /workspace/Galvatron/setup-env.sh
cd /workspace/Galvatron/galvatron/models/moe

# Profiles (steps 3-7) — edit MODEL_ARGS in each script first
docker exec hetu bash -lc "cd /root/Galvatron/galvatron/models/moe && bash scripts/profile_computation.sh"
docker exec hetu bash -lc "cd /root/Galvatron/galvatron/models/moe && NUM_GPUS_PER_NODE=2 bash scripts/profile_memory.sh"
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/profile_embedding_lmhead.py
# Step 6 (profile_computation_frozen.sh) moved to _legacy/ — see Section 6.
# Step 7 (profile_memory_frozen.sh)      moved to _legacy/ — see Section 7.

# Calibration sweeps (step 8) — main matrix covers gbsz ∈ {4, 2, 1};
# the chunks=2 sister runs separately for per-microbatch slopes.
docker exec hetu bash -lc "cd /root/Galvatron && bash galvatron/models/moe/scripts/cost_model_real_test.sh         > /tmp/step8.log  2>&1"
docker exec hetu bash -lc "cd /root/Galvatron && bash galvatron/models/moe/scripts/cost_model_real_test_chunks2.sh > /tmp/step8b.log 2>&1"

# Aggregate (step 9)
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/profile_cost_model_terms.py

# Drift verify (step 10)
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_split_regression.py
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_drift.py

# Search (step 11)
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
  --model qwen-30b-a3b-e128k8 --num-gpus 4 --num-layers 48 --global-bsz 4 --top-k 12

# Validate (step 11.3)
docker exec hetu bash -lc "cd /root/Galvatron && bash galvatron/models/moe/scripts/validate_top_config.sh"
```

---

## 15. Next steps

The asymmetric search Phase 1/2 work is unblocked by the now-validated
PP-aware time and memory predictions. See
`doc/asymmetric_search_plan.md` for the design of per-component
activation memory (Phase 1) and the asymmetric layer-split search
algorithm (Phase 2).
