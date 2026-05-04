# Workflow: Cost-model calibration + search for Qwen3-30B-A3B on 8×A100 80GB

Practical playbook for a fresh 8×A100 80GB cloud instance: build the
package, profile the model, calibrate the cost model, verify drift,
run the search. Target model is **Qwen3-30B-A3B** registered as
`qwen-30b-a3b-e128k8` in `galvatron/models/moe/meta_configs/`.

**Wall-clock estimate**: ~5 hours from blank instance to drift-checked
search output. Most of that is the calibration sweep (~75 min) and the
FSEP-on profiles (~2 h). Setup + drift checks are minutes.

**Skim time**: read top-to-bottom in 15 min before running anything;
each step has explicit expected outputs you should verify.

---

## 0. Pre-flight

### 0.1 Hardware

- 8× NVIDIA A100 80GB SXM4 (NVLink-connected, single node)
- ≥ 256 GB host RAM
- ≥ 500 GB local disk (calibration logs + profile JSONs accumulate)
- Verify cleanly: every GPU should show < 5 % memory used and 0 %
  utilization before starting.

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader
# Expected: 8 lines, each ~0 MB / 81920 MiB, 0 % util
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

### 0.3 Memory budget for Qwen3-30B-A3B on 8×80 GB

Under FSDP + ZeRO-3 + bf16 + Adam:

| Term | Total | Per-rank (÷8) |
| --- | ---: | ---: |
| bf16 params (30 B × 2) | 60 GB | 7.5 GB |
| bf16 grads | 60 GB | 7.5 GB |
| Adam fp32 m + v + master (30 B × 12) | 360 GB | 45 GB |
| **Model state subtotal** | 480 GB | **60 GB** |
| Available for activations | — | ~20 GB |

Tight but feasible. **Activation recomputation is mandatory** under any
realistic batch size; training requires `--global_checkpoint 1`.
Profiling at small `num_layers` (1–4) keeps headroom for the calibration
sweep itself.

### 0.4 Why this much profiling?

The cost model needs three layers of inputs to be accurate:

1. **Per-component computation slopes** (from the three-pass
   `profile_computation.sh`) — feed the asymmetric-attention/expert
   split.
2. **Per-block calibration anchors** (from the calibration sweep
   `cost_model_real_test.sh`) — pin the absolute time/memory at each
   `(tp, ep, micro_bsz, seq, fsep, pp)` shape; α + β × N fit when ≥ 2
   `num_layers` points exist.
3. **FSEP overhead profile** (derived from on/off pairs in the same
   sweep) — captures the FSEP-on dispatch cost per expert layer; used
   by the analytical fall-back when no FSEP-on calibration exists at a
   queried shape.

Skip any of these and the cost model degrades to fully analytical
predictions (typically ±20 % drift instead of the ±2–5 % we see at
calibrated rows).

### 0.5 Files you'll edit (reference)

| File | What you'll change |
| --- | --- |
| `galvatron/models/moe/scripts/profile_computation.sh` | model dims for Qwen3 |
| `galvatron/models/moe/scripts/profile_memory.sh` | model dims for Qwen3 |
| `galvatron/models/moe/scripts/profile_computation_frozen.sh` | model dims, EP/cap tuples for 8 GPUs × 128 experts |
| `galvatron/models/moe/scripts/profile_memory_frozen.sh` | same |
| `galvatron/models/moe/scripts/cost_model_real_test.sh` | CAP formula, model dims, DEFAULT_CONFIGS for 8 GPUs |
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

Failure means the runtime training won't work — but the **cost model
will**, because `galvatron/models/moe/__init__.py` import-guards the
runtime stack. You can do all the cost-model work on a CPU-only
machine if needed; only the calibration sweep + per-component profiles
require GPUs.

### 1.1 Set environment

```bash
cat > /workspace/Galvatron/setup-env.sh <<'EOF'
export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NODE_RANK=0
export OMP_NUM_THREADS=8

# NCCL — A100 NVLink is fine, leave P2P enabled
export NCCL_DEBUG=WARN
export TORCH_NCCL_AVOID_RECORD_STREAMS=1                 # required for FSEP correctness
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MPS bypass (harmless if no MPS server runs)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-such-mps

# Inductor warm-pool guard — see profile_computation_frozen_fixes.md
export TORCHINDUCTOR_COMPILE_THREADS=1

# LAER solver: off by default; enabled per-script when FSEP needs it
export ENABLE_SOLVER=0
EOF

source /workspace/Galvatron/setup-env.sh
```

Persist across shells if you SSH back in:
```bash
echo "source /workspace/Galvatron/setup-env.sh" >> ~/.bashrc
```

### 1.2 Verify the meta-config exists

```bash
ls /workspace/Galvatron/galvatron/models/moe/meta_configs/qwen-30b-a3b-e128k8.json
```

If missing, create it (this should already exist from the
qwen-30b-a3b-e128k8 work — see `meta_configs/config_utils.py:20` for
the path-dict registration). The contents:

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

**Output**: `hardware_configs/{allreduce_bandwidth,p2p_bandwidth,overlap_coefficient,sp_time}_1nodes_8gpus_per_node.json`

Then propagate the bandwidth values into the cost-model's
`network_config.json`:

```bash
cat > /workspace/Galvatron/galvatron/models/moe/configs/network_config.json <<'EOF'
{
    "intra_node": 600.0,
    "inter_node": 25.0
}
EOF
```

For 8×A100 80GB SXM4 with NVLink: ~600 GB/s intra-node is the right
ballpark. `inter_node` is unused on a single node but the field must
exist.

**Verify** by printing what the cost model picks up:

```bash
python3 -c "
import sys; sys.path.insert(0, '/workspace/Galvatron')
from galvatron.models.moe.cost_model import PPCostModel
cm = PPCostModel('qwen-30b-a3b-e128k8')
print(cm.intra.network)
"
```

Should print the dict you just wrote.

---

## 3. Per-component computation profile, FSEP-off (~30 min, 1 GPU)

Three-pass loop over `--profile_unit ∈ {all, attention, mlp}`. Produces
per-layer fwd-only time slopes the cost model uses to split per-block
time into attention vs expert components.

### 3.1 Edit `profile_computation.sh`

Open `galvatron/models/moe/scripts/profile_computation.sh`. The default
`MODEL_ARGS` block is for mixtral-8x7b. Replace with:

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

**Why pass dims explicitly even with `set_model_config_manually=0`?**
The profiler's argparser requires some of them for input-tensor shape
construction; the meta-config only kicks in deeper. Passing them
matches what `cost_model_real_test.sh` does at calibration time and
eliminates a class of subtle mismatches.

Leave everything else (BSZ_MIN/MAX, layernum_min/max, ENABLE_SOLVER=0,
the FSEP-on guard, the three-pass loop) alone.

### 3.2 Run

```bash
cd /workspace/Galvatron/galvatron/models/moe
bash scripts/profile_computation.sh 2>&1 | tee /tmp/compute_profile.log
```

**Expected runtime**: ~30 min (3 passes × ~10 min, single GPU).

### 3.3 Verify outputs

```bash
ls configs/computation_profiling_bf16_qwen-30b-a3b-e128k8*.json
```

Should show 1 file (`_seqlen4096.json`). At `NUM_GPUS_PER_NODE=1` the
profiler doesn't sweep tp/ep, so no per-(tp,ep) variants emerge.

```bash
python3 -c "
import json
with open('configs/computation_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096.json') as f:
    p = json.load(f)
keys = sorted(p)
print('all keys:', [k for k in keys if 'attention' not in k and 'mlp' not in k][:5], '...')
print('attn keys:', [k for k in keys if 'attention' in k][:3], '...')
print('mlp keys:', [k for k in keys if '_mlp' in k][:3], '...')
"
```

You should see three families of keys (`*_bsz4_seq4096`,
`*_bsz4_seq4096_attention`, `*_bsz4_seq4096_mlp`) — these are the
per-component slopes the cost model consumes.

---

## 4. Per-component memory profile, FSEP-off (~60 min, 2 GPUs)

Same three-pass concept, but for activation memory. Sweeps over
PROFILE_MODE (currently just `static`) × UNIT.

### 4.1 Edit `profile_memory.sh`

Same `MODEL_ARGS` substitution as above. Plus:

```bash
export NUM_GPUS_PER_NODE=2   # memory profile uses 2 GPUs to sweep tp ∈ {1, 2}
```

### 4.2 Run

```bash
bash scripts/profile_memory.sh 2>&1 | tee /tmp/memory_profile.log
```

**Expected runtime**: ~60 min (the memory profile is heavier — full
fwd+bwd+optimizer iters under zero3 at multiple `(tp, layernum, bsz)`
points).

### 4.3 Verify outputs

```bash
ls configs/memory_profiling_bf16_qwen-30b-a3b-e128k8*.json
ls configs/non-solver/memory_profiling_bf16_qwen-30b-a3b-e128k8.json
```

The cost model reads from the `non-solver/` processed file. If only the
raw per-(tp, ep) files exist and `non-solver/` is empty, the
processing step in `_process_memory_data` didn't fire — check the log
for "Already written processed memory" lines.

---

## 5. Embedding + LM-head standalone profile (~5 min, 1 GPU)

Times `nn.Embedding` and `nn.Linear(hidden, vocab)` in isolation.
Required because the calibration sweep measures them as part of the
full block, and the cost model needs to peel them off when scoring
configs that place embed/lm-head on specific PP stages.

### 5.1 Edit `profile_embedding_lmhead.py`

Top of `galvatron/models/moe/scripts/profile_embedding_lmhead.py`:

```python
MODEL = "qwen-30b-a3b-e128k8"   # was: mixtral-8x7b-e8k2
PRECISION = "bf16"
DTYPE = torch.bfloat16
```

The script reads the meta-config to get `vocab_size` (151936) and
`hidden_size` (2048), so no other edits needed.

### 5.2 Run

```bash
docker exec hetu python3 \
    /workspace/Galvatron/galvatron/models/moe/scripts/profile_embedding_lmhead.py
```

(Drop the `docker exec hetu` if you're not in a hetu-style container.)

**Output**:
`configs/embedding_lmhead_profiling_bf16_qwen-30b-a3b-e128k8.json`

The vocab is ~5× larger than mixtral's (151936 vs 32000), so embed +
lm-head time will be proportionally higher. Sanity check:

```bash
python3 -c "
import json
with open('configs/embedding_lmhead_profiling_bf16_qwen-30b-a3b-e128k8.json') as f:
    p = json.load(f)
for s in p['samples']:
    print(f\"  bsz={s['bsz']}  embed={s['embedding']['fwd_bwd_ms']:.2f} ms  lmhead={s['lmhead']['fwd_bwd_ms']:.2f} ms\")
"
```

LM-head fwd_bwd should dominate (it's a hidden→vocab GEMM of ~310 M
params: 2048 × 151936). Embed is just an indexed gather → backward
sparse-scatter; much smaller.

---

## 6. FSEP-on per-block computation profile (~2 hours, 8 GPUs)

This is the heavy one. Profile FSEP-on per-block time across the EP /
capacity space — the cost model's FSEP-on path uses these to anchor
the FSEP overhead.

The companion `profile_computation_frozen.sh` runs only the `all` pass
under FSEP-on (not the three-pass loop — see CLAUDE.md and
`doc/cost_model_guide.md` §1a for why: layer-stripping biases routing
distribution under FSEP).

### 6.1 Edit `profile_computation_frozen.sh`

Two blocks:

(a) `MODEL_ARGS` — same Qwen3 substitution as Step 3.1.

(b) `EP_CAP_TUPLES_DEFAULT` — currently `(1 8, 2 4, 4 2)` for 8-expert
mixtral on 4 GPUs. For 128-expert Qwen3 on 8 GPUs:

```bash
NUM_GPUS_PER_NODE=8

# CAP = num_experts_per_device. Under FSEP, total experts = num_experts
# = 128, sharded across EP. So per-rank expert count = 128 / EP.
# Tuples are (EP, CAP) where CAP = 128 / EP.
EP_CAP_TUPLES_DEFAULT=(
    "1 128"
    "2 64"
    "4 32"
    "8 16"
)
```

Plus update the FSEP feasibility check inside the inner loop. The
script's outer feasibility is `tp × ep == per_stage_world` — this is
checked by the profiler itself; the outer loop just enumerates EP.
The TP sweep happens internally via `--max_tp_deg`.

Set `--max_tp_deg ${NUM_GPUS_PER_NODE}` in `COMMON_PROFILE_ARGS` (it
already is).

### 6.2 Run

```bash
bash scripts/profile_computation_frozen.sh 2>&1 | tee /tmp/compute_frozen.log
```

**Expected runtime**: ~2 hours.
- 4 EP/cap tuples
- TP sweep within each: 1, 2, 4, 8 (~4 inner points each)
- BSZ sweep: 1–4 (4 inner points)
- Each inner point ~1–2 min

If a tuple times out (default `OUTER_TIMEOUT=5400` = 90 min per outer
config), the script aborts and asks you to restart and retry — see
"Troubleshooting" §11 for the cascade-prevention rationale.

### 6.3 Verify outputs

```bash
ls configs/computation_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096_tp*_ep*.json
```

Should show 8–10 files, one per (tp, ep) combo.

---

## 7. FSEP-on memory profile (~30 min, 8 GPUs)

```bash
# Edit profile_memory_frozen.sh:
#   - MODEL_ARGS: Qwen3 dims (Step 3.1)
#   - EP_CAP_TUPLES: same as Step 6.1
bash scripts/profile_memory_frozen.sh 2>&1 | tee /tmp/memory_frozen.log
```

**Output**: `configs/memory_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096_tp*_ep*.json`

---

## 8. Calibration sweep — runtime profile + FSEP overhead + optimizer step (~75 min, 8 GPUs)

This is what actually anchors the cost model's predictions on real
measurements. Each per-config run produces one log file in `logs/`
with the `[real_measure]`, `[stage_time]`, and `Average iteration time
is:` lines that `profile_cost_model_terms.py` aggregates.

### 8.1 Customize `cost_model_real_test.sh`

The default script has a 4-GPU 8-expert (mixtral) config matrix and
hardcoded model dims. For 8-GPU 128-expert Qwen3, edit:

(a) **Header env**:
```bash
export NUM_GPUS_PER_NODE=8
```

(b) **CAP formula** (currently hardcoded to `8 / EP`):
```bash
NUM_GLOBAL_EXPERTS=${NUM_GLOBAL_EXPERTS:-128}
CAP=$((NUM_GLOBAL_EXPERTS / EP))
```

(c) **Model launch args**: replace the mixtral block. Find the
`${LAUNCHER} train_dist_random.py` line and update the block of
`--model_size`, `--hidden_size`, etc. flags:

```bash
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
    --num_key_value_heads 4 --num_local_experts 128 \
    --vocab_size 151936 --rms_norm_eps 1e-06 --rope_theta 10000000.0 \
    --router_aux_loss_coef 0.001 --is_moe_model \
    --set_model_config_manually 0 --set_layernum_manually 1 --set_seqlen_manually 1 \
    ...rest unchanged...
```

(Note `train_dist_frozen.py` instead of `train_dist_random.py` —
provides static input + LAER freeze + the `[real_measure]` /
`[stage_time]` instrumentation.)

(d) **Config matrix** — the existing 4-GPU configs don't apply.
Replace `DEFAULT_CONFIGS` with the 8-GPU set:

```bash
DEFAULT_CONFIGS=(
    # FSEP-off track: standard MoE.
    # pp=1 layouts, dp×tp×ep = 8:
    "1 8 zero2sdp 8 off"   "1 8 zero3 8 off"     # tp=1 ep=8 dp=1
    "2 4 zero2sdp 8 off"   "2 4 zero3 8 off"     # tp=2 ep=4 dp=1
    "4 2 zero2sdp 8 off"   "4 2 zero3 8 off"     # tp=4 ep=2 dp=1
    "8 1 zero2sdp 8 off"   "8 1 zero3 8 off"     # tp=8 ep=1 dp=1

    # FSEP-on track: tp × ep == per-stage world; ep | num_experts.
    "1 8 zero2sdp 8 on"    "1 8 zero3 8 on"
    "2 4 zero2sdp 8 on"    "2 4 zero3 8 on"
    "4 2 zero2sdp 8 on"    "4 2 zero3 8 on"
    "8 1 zero2sdp 8 on"    "8 1 zero3 8 on"
)
```

This gives 16 configs at pp=1. Add pp=2 / pp=4 / pp=8 by re-invoking
the script with `PP=2`, `PP=4`, `PP=8` env (the script already supports
this).

(e) **Optional**: bump `EPOCHS=20` if you want more profiler-window
samples. The script already uses iters [10, 20) for the time average.

### 8.2 Run the sweep

```bash
# Default pp=1:
bash scripts/cost_model_real_test.sh 2>&1 | tee /tmp/calib_pp1.log

# pp=2 sweep (subset of layouts that fit on 4-GPU per-stage):
PP=2 bash scripts/cost_model_real_test.sh \
    1 4 zero2sdp 8 off    2 2 zero2sdp 8 off    4 1 zero2sdp 8 off \
    1 4 zero2sdp 8 on     2 2 zero2sdp 8 on     4 1 zero2sdp 8 on \
    2>&1 | tee /tmp/calib_pp2.log

# pp=4 (per-stage 2 GPUs):
PP=4 bash scripts/cost_model_real_test.sh \
    1 2 zero2sdp 8 off    2 1 zero2sdp 8 off \
    1 2 zero2sdp 8 on     2 1 zero2sdp 8 on \
    2>&1 | tee /tmp/calib_pp4.log
```

**Wall-clock estimate**: ~3 min per config × ~30 configs = ~75 min total.

### 8.3 Multi-N runs for the α/β fit

For one selected shape (typical: `tp=1, ep=8, zero2sdp, bsz=8, fsep=off`),
run with multiple `num_layers` to populate the α + β × N fit:

```bash
for NL in 2 4 6 8; do
    NUM_LAYERS=$NL bash scripts/cost_model_real_test.sh \
        1 8 zero2sdp 8 off    2>&1 | tee -a /tmp/calib_multi_n.log
done
```

The aggregator picks these up automatically: when ≥ 2 N values exist
for a shape, it fits α + β × N for every memory/time component and
exposes them via `runtime_profiling_*.json`.

### 8.4 Verify the calibration logs

```bash
ls galvatron/models/moe/logs/cost_model_real_*.log | wc -l
# Expected: ≥ 30 (16 pp=1 + ~6 pp=2 + ~4 pp=4 + multi-N)

# Spot-check one log:
grep -E "real_measure|stage_time|Average iteration" \
    galvatron/models/moe/logs/cost_model_real_tp1_ep8_zero2sdp_bsz8_fsepoff.log \
    | head
```

Should show:
```
[real_measure] params_mb=...
[real_measure] optimizer_mb=... activation_peak_mb=... cuda_peak_mb=...
[stage_time] fwd_bwd_ms=... opt_ms=... window=[10,20)
Average iteration time is: ... s
...
```

If any log is missing these — the run failed mid-way. Re-run the
specific config in isolation; cascade-prevention is on by default.

---

## 9. Aggregate calibration logs into JSON profiles (~10 sec, CPU)

### 9.1 Edit `profile_cost_model_terms.py`

Top of file:
```python
MODEL = "qwen-30b-a3b-e128k8"     # was: mixtral-8x7b-e8k2
PRECISION = "bf16"
SEQ_LEN = 4096
NUM_MOE_LAYERS = 4                # = the num_hidden_layers used by the calibration sweep
```

`NUM_MOE_LAYERS` matters because it gets recorded as
`num_layers_in_calibration` and is the divisor for the FSEP overhead
"per expert layer" math.

### 9.2 Run

```bash
docker exec hetu python3 scripts/profile_cost_model_terms.py
```

**Outputs** (in `galvatron/models/moe/configs/`):
- `runtime_profiling_bf16_qwen-30b-a3b-e128k8.json`
- `fsep_overhead_profiling_bf16_qwen-30b-a3b-e128k8.json`
- `optimizer_step_profiling_bf16_qwen-30b-a3b-e128k8.json`

The script prints a per-shape summary at the end. Sanity-check:

- Adam throughput: should be ~50 MB/ms (HBM-bandwidth bound; ≈ same on
  any A100 regardless of cluster).
- `optimizer_to_params_ratio_median`: should be ≈ 2.0 (bf16 + Adam
  fp32 m, fp32 v; no fp32 master).
- FSEP overhead: per-MoE-layer time delta should be in the hundreds of
  ms (FSEP-on adds the all-to-all dispatch path).

If any of these are wildly off, something is wrong with the calibration
logs — inspect the raw logs before proceeding.

---

## 10. Drift verification (no GPU, ~30 sec)

Four scripts. Run all four; failure of any one means the calibration
isn't sound.

### 10.1 Regression suite (24 invariants)

```bash
docker exec hetu python3 scripts/cost_model_split_regression.py
```

Expected: `# total: 24 checks, 0 failures`. If any check fails:

- **Symmetric identity** (1) failed → asymmetric kwargs path is broken;
  shouldn't happen unless cost_model code changed.
- **PP critical path** (2) failed → `iter_ms ≠ (n_micro + pp − 1) ×
  max_stage_ms + max_post_bwd_ms`; the breakdown's `stage_bottleneck_ms`
  is mis-populated.
- **FSEP attention invariance** (3) failed → the per-component
  computation profile produced an inconsistent attention slope vs.
  the FSEP-off runtime entry. Likely cause: sweep didn't run with
  matching shapes.
- **FSEP overhead reconciliation** (4) failed → the FSEP profile's
  `time_overhead_per_expert_layer_ms` doesn't match the runtime
  profile's on/off delta. Indicates aggregation bug (re-run Step 9).
- **Input validation** (5) failed → cost-model code regression;
  shouldn't happen.

### 10.2 Per-shape drift table

```bash
docker exec hetu python3 scripts/cost_model_drift.py
```

Compares the cost model's predictions to the recorded measurements at
every calibrated `(tp, ep, dp_mode, bsz, fsep)` shape. Healthy:

- iter-time drift: median signed near 0, mean abs ≤ 5 %, max abs ≤ 10 %
- cuda-peak drift: 0 % at every row (the runtime-profile shortcut
  returns the calibrated value directly)

### 10.3 PP critical-path drift

```bash
docker exec hetu python3 scripts/cost_model_pp_drift.py
```

Cross-checks pp ∈ {1, 2, 4, 8} predictions against measurements. If
the multi-PP sweep wasn't run, this prints "no real runs found" and
exits — populate by re-running step 8.2 with `PP=...` overrides.

### 10.4 α + β × N extrapolation drift

```bash
docker exec hetu python3 scripts/cost_model_alpha_beta.py
```

Validates the `peak_memory_mb(N) = α + β × N` linear fit. Needs the
multi-N runs from step 8.3. If the fit drifts > 5 % on the
empirical α / β, run more N values (e.g., add N=16 if the highest
profiled N is 8).

---

## 11. Search (~1 sec, CPU)

Now the cost model is calibrated. Search for the optimal
parallelization config under the 8×80 GB budget:

```bash
docker exec hetu python3 scripts/cost_model_search.py \
    --model qwen-30b-a3b-e128k8 \
    --num-gpus 8 \
    --num-layers 48 \
    --global-bsz 8 \
    --num-experts 128 \
    --gpu-memory-mb 80000 \
    --trust-source calibrated \
    --top-k 10
```

Flags:

- `--num-gpus 8`, `--num-experts 128` — match the cluster
- `--num-layers 48` — full Qwen3-30B-A3B
- `--gpu-memory-mb 80000` — 80 GB minus a safety margin (the cost model
  prediction has up to ±5 % drift; leaving 5 % headroom keeps you out
  of OOM territory)
- `--trust-source calibrated` — drops fully-analytical configs;
  ranks only configs with runtime-profile-anchored time + memory.
  Safe default for picking a config to actually run.

The output prints the top 10 configs with iter_ms, max_stage_ms,
peak_memory_mb, and provenance.

### 11.1 Probe the OOM margin with `num_stages_behind`

The calibrated peak_memory_mb doesn't include framework buffer
overhead (pipeline send/recv keep-alive buffers, FSDP per-stage
scratch, etc.). To probe: re-run the search with extra reserve and
see which optima drop out:

```bash
for NSB in 0 1 2; do
    echo "=== num_stages_behind=$NSB ==="
    docker exec hetu python3 scripts/cost_model_search.py \
        --model qwen-30b-a3b-e128k8 \
        --num-gpus 8 --num-layers 48 --global-bsz 8 \
        --num-experts 128 --gpu-memory-mb 80000 \
        --trust-source calibrated --top-k 5 \
        --num-stages-behind $NSB
done
```

Configs that survive `num_stages_behind=2` are robust to a 2-extra-
microbatch reserve. Configs that drop out at `num_stages_behind=1` are
on the OOM edge.

### 11.2 Asymmetric layer-count probe (optional)

Design-space probe: "what if we had one extra MoE layer?":

```bash
docker exec hetu python3 scripts/cost_model_search.py \
    --model qwen-30b-a3b-e128k8 \
    --num-gpus 8 --num-layers 48 --global-bsz 8 \
    --num-experts 128 --gpu-memory-mb 80000 \
    --trust-source calibrated \
    --asymmetry-range -2 2
```

Sweeps `num_expert_layers ∈ [46 .. 50]` at fixed `num_attention_layers
= 48`, picking the best config at each point. Useful for evaluating
whether layer count rebalancing is worth a re-pretraining run.

---

## 12. Artifact bundle for handoff (Facade integration)

After everything passes, the bundle to ship to the Facade-served
profile store:

```
galvatron/models/moe/configs/
├── network_config.json
├── computation_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096.json
├── computation_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096_tp*_ep*.json
├── memory_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096.json
├── memory_profiling_bf16_qwen-30b-a3b-e128k8_seqlen4096_tp*_ep*.json
├── non-solver/computation_profiling_*.json
├── non-solver/memory_profiling_*.json
├── runtime_profiling_bf16_qwen-30b-a3b-e128k8.json
├── fsep_overhead_profiling_bf16_qwen-30b-a3b-e128k8.json
├── optimizer_step_profiling_bf16_qwen-30b-a3b-e128k8.json
└── embedding_lmhead_profiling_bf16_qwen-30b-a3b-e128k8.json

galvatron/models/moe/meta_configs/qwen-30b-a3b-e128k8.json
```

Plus `meta_configs/config_utils.py:9-23` (the `path_dict`) — confirms
the `qwen-30b-a3b-e128k8` registration.

Bundle versioning: tag the directory with sweep-date and hardware-class
(`8xA100-NVLink-bf16/qwen-30b-a3b-e128k8/2026-MM-DD/`). See
`doc/cost_model_facade_integration.md` §3.5 for the recommended
layout.

**Do not** ship the calibration logs (`logs/cost_model_real_*.log`) —
they're large and only needed at re-aggregation time. Keep them on the
calibration host (or in a separate raw-data archive).

---

## 13. Troubleshooting

### NCCL hang during calibration

Symptom: a config in the sweep wedges; profile run never returns.

Likely cause: NCCL P2P contention or a competing process holding GPUs.

1. `pkill -KILL -f train_dist_frozen.py; pkill -KILL -f torchrun`
2. Wait 10 minutes (host driver state needs to quiesce — see
   `doc/profile_computation_frozen_fixes.md`)
3. Verify GPUs are clean: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
4. Re-run the failing config in isolation:
   ```bash
   bash scripts/cost_model_real_test.sh <tp> <ep> <dp_mode> <bsz> <fsep>
   ```

### OOM mid-iteration

Symptom: log contains `CUDA out of memory` on the calibration GPU.

For Qwen3-30B-A3B, the largest configs (pp=1 + ep=1: every rank holds
all 128 experts) approach 70+ GB/rank. Drop those:

- Skip `1 1 zero2sdp 8 off` and similar ep=1 layouts at pp=1.
- Use larger `pp` (= smaller per-stage param footprint) for ep=1.
- Increase EP (= shard experts further) for pp=1.

The cost model will still pick a feasible config from what survived;
the search's `--gpu-memory-mb 80000` budget filter handles the rest.

### "No computation profile found for tp=X, ep=Y" in search

Symptom: `cost_model_search.py` errors out on certain configs.

Cause: per-(tp, ep) computation profile missing at that shape. Either
the calibration sweep didn't include that point, or the FSEP-on
(`profile_computation_frozen.sh`) sweep didn't either.

Fix: re-run step 6 with the missing (EP, cap) tuple, OR drop those
configs from the search (`--trust-source calibrated` filters them out
automatically — they go to the infeasible bucket with
`error="filtered (trust=...)"`).

### Symmetric identity check fails after re-aggregation

Symptom: regression suite invariant 1 fails after re-running step 9.

Cause: the runtime profile's `num_layers_profiled` changed, but the
α / β fits weren't re-keyed. Re-aggregating from the same logs should
be deterministic — if it isn't, check that `NUM_MOE_LAYERS` in
`profile_cost_model_terms.py` matches the actual N used in the
calibration sweep (step 8). They must be consistent.

### Cost-model drift > 10 % at calibrated rows

Cause: the calibration sweep's measurements are noisy or your hardware
was contended during the sweep.

Diagnosis:

1. Inspect a high-drift row. Is the `n_dp_samples` for that shape > 1?
   If so, multiple dp_mode runs averaged together — check whether
   they agree (look at the raw logs).
2. Is `bwd_mult ≈ 2.0` a good fit for A100? Mixtral on A6000 saw 2.0;
   Qwen3 on A100 might see 2.1 due to different fwd/bwd kernel mix.
   Override per query: `--bwd-mult 2.1`.
3. Is the 80 GB of HBM bandwidth-bound? At very high `tp` the all-reduce
   becomes the bottleneck and the analytical formula under-counts.
   Verify by checking `ar_volume_ms` in the search breakdown.

### Sanity benchmark

After the full sweep, the cost model's prediction for the search's
chosen optimum should be within ±5 % of a held-out real measurement.
To verify:

1. Pick the search's optimal config.
2. Run it once via `cost_model_real_test.sh` with the same
   parallelization layout.
3. Compare predicted vs measured `iter_ms` and `cuda_peak_mb`.

Within 5 % → ship the bundle. Outside 5 % → investigate which step had
noisy data and re-run.

---

## 14. Recap timeline

| Step | Wall time | GPUs | Output |
| ---: | ---: | ---: | --- |
| 1 — Install + build | 30 min | 0 | extensions compiled |
| 2 — Hardware profile | 5 min | 8 | `network_config.json` |
| 3 — Compute profile (3-pass) | 30 min | 1 | `computation_profiling_*.json` |
| 4 — Memory profile (3-pass) | 60 min | 2 | `memory_profiling_*.json`, `non-solver/*` |
| 5 — Embedding/LM-head | 5 min | 1 | `embedding_lmhead_profiling_*.json` |
| 6 — FSEP-on compute | 120 min | 8 | per-(tp,ep) `computation_profiling_*.json` |
| 7 — FSEP-on memory | 30 min | 8 | per-(tp,ep) `memory_profiling_*.json` |
| 8 — Calibration sweep | 75 min | 8 | 30+ logs in `logs/` |
| 8.3 — Multi-N | 30 min | 8 | additional logs (one shape × 4 N values) |
| 9 — Aggregate | 10 sec | 0 | runtime / FSEP / optimizer JSONs |
| 10 — Drift checks | 30 sec | 0 | invariants + drift table |
| 11 — Search | 1 sec | 0 | top-K configs |

**Total**: ~5 hours from cold instance.

---

## 15. Quick-reference command summary

```bash
# Setup
cd /workspace/Galvatron && pip install -e . --no-build-isolation
source setup-env.sh

# Profiles (in order)
cd galvatron/profile_hardware && bash scripts/profile_hardware.sh
cd ../models/moe
bash scripts/profile_computation.sh
bash scripts/profile_memory.sh
python3 scripts/profile_embedding_lmhead.py
bash scripts/profile_computation_frozen.sh
bash scripts/profile_memory_frozen.sh

# Calibration sweep (~75 min)
bash scripts/cost_model_real_test.sh                  # pp=1
PP=2 bash scripts/cost_model_real_test.sh ...         # pp=2 subset
PP=4 bash scripts/cost_model_real_test.sh ...         # pp=4 subset
for NL in 2 4 6 8; do                                 # multi-N for α/β fit
    NUM_LAYERS=$NL bash scripts/cost_model_real_test.sh 1 8 zero2sdp 8 off
done

# Aggregate
python3 scripts/profile_cost_model_terms.py

# Verify
python3 scripts/cost_model_split_regression.py        # 24 invariants
python3 scripts/cost_model_drift.py                   # per-shape drift
python3 scripts/cost_model_pp_drift.py                # PP critical-path
python3 scripts/cost_model_alpha_beta.py              # extrapolation

# Search
python3 scripts/cost_model_search.py \
    --model qwen-30b-a3b-e128k8 --num-gpus 8 --num-layers 48 \
    --global-bsz 8 --num-experts 128 --gpu-memory-mb 80000 \
    --trust-source calibrated --top-k 10
```

---

## 16. Multi-node deltas (when you scale beyond one node)

Everything above assumes a single-node 8×A100 setup. If you later
target multi-node — e.g., 4 nodes × 2× H100 — see the dedicated
playbook **`doc/qwen3_4x2_h100_workflow.md`** for the full multi-node
adaptation. Quick summary of what changes:

### What stays the same

- **Cost-model package** (`cost_model/`) — CPU-only, model-agnostic,
  reads from JSONs.
- **Per-component computation profile** (`profile_computation.sh`) —
  single-rank, runs on any one node.
- **Embedding/LM-head profile** — single-GPU standalone.
- **Aggregation, drift, search** — CPU-only post-processing.
- **Memory budget math** — same ÷ 8 split (8 GPUs × 80 GB regardless
  of how they're partitioned across nodes).

### What changes

| Concern | Single-node 8×A100 | Multi-node (e.g. 4×2 H100) |
| --- | --- | --- |
| `NUM_NODES` / `NUM_GPUS_PER_NODE` | `1` / `8` | `4` / `2` |
| `MASTER_ADDR` | `127.0.0.1` | rendezvous-resolvable (K8s headless service / SLURM `srun` / static IP) |
| `NODE_RANK` | `0` | per-pod ordinal (set by orchestrator) |
| `network_config.json` `inter_node` | placeholder | **measured** — drives DP/EP comm timing |
| `profile_memory.sh` for `TP > 2` | implicit (single-node sweeps all `tp` ≤ 8) | requires multi-node launch (per-stage world spans nodes) |
| FSEP-on / calibration scripts | single-node `torchrun --standalone` | every node runs the same script with `--node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR}` |
| Calibration `DEFAULT_CONFIGS` matrix | flat layout space | topology-aware: PP=4 first (one stage per node), PP=1 worst-case |
| Static input file path | container-local OK | shared FS (NFS / FSx / GCSFuse) reachable from every rank |
| Drift expectations | ≤ 5 % iter mean abs | ≤ 8 % iter mean abs (inter-node jitter) |

### Container-specific gotchas (multi-node only)

1. **NCCL must see IB/RoCE**, not the CNI overlay. Verify with a
   small `all_reduce_perf` cross-node — should hit fabric spec
   (~25 GB/s for HDR IB), not < 5 GB/s.
2. **Pod networking**: typically `hostNetwork: true` or SR-IOV / IB
   CNI plugin. K8s default CNI overlays kill IB performance.
3. **NCCL env**: set `NCCL_IB_DISABLE=0`, `NCCL_IB_HCA=<your-HCAs>`,
   `NCCL_IB_GID_INDEX=<your-index>`.
4. **Hostfile is irrelevant for torchrun**. The `hostfile` in
   `profile_hardware/` is only consumed by mpirun-based NCCL test
   launches (`--backend nccl`). Torchrun uses env-var rendezvous —
   no hostfile in the workflow.

### When to use which doc

- **Doing single-node profiling now** → stay in this doc.
- **Spinning up a multi-node calibration** → switch to
  `doc/qwen3_4x2_h100_workflow.md` for the multi-node-specific
  setup (networking smoke test, rendezvous patterns, multi-node
  calibration matrix, multi-node-specific troubleshooting).
- **Both clusters in flight** → Facade-style integration with
  per-hardware-class profile bundles (one bundle per topology). See
  `doc/cost_model_facade_integration.md` §2.3.
