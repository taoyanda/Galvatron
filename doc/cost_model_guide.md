# MoE Cost Model — User Guide

End-to-end recipe for the Galvatron-MoE cost model: how to collect the
profiles it consumes, how to invoke it programmatically, and how to use
the brute-force search driver to pick a parallelization strategy for a
given model + cluster shape.

The cost model itself lives in `galvatron/models/moe/cost_model/`:

- `base.py` — `ICostModel` abstract interface + `CostEstimate` dataclass.
- `intra.py` — `IntraCostModel` (single pipeline stage; pp == 1).
- `pp.py` — `PPCostModel` (1F1B / PipeDream-Flush over `IntraCostModel`).
- `__init__.py` — public exports + `CostModel = PPCostModel` alias.

All numerical constants come from profile JSONs; nothing is hard-coded
in source. The pipeline has three layers:

```
real measurements → profile JSONs → cost model (.estimate(...)) → search
   (Step 1)          (Step 2 outputs)     (Step 2)                (Step 3)
```

---

## Conventions used throughout

- All Python / torch / training commands run **inside the `hetu`
  container**. The host `/home/yt522/Galvatron` maps to
  `/root/Galvatron`, so a host-side script at
  `galvatron/models/moe/scripts/foo.py` is invoked as
  `docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/foo.py`.
- Per the project memory, MPS bypass is mandatory: every launcher
  exports `CUDA_MPS_PIPE_DIRECTORY=/tmp/no-such-mps`.
- Single-letter abbreviations used here: **PP** = pipeline parallel
  degree; **DP** = data parallel; **TP** = tensor parallel; **EP** =
  expert parallel; **FSEP** = fast static expert parallelism (LAER's
  smart-routing kernel); **N** = `num_hidden_layers`; **micro_bsz** =
  per-DP-rank batch size = `global_bsz / dp / chunks`.

---

## 1. Collecting profiling results

The cost model consumes **six** profile artifacts. Three are pre-existing
Galvatron profiles you already produced before training. Three are new
artifacts derived from a small calibration sweep that we need to run on
the target cluster.

### 1a. Pre-existing Galvatron profiles (reused as-is)

| File (under `galvatron/models/moe/configs/`) | Produced by |
| --- | --- |
| `computation_profiling_<prec>_<model>_seqlen<S>[_tp<TP>_ep<EP>].json` | `scripts/profile_computation.sh` (or `_full`/`_frozen`) |
| `non-solver/memory_profiling_<prec>_<model>.json` | `scripts/profile_memory.sh` (or `_frozen`) |
| `network_config.json` (committed; per-cluster) | hand-tuned from `galvatron/profile_hardware/scripts/profile_hardware.sh` |

Run order on a fresh cluster:

```bash
# Inside the hetu container or on host with the correct env.
cd /root/Galvatron/galvatron/profile_hardware
bash scripts/profile_hardware.sh         # outputs hardware_configs/*

cd /root/Galvatron/galvatron/models/moe
bash scripts/profile_computation_full.sh # writes computation_profiling_*.json
bash scripts/profile_memory.sh           # writes memory_profiling_*.json
```

`profile_computation_full.sh` is the variant we ship — it runs three
passes (`all`, `attention`, `mlp`) into the same per-shape file so the
LAER solver and the cost model can both consume it. `profile_*_frozen.sh`
variants pin LAER's expert layout via `--laer_freeze_after_iter 5` and
are used when the live solver would perturb the per-iter timings.

### 1b. New cost-model-specific profiles (derived from a calibration sweep)

| File | Built by | Contents |
| --- | --- | --- |
| `embedding_lmhead_profiling_<prec>_<model>.json` | `scripts/profile_embedding_lmhead.py` | embedding & LM-head fwd/bwd ms (split, so PP can place each on the boundary stage) |
| `optimizer_step_profiling_<prec>_<model>.json` | `scripts/profile_cost_model_terms.py` | Adam step throughput (MB/ms) + empirical `optimizer_to_params_ratio` |
| `runtime_profiling_<prec>_<model>.json` | `scripts/profile_cost_model_terms.py` | full-iter `fwd_bwd_ms` / `opt_ms` / `cuda_peak_mb` per `(tp, ep, micro_bsz, seq, fsep[, pp])` shape, with α/β fits across `num_layers` |
| `fsep_overhead_profiling_<prec>_<model>.json` | `scripts/profile_cost_model_terms.py` | per-MoE-layer FSEP time + memory overhead, derived from matched `fsep=on/off` runs |

Three steps:

#### 1b-i. Profile the embedding + LM head

Standalone, no distributed launch. Reads `meta_configs/<model>.json`
(vocab size, hidden) and times `nn.Embedding` and `nn.Linear(hidden, vocab)`
in isolation.

**Inputs:**

| Source | What it controls |
| --- | --- |
| `meta_configs/<model>.json` | `vocab_size`, `hidden_size` |
| top of the script: `MODEL`, `PRECISION`, `BSZ_VALUES`, `SEQ_VALUES` | which shapes get profiled |

**Example command:**

```bash
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/profile_embedding_lmhead.py
```

Output: `configs/embedding_lmhead_profiling_bf16_<model>.json`.

#### 1b-ii. Run the calibration sweep (real measurements)

`scripts/cost_model_real_test.sh` launches `train_dist_random.py` for
each `(tp, ep, dp_mode, global_bsz, fsep)` configuration in its
`DEFAULT_CONFIGS` array and writes per-config logs to
`galvatron/models/moe/logs/cost_model_real_*.log`. Each run lasts 20
iterations; the profiler averages `[10, 20)` and emits the
`[real_measure]` and `[stage_time]` instrumentation lines that the
profile builder parses.

**Required env vars:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `NUM_GPUS_PER_NODE` | `4` | total GPUs available on the node |
| `MASTER_PORT` | `29500` | torchrun rendezvous |
| `CUDA_MPS_PIPE_DIRECTORY` | `/tmp/no-such-mps` | **MPS bypass — leave at the default** |
| `NCCL_P2P_DISABLE` | `1` | required for stability on this 4×A6000 host |
| `TORCH_NCCL_AVOID_RECORD_STREAMS` | `1` | required for FSEP correctness |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | matches `train.sh` |
| `ENABLE_SOLVER` | `1` | needed by the FSEP-on track; otherwise irrelevant |

**Optional sweep dimensions:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `NUM_LAYERS` | `4` | total `num_hidden_layers`. Tag this on the log filename so the runtime profile builder can fit α + β × N. |
| `PP` | `1` | pipeline parallel degree. Per-stage shape becomes `(tp×ep×dp) = NUM_GPUS_PER_NODE / PP`; caller must pick a feasible `(TP, EP, DP_MODE)`. |
| `SEQ_LEN` | `4096` | sequence length |
| `EPOCHS` | `20` | iters per config; profiler window is `[10, 20)` |

**Example commands:**

```bash
# Full default sweep (every (tp, ep, dp_mode, bsz, fsep) listed in the script):
bash /root/Galvatron/galvatron/models/moe/scripts/cost_model_real_test.sh \
  | tee /tmp/sweep.log

# Single config — positional args: tp ep dp_mode global_bsz [fsep_on_or_off]
bash /root/Galvatron/galvatron/models/moe/scripts/cost_model_real_test.sh \
    1 4 zero2sdp 4 off

# Multi-N sweep at one shape (used to fit α + β × num_layers):
for N in 2 4 6 8; do
    NUM_LAYERS=$N bash \
      /root/Galvatron/galvatron/models/moe/scripts/cost_model_real_test.sh \
      1 4 zero2sdp 4 off
done

# Multi-PP sweep at one shape:
PP=2 bash /root/Galvatron/galvatron/models/moe/scripts/cost_model_real_test.sh 1 2 zero2sdp 4 off
PP=4 bash /root/Galvatron/galvatron/models/moe/scripts/cost_model_real_test.sh 1 1 zero2sdp 4 off
```

Each invocation produces logs at:

```
galvatron/models/moe/logs/cost_model_real_tp{TP}_ep{EP}_{DP_MODE}_bsz{BSZ}_fsep{ON|OFF}[_nl{N}][_pp{P}].log
```

#### 1b-iii. Aggregate the calibration sweep into JSON profiles

Parse the logs into the three derived profiles in one go:

```bash
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/profile_cost_model_terms.py
```

This script reads every `cost_model_real_*.log` under `logs/` and emits:

- `optimizer_step_profiling_bf16_<model>.json`
- `runtime_profiling_bf16_<model>.json`
- `fsep_overhead_profiling_bf16_<model>.json`

Stdout summarises Adam throughput, optimizer-to-params ratio, FSEP
overhead per shape, and how many `num_layers` points back each shape
key in the runtime profile (more points → α + β × N fit; one point
falls back to per-layer linear extrapolation).

**Caveat (project memory):** if any sweep iteration times out or hangs,
`pkill` stragglers, `docker restart hetu`, wait ~5 min for the GPUs to
quiesce, then resume — never let a hung config feed bad data into the
profile aggregation.

### 1c. Validating the profiles end-to-end

Three diagnostic scripts cross-check the profiles against held-out real
measurements. None take arguments; all read logs in `logs/`.

```bash
# Drift table for each (shape, num_layers) row vs. real:
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_drift.py

# α + β × num_layers extrapolation check (multi-N sweep at one shape):
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_alpha_beta.py

# PP critical-path check (multi-PP sweep at one shape):
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_pp_drift.py
```

A healthy calibration produces ≤ ±5 % memory drift and ≤ ±5 %
iter-time drift on calibrated rows. See `doc/cost_model_optimizations.md`
for the optimization log that got us there.

---

## 2. Invoking the cost model

After step 1 the profile JSONs are present and the cost model can be
invoked from any Python process inside the container.

### 2a. Public API

```python
from galvatron.models.moe.cost_model import (
    PPCostModel,    # 1F1B-aware orchestrator (recommended)
    IntraCostModel, # single-stage; pp must equal 1
    CostEstimate,   # return type
    estimate_cost,  # one-shot wrapper
)
# Back-compat alias:
from galvatron.models.moe.cost_model import CostModel     # = PPCostModel
```

### 2b. Constructing the model

`PPCostModel(model_name, mixed_precision="bf16")` loads every profile
JSON it can find and falls back gracefully when one is missing
(analytical fall-back is documented in `intra.py` near `memory_source`).

```python
cm = PPCostModel("mixtral-8x7b-e8k2", mixed_precision="bf16")
```

### 2c. `estimate(...)` keyword arguments

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `num_layers` | int | yes | `num_hidden_layers` |
| `num_gpus` | int | yes | total GPU allocation |
| `dp` | int | yes | data-parallel degree |
| `pp` | int | yes | pipeline-parallel degree |
| `tp` | int | yes | tensor-parallel degree |
| `ep` | int | default `1` | expert-parallel degree |
| `micro_batch_size` | int | default `1` | per-DP-rank batch |
| `global_batch_size` | int or `None` | default `None` (= `micro_batch_size × dp`) | total batch |
| `seq_len` | int | default `4096` | sequence length |
| `gpus_per_node` | int | default `8` | used for intra/inter-node bandwidth lookup |
| `recompute` | bool | default `False` | activation recomputation |
| `zero_stage` | int | default `1` | ZeRO stage (1 / 2 / 3) |
| `sdp` | bool | default `False` | sequence-data-parallel (zero2sdp / zero3 set this true) |
| `sequence_parallel` | bool | default `True` | activation sharding over TP |
| `bwd_mult` | float | default `2.0` | bwd-vs-fwd ratio for analytical path |
| `fsep` | bool | default `False` | LAER smart routing |

Invariants: `dp × pp × tp × ep == num_gpus`; `num_layers % pp == 0`;
`global_batch_size % (dp × micro_batch_size) == 0`. Violations raise
`ValueError`.

### 2d. Return value

`CostEstimate(total_iter_ms, peak_memory_mb, breakdown)` —
`breakdown` is a free-form dict whose useful keys include
`time_source`, `memory_source`, `parameters_mb`, `optimizer_mb`,
`activations_mb`, `pipeline_iter_ms`, `stage_bottleneck_ms`,
`bottleneck_stage`, and the α/β fit coefficients used (when the
runtime profile had ≥ 2 N points for this shape).

### 2e. Example

```python
from galvatron.models.moe.cost_model import PPCostModel

cm = PPCostModel("mixtral-8x7b-e8k2")
est = cm.estimate(
    num_layers=4, num_gpus=4,
    dp=1, pp=1, tp=1, ep=4,
    micro_batch_size=4, global_batch_size=4,
    seq_len=4096, sequence_parallel=True,
    zero_stage=2, sdp=True,
    recompute=True, bwd_mult=2.0, fsep=False,
)
print(f"iter_ms={est.total_iter_ms:.0f}  peak_mb={est.peak_memory_mb:.0f}")
print(f"time_src={est.breakdown['time_source']}")
print(f"mem_src={est.breakdown['memory_source']}")
```

A pure-Python smoke test (no GPU needed):

```bash
docker exec hetu python3 -c "
from galvatron.models.moe.cost_model import PPCostModel
cm = PPCostModel('mixtral-8x7b-e8k2')
est = cm.estimate(num_layers=4, num_gpus=4, dp=1, pp=1, tp=1, ep=4,
                  micro_batch_size=4, global_batch_size=4, seq_len=4096,
                  sequence_parallel=True, zero_stage=2, sdp=True,
                  recompute=True, bwd_mult=2.0)
print(f'iter_ms={est.total_iter_ms:.0f} peak_mb={est.peak_memory_mb:.0f}')
"
```

---

## 3. Searching given a model definition

`scripts/cost_model_search.py` enumerates every legal
`(pp, dp, tp, ep, dp_mode, fsep)` tuple for the requested cluster shape,
calls `PPCostModel.estimate(...)` on each, and prints the top-K ranked
by predicted iteration time (ties broken by lower peak memory).

### 3a. Inputs / CLI flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `mixtral-8x7b-e8k2` | meta config name (must have `meta_configs/<name>.json`) |
| `--num-gpus` | `4` | total GPUs |
| `--num-layers` | `4` | `num_hidden_layers` |
| `--global-bsz` | `4` | global batch size |
| `--seq-len` | `4096` | sequence length |
| `--num-experts` | `8` | number of experts (gate for `ep` and FSEP feasibility) |
| `--gpu-memory-mb` | `45000` | OOM filter; configs whose predicted peak exceeds this are dropped. Pass `0` to disable. |
| `--top-k` | `10` | top-K to print |
| `--show-infeasible` | off | also print infeasible configs (and the reason) |
| `--trust-source` | `any` | `any` / `calibrated` / `sample` — see below |

### 3b. The `--trust-source` filter

Different configs end up with different cost-model provenance. The
trust filter lets the caller require a specific level of empirical
backing:

| Mode | Keeps configs whose estimate came from … |
| --- | --- |
| `any` | anything (analytical fall-back included) — **default** |
| `calibrated` | runtime profile lookup for both time **and** memory |
| `sample` | exact-N profile sample for both |

`calibrated` is the recommended setting when picking a config to
actually run, since it rules out the "analytical fall-back optimum"
trap (an FSEP-on config the model thinks is fast but never measured).

### 3c. Example commands

```bash
# Default search, top-10:
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py

# 8-GPU, 32-layer Mixtral-8x7B:
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
    --num-gpus 8 --num-layers 32 --global-bsz 16

# Only show configs backed by a runtime-profile measurement:
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
    --trust-source calibrated

# Disable the OOM filter and print why each rejected config was rejected:
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
    --gpu-memory-mb 0 --show-infeasible
```

### 3d. Side-by-side comparison

`scripts/cost_model_search_compare.py` runs the search four times
(baseline / option-1 trust filter / option-2 FSEP overhead profile /
both) and attaches real measurements where available. Useful for
verifying that the search picks the same optimum as ground truth.

```bash
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_search_compare.py
```

### 3e. Interpreting the output

```
rk pp dp tp ep   dp_mode fsep |   iter_ms   peak_mb |  params   optim     act |       time_src         mem_src
 1  1  1  1  4  zero2sdp  off |      1446     30357 |    6067   12134   12155 | sample(N_pts=2)    sample(N_pts=2)
```

- Configs with `time_src = sample(N_pts=...)` were anchored on a real
  measurement at the requested `num_layers` — high confidence.
- Configs with `α/β(N_pts=k)` are extrapolated linearly from the k
  profiled `num_layers` points.
- Configs with `fwd` / `anal` are pure analytical predictions; treat
  them as guidance only (or filter via `--trust-source calibrated`).

---

## Quick reference (full pipeline, fresh cluster)

```bash
# Hardware + computation + memory profiles (existing Galvatron path):
docker exec hetu bash /root/Galvatron/galvatron/profile_hardware/scripts/profile_hardware.sh
docker exec hetu bash /root/Galvatron/galvatron/models/moe/scripts/profile_computation_full.sh
docker exec hetu bash /root/Galvatron/galvatron/models/moe/scripts/profile_memory.sh

# Embedding + LM-head profile:
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/profile_embedding_lmhead.py

# Real-measurement calibration sweep (≈ 1 hour on 4× A6000):
docker exec hetu bash /root/Galvatron/galvatron/models/moe/scripts/cost_model_real_test.sh \
    | tee /tmp/sweep.log

# Aggregate calibration logs into the runtime / optimizer / FSEP overhead profiles:
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/profile_cost_model_terms.py

# Validate (optional but recommended):
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_drift.py
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_alpha_beta.py
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_pp_drift.py

# Pick a strategy:
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
    --trust-source calibrated --top-k 10
```
