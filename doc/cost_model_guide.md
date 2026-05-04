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
  expert parallel; **FSEP** = **Fully Sharded Expert Parallel** (the
  expert-MLP sharding strategy across the EP group; under LAER it pairs
  with the smart-routing solver but the two are orthogonal —
  fsep=on/off toggles the sharding pattern, not the solver);
  **N** = `num_hidden_layers`; **micro_bsz** = per-DP-rank batch size
  = `global_bsz / dp / chunks`.

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

`profile_computation.sh` and `profile_memory.sh` now run as a **three-pass
loop** over `--profile_unit ∈ {all, attention, mlp}`. The `attention`
and `mlp` passes write `_attention` / `_mlp` suffixed keys
(`layertype_0_bsz<B>_seq<S>_attention`, `_mlp`). The cost model picks
these up automatically and uses them to split per-block time/memory
into attention vs expert components — required for the asymmetric
layer-count API (§4 below). When only the `all` pass has run, the
cost model falls back to the full-block path; symmetric queries are
unaffected.

Both scripts drive `train_dist_frozen.py` (not the older
`train_dist_random.py`) with `--static_input + --laer_freeze_after_iter 5
+ --dropout_prob 0`. This is **required when FSEP (Fully Sharded
Expert Parallel) is enabled**: the LAER smart-routing solver makes
per-iteration token-routing decisions, and without a frozen batch
those decisions vary across iterations and produce non-stationary
per-component readings. At the FSEP-off baseline static input is
still used for cross-run reproducibility — the per-component time
and memory ratios stay bit-identical across calibration generations.

> **Layer-stripping bias under FSEP.** The `attention` and `mlp`
> passes ask the profiler to construct a model containing only that
> layer type (the other type is dropped). Under FSEP the router in the
> `mlp` pass sees the **raw static input** rather than post-attention
> features, which produces a different routing distribution → different
> per-rank token load → different per-expert wall time and activation
> memory. This makes the per-component slopes meaningless for FSEP-on
> calibration. So the canonical `profile_computation.sh` /
> `profile_memory.sh` are FSEP-off only (they refuse to run if
> `--use_fsep` is in the args). For FSEP-on, run the `.frozen`
> companions, which run only the `all` pass — that one builds the full
> block (attention + MoE) and sees the real routing distribution. The
> cost model's FSEP-on path uses the FSEP-off per-component ratio (via
> the attention-invariance rule) precisely to avoid relying on biased
> FSEP-on per-component data.

Run order on a fresh cluster:

```bash
# Inside the hetu container or on host with the correct env.
cd /root/Galvatron/galvatron/profile_hardware
bash scripts/profile_hardware.sh         # outputs hardware_configs/*

cd /root/Galvatron/galvatron/models/moe
bash scripts/profile_computation.sh      # writes computation_profiling_*.json
bash scripts/profile_memory.sh           # writes memory_profiling_*.json
```

To sweep FSEP-on (EP, capacity) tuples on top, run the companion
`.frozen` scripts after the canonical sweep:

```bash
bash scripts/profile_computation_frozen.sh   # adds --use_fsep + EP/cap loop
bash scripts/profile_memory_frozen.sh        # same
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
    PPCostModel,     # 1F1B-aware orchestrator (recommended)
    IntraCostModel,  # single-stage; pp must equal 1
    CostQuery,       # focused 3-metric return type — what most callers want
    CostEstimate,    # full diagnostic return type (all breakdown fields)
    query_cost,      # one-shot wrapper around .query()
    estimate_cost,   # one-shot wrapper around .estimate()
)
# Back-compat alias:
from galvatron.models.moe.cost_model import CostModel  # = PPCostModel
```

Two query entry points; pick by what you need:

| Method | Returns | Use when… |
| --- | --- | --- |
| `cm.query(...)` | `CostQuery` (iter_ms, max_stage_ms, peak_memory_mb + provenance) | external interface, planner, search loop — you want the headline numbers |
| `cm.estimate(...)` | `CostEstimate` (full breakdown dict) | drift / regression scripts, the cost model itself for stage composition — you want every diagnostic field |

`query()` is a thin wrapper over `estimate()` that unpacks the breakdown
into a focused dataclass; the underlying computation is identical.

### 2a-bis. The `CostQuery` shape

```python
@dataclass
class CostQuery:
    iter_ms: float           # 1F1B critical path + post-bwd terms
    max_stage_ms: float      # per-microbatch slowest-stage compute
    peak_memory_mb: float    # per-rank high-water mark

    bottleneck_stage: str    # "first" / "middle" / "last" / "single" / "uniform"
    memory_stage: str        # which stage hit peak_memory_mb
    time_source: str         # provenance — runtime_profile[...] / analytical_*
    memory_source: str       # provenance — same scheme

    num_attention_layers: int
    num_expert_layers: int
    asymmetric: bool         # True iff n_attn ≠ n_expert

    breakdown: dict          # full breakdown for callers that want it
```

For typical use only the first three fields matter; `bottleneck_stage` /
`memory_stage` help diagnose bubble vs. comm bottlenecks at large `pp`;
`time_source` / `memory_source` plug into the `--trust-source` filter
in the search driver.

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
| `fsep` | bool | default `False` | Fully Sharded Expert Parallel (expert-MLP sharding across EP) |
| `num_attention_layers` | int or `None` | default `None` (= `num_layers`) | asymmetric API; see §4 |
| `num_expert_layers` | int or `None` | default `None` (= `num_layers`) | asymmetric API; see §4 |

Invariants: `dp × pp × tp × ep == num_gpus`; `num_layers % pp == 0`;
`global_batch_size % (dp × micro_batch_size) == 0`. Violations raise
`ValueError`.

### 2d. Return values

- `cm.query(...)` returns a `CostQuery` (see §2a-bis above).
- `cm.estimate(...)` returns a `CostEstimate(total_iter_ms,
  peak_memory_mb, breakdown)` where `breakdown` is the full diagnostic
  dict. Useful keys include `time_source`, `memory_source`,
  `parameters_mb`, `optimizer_mb`, `activations_mb`, `pipeline_iter_ms`,
  `stage_bottleneck_ms`, `bottleneck_stage`, and the α/β fit
  coefficients (when the runtime profile had ≥ 2 N points for the
  queried shape).

### 2e. Example — programmatic interface

The recommended pattern: build the model **once**, query many times.
Each query is cheap (no profile reload).

```python
from galvatron.models.moe.cost_model import PPCostModel

cm = PPCostModel("mixtral-8x7b-e8k2")  # loads profile JSONs once

# Single query for the headline numbers:
result = cm.query(
    num_layers=4, num_gpus=4,
    dp=1, pp=1, tp=1, ep=4,
    micro_batch_size=4, global_batch_size=4,
    seq_len=4096, sequence_parallel=True,
    zero_stage=2, sdp=True,
    recompute=True, bwd_mult=2.0, fsep=False,
)
print(result.iter_ms, result.max_stage_ms, result.peak_memory_mb)
# 1445.61 1445.61 30356.86

# Sweep PP at the same per-stage shape:
for pp in (1, 2, 4):
    r = cm.query(num_layers=4, num_gpus=4,
                 dp=1, pp=pp, tp=1, ep=4 // pp,
                 micro_batch_size=4, global_batch_size=4, seq_len=4096,
                 sequence_parallel=True, zero_stage=2, sdp=True,
                 recompute=True, bwd_mult=2.0)
    print(f"pp={pp}: iter={r.iter_ms:.0f}  max_stage={r.max_stage_ms:.0f}"
          f"  peak={r.peak_memory_mb:.0f}  ({r.bottleneck_stage})")
```

For one-shot CLI-style calls where construction overhead doesn't
matter:

```python
from galvatron.models.moe.cost_model import query_cost

result = query_cost(
    "mixtral-8x7b-e8k2",
    num_layers=4, num_gpus=4, dp=1, pp=1, tp=1, ep=4,
    micro_batch_size=4, global_batch_size=4, seq_len=4096,
    sequence_parallel=True, zero_stage=2, sdp=True, recompute=True,
)
print(result)
# CostQuery(iter_ms=1445.6, max_stage_ms=1445.6, peak_memory_mb=30356.9, ...)
```

For drift/regression scripts that need the full diagnostic breakdown,
use `cm.estimate(...)` directly:

```python
est = cm.estimate(num_layers=4, num_gpus=4, dp=1, pp=1, tp=1, ep=4, ...)
print(est.breakdown["per_attention_layer_ms"],
      est.breakdown["per_expert_layer_ms"])
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

### 3f. Programmatic search interface

The CLI is a thin wrapper over :class:`MoESearcher` — the same class
external code (planners, design-space probes, "compose results across
partial models" loops) should import directly. `MoESearcher` holds a
long-lived :class:`PPCostModel`, and exposes per-call entry points
that an outer loop can drive without spawning processes (the inner
search is single-process by design — `query()` is too cheap to
parallelize, and cascading process pools across the search wastes CPU
without measurable speedup).

```python
from galvatron.models.moe.cost_model import (
    MoESearcher,    # the search object
    SearchResult,   # one scored config: cfg + CostQuery (or error)
    RankedSearch,   # output of MoESearcher.rank: viable + infeasible
)

searcher = MoESearcher("mixtral-8x7b-e8k2")  # builds PPCostModel once
```

**Three entry points:**

| Method | Returns | Use when… |
| --- | --- | --- |
| `searcher.score(cfg, **workload)` | `SearchResult` | scoring one specific layout (e.g. validating a known config) |
| `searcher.rank(**workload)` | `RankedSearch` | scoring the full enumerated config space and picking the best |
| `searcher.enumerate(num_gpus, num_layers, ...)` | iterator of cfg dicts | inspecting / pre-filtering the config space without running the cost model |

`SearchResult` is the per-config dataclass:

```python
@dataclass
class SearchResult:
    cfg: dict                  # {pp, dp, tp, ep, dp_mode, fsep}
    query: Optional[CostQuery] # None if errored
    error: Optional[str]       # None if viable
    num_attention_layers: int  # the layout this was scored against
    num_expert_layers: int

    @property
    def viable(self) -> bool: ...
    @property
    def iter_ms(self) -> float: ...      # NaN when not viable
    @property
    def max_stage_ms(self) -> float: ...
    @property
    def peak_memory_mb(self) -> float: ...
```

`RankedSearch` is the per-search dataclass:

```python
@dataclass
class RankedSearch:
    viable: List[SearchResult]      # sorted ascending by sort_key
    infeasible: List[SearchResult]  # filtered out (with `error` set)
    num_total: int                  # total enumerated
    sort_key: Callable              # the key that produced this order

    @property
    def best(self) -> Optional[SearchResult]: ...
    def top(self, k: int) -> List[SearchResult]: ...
    def resort(self, key) -> RankedSearch: ...
```

#### Pattern A — score one config (an outer loop validating known layouts)

```python
cfg = {"pp": 1, "dp": 1, "tp": 1, "ep": 4,
       "dp_mode": "zero2sdp", "fsep": "off"}
r = searcher.score(cfg, num_layers=4, num_gpus=4, global_bsz=4)
if r.viable:
    print(f"iter={r.iter_ms:.0f}  max_stage={r.max_stage_ms:.0f}  "
          f"peak={r.peak_memory_mb:.0f}")
else:
    print(f"rejected: {r.error}")
```

#### Pattern B — rank the full config space, custom sort key

```python
ranked = searcher.rank(
    num_gpus=4, num_layers=4, global_bsz=4,
    gpu_memory_mb=45000, trust="calibrated",
)
# Default ordering: (iter_ms, peak_memory_mb)
print(ranked.best.cfg, ranked.best.iter_ms)

# Re-sort by a different criterion (no re-scoring)
by_max_stage = ranked.resort(lambda r: r.max_stage_ms)
print(by_max_stage.best.cfg, by_max_stage.best.max_stage_ms)
```

#### Pattern C — outer loop composing results across partial models

The original use case for this API. The outer loop drives the
searcher across multiple "partial models" (different `num_layers`,
different shapes, different parallelization budgets) and combines the
per-partial best configs externally.

```python
# One searcher, many calls — profile JSONs load once.
searcher = MoESearcher("mixtral-8x7b-e8k2")

results = []
for partial_n in (4, 8, 12, 16):
    ranked = searcher.rank(
        num_gpus=8, num_layers=partial_n, global_bsz=8,
        gpu_memory_mb=80000, trust="calibrated",
    )
    if ranked.best:
        results.append((partial_n, ranked.best))

# Compose externally — the searcher is silent on what "compose" means.
total_iter_ms = sum(r.iter_ms for _, r in results)
print(f"end-to-end best iter time: {total_iter_ms:.0f} ms")
```

The inner search is single-process; if the outer loop runs across
many models on a multi-CPU host, parallelize at the **outer** level
(spawn a worker per model with its own `MoESearcher`) — never inside
`searcher.rank()`. See §5 below for why.

#### Pattern D — sharing a cost model across multiple searchers

When several searchers all target the same model, share one
`PPCostModel` to avoid re-loading the profile JSONs:

```python
from galvatron.models.moe.cost_model import PPCostModel, MoESearcher

shared_cm = PPCostModel("mixtral-8x7b-e8k2")
s_baseline = MoESearcher(cost_model=shared_cm)
s_strict   = MoESearcher(cost_model=shared_cm)

baseline = s_baseline.rank(num_gpus=4, num_layers=4, global_bsz=4,
                            trust="any")
strict   = s_strict.rank(num_gpus=4, num_layers=4, global_bsz=4,
                          trust="calibrated")
# Compare the rankings externally
```

#### Pattern E — externally-supplied configs (skip enumeration)

```python
# Outer loop has its own constraint logic (e.g. only configs with
# pp ≥ 2 because pp=1 is already known).
custom = [
    {"pp": 2, "dp": 1, "tp": 1, "ep": 2, "dp_mode": "zero2sdp", "fsep": "off"},
    {"pp": 2, "dp": 1, "tp": 1, "ep": 2, "dp_mode": "zero2sdp", "fsep": "on"},
    {"pp": 4, "dp": 1, "tp": 1, "ep": 1, "dp_mode": "zero2sdp", "fsep": "off"},
]
ranked = searcher.rank(
    num_gpus=4, num_layers=4, global_bsz=4,
    gpu_memory_mb=45000, configs=custom,
)
```

#### Back-compat

The historical module-level functions `search`, `estimate_one`, and
`enumerate_configs` in `scripts/cost_model_search.py` still work —
they're thin shims around `MoESearcher`. New code should use the
class API directly; only existing callers (e.g.
`cost_model_search_compare.py`) keep using the dict-shaped legacy
return.

---

## 4. Asymmetric attention vs expert layer counts

The runtime always trains 1:1 (one attention sublayer + one MoE
sublayer per transformer block). The cost model — but not the
runtime — supports projecting costs for hypothetical layouts where
`num_attention_layers ≠ num_expert_layers` (e.g. "what if the last PP
stage carries one extra MoE block?"). This is purely a design-space
exploration tool; the search and `estimate(...)` API expose it through
optional kwargs that default to symmetric.

### 4a. API

```python
cm.estimate(
    num_layers=4,
    num_attention_layers=4,
    num_expert_layers=5,   # one extra MoE layer
    num_gpus=4, dp=1, pp=1, tp=1, ep=4,
    micro_batch_size=4, global_batch_size=4, seq_len=4096,
    sequence_parallel=True, zero_stage=2, sdp=True,
    recompute=True, bwd_mult=2.0,
)
```

When `num_attention_layers == num_expert_layers == num_layers` the
result is bit-identical to today's symmetric path (verified by
`cost_model_split_regression.py`).

### 4b. Required artifact

The asymmetric API needs a per-component computation profile at the
queried `(tp, ep, micro_bsz, seq)` shape. Run
`profile_computation.sh` (now a three-pass loop) once per shape; the
`_attention` and `_mlp` slopes that result drive the per-component
time split.

When the per-component profile is missing at the queried shape and the
caller asks for `n_attn ≠ n_expert`, the cost model raises a `KeyError`
with a clear message. Symmetric queries are unaffected.

### 4c. Uniform-FSEP rule

FSEP (Fully Sharded Expert Parallel) only changes how expert MLPs
are sharded across the EP group; attention compute is unchanged. So
its overhead always multiplies `num_expert_layers`, never
`num_attention_layers`. The cost model anchors attention on the
matching `fsep=off` runtime-profile entry when available, then
attributes the residual of the queried per-layer time to the expert
component — keeping the total calibrated and making the FSEP
attention-invariance regression
(`cost_model_split_regression.py` invariant 3) hold.

There's no per-expert-layer FSEP toggle. FSEP is uniform across all
expert layers for a given query.

### 4d. Pipeline parallelism rules

`num_attention_layers` and `num_expert_layers` must each be divisible
by `pp`. Per stage, the cost model creates `n_attn // pp` attention
sublayers and `n_expert // pp` expert sublayers (uniform layout).
Heterogeneous per-stage layouts (e.g. last stage gets the extra MoE
block) aren't supported yet — the divisibility check raises a clear
`ValueError`.

### 4e. Search flags

```bash
# Single asymmetric query: project costs at n_attn=4, n_expert=5
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
    --num-attention-layers 4 --num-expert-layers 5

# Sweep n_expert ∈ [num_layers + LO .. num_layers + HI] at fixed n_attn
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
    --asymmetry-range -1 2
```

The sweep prints one summary row per `n_expert` value with the best
config's `iter_ms`, `max_stage_ms`, `peak_memory_mb`. Useful for
"is the extra layer worth it?" probes.

### 4-bis. The `num_stages_behind` hyperparameter

Independent of the asymmetric API, the cost model exposes a
`num_stages_behind` knob that adds reserve activation memory for
"extra stages behind the current stage" in 1F1B. Available
everywhere the cost-model API is exposed:
`cm.estimate(num_stages_behind=...)`,
`cm.query(num_stages_behind=...)`,
`searcher.score(..., num_stages_behind=...)`,
`searcher.rank(..., num_stages_behind=...)`,
and the CLI `--num-stages-behind INT` flag.

**Semantics — pure-additive count.** Under 1F1B, stage `k` (1-indexed)
naturally has `pp − k` stages behind it and holds `pp − k + 1`
microbatches of activation memory at steady state. With
`num_stages_behind = N`, every stage acts as if it had `N` more stages
behind it:

```
effective_n_behind[k]      = (pp − k) + num_stages_behind
in_flight_microbatches[k]  = (pp − k + 1) + num_stages_behind
                              ^ standard 1F1B   ^ uniform additive (default 0)
```

The reserve is **uniform across every stage** — first, middle, last,
and even the single stage at `pp == 1` all get the same `+N`
microbatches of activation memory. The original `n_behind` is not
used as a multiplier and not as a cap; the new value is simply
`n_behind + num_stages_behind`.

`num_stages_behind` is an `int`, default `0`. At `0` the cost model
behaves exactly as before.

**Memory-only knob.** `iter_ms` and `max_stage_ms` stay anchored on
the calibrated runtime profile when available; only the activation
reserve delta is computed analytically and stacked onto the
calibrated peak. The `memory_source` field is annotated
`+num_stages_behind(N)` so callers can see the reserve was applied.

**Use cases.**

  - **Framework buffer overhead.** Megatron's pipeline send/recv
    keep-alive buffers, FSDP per-stage scratch, etc., aren't captured
    by the analytical in-flight count. Dial `num_stages_behind` to
    bake in a calibrated overhead estimate.
  - **OOM-margin-sensitive search.** "What configs survive a more
    conservative memory budget?" Run the search with
    `num_stages_behind=1` or `2` and see which optima fall out — the
    OOM filter sees the inflated peak.
  - **Stress-testing rankings.** Sweeping `num_stages_behind` reveals
    which optima are robust to extra activation reserve and which
    only win in the calibrated baseline.

**Example.** CLI sweep:

```bash
# Default budget — show how the optimum's memory grows with the reserve:
docker exec hetu python3 scripts/cost_model_search.py
docker exec hetu python3 scripts/cost_model_search.py --num-stages-behind 1
docker exec hetu python3 scripts/cost_model_search.py --num-stages-behind 2

# Tighter budget — high-pp configs drop out as the reserve grows:
docker exec hetu python3 scripts/cost_model_search.py \
    --num-stages-behind 2 --gpu-memory-mb 32000
```

Programmatic — outer-loop sweep to find robust optima:

```python
searcher = MoESearcher("mixtral-8x7b-e8k2")
for nsb in (0, 1, 2, 3):
    ranked = searcher.rank(
        num_gpus=4, num_layers=4, global_bsz=4,
        gpu_memory_mb=32000, num_stages_behind=nsb,
    )
    if ranked.best:
        print(f"nsb={nsb}: {ranked.best.cfg}  "
              f"peak={ranked.best.peak_memory_mb:.0f}")
    else:
        print(f"nsb={nsb}: no viable configs")
```

**Linearity.** Each step of `num_stages_behind=+1` adds the same
fixed amount of activation memory at a given `pp`:
`per_microbatch_act × layers_per_stage`. Halves each `pp` doubling
since `layers_per_stage = num_layers / pp`.

### 4f. Validation

`cost_model_split_regression.py` runs a 5-invariant suite:

1. Symmetric identity — `estimate(num_layers=N)` and
   `estimate(num_attention_layers=N, num_expert_layers=N)` agree
   bit-identically.
2. PP critical path — `iter_ms == (n_micro + pp − 1) × max_stage_ms +
   max_post_bwd_ms`.
3. FSEP attention invariance — `per_attention_layer_ms` agrees across
   `fsep=on/off` at the same shape.
4. FSEP overhead reconciliation — the FSEP profile's
   `time_overhead_per_expert_layer_ms` matches
   `(per_layer_on − per_layer_off)` derived from the runtime profile.
5. Input validation — negative counts and non-divisible-by-pp counts
   raise.

Run after profile changes:

```bash
docker exec hetu python3 \
    /root/Galvatron/galvatron/models/moe/scripts/cost_model_split_regression.py
```

Exit code 0 = all invariants hold.

---

## 5. Multiprocessing notes

Both `cm.query()` and `searcher.score()`/`searcher.rank()` are
**multiprocessing-clean**: the `PPCostModel` and `MoESearcher` objects
pickle cleanly (~16 KB), `CostQuery` and `SearchResult` are tiny
(<1 KB), and neither method mutates any state after construction. So
`multiprocessing.Pool` works in every standard pattern (initializer,
fork-inherit, pickle-per-task).

That said, **don't parallelize the inner search**. A single
`searcher.score()` is ~20 µs of pure-Python arithmetic; per-task IPC
overhead in a Pool is ~10–20 µs. Best speedup we measured (50 000
queries, 4 GPUs / 48 CPU host) was only **1.7×** at 4 workers, then
dropped to **1.15×** at 8 workers as IPC dominated. The CLI search is
single-process by design, and `searcher.rank()` is single-process by
design.

If you want parallelism, do it at the **outer** level — one worker per
model, or one worker per partial-model sweep. The cost model load time
(~ms) parallelizes well; individual queries don't.

For huge config spaces (>1 M points), the right answer is to vectorize
the cost-model arithmetic with NumPy, not to spawn processes. Each
query does ~30 dict lookups + ~50 floating-point ops; a stacked
implementation would give ≫8× without IPC.

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

# Optional: design-space probe — does adding one MoE layer pay off?
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_search.py \
    --asymmetry-range -1 2

# Optional: regression after any profile / cost-model change:
docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_split_regression.py
```
