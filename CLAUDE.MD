# CLAUDE.md
<!-- Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed. -->

<!-- **Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment. -->

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:◊
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---
<!-- **These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes. -->

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Runtime environment

Python / torch / CUDA tooling for this repo lives **inside the `hetu` Docker container**, not on the host. The host path `/home/yt522/Galvatron` maps to `/root/Galvatron` inside the container. Any python, torch, profiler or training command must be run as:

```
docker exec hetu <command>           # e.g. docker exec hetu python3 /root/Galvatron/galvatron/models/moe/profiler.py --help
```

Running python on the host fails with `ModuleNotFoundError: No module named 'torch'`. File edits made on the host are visible inside the container at the translated path. The full software stack (CUDA 12.1, torch 2.1.0+cu121, flash-attn v2.5.8, apex 312acb4, TransformerEngine 7f2afaa, python 3.9.2) is reproduced in `Dockerfile`.

## Common commands

Build / install (inside container):

```
make                                 # builds csrc/dp_core.cpp -> galvatron/build/lib/galvatron_dp_core*.so
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

`setup.py` also builds two CUDA / C++ extensions out-of-tree at install time:
- `greedy_balancer` (csrc/greedy_balancer.cpp) — greedy expert-placement planner used by LAER.
- `moe_all_to_all_kernels` (csrc/moe_all_to_all_*.{cpp,cu}) — fused NCCL MoE all-to-all. Skipped if `GALVATRON_MOE_KERNELS_INSTALL=FALSE` or CUDA is unavailable.

Profile (must be done before training a new model/cluster):

```
cd galvatron/profile_hardware && bash scripts/profile_hardware.sh   # cluster bandwidth, p2p, allreduce, overlap
cd galvatron/models/moe       && bash scripts/profile_computation.sh
cd galvatron/models/moe       && bash scripts/profile_memory.sh
```

`profile_computation.sh` and `profile_memory.sh` are now **three-pass loops** over `--profile_unit ∈ {all, attention, mlp}` (refactored to feed the cost-model's per-component split — see "Cost model" below). The `mlp` slice is the one the LAER solver consumes (`solver.py:41`); the `all` slice is the back-compat full-block path. Profiling scripts set `ENABLE_SOLVER=0` so per-layer time/memory readings are stationary — preserve this when editing them.

Train (full args list — see `galvatron/models/moe/scripts/train.sh`):

```
bash galvatron/models/moe/scripts/train.sh <model_name> <layer_num> <aux_loss> <approach> <dataset> <capacity> <batch_size> <chunk> <seq_length> <profile>
```

`<approach>` ∈ {`LAER`, `FLEX`, `FSDP`} — this is mapped to `SOLVER` ∈ {`LAER`, `FLEX`, `NONE`} that the runtime reads.

Calibration sweep + cost-model entry points:

```
# Per-config calibration runs (~30 configs, ~3 min each on 4×A6000):
bash galvatron/models/moe/scripts/cost_model_real_test.sh

# Aggregate logs into runtime / FSEP / optimizer profiles (CPU-only):
docker exec hetu python3 galvatron/models/moe/scripts/profile_cost_model_terms.py

# Search the parallelization space:
docker exec hetu python3 galvatron/models/moe/scripts/cost_model_search.py [--num-stages-behind N] [--asymmetry-range LO HI]

# Verify split-arithmetic invariants (no GPU):
docker exec hetu python3 galvatron/models/moe/scripts/cost_model_split_regression.py
```

Artifact-evaluation (paper figures) entry points live in `scripts_ae/` (`e2e.sh`, `convergence.sh`, `breakdown.sh`, `token_counts.sh`, `planner.sh`, `ablation.sh`, `plot.sh`). Each plot script has a matching `<fig>_analysis.py` / `<fig>_plot.py`. `scripts_ae/env.sh` defines `NUM_NODES`, `NUM_GPUS_PER_NODE`, `MASTER_ADDR/PORT`, `NODE_RANK` and NCCL env — source it (or set the same vars) before any distributed launch.

Tests: `pytest.ini` declares markers (`distributed`, `model`, `parallel`, `search_engine`, `profiler`, `utils`) and `testpaths = tests`, but no `tests/` directory is committed. There is no working unit-test entry point for the legacy code; the cost-model package has its own regression suite at `galvatron/models/moe/scripts/cost_model_split_regression.py` (24 invariants, no GPU required).

## Architecture (big picture)

This repo is the LAER-MoE artifact built on top of Galvatron's hybrid-parallel runtime, which itself sits on Megatron-LM. Three concentric layers:

1. **`galvatron/core/runtime/`** — generic hybrid-parallel runtime: PP, TP, DP/SDP, sequence parallel, FSDP. Public API re-exported from `galvatron.core` (`construct_hybrid_parallel_model_api`, `get_hybrid_parallel_configs_api`, `init_empty_weights`, etc.). Includes a top-level FSDP monkey-patch in `runtime/__init__.py` (only when torch >= 2.1.0 < 2.2.0) that resets `handle._prefetched = False` on every reshard — required so explicit forward-refetch works under torch 2.1.

2. **`galvatron/core/runtime/moe/`** — MoE-specific runtime. Token dispatcher (`smart_routing.py`, `token_dispatcher.py`), router, fused kernels (`fused_kernel.py`), and the asynchronous expert-placement system in `prefetch/` (`async_linear_programming.py`, `solver.py`, `fsdp_patch.py`).

3. **`galvatron/core/profiler/`** — model and runtime profilers (`model_profiler.py`, `runtime_profiler.py`, `base_profiler.py`); CLI args in `arguments.py`. Outputs `computation_profiling_*.json` and `memory_profiling_*.json` consumed by the search/cost model. The `_attention` / `_mlp` per-component slices are emitted when `--profile_unit` ≠ `all`.

The MoE entry-point is **`galvatron/models/moe/`**:
- `MoEModel_hybrid_parallel.py` / `_sequential.py` / `_tensor_parallel.py` — model construction at each parallel granularity.
- `__init__.py` — package init wraps the runtime imports in `try / except ImportError` so the cost-model subpackage stays importable on CPU-only systems (where the compiled MoE kernels aren't built).
- `train_dist.py`, `train_dist_random.py`, `train_dist_with_profile.py`, `profile_dist_static.py` — launchers (selected by `train.sh` via `GALVATRON_PROFILE`).
- `meta_configs/*.json` — predefined model configs (mixtral-8x7b-{e8k2,e16k4}, mixtral-8x22b-*, qwen-8x7b-*, qwen-30b-a3b-e128k8, mixtral-tiny). Loaded via `meta_configs/config_utils.py`. Naming starts with a family prefix that gates per-model arg branches (`qwen-30…` → `qk_layernorm=True`; `qwen-8x…` → `add_qkv_bias=True`).
- `configs/` — profiled time/memory and Galvatron strategy JSONs that the runtime reads at training time. `network_config.json` carries link-bandwidth coefficients (`inter_node`, `intra_node`).
- `cost_model/` — the cost-model package. See "Cost model" below.

Hardware profiles consumed by the cost model live in `galvatron/profile_hardware/hardware_configs/` (`allreduce_bandwidth_*.json`, `p2p_bandwidth_*.json`, `overlap_coefficient.json`, `sp_time_*.json`). These must match the cluster shape (`{nodes}nodes_{gpus}gpus_per_node`) named in the filenames.

### LAER expert re-layout pipeline (the central novel piece)

End-to-end docs in `doc/LAER_MOE_NOTES.md`. Per-MoE-layer step:

1. `MoEAlltoAllSmartTokenDispatcher` (`smart_routing.py`) collects per-expert token counts, all-gathered over the EP group into a 5-iter rolling window.
2. `submit_lp_optimization(...)` (`prefetch/async_linear_programming.py`) hands the window to a worker process — non-blocking on the training stream.
3. The worker calls into the C++ `greedy_balancer` extension (`csrc/greedy_balancer.cpp`, entry `greedy_load_balancing_heuristic_complete`), which runs replica budgeting → LPT greedy placement (with topology tiebreak) → smart-routing simulator scoring.
4. Result is fetched via `get_lp_optimization_result(timeout=1.0)`; tensors (`global_expert_indices`, `global_expert_locations`, `inverse_expert_map`) are HtoD-copied on a side `cuda_htod_stream`.
5. `sync_htod()` swaps the new placement into the live FSDP handle; the next forward picks it up through the fused routing kernels.

If the solver isn't ready in time the previous layout is reused. `ENABLE_SOLVER=0` (env) disables the whole loop — required during profiling so per-iteration cost stays stationary.

### Cost model

`galvatron/models/moe/cost_model/` is a standalone CPU-only package that predicts per-iteration time, max-stage time, and peak memory for a parallelization config without launching a real run. Used by the search driver and intended for integration into a higher-level `Facade` (see `doc/cost_model_facade_integration.md`).

Three layers, mirroring the runtime:

- **`base.py`** — `ICostModel` abstract interface; `CostEstimate` (full diagnostic breakdown); `CostQuery` (focused 3-metric façade — `iter_ms`, `max_stage_ms`, `peak_memory_mb` + provenance).
- **`intra.py`** — `IntraCostModel`: cost of running `num_layers` transformer layers on one pipeline stage. Reads profile JSONs (computation, memory, runtime, FSEP overhead, optimizer step, embedding/lm-head, network) at construction; never mutates state thereafter.
- **`pp.py`** — `PPCostModel`: 1F1B-aware orchestrator over `IntraCostModel`. Has a runtime-profile shortcut at calibrated shapes that returns the measured `iter_ms` directly; per-stage compositional path for asymmetric or uncalibrated configs.
- **`search.py`** — `MoESearcher`, `SearchResult`, `RankedSearch`. Object-style search API. Holds a long-lived `PPCostModel`, exposes `score(cfg, …)` for one-config probes and `rank(…)` for batched + filtered + sorted results. Single-process by design; outer loops parallelize.

Both `IntraCostModel.estimate(...)` and `PPCostModel.estimate(...)` accept these high-leverage kwargs:

- `num_attention_layers` / `num_expert_layers` — asymmetric layer counts. Default to `num_layers` (1:1, the runtime layout). Diverging requires a per-component computation profile at the queried shape; uses the `_attention` / `_mlp` slopes to split per-layer time, applies the `_attention_param_share` analytical formula for memory.
- `num_stages_behind` — pure-additive integer. Each stage's effective in-flight count becomes `(pp − k + 1) + num_stages_behind`. Memory-only knob; iter_ms / max_stage_ms stay calibrated. Useful for modelling framework buffer overhead and OOM-margin-sensitive search probes.
- `fsep` — bool. Triggers the FSEP overhead profile (per-MoE-layer time + memory delta from matched on/off calibration runs). Uniform-FSEP rule: overhead scales with `num_expert_layers`, attention is fsep-invariant.

The `query()` method on every cost model returns a `CostQuery` dataclass — the right shape for external callers (planners, search loops). `estimate()` returns a `CostEstimate` with the full diagnostic breakdown — used by drift / regression scripts.

Profile artifacts under `galvatron/models/moe/configs/`:
- `runtime_profiling_<prec>_<model>.json` — per-`(tp, ep, micro_bsz, seq, fsep[, pp])` shape: `fwd_bwd_ms`, `opt_ms`, `iter_ms`, `params_mb`, `optimizer_mb`, `activation_peak_mb`, `cuda_peak_mb` + α/β fits across `num_layers` when ≥ 2 N samples exist.
- `fsep_overhead_profiling_<prec>_<model>.json` — per-MoE-layer FSEP delta. Field name is `time_overhead_per_expert_layer_ms` / `memory_overhead_per_expert_layer_mb` (legacy aliases without `_expert` are kept).
- `optimizer_step_profiling_<prec>_<model>.json` — Adam throughput + empirical `optimizer_to_params_ratio_median`.
- `embedding_lmhead_profiling_<prec>_<model>.json` — standalone embed + lm-head timings.
- `non-solver/computation_profiling_*.json`, `non-solver/memory_profiling_*.json` — processed per-layer slopes (unsuffixed = `all`, `_attention` / `_mlp` when the three-pass loop ran).

Programmatic usage is documented in `doc/cost_model_guide.md` (sections §2 estimate / query API, §3 search CLI, §3f programmatic search, §4 asymmetric layers, §4-bis `num_stages_behind`, §5 multiprocessing notes). Integration target is documented in `doc/cost_model_facade_integration.md`.

### Configuration modes

Two ways to drive parallelism (see `galvatron/models/README.md` for full details):

- **JSON config mode (recommended)**: `--galvatron_config_path ./configs/galvatron_config_xxx.json`. The JSON specifies layer-wise hybrid strategies; all GLOBAL-mode args are ignored.
- **GLOBAL config mode**: `galvatron_config_path` unset; specify `pp_deg`, `global_tp_deg`, `global_tp_consec`, `global_ep_deg`, `global_tp_of_ep_deg`, `sdp`, `chunks`, `global_checkpoint`, `pipeline_type`, `vocab_tp` directly. Used by `train.sh` defaults.

`profile_mode` ∈ {`static`, `batch`, `sequence`}. The training-time `profile_mode` should match the value used at profile time (except `sequence`, which generalises). Switching `profile_mode` requires regenerating the matching `*_profiling_*.json`.

## Conventions worth knowing

- The package historically also imports as `LAER-MoE` (see `save_checkpoint.sh` which `cd`s to `$BASE_DIR/LAER-MoE`). The Galvatron checkout itself is the source — checkpoint scripts assume a sibling `Megatron/` repo and sibling `tokenizers/`, `datasets/`, `checkpoints/` directories under `$BASE_DIR`.
- Checkpoints come in two formats: Megatron-LM (`checkpoints/megatron/<name>`) and LAER (`checkpoints/laer/<name>`). `galvatron/tools/convert_to_laer.py` converts between them; both formats start from the same Megatron checkpoint to keep convergence comparable across approaches.
- Solver disable: any time you change a profiling script or write one, set `ENABLE_SOLVER=0` so per-step time/memory readings aren't perturbed by online expert migration.
- Pre-built `.so` artefacts (`greedy_balancer.cpython-*.so`, `moe_all_to_all_kernels.cpython-*.so`) are checked in at the repo root for both py3.9 and py3.12 ABIs. `pip install -e .` will rebuild them; if you only edit Python you can avoid touching them.
- The cost-model package is **CPU-only**. `galvatron/models/moe/__init__.py` wraps the runtime stack import in a `try/except ImportError` so `from galvatron.models.moe.cost_model import …` works on machines without the compiled MoE kernels (fresh cloud instances, CI runners, design-space probes).
- Domain conventions kept in the cost model: `pp / dp / tp / ep` for parallel degrees, `fsep` for the LAER smart-routing kernel, `mb` suffix for "megabytes," `ms` suffix for "milliseconds." Avoid these as bare local names.
- Variable-naming preference: meaningful names over acronyms. `num_microbatches` not `n_micro`; `runtime_entry` not `rt_entry`; `attention_time_per_layer_ms` not `t_attn_per`. The exception is the small set of domain abbreviations above and field names that already ship in JSON profiles (don't churn the on-disk format without re-aggregating).
- Pre-existing snapshots live alongside the working tree under `~/galvatron-*-checkpoint-*` directories. Useful as recovery points when the working tree's untracked files (configs, logs, CLAUDE.md) get cleaned up.
