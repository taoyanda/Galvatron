# Plan: MoE Model Profiling, Inspired by LLaMA v2.4.0

Companion to [LAER_MOE_NOTES.md](LAER_MOE_NOTES.md) and [PP_PROFILING_PLAN.md](PP_PROFILING_PLAN.md). This plan covers the **v2.4.0-equivalent baseline** profiling pipeline (computation + memory) for MoE models. The PP-cost-model extensions in `PP_PROFILING_PLAN.md` build on top of it.

Status: **planned, not yet implemented**.

---

## Context

The `moe-integration` branch already has an in-training **`RuntimeProfiler`** (writes per-iter time and per-stage memory samples), and a substantial **`ModelProfiler`** ([galvatron/core/profiler/model_profiler.py](galvatron/core/profiler/model_profiler.py), ~1052 lines) that — at tag `tags/v2.4.0` — drives the LLaMA computation+memory profiling pipeline by sweeping `(layernum, bsz, seq_len, tp, pp, ckpt)`, launching `train_dist_random.py` for each point, then post-processing the raw JSON into per-layer numbers via linear fit.

**What's missing on `moe-integration` to make this pipeline run for MoE models:**

1. `ModelProfiler` is not exported from [galvatron/core/profiler/__init__.py](galvatron/core/profiler/__init__.py) (only `RuntimeProfiler` is).
2. The dedicated profile-args registration file `galvatron/core/profiler/arguments.py` exists in `tags/v2.4.0` but **is missing on this branch**, so flags like `--profile_mode`, `--profile_type`, `--layernum_min/max`, `--profile_min/max_batch_size`, `--max_tp_deg`, `--profile_dp_type`, `--profile_seq_length_list` never reach `argparse`.
3. `initialize_galvatron` ([galvatron/core/arguments.py:6-22](galvatron/core/arguments.py#L6-L22)) only knows the `"train_dist"` and `"train"` modes — there is no `"profile"` branch (and an unset `extra_args_provider` would NameError).
4. There is no model-side launcher `galvatron/models/moe/profiler.py`, no `scripts/profile_computation.sh`, no `scripts/profile_memory.sh`. The LLaMA equivalents in `tags/v2.4.0` are tiny (~20 lines and ~50 lines respectively).

**Intended outcome**: `bash scripts/profile_computation.sh` and `bash scripts/profile_memory.sh` from `galvatron/models/moe/` produce `configs/computation_profiling_<prec>_<model>.json` and `configs/memory_profiling_<prec>_<model>.json` — exactly as for LLaMA at v2.4.0.

A pre-existing [configs/computation_profiling_bf16_mixtral-8x7b.json](galvatron/models/moe/configs/computation_profiling_bf16_mixtral-8x7b.json) already uses the v2.4.0 schema (`layernum[N]_bsz<B>_seq<S>` raw + `layertype_<idx>_*` derived, plus `_attention` / `_mlp` sub-keys), so the schema is compatible — it just isn't being produced by an automated pipeline today.

---

## Files to modify / add

### Reuse (already present, no changes)
- [galvatron/core/profiler/model_profiler.py](galvatron/core/profiler/model_profiler.py) — `ModelProfiler` class, `set_profiler_launcher`, `launch_profiling_scripts`, `process_profiled_data` and the per-layer linear-fit derivation.
- [galvatron/core/profiler/runtime_profiler.py](galvatron/core/profiler/runtime_profiler.py) — `RuntimeProfiler`, already wired into all three MoE train scripts.
- [galvatron/core/profiler/utils.py](galvatron/core/profiler/utils.py) — `save_profiled_memory`, `save_profiled_time`, `print_peak_memory`.
- [galvatron/core/profiler/base_profiler.py](galvatron/core/profiler/base_profiler.py) — `BaseProfiler.memory_profiling_path()` / `time_profiling_path()`.
- [galvatron/models/moe/train_dist_random.py](galvatron/models/moe/train_dist_random.py) — already invokes the runtime profiler on synthetic data; this is what the launcher will spawn.
- [galvatron/models/moe/meta_configs/config_utils.py](galvatron/models/moe/meta_configs/config_utils.py) — `set_layernum_manually` flag already supports overriding `num_hidden_layers`.

### Modify
1. **[galvatron/core/profiler/__init__.py](galvatron/core/profiler/__init__.py)** — also export `ModelProfiler`.
2. **[galvatron/core/__init__.py](galvatron/core/__init__.py)** — re-export `ModelProfiler` so model entrypoints can `from galvatron.core import ModelProfiler`.
3. **[galvatron/core/arguments.py](galvatron/core/arguments.py)** — add a `"profile"` mode branch to `initialize_galvatron`. Mirror v2.4.0 behaviour: `extra_args_provider = [galvatron_profile_args]` (no megatron init), then call `parse_args(...)`. Also fix the latent bug where `extra_args_provider` is unset for unknown modes (initialise to `[]`).
4. **[galvatron/models/moe/arguments.py](galvatron/models/moe/arguments.py)** — add a `layernum_arg_names()` function returning `["num_hidden_layers"]` (matches LLaMA convention; required by `ModelProfiler.set_profiler_launcher`).
5. **[galvatron/models/moe/MoEModel_hybrid_parallel.py](galvatron/models/moe/MoEModel_hybrid_parallel.py)** — verify `get_moe_config(args, overwrite_args=False)` accepts the kwarg LLaMA uses; if not, add it (LLaMA `profiler.py` calls `get_llama_config(args, overwrite_args=False)` to skip `set_layernum_manually` overrides during profiling setup).

### Add
6. **`galvatron/core/profiler/arguments.py`** — port from `tags/v2.4.0`. Defines `galvatron_profile_args(parser)` registering: `--profile_mode {static,batch,sequence}`, `--profile_type {computation,memory}`, `--layernum_min`, `--layernum_max`, `--profile_batch_size`, `--profile_min/max_batch_size`, `--profile_batch_size_step`, `--profile_min/max_seq_length`, `--profile_seq_length_step`, `--profile_seq_length_list`, `--max_tp_deg`, `--profile_dp_type`, `--mixed_precision`, `--use-flash-attn`, `--sequence_parallel`, `--extra_args_str`. Source: `git show tags/v2.4.0:galvatron/core/profiler/arguments.py`.
7. **`galvatron/models/moe/profiler.py`** — ~20-line launcher modeled on v2.4.0 `llama_hf/profiler.py`:
   ```python
   from galvatron.core import ModelProfiler, initialize_galvatron
   from galvatron.models.moe.arguments import model_args, layernum_arg_names
   from galvatron.models.moe.MoEModel_hybrid_parallel import get_moe_config
   from galvatron.models.moe.meta_configs import model_name

   if __name__ == "__main__":
       args = initialize_galvatron(model_args, mode="profile")
       config = get_moe_config(args, overwrite_args=False)
       profiler = ModelProfiler(args)
       path = os.path.dirname(os.path.abspath(__file__))
       profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
       profiler.launch_profiling_scripts()
       profiler.process_profiled_data()
   ```
8. **`galvatron/models/moe/scripts/profile_computation.sh`** — port from v2.4.0 LLaMA, swap `MODEL_ARGS` for `--model_size mixtral-8x7b` and matching hidden/heads/seq. Set `PROFILE_TRAINER="train_dist_random.py"`. Default flags: `--profile_mode batch --profile_type computation --layernum_min 1 --layernum_max 2 --profile_min_batch_size 1 --profile_max_batch_size 4 --mixed_precision bf16 --use-flash-attn`. Export `ENABLE_SOLVER=0` (see subtleties).
9. **`galvatron/models/moe/scripts/profile_memory.sh`** — port from v2.4.0 LLaMA. Use `NUM_GPUS_PER_NODE=8`. Defaults: `--profile_mode sequence --profile_type memory --profile_batch_size 1 --layernum_min 1 --layernum_max 2 --max_tp_deg 8 --profile_dp_type zero3 --mixed_precision bf16 --sequence_parallel --use-flash-attn`. Export `ENABLE_SOLVER=0`.

---

## MoE-specific subtleties to handle in the implementation

1. **LAER off during profiling.** Set `ENABLE_SOLVER=0` in both shell scripts. Otherwise the async solver mutates expert placement across iterations and per-layer time becomes non-stationary, breaking the linear-fit assumption. (Static-LAER profile mode is covered separately in [PP_PROFILING_PLAN.md](PP_PROFILING_PLAN.md).)
2. **Random data is fine.** `train_dist_random.py` uses `random_collate_fn`; routing will be near-uniform on average — the right baseline for "default placement" profiling. Document this in the JSON `meta` so downstream consumers don't conflate it with skewed-load timings.
3. **Layer counts must fit.** Mixtral-8x7B is large; `layernum_min/max=1/2` keeps a single-GPU computation profile feasible. For memory profiling, target the 8-GPU run with TP=8 (`--max_tp_deg 8`) so expert weights shard across the node.
4. **Per-component sub-keys.** The existing JSON has `_attention` and `_mlp` sub-keys. Verify whether `model_profiler.py` already emits those (the current sample looks v2.4.0-compatible) and, if so, confirm derivation works with the MoE block (where MLP = router + dispatch + experts + combine bundled). If `model_profiler.py` only emits the bare `layertype_0_bsz<B>_seq<S>` per-block number, leave per-component split for a later PR — single per-MoE-layer cost is sufficient for an initial cost model.
5. **`model_name(config)`** for MoE needs to include enough variability that different `(precision, model_size, seqlen)` produce distinct filenames. Verify [galvatron/models/moe/meta_configs/__init__.py](galvatron/models/moe/meta_configs/__init__.py) exposes a `model_name()` matching the LLaMA convention; if it returns just the bare model_size, extend it (the existing JSON uses `mixtral-8x7b` only, which is acceptable but coarse).
6. **`model_layer_configs(config)`** — used by `RuntimeProfiler.set_profiler_dist`. For MoE the layer-config list should include hidden_size, num_attention_heads, seq_length, **and** num_experts / top_k so memory math doesn't undercount expert weights. Verify what `meta_configs/__init__.py` returns; extend if needed.
7. **`get_moe_config(args, overwrite_args=False)`** — the v2.4.0 LLaMA pattern passes this kwarg during profiling so `set_layernum_manually` (used by the launcher to vary `num_hidden_layers`) is honored without other overwrites. Verify the MoE equivalent supports this kwarg; add if missing.
8. **Dropout already 0.** Confirmed in [galvatron/models/moe/meta_configs/config_utils.py](galvatron/models/moe/meta_configs/config_utils.py); no extra step needed.
9. **`extra_args_provider` NameError fix.** While modifying `initialize_galvatron`, also fix the latent bug where modes outside `{"train_dist","train"}` leave the variable unbound. Initialize to `[]` at the top of the function.
10. **No hardware profiler.** v2.4.0 has `hardware_profiler.py` (bandwidth probing). Out of scope for this plan; the LAER constants in [csrc/greedy_balancer.cpp:206-208](csrc/greedy_balancer.cpp#L206-L208) remain authoritative for now.

---

## Verification

End-to-end smoke test once implemented:

```bash
cd galvatron/models/moe
ENABLE_SOLVER=0 bash scripts/profile_computation.sh
# Expected: configs/computation_profiling_bf16_mixtral-8x7b.json overwritten with
#   keys layernum[1]_bsz1_seq4096, layernum[2]_bsz1_seq4096, ...,
#   and derived layertype_0_bsz<B>_seq<S> entries.

ENABLE_SOLVER=0 bash scripts/profile_memory.sh
# Expected: configs/memory_profiling_bf16_mixtral-8x7b.json populated with
#   strategy keys (e.g., "1_8_1", "1_8_1_c", "1_8_1_sp"), each containing
#   layernum[N]_bsz<B>_seq<S>_rank<R>_{ms,act,act_peak} entries for ranks 0 and 7.
```

Sanity checks:
- Compare the produced `layertype_0_bsz1_seq4096` value against the value in the existing pre-profiled JSON — should agree within ~5% on the same hardware.
- For memory, `_act_peak ≥ _act > 0` and `_ms` is positive on every rank.
- Re-running with `--layernum_min 2 --layernum_max 4` should produce a self-consistent slope (per-layer time should be stable across choice of `(min,max)` if the linear assumption holds).

Lightweight unit test:
- Add or extend a smoke test that imports `from galvatron.core import ModelProfiler` and calls `initialize_galvatron(model_args, mode="profile")` in a no-op fashion to catch arg-registration regressions.

---

## Out of scope (deferred)

- Per-MoE-phase timing instrumentation (gate / dispatch-A2A / experts / combine-A2A) — see [PP_PROFILING_PLAN.md](PP_PROFILING_PLAN.md) Step 3.
- LAER freeze flag and static-input mode — see [PP_PROFILING_PLAN.md](PP_PROFILING_PLAN.md) Steps 1-2.
- Hardware profiling (`hardware_profiler.py` port).
- Per-rank memory expansion beyond ranks 0 and `world_size-1`.
- Per-stage activation memory under PP — see [PP_PROFILING_PLAN.md](PP_PROFILING_PLAN.md) Step 4.
