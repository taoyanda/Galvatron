# Comprehensive Profiling Plan for PP Integration with LAER-MoE

Implementation plan for extending Galvatron's profiler to emit enough information for a **1F1B pipeline-parallel cost model** layered on top of LAER-MoE. Companion to [LAER_MOE_NOTES.md](LAER_MOE_NOTES.md).

Status: **planned, not yet implemented**. 

---

## Locked-in Design Decisions

| # | Decision |
|---|---|
| 1 | Schedule: Consider **1F1B**. No GPipe / zero-bubble in scope. |
| 2 | Cost-model purpose: Estimate **per-stage execution times** used by a PP layer-to-stage assigner. Not a full end-to-end iter-time predictor. |
| 3 | **High accuracy required** — all sizes are static, make as little approximation as possible. as long as the value can be profiled in reasonable time. |
| 4 | **Static LAER**: warmup iters with deterministic fixed input, then expert layout and token dispatch are frozen. Eliminates dynamicity as a noise source. |
| 5 | **Consider EP all-to-all cost** |
| 6 | **Per-MoE-layer timing required.** MoE-layer-10 ≈ MoE-layer-11 under balanced token distribution with FSEP, but each layer's frozen `A` differs — emit per-layer numbers, let the PP planner approximates if it wants to. |
| 7 | **Separate output files** for the PP planner. Do not touch LAER-MoE profiling files named after `computation_profiling_*.json` (which the FSEP load-balancing planner reads). |

## Code-alignment corrections in this plan revision

- Profiling-mode flags should be added in [galvatron/core/runtime/arguments.py](galvatron/core/runtime/arguments.py), not in [galvatron/models/moe/arguments.py](galvatron/models/moe/arguments.py). The latter currently carries model-shape arguments.
- Main MoE training data path is `get_batch(...)` in [galvatron/models/moe/dataloader.py](galvatron/models/moe/dataloader.py), used by [galvatron/models/moe/train_dist.py](galvatron/models/moe/train_dist.py) and [galvatron/models/moe/train_dist_with_profile.py](galvatron/models/moe/train_dist_with_profile.py).
- LAER freeze logic is meaningful only when solver is enabled (`ENABLE_SOLVER=1`), since `sync_lp_solver()` is gated by `async_lp_solver_config["enabled"]` in [galvatron/core/runtime/moe/smart_routing.py](galvatron/core/runtime/moe/smart_routing.py).
- Verify compatibility with activation checkpointing and recompute modes. The profiler should be able to emit separate profiles for each mode, and the PP planner should be able to choose which one to use.

## Cost-model signature the profiler must serve

```
T_segment(fsep_cfg, seq_len, microbatch_size, num_microbatches, layer_list)
    = ( Σ_{layer ∈ list} t_fwd[layer] + t_bwd[layer] ) · num_microbatches
      + per_iter_overhead

M_segment(fsep_cfg, seq_len, microbatch_size, stage_idx, layer_list)
    = model_state(layer_list) + (PP − stage_idx) · Σ per_mbs_act[layer]
```

Everything below exists to populate the right-hand-side primitives.

---

## Step 1 — Deterministic, static training input

The whole "static LAER" premise rests on this. For the purpose of evaluating planning decisions, the planner and the token dispatcher must make deterministic decisions. Hence, the input tokens must remain the same across iterations.

- Add `--static_input` in [galvatron/core/runtime/arguments.py](galvatron/core/runtime/arguments.py) (runtime/profiling scope).
- In [galvatron/models/moe/dataloader.py](galvatron/models/moe/dataloader.py), implement static-input mode in `get_batch(...)` (the path used by current distributed training scripts):
  - Build one full batch once.
  - Broadcast rank-0's batch tensors across the DP group (`mpu.get_data_parallel_group()`).
  - Cache and return the same batch every iteration.
  <!-- - Keep the existing fake-tensor path for middle PP stages unchanged. -->
- Optionally mirror static mode in `train_dist_random.py` path (synthetic `DataLoaderForMoE`) for quick local checks.
- Seed handling is already partly present through `set_seed()` in [galvatron/utils/training_utils.py](galvatron/utils/training_utils.py); static mode should force deterministic seeding at startup for all ranks.
- Keep dropout invariant.
- Verify determinism: log `history_num_global_tokens_per_expert` from dispatcher state for several consecutive post-warmup iterations. If not bit-identical, stop and debug nondeterminism before profiling.

## Step 2 — Freeze the LAER layout after warmup

No freeze flag exists today. Add a corresponding flag to trigger this behavior.

The behavior should be: once the static input is established, the solution returned by LAER should remain the same. After a number of iterations, training performance should be measured without the overhead of planner and layout changes.

- Read into `async_lp_solver_config` at [smart_routing.py:126](galvatron/core/runtime/moe/smart_routing.py#L126).
- In `sync_lp_solver` at [smart_routing.py:514](galvatron/core/runtime/moe/smart_routing.py#L514), early-return when `solver_iter >= freeze_after_iter`. Prefetch polls continue to drain any in-flight task; no new tasks are submitted.
- Freeze must fire for **every MoE layer independently** — each layer has its own dispatcher and task pipeline. Log `[layer L] layout frozen at iter N` from each so you can confirm all layers are stable.
- Note: freeze is a no-op unless solver is enabled via environment (`ENABLE_SOLVER=1`).
- Optional but recommended: expose parser args for `moe_computation_config_path` / `moe_network_config_path` since `smart_routing.py` currently reads them via `getattr(..., default)` without explicit CLI registration.

## Step 3 — Profiling implementation

Under static input + frozen layout, per-layer times are deterministic up to CUDA scheduling jitter. Sample ~20 post-warmup iterations and take the median.

Reuse `RuntimeProfiler` as much as possible.

<!-- Add a `PhaseTimer` helper adjacent to [runtime_profiler.py](galvatron/core/profiler/runtime_profiler.py):

```python
class PhaseTimer:
    # keyed by (layer_idx, "fwd"|"bwd"|"recomp")
    # torch.cuda.Event pairs pre-allocated
    def start(self, key): ...
    def stop(self, key):  ...    # records elapsed_time into a per-key list
    def finalize(self):          # medians, p95s → dict
``` -->

Instrumentation points:
- **Attention block** fwd + bwd in `MoEAttention_tp.forward(...)` in [MoEModel_tensor_parallel.py](galvatron/models/moe/MoEModel_tensor_parallel.py).
- **MoE block** fwd + bwd in `MoELayer_tp.forward(...)` / `MoEMLP_tp.forward(...)` in [MoEModel_tensor_parallel.py](galvatron/models/moe/MoEModel_tensor_parallel.py): one combined timer for gate + dispatch + expert compute + combine (+ TP collectives), per decision 5.
- **Recompute variants** should be profiled as separate modes for `--global_checkpoint` and `--recompute_communication`; if Megatron recompute granularity is used, capture that mode in metadata too.

Subtleties:
- Do not call `torch.cuda.synchronize()` inside layer hooks. Synchronize only at controlled collection boundaries.
- Because layers are wrapped by FSDP/checkpoint wrappers in runtime construction, ensure backward instrumentation survives wrapping (attach hooks to wrapped modules or use output-tensor grad hooks). Do not assume raw module hooks alone are sufficient.
- TP all-reduces inside attention/MoE are implicitly counted inside the layer time (per decision 5). Document clearly in the output schema so the PP planner does not add comm separately.
- Per-layer MoE times will differ across layers even under balanced load because each layer's frozen `A` is different. Emit per-layer-index numbers.
- Keep current `RuntimeProfiler` iteration-level timing separate from per-layer phase timing; do not overload existing `profile_time_start/end` semantics.

## Step 4 — Per-layer activation memory

For 1F1B the cost model needs `per_mbs_act[layer]` so it can compute `(PP − stage_idx) × Σ per_mbs_act[layer_list]`.

Do both and cross-check:

(a) **Layer-sweep fit** (reuses Galvatron's existing approach). Profile several `--num_hidden_layers` values (e.g. 2, 4, 6, 8) at the smallest microbatch size that still fires all experts. Fit linear: slope = per-layer activation, intercept = non-layer state.

(b) **Direct measurement** via `torch.cuda.memory_allocated` deltas around layer forward boundaries (`MoELayer_tp`/attention path), holding outputs so they are not freed early. Faster; verify against (a) on one config.

Record separately:
- `act[layer_idx]` for dense layers.
- `act[layer_idx]` for MoE layers, **per rank** — under static LAER different ranks hold different expert replicas and route different token counts. Output the distribution, not just the mean; the PP planner uses the **max** for the OOM check.

Subtleties:
- `reset_peak_memory_stats()` is global. Call once at iter start, read max once at iter end; don't reset mid-iteration.
- FSDP gather buffers spike memory transiently. Put them in `model_state`, not per-layer `act`, to keep the per-layer activation interpretable.
- Under checkpoint/recompute modes, "activation memory" means *stored* activations only; keep a separate field for recompute-related transient peak.
- Existing runtime defaults are narrow for this purpose: `RuntimeProfiler` currently defaults to `profile_ranks=[0, world_size-1]` and `max_profile_iter=5`. Widen rank coverage to represent PP stages and EP groups for Step 4 outputs.

## Step 5 — P2P calibration (cheap, one-time)

Separate micro-benchmark, not in the training loop. Write `galvatron/tools/bench_p2p.py` that does `torch.distributed.send/recv` at exact boundary tensor sizes (`hidden × seq × microbatch × dtype`) and reports effective bandwidth/latency after warmup. Store as `pp_p2p_<cluster>.json`.

Keep decoupled from model-level profiling — P2P characteristics are cluster-global, not model-specific.

## Step 6 — Output schema

Write to `galvatron/models/moe/configs/pp_profiling_<prec>_<model>_<fsep_cfg>.json`. **Do not touch** the existing `computation_profiling_*.json` that the LAER solver reads via `computation_config_path`.

Suggested schema:

```json
{
  "meta": {
    "model": "mixtral-8x7b", "precision": "bf16",
    "fsep": {"tp": 2, "ep": 8, "dp": 2},
    "seq_len": 4096, "microbatch_size": 1,
    "warmup_iters": 50, "profile_iters": 30,
    "laer_frozen_at_iter": 50,
    "pipeline_type": "pipedream_flush",
    "profile_flags": {
      "global_checkpoint": true,
      "recompute_communication": false,
      "recompute_granularity": null
    },
    "units": {"time": "ms", "memory": "MB"}
  },
  "layers": [
    {"idx": 0, "type": "attn",
     "t_fwd_ms": ..., "t_bwd_ms": ..., "t_recomp_ms": ...,
     "act_mb_per_mbs": ..., "model_state_mb": ...},
    {"idx": 1, "type": "moe",
     "t_fwd_ms": ..., "t_bwd_ms": ..., "t_recomp_ms": ...,
     "act_mb_per_mbs": {"rank_0": ..., "rank_max": ...},
     "model_state_mb_per_rank": [...]}
  ],
  "per_iter_overhead_ms": {
    "clip_grad_norm": ...,
    "optimizer_step": ..., "grad_norm": ..., "misc": ...
  }
}
```

One file per `(model, precision, fsep_cfg, seq_len, microbatch_size)`. The PP planner loads the matching file and sums.

## Step 7 — Validation harness

For at least 3 (PP, TP, EP) configs with different layer-to-stage partitions:
- Run real training for 100 iters post-warmup with `--pipeline_type pipedream_flush`; record wall-clock iter time.
- Predict iter time from the cost model:
  ```
  T_iter ≈ T_bubble
         + max_stage( Σ layers t_fwd + t_bwd ) · num_microbatches
         + T_p2p
         + T_per_iter_overhead
  ```
- Target ≤5% relative error (per decision 3). If you miss, usual culprits: P2P at small messages, recompute mode mismatch, missing clip/optimizer/scheduler overhead, or incomplete freeze.

---

## Additional Subtleties

1. **"Static input" is stricter than it sounds.** If any rank sees a different micro-batch (e.g. from DP sharding), routing decisions diverge and the global layout will not stabilise. Broadcast the frozen batch from rank 0 across the DP group.

2. **Warmup length.** Rolling window is 5, but the solver also takes several iters to stop flipping between near-cost layouts. Target ≥50 warmup iters before freezing; verify by logging "layout changed" events from `_process_lp_result` at [smart_routing.py:596](galvatron/core/runtime/moe/smart_routing.py#L596).

3. **Per-layer time under 1F1B is schedule-agnostic by construction**, but `t_bwd` must be measured on a *steady-state* microbatch's backward, not the final microbatch that triggers the optimizer step.

4. **Optimizer step and grad-norm are per-iter, not per-microbatch or per-layer.** Record once in `per_iter_overhead_ms`. Do not let the PP planner attribute them to a stage.

5. **Recompute is orthogonal.** Profile with recompute-on and recompute-off separately; the PP planner may want to choose per-layer. If you only profile one mode now, document the limitation in the JSON `meta`.

6. **Per-MoE-layer cost is not i.i.d. across layers even at steady state.** Layer 10 and layer 11 will be *close* under balanced load but not identical, because the frozen `A` matrices differ. Keep per-layer numbers; average in the planner, not in the profiler.

7. **Cross-file consistency.** When the PP planner combines `pp_profiling_*.json` with the FSEP solver's `computation_profiling_*.json`, units and conventions must match. Add a `units` field to each JSON and a sanity check that loads both and asserts an invariant (e.g. sum of per-layer times under single-stage PP ≈ iter-time in the other file).

8. **CUDA-event overhead at short phases.** Event pairs around ops <50μs have measurable overhead. If any MoE sub-phase is that short, coalesce events at the block level rather than at sub-phase level.

9. **Cluster constants drift.** `v_comp`, `v_intra`, `v_inter` are hardcoded at [greedy_balancer.cpp:206-208](csrc/greedy_balancer.cpp#L206-L208). Once you have measured P2P and per-layer comm numbers, cross-check against these — if they drift >20%, the LAER planner itself is running on stale constants and should be updated.

10. **Memory profiler defaults are too narrow.** Current `RuntimeProfiler` caps at 5 iters and 2 ranks (rank 0 + world_size−1). For per-PP-stage numbers you must widen `profile_ranks` to one rank per (pp_stage, ep_group). Do this as part of Step 4.

11. **Current CLI surface gap.** `smart_routing.py` supports `moe_computation_config_path` and `moe_network_config_path` via `getattr`, but parser registration is missing. Add explicit args before relying on per-run overrides.

---

## Execution order

1. Steps 1 + 2 together (the "profiling mode" on-switch).
2. Step 3 (time primitives).
3. Step 4 (memory primitives).
4. Step 5 (P2P calibration — can run in parallel with 3-4).
5. Step 6 (schema + writer).
6. Step 7 (validation).

Do not start the PP planner itself until Step 7 passes the ≤5% bar — a cost model built on a noisy profiler is worse than no cost model.
