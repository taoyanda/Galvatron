# LAER-MoE: Executor & Greedy Planner — Implementation Notes

Working notes on how **LAER-MoE** ("Load-Adaptive Expert Re-layout") integrates an online greedy placement planner with an asynchronous executor into Galvatron's expert-parallel training loop. All file/line references are against this repo.

---

## 1. End-to-End Dataflow

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Training loop  (galvatron/models/moe/train_dist.py:59-90)                 │
│    for iter in ...:                                                        │
│      loss = model.forward_backward(...)    ← per-step forward+backward     │
│      clip_grad_norm; optimizer.step(); zero_grad()                         │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  MoE layer fwd/bwd  (MoEModel_tensor_parallel.py:332-343)                  │
│                                                                            │
│  Router ─► Token Dispatcher (smart_routing.py)                             │
│                │                                                           │
│     ┌──────────┼───────────────────┐                                       │
│     ▼          ▼                   ▼                                       │
│  permute  all-to-all dispatch   grouped GEMM on experts                    │
│            (uses current        (uses inverse_expert_map,                  │
│             global_placement)     global_expert_locations)                 │
│     ▲          ▲                                                           │
│     │          │                                                           │
│   unpermute  all-to-all combine                                            │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │   at the end of each MoE layer step:
                                ▼
 (A) COLLECT LOAD                                  (smart_routing.py:137-178)
     per-expert token counts, all-gathered over EP group
     pushed into a 5-iter rolling window
                                │
                                ▼
 (B) SUBMIT PLAN (async, non-blocking)                  (smart_routing.py:514)
     sync_lp_solver():
       submit_lp_optimization(history_data, layer_number, C_e, ...)
       → task placed on worker-process Queue
       (async_linear_programming.py:20)
                                │
                                ▼
 (C) PLANNER  —  runs in a background worker process
     MoEOptimizer.greedy_load_balancing_heuristic
       → C++ kernel greedy_load_balancing_heuristic_complete
         in csrc/greedy_balancer.cpp:402-506
                                │
                                ▼  result: A_res = list[device] → list[expert_id]
 (D) EXECUTOR PREFETCH + APPLY                     (smart_routing.py:588-630)
     _async_lp_prefetch_logic():
       get_lp_optimization_result(task_id, timeout=1.0)
     _process_lp_result():
       global_expert_indices, global_expert_locations, inverse_expert_map
       → HtoD copy on side stream cuda_htod_stream
     sync_htod():
       fsdp_handle.global_placement         = new placement
       fsdp_handle.global_expert_locations  = …
                                │
                                ▼
 (E) Next forward uses the new layout — fused routing kernels in
     galvatron/core/runtime/moe/fused_kernel.py pick the new mapping up
     via global_expert_indices / inverse_expert_map.
```

Key architectural decision: steps (B)→(C)→(D) run **concurrently** with training on a side CUDA stream + a worker process, so the planner's cost is hidden behind compute. If the solver isn't ready when the next forward needs a layout, the previous layout is reused (1s poll timeout).

---

## 2. The Greedy Planner

Lives in [csrc/greedy_balancer.cpp:402-506](csrc/greedy_balancer.cpp#L402-L506). Entry point: `greedy_load_balancing_heuristic_complete`.

**Inputs**
- `E ∈ R^{n_device × n_expert}` — profiled token counts (summed over rolling window).
- `C_e` — expert slots per device.
- Topology constants: `v_comp`, `v_intra`, `v_inter`, `global_checkpoint`.

**Three stages**

### Stage 1 — Replica budgeting
Decide `r_j` = number of copies of expert *j*, with `Σ r_j = C_e · n_device`. Two candidate strategies enumerated:

- **Average**: `r_j = C_e · n_device / n_expert` when it divides evenly.
- **Precise** (`allocate_expert_replicas_precise`): max-heap proportional allocation. Iteratively gives an extra replica to whichever expert currently has the highest `load_j / r_j`. Hot experts get more copies.

Both are perturbed (`generate_perturbed_replicas`) to produce neighbour candidates. All candidates are scored later.

### Stage 2 — Greedy placement ([greedy_balancer.cpp:141-198](csrc/greedy_balancer.cpp#L141-L198))
Given `r_j`, explode each expert into `r_j` replica-items, each weighted by its share of the expert's load (from `distribute_expert_load_precise`). Sort **all** replica-items by load **descending** (LPT / longest-first list-scheduling). Then for each item:

1. Candidate devices ← those with `device_expert_count < C_e`.
2. **Topology tiebreak**: among those, prefer devices on the node that currently holds the *fewest* replicas of this expert — spreads replicas across nodes to reduce inter-node all-to-all.
3. Among that preferred set, pick the device with **minimum current load**.
4. Assign: `device_loads[best] += load`, `device_expert_count[best] += 1`.

### Stage 3 — Evaluate & select best
For every candidate placement `A`, `generate_smart_routing_and_calculate_time` ([greedy_balancer.cpp:200-345](csrc/greedy_balancer.cpp#L200-L345)) simulates smart routing: each source device splits its tokens for expert *j* across *j*'s replicas, preferring intra-node targets first, then inter-node, weighted by replica weights. Accumulates:

- `comm_times[src] += M_token · 1e-9 · tokens / bw(src,dst)`  (bw = `v_intra` or `v_inter`)
- `comp_times[dst] += tokens · v_comp / 1000`

Objective at [greedy_balancer.cpp:342](csrc/greedy_balancer.cpp#L342):

```
total_time = 4 · Σ_src comm_times[src]  +  (3 + global_ckpt) · max_dst comp_times[dst]
```

`4×` reflects forward + two backward passes + recompute per layer; comm is summed (serial per source in all-to-all), compute is `max` (experts run in parallel). Best-scoring placement wins; returned as `A_res : list[device] → list[expert_id]`.

**Ablation knobs** at [async_linear_programming.py:99-116](galvatron/core/runtime/moe/prefetch/async_linear_programming.py#L99-L116):
- `no_even` — drops the average strategy.
- `no_pq` — drops the precise/max-heap strategy.

These correspond to the ablations in the paper.

---

## 3. The Executor

The "executor" is the runtime glue that turns a planner result into a live routing configuration while staying off the critical path. Lives inside the token dispatcher + FSDP handle.

### Submission
[`sync_lp_solver()` at smart_routing.py:514](galvatron/core/runtime/moe/smart_routing.py#L514) is called at the end of each MoE layer's fwd/bwd ([MoEModel_tensor_parallel.py:342](galvatron/models/moe/MoEModel_tensor_parallel.py#L342)).
- Synchronises `cuda_dtoh_stream` (so load stats are on CPU).
- Submits a task with `history_data`, `layer_number`, `C_e`, and the current `global_expert_indices_numpy` (lets the solver favour placements close to existing layout).

### Prefetch / apply
[`_async_lp_prefetch_logic()` at smart_routing.py:588](galvatron/core/runtime/moe/smart_routing.py#L588) polls with a 1s timeout. On success, [`_process_lp_result`](galvatron/core/runtime/moe/smart_routing.py#L596) unpacks:

- `global_expert_indices` — which expert IDs live on each device (the placement).
- `global_expert_locations[expert, k]` — inverse map: physical slots where expert *e* is replicated.
- `inverse_expert_map[slot] → expert_id` — used by the all-to-all permutation kernels.

All three are H→D copied on `cuda_htod_stream`, and `need_to_sync = True` is raised.

### Commit
[`sync_htod()` at smart_routing.py:625](galvatron/core/runtime/moe/smart_routing.py#L625) is called at the start of the next layer's forward. Waits on `cuda_htod_stream`, then installs new maps into `fsdp_handle.global_placement` and `global_expert_locations`. The fused routing kernels in [fused_kernel.py](galvatron/core/runtime/moe/fused_kernel.py) read those tensors when constructing dispatch/combine, so tokens flow to the new locations from the next forward onwards.

Subtle point: expert **weights** follow the placement because `fsdp_handle.global_placement` is the ground truth FSDP uses for all-gathering/shuffling expert parameters. The executor does not copy weights synchronously — FSDP's standard prefetch picks up the new placement on its next gather, amortizing migration.

---

## 4. Timing Semantics & Warmup

The layout installed for iteration `N+1` is the planner's answer to the **rolling window ending at iteration N** (not iteration N alone). Per-layer cadence:

| Point in time | What happens |
|---|---|
| Iter N, during fwd | `get_smart_routing` appends this step's token counts into rolling window (capacity 5) at [smart_routing.py:159-163](galvatron/core/runtime/moe/smart_routing.py#L159-L163). |
| Iter N, during fwd | `nxt_dispatcher._async_lp_prefetch_logic()` polls for a previously submitted plan and, if ready, commits it via `sync_htod`. |
| Iter N, end of layer | `sync_lp_solver` submits a new task using the **summed last-≤5-iter history** at [smart_routing.py:168,522](galvatron/core/runtime/moe/smart_routing.py#L168). |
| Iter N+1, next fwd | That plan is (normally) ready and gets installed at the top of the forward. |

If the worker hasn't finished by the 1s poll, the previous layout is kept and the task is picked up on a later forward.

### Measurement — you need a warmup, for three reasons

1. **History warm-up (5 iters).** For iters 0-4 the rolling window is shorter than 5, so early plans are computed on a smaller sample and are noisier. Skip ≥5 iters.
2. **First-plan latency.** The first-ever plan pays worker-process startup + initial file loads ([async_linear_programming.py:26-33](galvatron/core/runtime/moe/prefetch/async_linear_programming.py#L26-L33)). That can push the first applied layout a step or two later than steady-state cadence would suggest.
3. **Migration cost at commit.** When a new layout is installed, FSDP has to gather expert weights under the new `global_placement` on its next all-gather. The first post-re-layout step is slower than steady state, so averaging over it biases results the wrong way.

### Practical recipe
- Drop the first **~10-20 iterations** from timing averages.
- Use the same warmup cutoff for baseline and LAER.
- Report mean iteration time over a long enough tail for the layout to have settled.
- Optionally log how often `_process_lp_result` actually mutates the placement — once mutations taper off, you're in steady state.

---

## 5. Why This Design Works

- **Load adaptivity.** Stats come from the last 5 iterations, so the layout tracks token-drift across training rather than committing to an offline plan.
- **Topology-aware.** The `/8` node grouping in `get_greedy_placement` and the `v_intra` / `v_inter` split in the cost model reflect the A100 4×8 cluster (NVLink intra-node, IB inter-node).
- **Cheap online solver.** Replica budgeting + LPT + cost eval are all ~`O(n_expert · n_device)` with small constants. The C++ planner finishes in tens of ms — easily hidden by one MoE step's compute, enabling per-layer re-planning without regret.

---

## 6. Key File Index

| Component | File | Lines |
|---|---|---|
| Training loop | [galvatron/models/moe/train_dist.py](galvatron/models/moe/train_dist.py) | 29-99 |
| MoE layer fwd/bwd + `sync_lp_solver` call | [galvatron/models/moe/MoEModel_tensor_parallel.py](galvatron/models/moe/MoEModel_tensor_parallel.py) | 332-343 |
| Load stats collection, rolling window | [galvatron/core/runtime/moe/smart_routing.py](galvatron/core/runtime/moe/smart_routing.py) | 137-178 |
| Plan submission (`sync_lp_solver`) | [galvatron/core/runtime/moe/smart_routing.py](galvatron/core/runtime/moe/smart_routing.py) | 514-536 |
| Async worker | [galvatron/core/runtime/moe/prefetch/async_linear_programming.py](galvatron/core/runtime/moe/prefetch/async_linear_programming.py) | 20-167 |
| Greedy C++ planner | [csrc/greedy_balancer.cpp](csrc/greedy_balancer.cpp) | 141-506 |
| Prefetch + commit (`_async_lp_prefetch_logic`, `sync_htod`) | [galvatron/core/runtime/moe/smart_routing.py](galvatron/core/runtime/moe/smart_routing.py) | 588-630 |
| Routing kernels | [galvatron/core/runtime/moe/fused_kernel.py](galvatron/core/runtime/moe/fused_kernel.py) | — |
