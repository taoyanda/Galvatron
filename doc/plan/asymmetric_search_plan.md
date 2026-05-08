# Asymmetric Search Plan

Decision plan for adding an asymmetric per-stage layer-split search to the MoE
cost model: each PP stage may carry a different number of attention vs expert
blocks, with `±1` layer moves as the primary search perturbation.

Activation memory under PP is called out as a prerequisite — peak activation
memory at the bottleneck stage scales with PP depth (1F1B keeps `≈pp`
microbatches' activations live), and the current cost model's PP-aware path
must be verified before any asymmetric search is meaningful.

## Status (as of latest revision)

**Phase 0 (prerequisite)** — **all sub-items complete**:

- 0a (audit `peak_memory_mb` vs PP depth): ✅ shipped via
  `galvatron/models/moe/scripts/audit_pp_peak_memory.py`. Verified
  `extra_reserve_mb = num_stages_behind × per_microbatch_act_mb` is
  linear and matches 1F1B critical path.
- 0b (auto-derive `natural_n_behind` from `(pp, num_microbatches)`):
  ✅ landed in `pp.py` shortcut path. Adds the right reserve when
  the query's num_microbatches > the calibration's.
- 0c (validate PP=2 peak prediction): ✅ predicted vs measured PP=2
  peak agrees within 0.5 % across 4 shape pairs; chunks=32 validation
  agrees within 0.4 %.
- 0e (fix iter_ms scaling in PP shortcut): ✅ Alpa-style
  `bottleneck × (num_mb − 1) + Σ stages + opt`. Per-stage individual
  prediction (no first/middle/last representatives).
- 0f (model per-microbatch overhead): ✅ shipped as the
  `chunks_overhead_profiling_*.json` empirical path. Closes the
  iter_ms gap from +34 % to −2.3 % at the validated top config
  (gbsz=128 chunks=32). The analytical-comm + per-(PP, FSEP) regime
  constant alternative was investigated
  (`hybrid_overhead_smoke_test.py`) and shown to close only ~4 pp of
  the gap on its own — per-shape empirical anchor is structurally
  necessary.

**Phase 1 / Phase 2**: not started. Phase 0 unblocks both — the
PP-aware time and memory predictions are now validated end-to-end.

## Phase 0 (prerequisite): PP-depth-aware peak activation memory

### Problem

Under 1F1B with depth `pp`, the bottleneck stage holds `≈ pp − stage_idx`
microbatches' activations live at peak. The bottleneck is stage 0 with `pp`
microbatches stacked. Today the cost model:

- Calibrates `cuda_peak_mb` per shape at the PP value it was profiled at
  (matrix has PP ∈ {1, 2} → entries at `pp=1` and `pp=2`).
- Adds `num_stages_behind × per_microbatch_activation_mb` as
  `extra_reserve_mb` for stages with `n_behind > 0`, where
  `per_microbatch_activation_mb` is computed **analytically** from the cost
  model's per-layer activation formula.

Two failure modes are possible:

1. **Analytical `per_microbatch_activation_mb` is inconsistent with the
   empirical activation slope** (different recompute interactions, different
   SP layout, etc.) → predicted peak drifts from real peak as `pp` grows.
2. **Extrapolation to PP not in the calibration matrix** (e.g., PP=4 or PP=8
   on a larger box) silently uses the analytical formula end-to-end with no
   empirical anchor.

### Options

| | Approach | Pros | Cons |
|---|---|---|---|
| **0a** | Audit + regression test: pick a known shape, compare cost-model `peak_memory_mb` predictions across PP ∈ {1, 2, 4, 8} against an analytical 1F1B reference. Confirm current `extra_reserve_mb` computation is right. | Cheap, defensive | Doesn't catch model drift if analytical reference is also wrong |
| **0b** | Extract `per_microbatch_activation_mb_measured` empirically from the PP=1 runtime profile (`activation_peak − model_state_mb`) and use **it** in `extra_reserve_mb`, instead of the analytical formula. | Anchors the multiplier on a real measurement | Still extrapolates by linear scaling; assumes 1F1B microbatches stack uniformly |
| **0c** | Cross-check 0b's prediction at PP=2 (which we *do* calibrate) against the measured PP=2 peak. The residual = analytical-model error budget. | Direct empirical validation of the PP scaling factor | Only validates ratio between PP=1 and PP=2; PP=4+ stays extrapolated until a larger box is available |
| **0d** | Add `pp_depth_factor` to the runtime profile schema: an explicit `(pp, alpha_pp, beta_pp)` curve fit so the cost model can interpolate to arbitrary PP without re-running the fixed-PP analytical path. | Cleanest separation: empirical curve replaces analytical formula entirely | Needs ≥2 PP samples per shape (we have those for some shapes, missing for others) |

**Recommendation:** **0b + 0c together.** Replace the analytical
`per_microbatch_activation_mb` with the empirical PP=1 value (0b), then run a
regression check against PP=2 to bound the error (0c). 0d is the principled
long-term fix but requires a wider PP matrix than we can calibrate on 4 GPUs.

### Acceptance criteria

- A small test (synthetic shape, no GPU) that asserts
  `cost_model.peak_memory_mb(pp=N)` increases linearly in `N` with the right
  slope.
- Predicted PP=2 peak matches measured PP=2 peak within X% on at least 4
  shapes from the current calibration.

---

## Phase 1: Per-component activation memory

### Problem

The asymmetric search wants `peak(n_attn, n_expert)` for non-symmetric
splits. Today: analytical params/optim per component (clean), single
empirical activation slope per shape.

### Options

| | Approach | Pros | Cons |
|---|---|---|---|
| **1a** | Stay analytical. Per-attention-layer and per-expert-layer activation memory derived from `hidden_size`, `intermediate_size`, `num_local_experts`, recompute, SP. | Zero new calibration; DP-mode-invariant by construction; no artifact risk | Ignores empirical recompute / SP / FSEP interactions; calibrated activation is a single combined β |
| **1b** | Add a small **asymmetric multi-N** sweep to `cost_model_real_test.sh`. For each focus shape, run e.g. `(n_attn=4, n_exp=2)` and `(n_attn=2, n_exp=4)` alongside the existing symmetric `(n_attn=4, n_exp=4)`. Solve `α + β_attn·n_attn + β_exp·n_exp = act_peak` per shape from 3+ measurements. | Empirical β_attn vs β_expert with all interactions baked in | Adds N runs to the matrix; trainer must support asymmetric `num_attention_layers ≠ num_expert_layers` (verify it does) |
| **1c** | Hybrid: analytical β + empirical scale factor (single multiplicative correction per shape from full-iter calibration). | Cheap empirical anchor; no asymmetric runs | Doesn't separate β_attn from β_expert — only validates the sum |
| **1d** | Stripped-model deltas (`unit=attention` and `unit=mlp` peaks). | Reuses logs we already have | **Embedding artifact leaks in** (~622 MB phantom under zero2sdp from unsharded root embedding); rejected per prior discussion |

**Recommendation:** Start with **1a** (unblocks Phase 2 immediately), upgrade
to **1b** if Phase 2's validation reveals systematic prediction error. 1d
stays off the table.

### Acceptance criteria

- `cost_model.predict_activation_mb(n_attn=4, n_exp=4)` matches the symmetric
  runtime_profile within X% (basic sanity).
- (1b only) Predicted `act_mb(n_attn=3, n_exp=5)` matches a held-out
  asymmetric calibration run within X%.

---

## Phase 2: Asymmetric search algorithm

### Problem

Decision variable: per-stage `(n_attn[i], n_expert[i])` for `i ∈ [0, pp)`,
with constraints:

- `Σ n_attn[i] = N_attention_total`, `Σ n_expert[i] = N_expert_total`
  (model-fixed).
- Per-stage
  `peak_memory(n_attn[i], n_expert[i], stage_idx=i, pp) ≤ device_memory`.
- Objective: minimize `total_iter_ms` (PP-bubble-aware).

The specified search move is **±1 layer per perturbation**.

### Options

| | Approach | Pros | Cons |
|---|---|---|---|
| **2a** | **Local hill-climb on ±1 moves.** Start symmetric. At each step try every "move 1 attention from stage `i` to stage `j`" + "move 1 expert from stage `i` to stage `j`" perturbation. Accept if feasible (memory) and reduces predicted iter_ms. Stop at local optimum. | Matches the natural move set; small search space; trivial to implement | Local optima — may miss a globally-better split that requires a 2-layer move |
| **2b** | **Bounded enumeration with pruning.** Enumerate all `(n_attn[0..pp-1])` tuples with `Σ = N_attn` and similarly for expert. For PP=2 and N=4, that's 5·5 = 25 candidates — exhaustive and fast. Prune by memory feasibility, score by iter_ms, return top-k. | Globally optimal (within precision of the model); easy to validate against 2a | Blows up combinatorially for larger PP × N |
| **2c** | **2-stage: enumerate + local refine.** Do 2b for the (n_attn_per_stage, n_expert_per_stage) split, then run 2a-style ±1 moves within each accepted candidate to find sub-stage improvements. | Catches what 2a misses while staying tractable | Slightly more code |
| **2d** | **LP relaxation + rounding.** Treat splits as continuous, solve a min-bottleneck LP, round to integers, repair feasibility. | Tighter when bubble + memory are both binding | Heavier machinery; the integrality gap on small problems is what 2c catches anyway |

**Recommendation:** **2b for PP ≤ 4** (exhaustive is easy). Add **2a** as a
tighten-up pass *only if* 2b shows ties that 2a can break. 2c/2d is overkill
at this scale.

### Acceptance criteria

- Symmetric layout is the optimum on a model where it should be (e.g.,
  balanced attention/expert costs, no memory pressure asymmetry).
- A model with skewed attention/expert costs picks the predicted asymmetric
  layout (regression test on a fixture).
- Predicted top-1 layout's iter_ms matches a measured run within X%.

---

## Sequencing

```
Phase 0 (prereq, blocking) ──┬── Phase 1a (analytical) ── Phase 2b (search)
                             │                              │
                             └── Phase 1b (empirical) ──────┘  (upgrade if Phase 2 shows drift)
```

Phase 0 first. Phase 1a unblocks Phase 2 immediately; Phase 1b is a
deferrable upgrade. Phase 2 starts with 2b's enumeration.

## Risks

- **Trainer support for `n_attn ≠ n_expert`** — needs verification before
  Phase 1b is feasible. Galvatron's MoE model class may assume 1:1 layer
  mapping in places; surface that early.
- **`num_stages_behind` in the cost model already exists** but its
  derivation is buried in `pp.py:300+` — Phase 0a's audit should make sure
  the formula matches the 1F1B critical path, not just match a previous
  test.
- **Memory feasibility check at search time** must use the *first stage*
  peak (the bottleneck), not the average — easy to get wrong by aggregating
  across stages.
