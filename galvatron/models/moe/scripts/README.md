# MoE scripts — index

First stop on a fresh instance. Maps every script in this directory to
its role in the cost-model calibration + search workflow. The end-to-end
recipe is in [`doc/qwen3_8xa100_workflow.md`](../../../../doc/qwen3_8xa100_workflow.md);
this README is the at-a-glance reference for "what does each file do?".

## Cost-model calibration + search workflow

Numbered steps mirror the workflow doc.

| Step | Script | Role |
| --- | --- | --- |
| 2 | `../../../profile_hardware/scripts/profile_hardware.sh` | NVLink/PCIe bandwidth profile (one-time per cluster) |
| 3 | `profile_computation.sh` | Per-component fwd-only compute profile (FSEP-off, three-pass attn/mlp/all) — feeds the FSEP smart-routing solver |
| 4 | `profile_memory.sh` | Per-component memory profile (FSEP-off) |
| 5 | `profile_embedding_lmhead.py` | Embedding + LM-head standalone time/memory profile |
| ~~6~~ | ~~`profile_computation_frozen.sh`~~ | **Legacy** — moved to `_legacy/`. FSEP-on per-block fwd-only compute profile. Subsumed by Step 8: runtime_profile captures FSEP-on full-iter measurements at every (tp, ep, micro_bsz, fsep=on, pp) shape, and `fsep_overhead_profile` is built from those FSEP-on/off pairs (not from this script's output). |
| ~~7~~ | ~~`profile_memory_frozen.sh`~~ | **Legacy** — moved to `_legacy/`. FSEP-on memory profile. Output (`memory_profiling_*_fsep.json` and `memory_profiling_*_tp{T}_ep{E}_fsep.json`) was never read by the cost model — every `memory_profile` lookup in `intra.py` uses non-fsep filenames. FSEP-on memory predictions are sourced from runtime_profile (full-iter `cuda_peak_mb` with α + β · N fitting) and `fsep_overhead_profile`. |
| 8 | `cost_model_real_test.sh` | **Calibration sweep** — per-config (pp, tp, ep, dp_mode, micro_bsz, fsep, profile_unit) full-iter measurements. Per-component fan-out (all/attention/mlp); covers gbsz ∈ {4, 2, 1} (gap-fill folded in); sweeps `NUM_LAYERS_LIST="2 4"` for the α + β × N fit |
| 8b | `cost_model_real_test_chunks2.sh` | **Per-microbatch overhead calibration** — runs the matrix at chunks=2 to derive `chunks_overhead_profiling_*.json` |
| ~~8c~~ | ~~`cost_model_real_test_gap_fill.sh`~~ | **Folded into Step 8** — moved to `_legacy/`. The gbsz ∈ {1, 2} entries are now part of `DEFAULT_CONFIGS_BASE` in `cost_model_real_test.sh`, so the main sweep covers shape gaps directly. zero3 entries dropped on merge (consistent with the zero2sdp-only directive). |
| 9 | `profile_cost_model_terms.py` | Aggregator: turns step-8 logs into `optimizer_step_profiling`, `runtime_profiling`, `fsep_overhead_profiling`, `chunks_overhead_profiling` JSONs |
| 10 | `cost_model_alpha_beta.py` | α + β × N extrapolation drift check |
| 10 | `cost_model_pp_drift.py` | PP critical-path drift check |
| 10 | `cost_model_split_regression.py` | Symmetric identity / regression invariants (24 checks) |
| 10 | `cost_model_drift.py` | Per-shape drift table |
| 11 | `cost_model_search.py` | Brute-force search over the (pp, tp, ep, dp, dp_mode, fsep) layout space |
| 11b | `cost_model_search_micro_bsz_sweep.py` | Compares optimal config across micro_bsz ∈ {1, 2, 4} at fixed gbsz |
| 11c | `validate_top_config.sh` | End-to-end validation: real training run at the search's top config; compares measured iter_ms against predicted |

## Helpers

| Script | Role |
| --- | --- |
| `detect_p2p_island_size.py` | Probes the NVLink-island topology (used by `cost_model_real_test.sh` to decide when to set `NCCL_P2P_DISABLE`) |
| `env_nccl_nvl.sh` | Sourceable env block for NVLink-friendly NCCL settings |
| `generate_static_input.sh` | Regenerates the per-bsz frozen-input tensors under `static_inputs/` |

## Diagnostic / one-off (kept for reference)

These ran during Phase 0 of the asymmetric search prerequisite work
(see [`doc/asymmetric_search_plan.md`](../../../../doc/asymmetric_search_plan.md)).
They're not part of the standard workflow — keep on hand for re-investigation.

| Script | What it audits |
| --- | --- |
| `audit_pp_peak_memory.py` | PP-depth-aware peak memory; validates the `natural_n_behind` reserve and predicted-vs-measured PP=2 peak |
| `analyze_chunks_overhead.py` | Per-shape decomposition of chunks-overhead residuals (slope − bottleneck) into analytical comm + scheduler/Python residual |
| `hybrid_overhead_smoke_test.py` | Held-out smoke test for an analytical+regime-constant alternative to the per-shape chunks_overhead empirical calibration |

## Tests

| Script | Coverage |
| --- | --- |
| `test_per_component_aggregator.py` | Equivalence tests for `profile_cost_model_terms.py` — covers shape-axis, FSEP-pair, DP-mode-isolation invariants. Run via `python -m pytest test_per_component_aggregator.py` (needs pytest in the Python env) |

## Training launchers (separate from cost-model workflow)

`train.sh`, `train_ablation.sh`, `train_convergence.sh`, `train_dist_fsep.sh`,
`train_dist_fsep_32gpus.sh` — end-user training entry points, not part of
the cost-model pipeline. Use directly when running production training.

## Legacy

`_legacy/` contains older comparison/baseline tools (`cost_model_collect.py`,
`cost_model_compare.py`, `cost_model_search_compare.py`, `cost_model_sweep.py`,
`profile_computation_full.sh`) that the production workflow doesn't use but
that older docs (`doc/cost_model_optimizations.md`, `doc/cost_model_guide.md`)
still reference. Kept on hand for context, not part of the recommended path.

Also in `_legacy/`:

- `profile_computation_frozen.sh` — was Step 6 (FSEP-on per-block fwd-only
  compute). Removed from the workflow because:
  1. The runtime calibration sweep (Step 8) now produces FSEP-on full-iter
     measurements at every shape, with an `α + β × N` fit covering layer-count
     extrapolation. The PP shortcut path uses runtime_profile directly.
  2. `fsep_overhead_profile` is built from FSEP-on/off pairs in
     `runtime_profiling`, not from `computation_profiling_*_tp{T}_ep{E}.json`
     (which is what this script produced).
  3. `v_comp` (greedy balancer) and the FSEP smart-routing solver read the
     FSEP-**off** computation profile (single-MLP per-token cost, FSEP-invariant).
  The cost-model loader still has the `_tp{T}_ep{E}`-suffixed lookup path
  (`intra.py:_compute_profile_path`) as a tertiary fall-back, so this script
  can be re-run if the search ever needs to query shapes outside the
  calibration matrix.

- `cost_model_real_test_gap_fill.sh` — was Step 8c (gbsz ∈ {1, 2}
  shape coverage). The zero2sdp entries are now folded into
  `DEFAULT_CONFIGS_BASE` in `cost_model_real_test.sh`, eliminating the
  wrapper. zero3 entries from the original gap-fill matrix were dropped
  on merge (consistent with the project decision to stop extending
  zero3; existing zero3 entries in the runtime profile remain valid
  but no new ones get produced). Re-run the legacy script if you need
  zero3 gap-fill data; otherwise the merged main sweep covers
  everything.

- `profile_memory_frozen.sh` — was Step 7 (FSEP-on memory profile).
  Removed from the workflow because:
  1. The cost model has zero code paths that load `_fsep`-suffixed memory
     JSONs. Both `memory_profile` lookup paths in `intra.py` (the main
     loader at lines 85-102 and `_raw_memory_path` at 521-530) build
     non-fsep filenames; grep for `_fsep.json` in `cost_model/` returns
     no matches. The output files this script produced were never read.
  2. runtime_profile captures FSEP-on `cuda_peak_mb` / `params_mb` /
     `optimizer_mb` / `activation_peak_mb` at every shape with
     `α + β × N` fitting, covering production num_layers extrapolation.
  3. `fsep_overhead_profile` (built from runtime FSEP on/off pairs)
     supplies per-MoE-layer memory delta for the analytical fall-back.
  Pre-existing `_fsep`-suffixed JSONs in `configs/` are harmless — leave
  them in place or `git rm` separately.
