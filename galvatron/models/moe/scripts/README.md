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
| 6 | `profile_computation_frozen.sh` | FSEP-on per-block compute profile (legacy step; mostly subsumed by step 8 since it also captures FSEP-on) |
| 7 | `profile_memory_frozen.sh` | FSEP-on memory profile |
| 8 | `cost_model_real_test.sh` | **Calibration sweep** — per-config (pp, tp, ep, dp_mode, micro_bsz, fsep, profile_unit) full-iter measurements. Now per-component (fans into all/attention/mlp) |
| 8b | `cost_model_real_test_chunks2.sh` | **Per-microbatch overhead calibration** — runs the matrix at chunks=2 to derive `chunks_overhead_profiling_*.json` |
| 8c | `cost_model_real_test_gap_fill.sh` | Fills shape-coverage gaps at gbsz ∈ {1, 2} (per-rank micro_bsz < 4) |
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
