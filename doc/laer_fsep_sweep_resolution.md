# LAER + FSEP profile sweep resolution

This document summarizes the work to make `profile_computation_frozen.sh`
produce a complete computation-profile JSON for the Mixtral-8x7B-e8k2 model
under the LAER (Load-Adaptive Expert Re-layout) solver and FSEP (Fast Static
Expert Parallelism) paradigm on the hetu host (4× RTX A6000, 48 GB each;
NVLink pairs `(0,1)` and `(2,3)`; cross-pair PCIe NODE).

## Goal

Given Galvatron's hybrid-parallel runtime patched with FSEP smart routing
(`MoEAlltoAllSmartTokenDispatcher`) and the async LP solver
(`AsyncLinearProgrammingSolver`), drive the full sweep:

```
EP_CAP_TUPLES = [(1,8), (2,4), (4,2)]
bsz   ∈ {1, 2, 3, 4}
tp    ∈ {1, 2, 4}        (filtered by max_dp compatibility)
layernum ∈ {2, 4}
```

End-to-end success means producing the merged
`computation_profiling_bf16_mixtral-8x7b-e8k2_seqlen4096{,_tpX_epY}.json`
files that the LAER cost model consumes (`solver.py:41`).

## Fixes applied

The fixes are layered. Each addresses a distinct failure mode that surfaced
when the previous one was lifted.

### 1. `--use_fsep` / `--use-fsep` argparse mismatch

`profiler.py` only accepts the underscore form. The original frozen script
passed the hyphen form and never reached CUDA work. Renamed in the script.

### 2. torch._inductor warm-pool stealing CUDA contexts

Megatron's `@torch.compile def gelu_impl` triggers `AsyncCompile.warm_pool()`
at import time, forking 32 workers under the parent profiler.py. The
worker FDs alias `/dev/nvidia*` and the next torchrun launch sees
`set_device` fail with `device(s) is/are busy or unavailable`. Fixed via
`TORCHINDUCTOR_COMPILE_THREADS=1` in the script.

### 3. Per-iteration timeouts (defense)

Each inner `os.system()` from `model_profiler.py:_launch_*` is wrapped in
`timeout --kill-after=30 ${GALVATRON_PROFILE_INNER_TIMEOUT}`. The outer
shell loop is wrapped in `timeout --kill-after=${OUTER_KILL_AFTER:-120}
${OUTER_TIMEOUT}`. A hung inner config can no longer strand the whole sweep.

### 4. user-NCCL `getcomm` storeKey collision

`csrc/moe_all_to_all_binding.cpp` `HackNCCLGroup::getcomm` broadcasts the
NCCL unique ID through the c10d Store with a fixed key
`"prefetch_all_to_all_comm"`. With multiple disjoint sub-groups (e.g.
`{0,2}` and `{1,3}` under tp=2 dp=2), both groups race on the same store
slot and corrupt each other's IDs — `ncclCommInitRank` succeeds with a
mismatched ID and the first send/recv hangs. Disambiguated with a
caller-supplied `key_suffix` (sorted group ranks), wired through
`fsdp_patch.py`'s call site.

### 5. `MoEEmbeddings_.forward` TP all-reduce drain

Galvatron's embedding wraps Megatron's `VocabParallelEmbedding`, which
queues a TP all-reduce on `self.tp_group`. Without an explicit drain the
all-reduce stays in flight while the next layer's collectives are queued,
causing `tp=4 dp=1` to hang at the first MoE layer's internal sync. Added
`torch.cuda.synchronize()` + `torch.distributed.barrier(group=self.tp_group)`
inside `MoEEmbeddings_.forward` after `embed_tokens(tokens)`. Required.

### 6. atexit + SIGTERM cleanup hooks

`train_dist_random.py` registers an atexit handler and SIGTERM/SIGINT
signal handlers that destroy the user-NCCL singleton, drain the async LP
solver worker pool, and call `destroy_process_group()`. Without this, a
SIGKILL'd inner config left dirty CUDA contexts on the host driver for
~10 minutes (Open 2 below).

### 7. Cascade-prevention: abort the sweep on any hang

`model_profiler.py:_report_inner_rc` raises `SystemExit(2)` on any non-zero
inner rc. The outer shell loop, on receiving non-zero from `python3
profiler.py`, kills stragglers and `exit "${rc}"` instead of marching to
the next (EP, CAP). Combined with Open 2 below, this prevents a single
broken config from corrupting the rest of the sweep.

### 8. MPS bypass

The host runs `nvidia-cuda-mps-server` (pid 9404). Container processes
connect to MPS by default. SIGKILL'd MPS clients leave stale contexts that
block fresh clients on whichever GPUs they touched, returning
`Error 700: illegal memory access` from `cudaGetDeviceCount`. The dirty
state survives `docker restart hetu` (host-level) and decays over tens of
minutes. Fixed by exporting
`CUDA_MPS_PIPE_DIRECTORY=/tmp/no-such-mps` in the script — CUDA fails to
attach to MPS and falls back to direct per-process contexts.

### 9. `_ExecOrderData._check_order` no-op patch (the real Open-1 fix)

Upstream FSDP's first-iteration order-consistency check issues an
`all_gather_into_tensor(group=self.process_group)` from inside
`_pre_forward`. Under tp=2 dp=2, the root FSDP wrap's `process_group` is a
2-rank DP group; this all_gather wedges deterministically on this hardware
(faulthandler stacks pin all 4 ranks at `_runtime_utils.py:425`). The
check is purely a consistency probe, not functional, so we monkey-patch
`_ExecOrderData._check_order` to a no-op in `fsdp_patch.py`.

### 10. `NCCL_P2P_DISABLE=1` for the profiling sweep

After Fix 9, ANY 2-rank NCCL P2P collective hangs on this host —
confirmed with a minimal 4-rank torchrun repro: `dist.barrier` on a
2-rank subgroup wedges with default P2P, completes in <1 s with
`NCCL_P2P_DISABLE=1`. Verified the bug is below the abstraction layer
that `NCCL_PROTO`, `NCCL_LAUNCH_MODE`, `NCCL_IB_DISABLE`,
`NCCL_NET_PLUGIN`, and `NCCL_NET=Socket` can shift around: those four
fix a minimal toy but don't generalize to the full workload.
`NCCL_P2P_DISABLE=1` is the only reliable fix observed for sustained
2-rank P2P traffic. Wired into the profile script.

For *production training* (where the full P2P perf matters), the right
recipe is `NCCL_P2P_LEVEL=NVL` + `NCCL_MAX_NCHANNELS=4`, which preserves
NVLink P2P on TP groups (1.4–1.7× collective speedup) without OOMing on
typical training shapes. Use `source
galvatron/models/moe/scripts/env_nccl_nvl.sh` for that path. The
profiling sweep deliberately exercises memory-edge configs that NVL+MAX4
can't fit on 48 GB, so the sweep keeps `P2P_DISABLE=1`.

### 11. WORLD broadcast input tokens at iter start

`distributed_dataloader` shards data by DP group, so TP-mates within the
same DP group end up with different microbatches. The router/LP solver
respond asymmetrically to the divergent token streams, surfacing as a
NCCL watchdog timeout on the per-iter WORLD barrier around iteration 2.
Cost-model only consumes per-op timing, so collapsing the batch is fine
for profiling: `dist.broadcast(tokens, src=0)` after the dataloader
fetch.

### 12. `smart_routing.preprocess` UnboundLocalError under tp=ep=1

`new_routing_map` / `new_probs` were only assigned inside
`if self.ep_size > 1 or self.tp_size > 1:`. Under tp=1 ep=1 the else
branch left them undefined and `return` raised `UnboundLocalError`.
Initialized to `routing_map, probs` before the branch.

### 13. Profile post-processing tolerates skipped (bsz, tp) cells

`_process_computation_data` iterates the JSON expecting all (bsz, tp) keys
to be present. The inner sweep skips combinations where `max_dp` doesn't
divide bsz evenly (e.g. tp=2 + bsz=3). Patched: skip cells with missing
keys; buffer derived layertype writes and only commit them when all
layernum keys for the bsz are present.

## Verification of FSEP smart routing and LAER solver

Three families of prints emit during the sweep (rank 0 only). Their
presence confirms the FSEP+LAER pipeline is wired and exercising:

| print | when | confirms |
|---|---|---|
| `[fsep_verify]` block | once per inner config, after model build | smart-routing dispatcher class is `MoEAlltoAllSmartTokenDispatcher`; `use_fsep=True`; `solver_enabled=True`; freeze threshold; ep/tp/cap |
| `[solver_init]` | once per inner config, layer 0 | LP solver constructed with config paths, hidden size, expert capacity |
| `[solver_submit] layer=N iter=K` | first 3 iters and every 5th iter, per layer | `submit_lp_optimization` was actually called with a non-empty token-history, i.e. the solver is processing real workload |
| `[layer N] layout frozen at iter K` | once per layer when `solver_iter ≥ laer_freeze_after_iter` | LAER converged and the placement is stationary (required for the cost model to consume timing) |

In a healthy run you should see, for each inner config:

1. A `[fsep_verify]` banner,
2. Then `[solver_init] enabled=True freeze_after_iter=5 ...`,
3. `[solver_submit] layer=K iter=1 hist_tokens_sum=N` for K = 0..L-1,
4. Repeated `[solver_submit]` at iters 2, 3, then 5, 10, 15, 20,
5. `[layer K] layout frozen at iter 5` for K = 0..L-1,
6. Twenty iteration-loss prints, then `Average iteration time is: …`,
7. `Already written profiled time into config file …`.

## Operational notes

- See `galvatron/models/moe/scripts/env_nccl_nvl.sh` for the production
  NCCL recipe (NVL + MAX_NCHANNELS=4) — different envelope than the
  profile sweep.
- See `doc/profile_computation_frozen_fixes.md` for the detailed
  per-fix history and open issues (Open 2: host MPS/driver decay).
- Two memories in `~/.claude/projects/.../memory/`:
  - `env_mps_bypass.md` — mandatory `CUDA_MPS_PIPE_DIRECTORY` env;
    P2P_DISABLE vs NVL trade-off
  - `feedback_hang_stop_restart.md` — operational rule: on any
    sweep-iter hang, abort the whole sweep, restart hetu, wait 10 min
    before retrying, isolate the failing config first
