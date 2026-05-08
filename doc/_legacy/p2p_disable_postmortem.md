# Post-mortem: cross-NUMA NCCL ring-construction failure on PCIe-A100

**Date**: 2026-05-04
**Affected workflows**: `profile_computation_frozen.sh`, `profile_memory_frozen.sh`,
`cost_model_real_test.sh` (FSEP-on profiling and the real-measurement
calibration sweep) on 4-GPU PCIe-A100 hosts whose 4 GPUs split into two
2-GPU NVLink islands without a cross-island P2P fabric.
**Resolution**: launch-time auto-detect of the host's P2P-island size
and conditional `NCCL_P2P_DISABLE=1` for inner configs whose raw DP
group spans both islands.

---

## TL;DR

Half a 4-GPU sweep crashed during NCCL initialisation with

```
NCCL WARN Error : ring 1 does not contain rank 0
NCCL WARN Error : ring 1 does not loop back to start (3 != 1)
```

followed by SIGSEGV inside FSDP's init-time barrier. Triggered for every
inner config whose raw DP group spans both NVLink islands (raw_dp = world_size /
(pp × tp) ≥ 3). NCCL's multi-channel ring builder constructs an
inconsistent layout when the cross-island legs have no P2P route.

We compared three NCCL env-var workarounds on the affected dp=4 configs.
`NCCL_P2P_DISABLE=1` is the only one that fixes every config, with a
0–15 % perf win over `NCCL_MAX_NCHANNELS=1`. The fix is gated on a
launch-time topology probe so the rest of the sweep keeps default NCCL
behaviour and intra-island NVLink P2P.

---

## 1. Topology

The host's P2P matrix from `nvidia-smi topo -m` and
`torch.cuda.can_device_access_peer`:

```
       GPU0   GPU1   GPU2   GPU3
GPU0    -    NV12   SYS    SYS
GPU1   NV12   -     SYS    SYS
GPU2   SYS   SYS    -     NV12
GPU3   SYS   SYS   NV12    -
```

`SYS = cross-NUMA, no P2P route`. `can_device_access_peer(0, 2)` returns
False. So the 4 GPUs form **two disjoint P2P islands** {GPU0, GPU1} and
{GPU2, GPU3}; any 4-rank collective must cross the gap on SHM.

## 2. Symptom

Inner configs of `profile_computation_frozen.sh` Step 6 (FSEP-on
computation profile) crashed during NCCL warm-up for every shape with
raw DP ≥ 3. Stack trace excerpt:

```
hetml-a100:75162:75162 [0] graph/rings.cc:38 NCCL WARN Error : ring 1 does not loop back to start (1 != 0)
hetml-a100:75164:75164 [2] graph/rings.cc:38 NCCL WARN Error : ring 1 does not loop back to start (-1 != 2)
...
Fatal Python error: Segmentation fault
File ".../torch/distributed/distributed_c10d.py", line 3698 in barrier
File ".../galvatron/core/runtime/moe/prefetch/fsdp_patch.py", line 164 in _new_init
File ".../torch/distributed/fsdp/fully_sharded_data_parallel.py", line 487 in __init__
```

Channel 0 of the all-reduce ring closes; channel 1's layout is internally
inconsistent (rank 1's `next` pointer is the −1 sentinel, while ranks 0,
2, 3 all point at rank 1) — NCCL fails its ring validation and the
process aborts when the FSDP init-barrier issues its first all-reduce.

## 3. Why the failure is layout-shape-dependent

NCCL chooses the number of ring channels based on the buffer size of
the collective. Small buffers → 1 channel (channel 0 alone closes
fine); larger buffers → ≥ 2 channels. With our 2×2-island topology:

- **Channel 0** is laid out NVLink-first within each island, then SHM
  across the gap. Builds and closes fine.
- **Channel 1** uses a rotated layout. NCCL's builder assumes the
  P2P-edges set is symmetric and dense enough to support multiple
  ring rotations. With the cross-island gap, it elides the missing
  edges silently and ends up with a ring where one rank has no
  successor.

That explains why **TP=4 always worked** in our sweep but **DP=4 always
failed**:

| Inner config | raw_dp = world/(pp×tp) | Per-iter all-reduce buffer | NCCL channels | Outcome |
|---|---|---|---|---|
| pp=1 tp=4 dp=1 | 1 | small (TP-sharded activation) | 1 | ✅ closes |
| pp=1 tp=2 dp=2 | 2 | within-island, small | 1 | ✅ closes |
| pp=1 tp=1 dp=4 | 4 | full-grad bucket | 2 | ❌ ring 1 fails |
| pp=1 tp=1 ep=2 | 4 (raw) | FSDP init-barrier on raw-DP group | 2 | ❌ ring 1 fails |

Even at ep=2 the failure persists: FSDP wraps parameter groups using
the **raw DP group of size `world/(pp×tp)`**, not the EP-sharded
`world/(pp×tp×ep)`. The init-time barrier therefore runs on a 4-rank
group regardless of ep, and that's the group that hits the multi-channel
ring construction failure.

## 4. Workarounds tested

We re-ran the 6 ring-spanning configs (pp=1 tp=1 × ep ∈ {1, 2, 4} ×
num_hidden_layers ∈ {2, 4}, all bsz=4) under three NCCL regimes.
Numbers are `[stage_time] fwd_bwd_ms` averaged over iters [10, 20).

| ep | cap | nl | `NCCL_MAX_NCHANNELS=1` | `NCCL_P2P_DISABLE=1` | `NCCL_IGNORE_DISABLED_P2P=1` |
|---|---|---|---|---|---|
| 1 | 128 | 2 | 101.78 | **98.57** | ❌ FAIL (Msg truncated 512 vs 256) |
| 1 | 128 | 4 | **164.08** | 169.26 | ❌ FAIL (ring 1) |
| 2 | 64 | 2 | 84.51 | **81.75** | ❌ FAIL (ring 1) |
| 2 | 64 | 4 | 141.77 | 128.85 | **111.67** |
| 4 | 32 | 2 | 78.58 | 69.37 | **64.78** |
| 4 | 32 | 4 | 142.71 | 120.70 | **105.43** |

**Verdicts:**

- `NCCL_MAX_NCHANNELS=1` — works on every config. Caps NCCL to a single
  ring; channel 1 never gets built. Keeps intra-island NVLink P2P.
  Slowest of the three because the single ring is bottlenecked by the
  SHM hops.
- `NCCL_P2P_DISABLE=1` — works on every config. Forces SHM uniformly,
  loses intra-island NVLink. Surprisingly the **fastest reliable
  option**: 0–15 % faster than MAX_NCHANNELS=1 on these dp=4 configs
  because two parallel SHM channels beat a single mixed NVLink+SHM
  ring at the message sizes we measured (full-grad-bucket all-reduce
  on 1.25 B params per layer of model state).
- `NCCL_IGNORE_DISABLED_P2P=1` — partial. Intended to skip topology
  validation on disabled-P2P edges, but on NCCL 2.17.1 it doesn't help
  in the layout-construction step (where our failure actually happens).
  Half the configs still crash. When it works (3/6 configs), it's the
  fastest of the three because NCCL keeps multi-channel + NVLink P2P
  enabled. Not reliable enough to ship.

Other things we considered but did not adopt:

- Forcing `NCCL_P2P_LEVEL=NVL` — caused the same ring-1 failure (this
  was actually the original setting that surfaced the bug).
- Unsetting `NCCL_P2P_LEVEL` (auto-discovery) — same failure; the auto-
  discovered transport list is identical to the NVL-explicit one.
- `NCCL_ALGO=Tree` — would side-step ring construction but degrades
  large-message all-reduce throughput across the board.

## 5. Decision

We auto-detect the host's P2P-island size at launch time and
conditionally apply the workaround:

1. **`galvatron/models/moe/scripts/detect_p2p_island_size.py`** — walks
   `torch.cuda.can_device_access_peer` over all pairs, finds the
   largest connected component, prints its size.
2. **`profile_computation_frozen.sh`, `profile_memory_frozen.sh`,
   `cost_model_real_test.sh`** — at the top of each script call the
   detector and export `GALVATRON_P2P_ISLAND_SIZE=<n>` if the island
   size is smaller than the world.
3. **`_maybe_cap_nccl_channels()` in `model_profiler.py`** — for each
   inner launch, if `GALVATRON_P2P_ISLAND_SIZE` is set and
   `raw_dp = world_size / (pp_deg × tp_deg) > island_size`, prepend
   `NCCL_P2P_DISABLE=1 ` to the inner CMD.

Configs whose raw_dp fits inside an island stay on default NCCL
transports (NVLink P2P intact). Configs that need to span the gap pay
the SHM-only cost — but they were going to lose intra-island NVLink in
any working workaround.

The post-fix ring-spanning launches show up in logs prefixed with
`NCCL_P2P_DISABLE=1 timeout --kill-after=...` so it is easy to grep for
which inner configs took the workaround.

## 6. What still needs doing

- The detector currently only checks **pairwise** P2P. On larger,
  more heterogeneous fabrics (e.g. 4-GPU islands with partial NVLink),
  the largest-connected-component logic is a coarse approximation.
  Refine if/when we run on those.
- We didn't test NCCL > 2.17.1 in this environment. Newer NCCL might
  fix the multi-channel ring construction directly, in which case the
  workaround can be retired. Re-test on the next NCCL bump.

## 7. Reproduction

To reproduce the original failure:

```bash
# Force the buggy regime by setting NVL explicitly:
export NCCL_P2P_LEVEL=NVL
unset GALVATRON_P2P_ISLAND_SIZE
bash galvatron/models/moe/scripts/profile_computation_frozen.sh
# Crashes at the first pp=1 tp=1 inner config with the ring-1 error.
```

To verify the fix:

```bash
# Default (auto-detected) regime:
bash galvatron/models/moe/scripts/profile_computation_frozen.sh
# All 22 inner configs complete; the 6 raw_dp=4 launches show
# NCCL_P2P_DISABLE=1 prepended in the log.
```

## Appendix: log evidence

Original failure (Step 6, pp=1 tp=1 ep=1, NCCL_P2P_LEVEL=NVL):

```
hetml-a100:69965:69965 [1] graph/rings.cc:38 NCCL WARN Error : ring 1 does not loop back to start (3 != 1)
hetml-a100:69967:69967 [3] graph/rings.cc:51 NCCL WARN Error : ring 1 does not contain rank 0
hetml-a100:69966:69966 [2] graph/rings.cc:51 NCCL WARN Error : ring 1 does not contain rank 0
hetml-a100:69964:69964 [0] graph/rings.cc:38 NCCL WARN Error : ring 1 does not loop back to start (2 != 0)
[2026-05-04 17:40:52] torch.distributed.elastic.multiprocessing.api: [ERROR]
  failed (exitcode: -11) local_rank: 2 (pid: 69966) of binary: /opt/conda/bin/python
```

Post-fix (Step 6 final attempt, raw_dp=4 launches with prepended env var):

```
[p2p] detected island_size=2 world=4; will cap NCCL_P2P_DISABLE=1 for dp>island launches
NCCL_P2P_DISABLE=1 timeout --kill-after=30 900 torchrun ... --global_tp_deg 1 ... --global_ep_deg 1 ...
Already written profiled time into config file ..._tp1_ep1.json!
```
