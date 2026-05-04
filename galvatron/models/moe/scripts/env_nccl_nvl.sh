#!/usr/bin/env bash
# NCCL tuning for production training on the hetu host.
# Source this BEFORE invoking train.sh to enable NVLink P2P on TP groups while
# avoiding the channel-count blowup that OOMs the largest configs.
#
# Usage:
#   source galvatron/models/moe/scripts/env_nccl_nvl.sh
#   bash galvatron/models/moe/scripts/train.sh ...
#
# Hardware context (RTX A6000 4-GPU node, NVLink pairs (0,1) and (2,3)):
#   - TP groups {0,1} and {2,3} are NV4 NVLink — high-bandwidth hot path.
#   - DP groups {0,2} and {1,3} are PCIe NODE — no NVLink available.
#   - With default NCCL settings the two PCIe DP groups behave asymmetrically
#     (DP-A wedges on first 2-rank collective). See
#     doc/profile_computation_frozen_fixes.md Open 1 / Fix 9.
#
# Performance evaluation (bsz=4, layernum=2 inner configs):
#                            DISABLE   NVL+MAX4
#     tp=1 dp=4 (no TP coll)  0.114 s   0.115 s    1.00x
#     tp=2 dp=2 (NVLink TP)   0.312 s   0.179 s    1.74x
#     tp=4 dp=1 (NVLink TP)   0.792 s   0.704 s    1.12x
#
# So this env gives a 1.1x-1.7x speedup on TP-heavy collectives. The profiling
# sweep (profile_computation_frozen.sh) keeps NCCL_P2P_DISABLE=1 because it
# intentionally exercises memory-edge configs — see that script for details.

# Force P2P only on NVLink pairs. Without this, NCCL also tries P2P on the
# PCIe-NODE DP links, which deterministically wedges DP group {0,2} on this
# host while DP group {1,3} completes. Restricting to NVL keeps the TP groups
# on their native NVLink (~2x throughput vs SHM fallback) while sending the
# cross-pair DP traffic through SHM/socket — which they would have used
# anyway since they have no NVLink.
# export NCCL_P2P_LEVEL=NVL

# Cap NCCL channels to 4 per multi-rank comm. With NVL enabled, NCCL's
# default selects 8 channels per pair (vs 4 under DISABLE), allocating ~16 MB
# of extra GPU-resident scratch per comm. Across ~24 comms per training run
# (root FSDP wrap + per-layer FSDP wraps + WORLD + TP + DP + EP), that adds
# up to ~400 MB and OOMs the largest model configs. Setting MAX_NCHANNELS=4
# matches the channel count to DISABLE while keeping NVLink P2P enabled.
# Empirically this fits on 48 GB cards for normal training (only the
# profiler's deliberately-edge configs still OOM).
# export NCCL_MAX_NCHANNELS=4

# Don't override NCCL_BUFFSIZE here — the default (4 MB) is correct.
# Halving it to 2 MB makes tp=4 ~13 % slower (more comm rounds) and triggers
# a retry-spiral instead of a clean OOM on the largest configs (10k+
# enqueue.cc:115 warnings). Documented in
# doc/profile_computation_frozen_fixes.md.
