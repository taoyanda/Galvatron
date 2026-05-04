"""Print the largest P2P-connected island size on this host.

NCCL on PCIe-A100 boxes with split NVLink islands (e.g. {GPU0, GPU1} NV12 +
{GPU2, GPU3} NV12, no cross-island P2P) cannot reliably build multi-channel
rings for groups whose rank count exceeds an island — channel-1 ring
construction fails with "ring N does not contain rank 0" / "ring N does not
loop back to start" regardless of NCCL_P2P_LEVEL. The minimal-impact fix is
to cap NCCL_MAX_NCHANNELS=1 only for those ring-spanning groups; default
NCCL behaviour stays in place for everything else.

This helper detects the topology by walking ``torch.cuda.can_device_access_peer``
across every pair, building an undirected graph, and printing the size of the
largest connected component. When the printed value is less than the world
size, the host has a cross-island gap and any group whose rank count exceeds
the island size needs the workaround.

Usage (called by the profile launch scripts):

    ISLAND=$(python3 detect_p2p_island_size.py)
    WORLD=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
    if [ "$ISLAND" -lt "$WORLD" ]; then
        export GALVATRON_P2P_ISLAND_SIZE=$ISLAND
    fi
"""
from __future__ import annotations

import sys

try:
    import torch
except Exception as e:  # pragma: no cover
    print(f"# detect_p2p_island_size: torch import failed: {e}", file=sys.stderr)
    print(0)
    sys.exit(0)

if not torch.cuda.is_available():
    print(0)
    sys.exit(0)

n = torch.cuda.device_count()
if n <= 1:
    print(n)
    sys.exit(0)

# Adjacency: i ↔ j when both directions of can_device_access_peer agree, plus
# self loops. Asymmetric P2P is unusual but we conservatively require both.
adj = [[False] * n for _ in range(n)]
for i in range(n):
    adj[i][i] = True
    for j in range(i + 1, n):
        try:
            ok = bool(torch.cuda.can_device_access_peer(i, j)) and bool(
                torch.cuda.can_device_access_peer(j, i)
            )
        except Exception:
            ok = False
        adj[i][j] = adj[j][i] = ok

# Largest connected component via DFS.
visited = [False] * n
best = 0
for start in range(n):
    if visited[start]:
        continue
    stack = [start]
    size = 0
    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        size += 1
        for v in range(n):
            if adj[u][v] and not visited[v]:
                stack.append(v)
    best = max(best, size)

print(best)
