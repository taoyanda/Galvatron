def gen_dp_groups(world_size, tp_of_ep_size, pp_size):
    """Mirrors gen_dp_group_dist(consecutive=False) — the FSEP dp_of_ep_group."""
    num_pp_groups = world_size // pp_size
    groups = []
    for i in range(pp_size):
        start, end = i * num_pp_groups, (i + 1) * num_pp_groups
        for j in range(tp_of_ep_size):
            groups.append(list(range(start + j, end, tp_of_ep_size)))
    return groups


def gen_ep_groups(world_size, ep_size, tp_of_ep_size, pp_size):
    """Mirrors gen_ep_group_dist."""
    num_pp_groups = world_size // pp_size
    dp_size = world_size // ep_size // tp_of_ep_size // pp_size
    num_dp_groups = ep_size * tp_of_ep_size
    groups = []
    for i in range(pp_size):
        base = i * num_pp_groups
        for j in range(dp_size):
            start, end = base + num_dp_groups * j, base + num_dp_groups * (j + 1)
            for k in range(tp_of_ep_size):
                groups.append(list(range(start + k, end, tp_of_ep_size)))
    return groups


def verify(world_size, pp_size, ep_size, tp_of_ep_size):
    dp_of_ep = gen_dp_groups(world_size, tp_of_ep_size, pp_size)
    eps = gen_ep_groups(world_size, ep_size, tp_of_ep_size, pp_size)
    rank_to_local = {r: g.index(r) for g in dp_of_ep for r in g}
    rank_to_ep = {r: g.index(r) for g in eps for r in g}
    miss = []
    for r in range(world_size):
        local = rank_to_local[r]
        truth = rank_to_ep[r]  # what mpu.get_expert_model_parallel_rank() returns
        guess = local % ep_size  # what MoEModel_tensor_parallel.py uses
        if truth != guess:
            miss.append((r, local, truth, guess))
    return miss


CONFIGS = [
    # (world, pp, ep, tp_of_ep)
    (4, 1, 4, 1),
    (4, 1, 2, 2),
    (4, 2, 2, 1),
    (8, 1, 8, 1),
    (8, 1, 4, 2),
    (8, 1, 2, 4),
    (8, 2, 4, 1),
    (8, 2, 2, 2),
    (8, 4, 2, 1),
    (16, 1, 8, 2),
    (16, 2, 4, 2),
    (16, 4, 2, 2),
    (32, 1, 8, 4),
    (32, 2, 8, 2),
    (32, 4, 4, 2),
]
for cfg in CONFIGS:
    miss = verify(*cfg)
    print(
        f"world={cfg[0]} pp={cfg[1]} ep={cfg[2]} tp_of_ep={cfg[3]}: "
        f"{'PASS' if not miss else f'FAIL {miss[:3]}'}"
    )
