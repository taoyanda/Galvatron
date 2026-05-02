"""Profiling trainer that consumes a deterministic, file-backed static batch.

Unlike train_dist_random.py (which samples random tokens in
DataLoaderForMoE.__init__ and then optionally caches the first batch in-memory),
this trainer calls dataloader.get_batch() every iteration. Under --static_input
that path either loads batch tensors from --static_input_path or builds them
from a fixed seed and writes them to that path, giving bit-identical tokens
across invocations.

Launched by the frozen profiling scripts via PROFILE_TRAINER=profile_dist_static.py.
"""

import os

import torch
from torch.optim import Adam

from galvatron.core import initialize_galvatron
from galvatron.models.moe.arguments import model_args
from galvatron.models.moe.dataloader import get_batch
from galvatron.models.moe.MoEModel_hybrid_parallel import (
    get_moe_config,
    get_runtime_profiler,
    moe_model_hp,
)
from galvatron.utils import print_loss, set_seed
from megatron.training.arguments import _print_args


def train(args):
    if not getattr(args, "static_input", False):
        raise RuntimeError(
            "profile_dist_static.py requires --static_input. Use train_dist_random.py "
            "for non-deterministic profiling."
        )

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    config = get_moe_config(args)
    model = moe_model_hp(config, args)

    if local_rank == 0:
        _print_args("arguments", args)
        print(
            f"[profile_dist_static] static_input_path={getattr(args, 'static_input_path', '')!r} "
            f"laer_freeze_after_iter={getattr(args, 'laer_freeze_after_iter', -1)} "
            f"dropout_prob={getattr(args, 'dropout_prob', None)}",
            flush=True,
        )

    optimizer = Adam(
        model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay
    )

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...", flush=True)

    if args.profile_forward:
        torch.set_grad_enabled(False)

    # The runtime profiler calls exit(0) once its averaging window completes
    # (end_iter=20 by default), so an oversized cap is fine here.
    MAX_ITERS_PER_EPOCH = 64
    for ep in range(args.epochs):
        for iter in range(MAX_ITERS_PER_EPOCH):
            tokens, kwargs, loss_func = get_batch(None)

            profiler.profile_time_start(iter)
            profiler.profile_memory(iter, "Before Forward")

            batch = [tokens]
            loss = model.forward_backward(
                batch, iter, profiler, loss_func=loss_func, **kwargs
            )

            profiler.profile_memory(iter, "After Backward")
            optimizer.step()
            profiler.profile_memory(iter, "After optimizer_step")
            optimizer.zero_grad()

            print_loss(args, loss, ep, iter)

            profiler.post_profile_memory(iter)
            profiler.profile_time_end(iter)

            torch.distributed.barrier()


if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
