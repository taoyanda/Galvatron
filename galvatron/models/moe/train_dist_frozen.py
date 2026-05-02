import os
import atexit
import faulthandler
import signal
import sys
import torch

# faulthandler dumps tracebacks even when Python is stuck in a C extension
# (e.g. spinning in NCCL polling). Install before set_device so it's armed
# from the very start of the process.
faulthandler.enable()
faulthandler.register(signal.SIGUSR2, all_threads=True)
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

from torch.optim import Adam
from tqdm import tqdm

from galvatron.core import initialize_galvatron
from galvatron.models.moe.arguments import model_args
from galvatron.models.moe.dataloader import DataLoaderForMoE, random_collate_fn
from galvatron.models.moe.MoEModel_hybrid_parallel import (
    get_moe_config,
    get_runtime_profiler,
    moe_model_hp,
)
from galvatron.utils import distributed_dataloader, print_loss, set_seed
from megatron.training.arguments import _print_args

# Graceful-shutdown hooks — best effort. SIGTERM under timeout, normal
# Python exit, and uncaught-exception paths all funnel here. Goal:
# release c10d NCCL comms and the user-NCCL singleton in
# moe_all_to_all_kernels so the host driver doesn't hold dirty CUDA
# context state (~10 min decay window) into the next torchrun launch.
# See doc/profile_computation_frozen_fixes.md (Open 2).
_shutdown_done = False


def _galvatron_graceful_shutdown(reason="atexit"):
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True
    rank = -1
    try:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    except Exception:
        pass
    # 1. User-NCCL singleton from moe_all_to_all_kernels (FSEP A2A comm).
    try:
        import moe_all_to_all_kernels  # already imported elsewhere; safe

        moe_all_to_all_kernels.destroy_nccl_comm()
    except Exception:
        pass
    # 2. The async LP solver worker pool.
    try:
        from galvatron.core.runtime.moe.prefetch.async_linear_programming import (
            cleanup_global_lp_solver,
        )

        cleanup_global_lp_solver()
    except Exception:
        pass
    # 3. c10d process groups.
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    # 4. Force CUDA cleanup.
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _galvatron_signal_handler(signum, frame):
    _galvatron_graceful_shutdown(reason=f"signal-{signum}")
    # Exit 0 — we treat a graceful SIGTERM as a successful shutdown so
    # the parent profiler.py's outer-rc handling can distinguish a
    # timeout-with-cleanup (rc 0) from a hard error (rc != 0).
    # NB: torchrun's elastic agent maps SIGTERM to "expected", so this
    # path is safe.
    sys.exit(0)


atexit.register(_galvatron_graceful_shutdown, reason="atexit")
signal.signal(signal.SIGTERM, _galvatron_signal_handler)
signal.signal(signal.SIGINT, _galvatron_signal_handler)


def _dump_stacks(signum, frame):
    """SIGUSR1 handler: print every thread's Python stack. Used to
    diagnose hangs (FSDP root pre-forward) when py-spy is blocked by
    container ptrace policy. Sender side: `kill -USR1 <pid>`."""
    import traceback as _tb
    import sys as _sys

    _r = -1
    try:
        if torch.distributed.is_initialized():
            _r = torch.distributed.get_rank()
    except Exception:
        pass
    print(f"[stackdump r{_r}] === SIGUSR1 stack dump begin ===", flush=True)
    for tid, fr in _sys._current_frames().items():
        print(f"[stackdump r{_r}] thread tid={tid}", flush=True)
        for ln in _tb.format_stack(fr):
            print(f"[stackdump r{_r}]   {ln.rstrip()}", flush=True)
    print(f"[stackdump r{_r}] === SIGUSR1 stack dump end ===", flush=True)


signal.signal(signal.SIGUSR1, _dump_stacks)


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_moe_config(args)
    model = moe_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    trainloader = distributed_dataloader(
        dataset=DataLoaderForMoE(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group=model.dp_groups_whole[0].group,
        collate_fn=random_collate_fn,
    )

    if local_rank == 0:
        _print_args("arguments", args)

    optimizer = Adam(
        model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay
    )

    # ============================================================
    # Real-measurement instrumentation for cost-model validation.
    #   - params_mb       : MEASURED — sum p.numel()*p.element_size() over
    #                       live model parameters (post-shard, per-rank).
    #   - optimizer_mb    : PREDICTED — Adam keeps 2 fp32 momentums (m, v)
    #                       per param = 8 bytes/param, plus bf16 grads
    #                       (2 bytes/param when allocated by autograd).
    #                       We don't force-allocate the actual Adam state
    #                       because under --profile_forward 1 it would
    #                       require a real .step() with grads, and for the
    #                       unsharded dp=4 tp=1 ep=1 case the 24 GB param
    #                       footprint × 4 (m+v fp32) = 96 GB exceeds the
    #                       48 GB GPU. Predicted size is deterministic
    #                       from the param count.
    #   - activation_mb   : MEASURED — torch.cuda.max_memory_allocated()
    #                       delta after first forward, minus params_mb.
    # rank-0 only to avoid interleaved output.
    # ============================================================
    _rm_n_params = sum(p.numel() for p in model.parameters())
    _rm_params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    # 2 fp32 momentums (Adam m, v) + bf16 grad ≈ 10 bytes/param.
    _rm_optimizer_mb_pred = _rm_n_params * (4 + 4 + 2) / 1e6
    if rank == 0:
        print(f"[real_measure] params_mb={_rm_params_mb:.2f}", flush=True)
        print(
            f"[real_measure] optimizer_mb_pred={_rm_optimizer_mb_pred:.2f}", flush=True
        )
    _rm_act_logged = False

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config)

    profiler.profile_memory(0, "After creating model")

    # ============================================================
    # FSEP + LAER verification banner (rank 0 only). Confirms the
    # smart-routing dispatcher and async LP solver are wired into
    # this run. Per-iter verification is provided by the
    # [solver_init] / [solver_submit] / [layer N layout frozen]
    # prints emitted from MoEAlltoAllSmartTokenDispatcher.
    # ============================================================
    if rank == 0:
        try:
            use_fsep = bool(getattr(args, "use_fsep", False))
            solver_enabled = os.environ.get("ENABLE_SOLVER", "0") == "1"
            freeze_iter = int(getattr(args, "laer_freeze_after_iter", -1))
            ep_deg = int(getattr(args, "global_ep_deg", 1))
            tp_of_ep = int(getattr(args, "global_tp_of_ep_deg", 1))
            cap = int(getattr(args, "expert_capacity_per_device", 0))
            n_moe_layers = 0
            dispatcher_cls = None
            for m in model.modules():
                td = getattr(m, "token_dispatcher", None)
                if td is not None:
                    n_moe_layers += 1
                    if dispatcher_cls is None:
                        dispatcher_cls = type(td).__name__
            print("=" * 64, flush=True)
            print(
                f"[fsep_verify] use_fsep={use_fsep}  solver_enabled={solver_enabled}",
                flush=True,
            )
            print(
                f"[fsep_verify] dispatcher={dispatcher_cls}  moe_layers={n_moe_layers}",
                flush=True,
            )
            print(
                f"[fsep_verify] ep_deg={ep_deg} tp_of_ep_deg={tp_of_ep} cap_per_device={cap}",
                flush=True,
            )
            print(f"[fsep_verify] laer_freeze_after_iter={freeze_iter}", flush=True)
            print("=" * 64, flush=True)
        except Exception as _e:
            print(f"[fsep_verify] failed: {_e}", flush=True)

    if local_rank == 0:
        print("Start training...")

    if args.profile_forward:
        torch.set_grad_enabled(False)

    # Stage-time + memory-evolution probes (rank 0 only).
    #   stage_time : CUDA events around forward_backward (PipelineParallel
    #                folds fwd+bwd into one call) and around optimizer.step.
    #                Mean computed over iters [10, 20) to match the runtime
    #                profiler's averaging window.
    #   mem_evo    : torch.cuda.max_memory_allocated() captured at five
    #                checkpoints around iter 10 (post-warmup) — pre_fwd,
    #                post_fwd_bwd, post_opt_step, post_zero_grad — plus
    #                post_construct logged before the loop.
    _fb_samples: list = []
    _opt_samples: list = []
    _MEM_ITER = 10
    _mem_logged = False
    _stage_logged = False
    if rank == 0:
        print(
            f"[mem_evo] post_construct_mb={torch.cuda.max_memory_allocated()/1e6:.2f}",
            flush=True,
        )

    static_input = getattr(args, "static_input", False)
    cached_batch = None
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            if static_input:
                if cached_batch is None:
                    cached_batch = batch
                else:
                    batch = cached_batch
            tokens, kwargs, loss_func = batch
            # Replicate input tokens across WORLD so the LP solver and MoE
            # router see the same input on every rank (the dataloader otherwise
            # shards by DP group, causing async drift that surfaces as a NCCL
            # watchdog timeout on the per-iter WORLD barrier).
            torch.distributed.broadcast(tokens, src=0)
            profiler.profile_time_start(iter)
            profiler.profile_memory(iter, "Before Forward")

            input_ids = tokens
            batch = [input_ids]

            if rank == 0 and iter == _MEM_ITER and not _mem_logged:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                print(
                    f"[mem_evo] pre_fwd_mb={torch.cuda.max_memory_allocated()/1e6:.2f}",
                    flush=True,
                )

            _fb_s = torch.cuda.Event(enable_timing=True) if rank == 0 else None
            _fb_e = torch.cuda.Event(enable_timing=True) if rank == 0 else None
            if _fb_s is not None:
                _fb_s.record()
            loss = model.forward_backward(
                batch, iter, profiler, loss_func=loss_func, **kwargs
            )
            if _fb_e is not None:
                _fb_e.record()

            profiler.profile_memory(iter, "After Backward")
            if rank == 0 and iter == _MEM_ITER and not _mem_logged:
                torch.cuda.synchronize()
                print(
                    f"[mem_evo] post_fwd_bwd_mb={torch.cuda.max_memory_allocated()/1e6:.2f}",
                    flush=True,
                )

            _op_s = torch.cuda.Event(enable_timing=True) if rank == 0 else None
            _op_e = torch.cuda.Event(enable_timing=True) if rank == 0 else None
            if _op_s is not None:
                _op_s.record()
            optimizer.step()
            if _op_e is not None:
                _op_e.record()

            profiler.profile_memory(iter, "After optimizer_step")
            if rank == 0 and iter == _MEM_ITER and not _mem_logged:
                torch.cuda.synchronize()
                print(
                    f"[mem_evo] post_opt_step_mb={torch.cuda.max_memory_allocated()/1e6:.2f}",
                    flush=True,
                )

            if rank == 0 and _fb_s is not None:
                torch.cuda.synchronize()
                _fb_samples.append(_fb_s.elapsed_time(_fb_e))
                _opt_samples.append(_op_s.elapsed_time(_op_e))

            # After the first full iteration, capture activation peak and
            # the actual optimizer-state byte count (Adam allocates m/v on
            # first step). When running --profile_forward 1, .step() is a
            # no-op, so optimizer.state stays empty and we report 0; the
            # analytical optimizer_mb_pred from above is the value to compare
            # against in that mode.
            if not _rm_act_logged and rank == 0:
                _peak_mb = torch.cuda.max_memory_allocated() / 1e6
                _opt_actual_mb = (
                    sum(
                        t.numel() * t.element_size()
                        for st in optimizer.state.values()
                        for t in st.values()
                        if torch.is_tensor(t)
                    )
                    / 1e6
                )
                _act_peak_mb = max(0.0, _peak_mb - _rm_params_mb - _opt_actual_mb)
                print(
                    f"[real_measure] optimizer_mb={_opt_actual_mb:.2f} "
                    f"activation_peak_mb={_act_peak_mb:.2f} "
                    f"cuda_peak_mb={_peak_mb:.2f}",
                    flush=True,
                )
                _rm_act_logged = True

            optimizer.zero_grad()

            if rank == 0 and iter == _MEM_ITER and not _mem_logged:
                torch.cuda.synchronize()
                print(
                    f"[mem_evo] post_zero_grad_mb={torch.cuda.max_memory_allocated()/1e6:.2f}",
                    flush=True,
                )
                _mem_logged = True

            # Emit the stage-time summary as soon as we have all the samples
            # in the runtime profiler's averaging window. We can't wait until
            # the iter loop ends — `profile_time_end` at iter=end_iter-1 may
            # terminate the process via save_profiled_time().
            if rank == 0 and not _stage_logged:
                _s, _e = getattr(profiler, "start_iter", 10), getattr(
                    profiler, "end_iter", 20
                )
                if iter + 1 >= _e and len(_fb_samples) >= _e:
                    _fb_win = _fb_samples[_s:_e]
                    _op_win = _opt_samples[_s:_e]
                    if _fb_win:
                        print(
                            f"[stage_time] fwd_bwd_ms={sum(_fb_win)/len(_fb_win):.4f} "
                            f"opt_ms={sum(_op_win)/len(_op_win):.4f} "
                            f"window=[{_s},{_e})",
                            flush=True,
                        )
                        _stage_logged = True

            print_loss(args, loss, ep, iter)

            profiler.post_profile_memory(iter)
            profiler.profile_time_end(iter)

            torch.distributed.barrier()
    # Stage-time summary (rank 0): mean over the runtime profiler's
    # averaging window so the numbers are directly comparable to its
    # printed `Average iteration time is: X s`.
    if rank == 0 and _fb_samples:
        s, e = getattr(profiler, "start_iter", 10), getattr(profiler, "end_iter", 20)
        fb = _fb_samples[s:e]
        op = _opt_samples[s:e]
        if fb:
            print(
                f"[stage_time] fwd_bwd_ms={sum(fb)/len(fb):.4f} "
                f"opt_ms={sum(op)/len(op):.4f} "
                f"window=[{s},{e})",
                flush=True,
            )


if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
