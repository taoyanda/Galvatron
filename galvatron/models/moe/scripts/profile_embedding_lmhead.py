"""Standalone profile of the embedding + LM-head forward/backward timings.

Both modules are paid once per iteration regardless of ``num_hidden_layers``.
The forward-only computation profile (``layertype_other_bsz<B>_seq<S>``)
under-counts them because the LM head's backward (a hidden→vocab GEMM) is
much heavier than its forward when measured end-to-end, and the embedding
backward involves a sparse scatter that doesn't track its forward 1:1.

Profiling them in isolation lets the cost model:

  1. Use accurate ``other_full_ms`` in the analytical fall-back path instead
     of multiplying forward-only ``other_ms`` by (1 + bwd_mult).
  2. Correctly extrapolate the runtime profile to a different
     ``num_hidden_layers``: subtract embedding+lm-head from the measured
     ``fwd_bwd_ms``, scale the residual per-layer cost, add embed/lm-head back.

Output: ``galvatron/models/moe/configs/embedding_lmhead_profiling_<prec>_<model>.json``

Run:
    docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/profile_embedding_lmhead.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(_HERE, ".."))
META_DIR = os.path.join(MODEL_DIR, "meta_configs")
CONFIGS_DIR = os.path.join(MODEL_DIR, "configs")

MODEL = "qwen-30b-a3b-e128k8"
PRECISION = "bf16"
DTYPE = torch.bfloat16

# Shapes to profile. Mirrors the (bsz, seq) coverage of the runtime profile
# so both artifacts share a key space; cost model interpolates if needed.
BSZ_VALUES = [1, 2, 4]
SEQ_VALUES = [4096]

WARMUP_ITERS = 3
TIMED_ITERS = 10


def cuda_time_ms(fn, *args, **kwargs) -> float:
    """Run ``fn`` ``TIMED_ITERS`` times after a warm-up, return median ms."""
    for _ in range(WARMUP_ITERS):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(TIMED_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def profile_embedding(vocab: int, hidden: int, bsz: int, seq: int, device: str
                      ) -> Dict[str, float]:
    embed = torch.nn.Embedding(vocab, hidden, dtype=DTYPE, device=device)
    embed.weight.requires_grad_(True)
    tokens = torch.randint(0, vocab, (bsz, seq), device=device, dtype=torch.long)
    grad_out = torch.randn(bsz, seq, hidden, device=device, dtype=DTYPE)

    def fwd():
        return embed(tokens)

    def fwd_bwd():
        embed.zero_grad(set_to_none=True)
        out = embed(tokens)
        out.backward(grad_out)

    fwd_ms = cuda_time_ms(fwd)
    fb_ms = cuda_time_ms(fwd_bwd)
    bwd_ms = max(0.0, fb_ms - fwd_ms)
    del embed, tokens, grad_out
    torch.cuda.empty_cache()
    return {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "fwd_bwd_ms": fb_ms}


def profile_lmhead(vocab: int, hidden: int, bsz: int, seq: int, device: str
                   ) -> Dict[str, float]:
    # LM head is a plain hidden→vocab linear without bias (Mixtral convention).
    lin = torch.nn.Linear(hidden, vocab, bias=False, dtype=DTYPE, device=device)
    lin.weight.requires_grad_(True)
    x = torch.randn(bsz, seq, hidden, device=device, dtype=DTYPE, requires_grad=True)
    grad_out = torch.randn(bsz, seq, vocab, device=device, dtype=DTYPE)

    def fwd():
        return lin(x)

    def fwd_bwd():
        lin.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        out = lin(x)
        out.backward(grad_out)

    fwd_ms = cuda_time_ms(fwd)
    fb_ms = cuda_time_ms(fwd_bwd)
    bwd_ms = max(0.0, fb_ms - fwd_ms)
    del lin, x, grad_out
    torch.cuda.empty_cache()
    return {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "fwd_bwd_ms": fb_ms}


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)
    device = "cuda:0"
    meta = json.load(open(os.path.join(META_DIR, f"{MODEL}.json")))
    vocab = int(meta["vocab_size"])
    hidden = int(meta["hidden_size"])
    print(f"profiling embedding/lmhead for {MODEL}: vocab={vocab} hidden={hidden}")

    samples: List[Dict] = []
    by_shape: Dict[str, Dict] = {}

    for seq in SEQ_VALUES:
        for bsz in BSZ_VALUES:
            print(f"  shape: bsz={bsz} seq={seq}", flush=True)
            embed = profile_embedding(vocab, hidden, bsz, seq, device)
            lmhead = profile_lmhead(vocab, hidden, bsz, seq, device)
            entry = {
                "bsz": bsz, "seq": seq,
                "embedding": embed,
                "lmhead": lmhead,
                "total_fwd_bwd_ms": embed["fwd_bwd_ms"] + lmhead["fwd_bwd_ms"],
            }
            samples.append(entry)
            by_shape[f"bsz{bsz}_seq{seq}"] = entry
            print(
                f"    embedding fwd={embed['fwd_ms']:.2f} bwd={embed['bwd_ms']:.2f}"
                f" lmhead fwd={lmhead['fwd_ms']:.2f} bwd={lmhead['bwd_ms']:.2f}"
                f" total fwd_bwd={entry['total_fwd_bwd_ms']:.2f} ms"
            )

    out = {
        "model": MODEL, "precision": PRECISION,
        "vocab_size": vocab, "hidden_size": hidden,
        "warmup_iters": WARMUP_ITERS, "timed_iters": TIMED_ITERS,
        "by_shape": by_shape,
        "samples": samples,
    }
    out_path = os.path.join(
        CONFIGS_DIR, f"embedding_lmhead_profiling_{PRECISION}_{MODEL}.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
