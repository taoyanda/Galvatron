"""Generate a deterministic synthetic batch and save it to a .pt file.

The output file is consumable by the MoE dataloader when --static_input and
--static_input_path point at this path (see galvatron/models/moe/dataloader.py).

Batch layout (matches _build_static_synthetic_batch):
    tokens:         [batch_size, seq_length]         int64
    labels:         [batch_size, seq_length]         int64   (next-token shifted)
    loss_mask:      [batch_size, seq_length]         float32 (cast to --precision)
    position_ids:   [batch_size, seq_length]         int64
    attention_mask: [1, 1, seq_length, seq_length]   bool, or None with flash-attn

Either pass --model_size to pull (seq_length, vocab_size) from the bundled
meta-config JSON, or override them explicitly with --seq_length/--vocab_size.
"""

import argparse
import json
import os
from pathlib import Path

import torch

PRECISION_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_meta_config(model_size: str) -> dict:
    meta_dir = Path(__file__).resolve().parent.parent / "meta_configs"
    path = meta_dir / f"{model_size}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Meta config {path} not found. Available: "
            f"{[p.stem for p in meta_dir.glob('*.json')]}"
        )
    with open(path) as f:
        return json.load(f)


def build_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    seed: int,
    dtype: torch.dtype,
    use_flash_attn: bool,
) -> dict:
    gen = torch.Generator()
    gen.manual_seed(seed)
    tokens_full = torch.randint(
        0, vocab_size, (batch_size, seq_length + 1), generator=gen, dtype=torch.long
    )
    tokens = tokens_full[:, :-1].contiguous()
    labels = tokens_full[:, 1:].contiguous()
    loss_mask = torch.ones((batch_size, seq_length), dtype=dtype)
    position_ids = (
        torch.arange(seq_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    if use_flash_attn:
        attention_mask = None
    else:
        mask = torch.tril(torch.ones((1, seq_length, seq_length))).view(
            1, 1, seq_length, seq_length
        )
        attention_mask = mask < 0.5
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Destination .pt file (parent dirs created if missing).",
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="Optional: meta-config name (e.g. mixtral-8x7b-e8k2). "
        "Supplies defaults for --seq_length and --vocab_size.",
    )
    parser.add_argument("--seq_length", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=list(PRECISION_MAP.keys()),
        help="Dtype for loss_mask. Tokens/labels/position_ids are int64.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Set to skip the attention_mask tensor (matches runtime).",
    )
    args = parser.parse_args()

    seq_length = args.seq_length
    vocab_size = args.vocab_size
    if args.model_size is not None:
        meta = load_meta_config(args.model_size)
        if seq_length is None:
            seq_length = meta["max_position_embeddings"]
        if vocab_size is None:
            vocab_size = meta["vocab_size"]

    if seq_length is None or vocab_size is None:
        parser.error(
            "--seq_length and --vocab_size are required unless --model_size is given."
        )

    dtype = PRECISION_MAP[args.precision]
    batch = build_batch(
        args.batch_size, seq_length, vocab_size, args.seed, dtype, args.use_flash_attn
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(batch, args.output_path)

    print(f"Wrote {args.output_path}")
    print(
        f"  batch_size={args.batch_size} seq_length={seq_length} "
        f"vocab_size={vocab_size} precision={args.precision} "
        f"seed={args.seed} use_flash_attn={args.use_flash_attn}"
    )
    print(
        f"  tokens.shape={tuple(batch['tokens'].shape)} "
        f"loss_mask.dtype={batch['loss_mask'].dtype} "
        f"attention_mask={'None' if batch['attention_mask'] is None else tuple(batch['attention_mask'].shape)}"
    )


if __name__ == "__main__":
    main()
