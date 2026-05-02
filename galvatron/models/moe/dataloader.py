import os
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from torch import Tensor
from torch.utils.data import Dataset

from galvatron.core.runtime.hybrid_parallel_config import get_chunks
from galvatron.core.runtime.pipeline.utils import chunk_batch


def random_get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""
    micro_batch_size, seq_length = data.size()
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # position_ids = torch.arange(seq_length, dtype=torch.long,
    #                             device=data.device)
    # position_ids = position_ids.unsqueeze(0).expand_as(data)
    attention_mask = attention_mask < 0.5

    return attention_mask  # , position_ids


def random_collate_fn(batch):
    tokens_ = torch.stack(batch, dim=0)
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    args = get_args()
    rotary_pos_emb = RotaryEmbedding(
            args.hidden_size // args.num_attention_heads, 
            args.rotary_percent, 
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            rotary_base=args.rotary_base
        )
    rotary_embedding = rotary_pos_emb(
                        tokens.shape[-1]
                    )
    if not args.use_flash_attn:
        attention_mask = random_get_ltor_masks_and_position_ids(tokens)
    else:
        attention_mask = None
    return tokens, {"attention_mask": attention_mask, "labels": labels, "rotary_embedding": rotary_embedding}, None


class DataLoaderForMoE(Dataset):
    def __init__(self, args, device, dataset_size=2560 * 16):
        self.vocab_size = args.vocab_size
        self.sentence_length = args.seq_length
        self.dataset_size = dataset_size
        self.data_length = np.random.randint(1, self.sentence_length + 1, (self.dataset_size,))
        self.device = device

        self.input_ids = []
        for i in range(self.dataset_size):
            sentence = np.random.randint(0, self.vocab_size, (self.sentence_length,))
            sentence[self.data_length[i] :] = 0
            mask = np.ones((self.sentence_length,))
            mask[self.data_length[i] :] = 0

            padding_sentence = np.zeros(self.sentence_length + 1, dtype=sentence.dtype)
            padding_sentence[: self.sentence_length] = sentence
            self.input_ids.append(padding_sentence)

        self.input_ids = np.array(self.input_ids)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input_ids = torch.LongTensor(self.input_ids[idx]).to(self.device)
        return input_ids


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset, train_val_test_num_samples, is_dataset_built_on_rank, core_gpt_dataset_config_from_args(args)
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_train_valid_test_data_iterators():
    train_valid_test_datasets_provider.is_distributed = True
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )
    return train_data_iterator, valid_data_iterator, test_data_iterator


def fake_tensor(bsz):
    return torch.zeros([bsz, 1], device="cuda")


# Cache for --static_input mode. Populated on the first call to get_batch and
# reused thereafter. Raw batch tensors + rotary embedding are cached; the
# chunked micro_lossmask must be rebuilt per call because loss_func pops from
# it.
_static_batch_cache = {
    "cached": False,
    "batch": None,
    "rotary_embedding": None,
    "hits": 0,
}


def _build_static_synthetic_batch(args, batch_size):
    """Build a deterministic synthetic batch without touching the data iterator.

    Uses a CPU-side torch.Generator seeded from args.seed so every rank produces
    bit-identical tokens independent of CUDA RNG state (which is perturbed by
    model init, dropout, etc.).
    """
    seq_len = args.seq_length
    vocab_size = args.vocab_size
    device = torch.cuda.current_device()
    seed = getattr(args, "seed", 1234)
    gen = torch.Generator()
    gen.manual_seed(seed)
    tokens_full = torch.randint(
        0, vocab_size, (batch_size, seq_len + 1), generator=gen, dtype=torch.long
    ).to(device)
    tokens = tokens_full[:, :-1].contiguous()
    labels = tokens_full[:, 1:].contiguous()
    loss_mask = torch.ones((batch_size, seq_len), dtype=torch.float32, device=device)
    position_ids = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )
    if getattr(args, "use_flash_attn", False):
        attention_mask = None
    else:
        mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device)).view(
            1, 1, seq_len, seq_len
        )
        attention_mask = mask < 0.5
    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def _broadcast_batch_across_dp(batch):
    """Broadcast DP-rank-0's synthetic tensors across the DP group. With the
    deterministic CPU generator above all ranks already produce identical
    tokens; this is a safety net against residual float-order divergence."""
    if mpu.get_data_parallel_world_size() <= 1:
        return
    dp_group = mpu.get_data_parallel_group()
    src_global_rank = torch.distributed.get_global_rank(dp_group, 0)
    for k in ("tokens", "labels", "loss_mask", "position_ids", "attention_mask"):
        t = batch.get(k)
        if isinstance(t, torch.Tensor):
            torch.distributed.broadcast(t, src=src_global_rank, group=dp_group)


def get_batch(data_iterator):
    """Generate a batch."""

    args = get_args()
    # TODO: this is pretty hacky, find a better way
    batch_size = args.global_train_batch_size // mpu.get_data_parallel_world_size()
    static_input = getattr(args, "static_input", False)
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return fake_tensor(batch_size), {}, None
        # return torch.empty(args.micro_batch_size,args.seq_length+1).cuda().long()

    if static_input:
        if not _static_batch_cache["cached"]:
            static_path = getattr(args, "static_input_path", "") or ""
            batch = None
            source = None
            if static_path and os.path.isfile(static_path):
                device = torch.cuda.current_device()
                loaded = torch.load(static_path, map_location="cpu")
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in loaded.items()
                }
                assert batch["tokens"].shape[0] == batch_size, (
                    f"static_input file {static_path} has batch_size "
                    f"{batch['tokens'].shape[0]}, expected {batch_size}"
                )
                assert batch["tokens"].shape[1] == args.seq_length, (
                    f"static_input file {static_path} has seq_length "
                    f"{batch['tokens'].shape[1]}, expected {args.seq_length}"
                )
                source = f"loaded from {static_path}"
            if batch is None:
                batch = _build_static_synthetic_batch(args, batch_size)
                _broadcast_batch_across_dp(batch)
                if static_path and torch.distributed.get_rank() == 0:
                    os.makedirs(
                        os.path.dirname(os.path.abspath(static_path)), exist_ok=True
                    )
                    torch.save(
                        {
                            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                            for k, v in batch.items()
                        },
                        static_path,
                    )
                source = f"built from seed={getattr(args, 'seed', 1234)}" + (
                    f", saved to {static_path}" if static_path else ""
                )
            if torch.distributed.get_rank() == 0:
                tok = batch["tokens"]
                print(
                    f"[static_input] FIRST-BATCH {source} | "
                    f"tokens.shape={tuple(tok.shape)} dtype={tok.dtype} "
                    f"sum={tok.sum().item()} first8={tok.flatten()[:8].tolist()} "
                    f"attention_mask={'None' if batch['attention_mask'] is None else tuple(batch['attention_mask'].shape)}",
                    flush=True,
                )
            rotary_pos_emb = RotaryEmbedding(
                args.hidden_size // args.num_attention_heads,
                args.rotary_percent,
                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
                rotary_base=args.rotary_base,
            )
            rotary_embedding = rotary_pos_emb(args.seq_length)
            _static_batch_cache["cached"] = True
            _static_batch_cache["batch"] = batch
            _static_batch_cache["rotary_embedding"] = rotary_embedding
        batch = _static_batch_cache["batch"]
        rotary_embedding = _static_batch_cache["rotary_embedding"]
        _static_batch_cache["hits"] += 1
        if torch.distributed.get_rank() == 0:
            hits = _static_batch_cache["hits"]
            if hits <= 3 or hits % 10 == 0:
                tok = batch["tokens"]
                print(
                    f"[static_input] CACHE-HIT #{hits} "
                    f"tokens.sum={tok.sum().item()} first4={tok.flatten()[:4].tolist()}",
                    flush=True,
                )
        micro_lossmask = chunk_batch([batch["loss_mask"]], get_chunks(args))
        return (
            batch["tokens"],
            {
                "position_ids": batch["position_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "rotary_embedding": rotary_embedding,
            },
            partial(loss_func, micro_lossmask),
        )

    batch = get_batch_on_this_tp_rank(data_iterator)

    rotary_pos_emb = RotaryEmbedding(
            args.hidden_size // args.num_attention_heads,
            args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            rotary_base=args.rotary_base
        )
    rotary_embedding = rotary_pos_emb(
                        args.seq_length
                    )
    micro_lossmask = chunk_batch([batch["loss_mask"]], get_chunks(args))
    # print(f"Rank {torch.cuda.current_device()} with input {tokens}")
    if batch["tokens"] == None:
        batch["tokens"] = fake_tensor(batch_size)
    return (
        batch["tokens"],
        {
            "position_ids": batch["position_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "rotary_embedding": rotary_embedding
        },
        partial(loss_func, micro_lossmask),
    )


def loss_func(micro_lossmask: Tensor, label: List, output_tensor: List):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """
    loss_mask = micro_lossmask[0][0]
    args = get_args()
    output_tensor = output_tensor[0]
    losses = output_tensor.float()
    # if torch.cuda.current_device()==0:
    #     print(f"loss {losses}")
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    averaged_loss = average_losses_across_data_parallel_group([loss])

    micro_lossmask.pop(0)
    return loss, averaged_loss[0]
