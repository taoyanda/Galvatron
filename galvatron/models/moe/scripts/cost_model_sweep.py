"""Cost-model estimator sweep for the FSDP-only validation matrix.

Tuple format: ``(tp, ep, dp_mode, bsz)`` with ``dp_mode ∈ {zero2sdp, zero3}``
and ``dp = 4 / (tp * ep)``. Mirrors ``cost_model_real_test.sh`` shapes
(modulo the FSEP on/off toggle, which the analytical cost model does not
distinguish — see the comparison script for the empirical FSEP delta).

Run inside the container so galvatron imports work:
    docker exec hetu python3 /root/Galvatron/galvatron/models/moe/scripts/cost_model_sweep.py

Writes ``configs/cost_model_estimate_4gpu_4layer.json`` and prints a table.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple


_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
CONFIGS_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))
sys.path.insert(0, REPO_ROOT)

from galvatron.models.moe.cost_model import CostModel  # noqa: E402

# Same (tp, ep, dp_mode, bsz, fsep) shapes as the real-test sweep so the
# estimates can match real measurements 1-to-1 along the FSEP axis.
# zero2sdp → zero_stage=2 (with SDP enabled), zero3 → zero_stage=3.
# fsep="on" enables runtime-profile lookup (full-iter FSEP-on timing);
# fsep="off" uses the FSEP-off runtime-profile entry (or falls back to the
# forward-only computation profile if no runtime sample exists).
CONFIGS: List[Tuple[int, int, str, int, str]] = [
    # FSEP-on rows: ep≥2, satisfies ep×cap≥num_experts.
    (1, 4, "zero2sdp", 4, "on"),
    (1, 4, "zero3",    4, "on"),
    (2, 2, "zero2sdp", 4, "on"),
    (2, 2, "zero3",    4, "on"),
    # FSEP-off rows: same shapes head-to-head.
    (1, 4, "zero2sdp", 4, "off"),
    (1, 4, "zero3",    4, "off"),
    (2, 2, "zero2sdp", 4, "off"),
    (2, 2, "zero3",    4, "off"),
    # FSEP-off ep=1 (FSEP can't satisfy its constraint here).
    (2, 1, "zero2sdp", 2, "off"), (2, 1, "zero2sdp", 4, "off"),
    (2, 1, "zero3",    2, "off"), (2, 1, "zero3",    4, "off"),
    (4, 1, "zero2sdp", 4, "off"),
    (4, 1, "zero3",    4, "off"),
]

NUM_GPUS = 4
NUM_LAYERS = 4
SEQ_LEN = 4096


def _zero_stage(dp_mode: str) -> int:
    return 2 if dp_mode == "zero2sdp" else 3


def main() -> None:
    cm = CostModel("mixtral-8x7b-e8k2")
    out: Dict[str, object] = {"configs": []}
    print(
        f"{'tp':>2} {'ep':>2} {'dp_mode':>9} {'bsz':>3} {'fsep':>4} {'dp':>2} | "
        f"{'iter_ms':>9} {'fwd_bwd_ms':>10} {'opt_ms':>7} | "
        f"{'params':>8} {'optim':>8} {'act':>8} {'peak':>9}  src"
    )
    print("-" * 130)
    for tp, ep, dp_mode, global_bsz, fsep_mode in CONFIGS:
        dp = NUM_GPUS // (tp * ep)
        micro_bsz = max(1, global_bsz // dp)
        if dp * micro_bsz != global_bsz:
            print(f"# skip tp={tp} ep={ep} {dp_mode} bsz={global_bsz} dp={dp} (bsz%dp != 0)")
            continue
        try:
            est = cm.estimate(
                num_layers=NUM_LAYERS,
                num_gpus=NUM_GPUS,
                dp=dp,
                pp=1,
                tp=tp,
                ep=ep,
                micro_batch_size=micro_bsz,
                global_batch_size=global_bsz,
                seq_len=SEQ_LEN,
                sequence_parallel=True,
                pp_schedule="1f1b",
                zero_stage=_zero_stage(dp_mode),
                # Both dp_modes shard model state across DP: zero2sdp shards
                # the optimizer slice; zero3 shards params+optimizer. The
                # cost model uses ``sdp=True`` together with ``zero_stage``
                # to pick the right shard factors.
                sdp=dp_mode in ("zero2sdp", "zero3"),
                # Real test runs with --global_checkpoint 1 → activation
                # recompute is on. Only used by the analytical fall-back;
                # the runtime profile already includes recompute cost.
                recompute=True,
                bwd_mult=2.0,
                fsep=(fsep_mode == "on"),
            )
        except KeyError as e:
            print(f"# skip tp={tp} ep={ep} {dp_mode} bsz={global_bsz} fsep={fsep_mode} (no profile: {e})")
            continue
        bd = est.breakdown
        # When the runtime profile supplies the time, ``pipeline_iter_ms``
        # IS the fwd+bwd estimate (it embeds the full-iter measurement).
        # In the analytical fall-back, the same identity holds because we
        # no longer add opt_step_ms_per_layer into per_layer_ms.
        fwd_bwd_ms_est = bd["pipeline_iter_ms"]
        opt_ms_est = bd.get("opt_step_ms", 0.0)
        row = {
            "tp": tp, "ep": ep, "dp_mode": dp_mode, "global_bsz": global_bsz,
            "fsep": fsep_mode, "dp": dp,
            "iter_ms": est.total_iter_ms,
            "fwd_bwd_ms": fwd_bwd_ms_est,
            "opt_step_ms": opt_ms_est,
            "peak_memory_mb": est.peak_memory_mb,
            "parameters_mb": bd["parameters_mb"],
            "optimizer_mb": bd["optimizer_mb"],
            "activations_mb": bd["activations_mb"],
            "time_source": bd.get("time_source", "?"),
            "breakdown": {k: v for k, v in bd.items() if k != "time_source"},
        }
        out["configs"].append(row)
        src = bd.get("time_source", "?")
        src_short = "rt" if src.startswith("runtime_profile") else "fwd"
        print(
            f"{tp:>2} {ep:>2} {dp_mode:>9} {global_bsz:>3} {fsep_mode:>4} {dp:>2} | "
            f"{est.total_iter_ms:>9.1f} {fwd_bwd_ms_est:>10.1f} {opt_ms_est:>7.1f} | "
            f"{bd['parameters_mb']:>8.0f} {bd['optimizer_mb']:>8.0f}"
            f" {bd['activations_mb']:>8.0f} {est.peak_memory_mb:>9.0f}  {src_short}"
        )

    out_path = os.path.join(CONFIGS_DIR, "cost_model_estimate_4gpu_4layer.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
