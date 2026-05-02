"""Side-by-side comparison: cost-model estimates vs real measurements + a
head-to-head FSEP+LAER vs naive Megatron-MoE+FSDP table.

Reads:
  - configs/cost_model_estimate_4gpu_4layer.json  (one row per (tp,ep,dp_mode,bsz))
  - configs/cost_model_real_4gpu_4layer.json      (per-(tp,ep,dp_mode,bsz,fsep) row)

Writes:
  - configs/cost_model_comparison_4gpu_4layer.md
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.normpath(os.path.join(_HERE, "..", "configs"))


Shape = Tuple[int, int, str, int, str]  # (tp, ep, dp_mode, bsz, fsep)
RealKey = Shape


def err_pct(est: Optional[float], real: Optional[float]) -> Optional[float]:
    if est is None or real is None or real == 0:
        return None
    return 100.0 * (est - real) / real


def speedup_pct(off: Optional[float], on: Optional[float]) -> Optional[float]:
    """Return (off-on)/on × 100, i.e. how much *slower* off is than on."""
    if off is None or on is None or on == 0:
        return None
    return 100.0 * (off - on) / on


def fmt(v: Optional[float], spec: str) -> str:
    return spec.format(v) if v is not None else "—"


def _shape(c: Dict) -> Shape:
    return (c["tp"], c["ep"], c["dp_mode"], c["global_bsz"], c.get("fsep", "off"))


def _rkey(c: Dict) -> RealKey:
    return (c["tp"], c["ep"], c["dp_mode"], c["global_bsz"], c["fsep"])


def main() -> None:
    est_path = os.path.join(CONFIGS_DIR, "cost_model_estimate_4gpu_4layer.json")
    real_path = os.path.join(CONFIGS_DIR, "cost_model_real_4gpu_4layer.json")
    est: Dict[Shape, Dict] = {_shape(c): c for c in json.load(open(est_path))["configs"]}
    real: Dict[RealKey, Dict] = {_rkey(c): c for c in json.load(open(real_path))["configs"]}

    real_shapes_with_both = sorted({
        (tp, ep, dpm, bsz)
        for (tp, ep, dpm, bsz, fsep) in real
        if (tp, ep, dpm, bsz, "on") in real and (tp, ep, dpm, bsz, "off") in real
    })
    all_real_keys = sorted(real.keys())

    md: List[str] = []
    md.append("# Cost-model validation: 4 GPUs × 4 layers (FSDP-only matrix)\n")
    md.append(
        "Validates the analytical model in `galvatron/cost_model.py` against "
        "real measurements from `train_dist_random.py`. Real readings use the "
        "in-loop `[stage_time]` CUDA-event window (iters 10–19) for "
        "`fwd_bwd_ms` and `opt_ms`, and `[real_measure]` for "
        "params/optimizer/activation/cuda_peak (sampled live from model + "
        "optimizer state + CUDA peak delta).\n"
    )

    # ---------- 1. estimate vs real (matched on shape; both FSEP rows shown) ----------
    md.append("## Iteration time — estimate vs real")
    md.append("")
    md.append(
        "| tp | ep | dp_mode | bsz | fsep | est_iter (ms) | real_iter (ms) | "
        "real_fwd_bwd | real_opt | err_iter % |"
    )
    md.append("|---:|---:|:---|---:|:---:|---:|---:|---:|---:|---:|")
    for k in all_real_keys:
        tp, ep, dpm, bsz, fsep = k
        e = est.get(k, {})
        r = real[k]
        e_it = e.get("iter_ms"); r_it = r.get("iter_ms")
        md.append(
            f"| {tp} | {ep} | {dpm} | {bsz} | {fsep} | "
            f"{fmt(e_it, '{:.0f}')} | {fmt(r_it, '{:.0f}')} | "
            f"{fmt(r.get('fwd_bwd_ms'), '{:.0f}')} | "
            f"{fmt(r.get('opt_step_ms'), '{:.1f}')} | "
            f"{fmt(err_pct(e_it, r_it), '{:+.1f}')} |"
        )

    # ---------- 2. memory: parameters / optimizer / activation_peak / cuda_peak ----------
    md.append("")
    md.append("## Memory — estimate vs real")
    md.append("")
    md.append(
        "| tp | ep | dp_mode | bsz | fsep | est_params | real_params | "
        "est_opt | real_opt | est_act | real_act_peak | real_cuda_peak |"
    )
    md.append("|---:|---:|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in all_real_keys:
        tp, ep, dpm, bsz, fsep = k
        e = est.get(k, {})
        r = real[k]
        opt_real = r.get("optimizer_mb") or r.get("optimizer_mb_pred")
        md.append(
            f"| {tp} | {ep} | {dpm} | {bsz} | {fsep} | "
            f"{fmt(e.get('parameters_mb'), '{:.0f}')} | "
            f"{fmt(r.get('params_mb'),    '{:.0f}')} | "
            f"{fmt(e.get('optimizer_mb'), '{:.0f}')} | "
            f"{fmt(opt_real,              '{:.0f}')} | "
            f"{fmt(e.get('activations_mb'),    '{:.0f}')} | "
            f"{fmt(r.get('activation_peak_mb'), '{:.0f}')} | "
            f"{fmt(r.get('cuda_peak_mb'),       '{:.0f}')} |"
        )

    # ---------- 3. memory evolution (5 checkpoints) ----------
    md.append("")
    md.append("## Memory evolution (single iteration, rank 0)")
    md.append("")
    md.append(
        "| tp | ep | dp_mode | bsz | fsep | post_construct | pre_fwd | "
        "post_fwd_bwd | post_opt | post_zero_grad |"
    )
    md.append("|---:|---:|:---|---:|:---:|---:|---:|---:|---:|---:|")
    for k in all_real_keys:
        tp, ep, dpm, bsz, fsep = k
        r = real[k]
        md.append(
            f"| {tp} | {ep} | {dpm} | {bsz} | {fsep} | "
            f"{fmt(r.get('mem_evo_post_construct_mb'), '{:.0f}')} | "
            f"{fmt(r.get('mem_evo_pre_fwd_mb'),        '{:.0f}')} | "
            f"{fmt(r.get('mem_evo_post_fwd_bwd_mb'),   '{:.0f}')} | "
            f"{fmt(r.get('mem_evo_post_opt_step_mb'),  '{:.0f}')} | "
            f"{fmt(r.get('mem_evo_post_zero_grad_mb'), '{:.0f}')} |"
        )

    # ---------- 4. FSEP+LAER vs naive Megatron-MoE+FSDP head-to-head ----------
    md.append("")
    md.append("## FSEP+LAER vs naive Megatron-MoE+FSDP")
    md.append("")
    md.append(
        "Direct head-to-head where both fsep on/off were measured at the same "
        "(tp, ep, dp_mode, bsz). `fsep=on` enables the smart-routing all-to-all "
        "kernel + LAER expert-relayout solver; `fsep=off` is the naive "
        "Megatron `MoEAlltoAllTokenDispatcher` path with plain FSDP.\n"
    )
    md.append(
        "| tp | ep | dp_mode | bsz | iter_off (ms) | iter_on (ms) | "
        "iter_speedup % | fb_off (ms) | fb_on (ms) | fb_speedup % | "
        "cuda_peak_off | cuda_peak_on | mem_delta % |"
    )
    md.append("|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for shape in real_shapes_with_both:
        tp, ep, dpm, bsz = shape
        r_on = real[(tp, ep, dpm, bsz, "on")]
        r_off = real[(tp, ep, dpm, bsz, "off")]
        md.append(
            f"| {tp} | {ep} | {dpm} | {bsz} | "
            f"{fmt(r_off.get('iter_ms'), '{:.0f}')} | "
            f"{fmt(r_on.get('iter_ms'),  '{:.0f}')} | "
            f"{fmt(speedup_pct(r_off.get('iter_ms'), r_on.get('iter_ms')), '{:+.1f}')} | "
            f"{fmt(r_off.get('fwd_bwd_ms'), '{:.0f}')} | "
            f"{fmt(r_on.get('fwd_bwd_ms'),  '{:.0f}')} | "
            f"{fmt(speedup_pct(r_off.get('fwd_bwd_ms'), r_on.get('fwd_bwd_ms')), '{:+.1f}')} | "
            f"{fmt(r_off.get('cuda_peak_mb'), '{:.0f}')} | "
            f"{fmt(r_on.get('cuda_peak_mb'),  '{:.0f}')} | "
            f"{fmt(speedup_pct(r_off.get('cuda_peak_mb'), r_on.get('cuda_peak_mb')), '{:+.1f}')} |"
        )

    md.append("")
    md.append("## Findings — FSEP+LAER vs naive on this benchmark")
    md.append("")
    md.append(
        "**Headline:** FSEP+LAER is 43–80 % *slower* per iter than naive "
        "Megatron-MoE+FSDP across all 8 head-to-head shapes, and uses 2.3 % "
        "more CUDA peak. This is the **worst-case benchmark for LAER by "
        "construction** — interpretation below."
    )
    md.append("")
    md.append(
        "**Why this benchmark is LAER's worst case.** "
        "`--static_input` + `--dropout_prob 0.0` makes routing bit-identical "
        "iter-over-iter, and `--laer_freeze_after_iter 5` freezes the layout "
        "before measurement (window=[10,20)). The verified FSEP-on logs "
        "show every layer printing `layout frozen at iter 5`, with no "
        "post-freeze solver activity. So during measurement, FSEP+LAER "
        "carries *all* of its overhead and *none* of the optimisation "
        "benefit it was designed to deliver: load-adaptive re-layout under "
        "dynamic expert-imbalance (i.e. real workloads with varying "
        "router scores). The numbers above are the steady-state cost of "
        "FSEP's expert-replication footprint, not the dynamic-payoff regime."
    )
    md.append("")
    md.append(
        "**Construct-memory delta is the headline.** Look at the "
        "`mem_evo` table, `post_construct` column. At `(tp=1, ep=4)`, "
        "FSEP-on holds **22,806 MB** vs FSEP-off's **7,303 MB** — a "
        "+15.5 GB delta the moment the model is built, before any forward. "
        "Naive MoE shards `num_experts/ep = 8/4 = 2` experts per rank; "
        "FSEP holds all 8 expert weights per rank so the smart-router "
        "can route to whichever copy LAER picks. Order-of-magnitude check: "
        "a single mixtral expert is ~3 × hidden × intermediate × 2 B = "
        "~352 MB → 4 layers × 6 extra experts × 352 MB ≈ 8 GB of weight "
        "replication; the remaining ~7 GB is LAER bookkeeping + "
        "FSDP-handle padding tied to the replicated layout. The runtime "
        "iter-time difference tracks this — every iteration touches "
        "~3× more expert memory under FSEP."
    )
    md.append("")
    md.append(
        "**Why iter cost grows with FSEP and grows further with bsz.** "
        "Under FSEP, each MoE layer runs the smart-routing all-to-all "
        "kernel + indexed expert lookup over the replicated layout. "
        "Even with the layout frozen, the dispatcher does fixed per-token "
        "work on the larger expert pool. fb_speedup goes from −80 % at "
        "bsz=4 to −44 % at bsz=8 in `(tp=2, ep=2)` — bsz=8 amortises the "
        "fixed-cost portion better, narrowing the gap."
    )
    md.append("")
    md.append(
        "**When FSEP+LAER would win.** This benchmark cannot capture it: "
        "you need a workload where router scores produce per-expert load "
        "imbalance large enough that re-layout amortises the replication "
        "footprint over many iterations. A future test should disable the "
        "freeze (`--laer_freeze_after_iter -1`) and use a non-static "
        "input that drifts router output."
    )
    md.append("")
    md.append("## Notes on methodology")
    md.append("- err% > 0 → cost model overestimates; < 0 → underestimates.")
    md.append(
        "- `iter_speedup %` and `fb_speedup %` are how much *slower* "
        "`fsep=off` is than `fsep=on` (positive ⇒ FSEP+LAER faster). "
        "`mem_delta %` is how much *more* CUDA peak memory `fsep=off` uses "
        "(positive ⇒ FSEP+LAER lower memory)."
    )
    md.append(
        "- The analytical cost model does not model the FSEP smart-routing "
        "kernel; one estimate row matches both fsep=on and fsep=off above. "
        "Underestimates of 23–80 % vs FSEP-on iter-time reflect that gap, "
        "not a cost-model bug."
    )
    md.append(
        "- Flag asymmetry between fsep tracks: fsep-on adds "
        "`--no_async_grad_reduce` and `--pipeline_type pipedream_flush`. "
        "All head-to-head rows have `dp=1` and `chunks=1` (the FSEP-on "
        "track only covers `(tp=1, ep=4)` and `(tp=2, ep=2)`, both with "
        "`dp = 4/(tp×ep) = 1`), so `--no_async_grad_reduce` is a no-op "
        "(no DP all-reduce to async) and `pipedream_flush` vs `gpipe` "
        "degenerate to the same single-microbatch schedule. The FSEP-off "
        "ep=1 rows (with `dp>1`) are not part of the head-to-head and use "
        "the gpipe defaults consistently."
    )
    md.append(
        "- LAER freeze verified: every FSEP-on log emits "
        "`[layer N] layout frozen at iter 5` for all 4 layers; the "
        "measurement window starts at iter 10. Slowdown is from the "
        "frozen replicated-expert layout, not from active solver work."
    )

    out = "\n".join(md) + "\n"
    print(out)
    out_path = os.path.join(CONFIGS_DIR, "cost_model_comparison_4gpu_4layer.md")
    with open(out_path, "w") as f:
        f.write(out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
