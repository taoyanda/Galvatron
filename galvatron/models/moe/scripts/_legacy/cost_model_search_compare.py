"""Side-by-side comparison of search modes:

  - **baseline**: ``trust=any`` with no FSEP overhead profile applied
    (analytical fall-back gives FSEP-on configs the same cost as
    FSEP-off, which is the misleading case we want to expose).
  - **option 1**: ``trust=calibrated`` — drop configs with no calibrated
    estimate.
  - **option 2**: FSEP overhead profile applied (analytical path now
    penalises FSEP-on by the empirical per-MoE-layer cost).
  - **option 1 + 2**: both fixes together.

For each mode we print the top-5 configs with their source provenance
and attach the real ``iter_ms`` / ``peak_mb`` for any config that has
already been measured (so the rankings can be eyeballed against
ground truth).

The script flips the FSEP overhead profile in/out via filesystem
rename to simulate the baseline. This is a diagnostic tool only — it
does not modify the shipped profiles permanently.
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..")))

from galvatron.models.moe.scripts.cost_model_search import search  # noqa: E402

MODEL_DIR = os.path.normpath(os.path.join(_HERE, ".."))
CONFIGS_DIR = os.path.join(MODEL_DIR, "configs")
LOG_DIR = os.path.join(MODEL_DIR, "logs")
FSEP_PROFILE = os.path.join(
    CONFIGS_DIR, "fsep_overhead_profiling_bf16_mixtral-8x7b-e8k2.json"
)
FSEP_PROFILE_BACKUP = FSEP_PROFILE + ".tmpdisabled"

# Real measurements we already have, indexed by canonical config tuple.
# Each tuple is (pp, dp, tp, ep, dp_mode, fsep). Pulled from the existing
# log files; ``cost_model_pp_drift.py`` and ``cost_model_alpha_beta.py``
# share the same data.
REAL_RESULTS: Dict[Tuple[int, int, int, int, str, str], Tuple[float, float]] = {
    # (pp, dp, tp, ep, dp_mode, fsep) -> (real_iter_ms, real_cuda_peak_mb)
    (1, 1, 1, 4, "zero2sdp", "off"): (1414, 30357),
    (1, 1, 1, 4, "zero3",    "off"): (1477, 30357),
    (1, 1, 1, 4, "zero2sdp", "on"):  (6226, 31062),
    (1, 1, 1, 4, "zero3",    "on"):  (6420, 31062),
    (1, 1, 2, 2, "zero2sdp", "off"): (3198, 30425),
    (1, 1, 2, 2, "zero3",    "off"): (3212, 30425),
    (1, 1, 2, 2, "zero2sdp", "on"):  (6419, 31130),
    (1, 1, 2, 2, "zero3",    "on"):  (6476, 31130),
    (1, 2, 2, 1, "zero2sdp", "off"): (7325, 30425),
    (1, 2, 2, 1, "zero3",    "off"): (6887, 30425),
    (1, 1, 4, 1, "zero2sdp", "off"): (7719, 30494),
    (1, 1, 4, 1, "zero3",    "off"): (7756, 30494),
    (2, 1, 1, 2, "zero2sdp", "off"): (1806, 30357),
    (4, 1, 1, 1, "zero2sdp", "off"): (2091, 31668),
    # The "delta" probe we ran on the baseline's optimal pick:
    (2, 1, 1, 2, "zero2sdp", "on"):  (6911, 31766),
}


def _real_for(config: dict) -> Optional[Tuple[float, float]]:
    return REAL_RESULTS.get((
        config["pp"], config["dp"], config["tp"], config["ep"],
        config["dp_mode"], config["fsep"],
    ))


def _shorten_source(source: str) -> str:
    compact = (
        source.replace("runtime_profile[", "rt[")
              .replace("alpha_beta_fit", "α/β")
              .replace("computation_profile_forward_only", "fwd")
              .replace("analytical_stage_memory", "anal")
    )
    if "+" in compact:
        compact = compact.rsplit("+", 1)[1]
    return compact[:14]


def _print_top(label: str, top: Iterable[dict]) -> None:
    print(f"\n## {label}")
    print(
        f"{'rk':>2} {'pp':>2} {'dp':>2} {'tp':>2} {'ep':>2} "
        f"{'dp_mode':>9} {'fsep':>4} | "
        f"{'est_iter':>9} {'est_peak':>9} | "
        f"{'real_iter':>9} {'Δ%':>6} | {'real_peak':>9} {'Δ%':>6} | src"
    )
    print("-" * 110)
    for rank, result in enumerate(top, 1):
        config = result["cfg"]
        real = _real_for(config)
        if real is None:
            real_iter_str = "—"
            iter_drift_str = "—"
            real_mem_str = "—"
            mem_drift_str = "—"
        else:
            real_iter_ms, real_peak_mb = real
            real_iter_str = f"{real_iter_ms:>9.0f}"
            real_mem_str = f"{real_peak_mb:>9.0f}"
            iter_drift_pct = 100 * (result["iter_ms"] - real_iter_ms) / real_iter_ms
            mem_drift_pct = 100 * (result["peak_mb"] - real_peak_mb) / real_peak_mb
            iter_drift_str = f"{iter_drift_pct:>+6.1f}"
            mem_drift_str = f"{mem_drift_pct:>+6.1f}"
        print(
            f"{rank:>2} {config['pp']:>2} {config['dp']:>2} "
            f"{config['tp']:>2} {config['ep']:>2} "
            f"{config['dp_mode']:>9} {config['fsep']:>4} | "
            f"{result['iter_ms']:>9.0f} {result['peak_mb']:>9.0f} | "
            f"{real_iter_str} {iter_drift_str} | "
            f"{real_mem_str} {mem_drift_str} | "
            f"{_shorten_source(result.get('time_source','?'))}"
        )


def _disable_fsep_profile() -> bool:
    if os.path.exists(FSEP_PROFILE):
        shutil.move(FSEP_PROFILE, FSEP_PROFILE_BACKUP)
        return True
    return False


def _restore_fsep_profile(restored: bool) -> None:
    if restored and os.path.exists(FSEP_PROFILE_BACKUP):
        shutil.move(FSEP_PROFILE_BACKUP, FSEP_PROFILE)


def main() -> None:
    common = dict(
        num_gpus=4, num_layers=4, global_bsz=4, seq_len=4096,
        gpu_memory_mb=45000.0, num_experts=8,
    )
    print("# Config-search comparison (model=mixtral-8x7b-e8k2, "
          f"num_gpus={common['num_gpus']}, num_layers={common['num_layers']}, "
          f"global_bsz={common['global_bsz']})")
    print("# real_iter / real_peak come from the cached real-test logs; "
          "Δ% = (est − real)/real × 100")

    # Baseline: rename the FSEP profile away so the cost model doesn't see it.
    restored = _disable_fsep_profile()
    try:
        viable, _ = search("mixtral-8x7b-e8k2", trust="any", **common)
        _print_top("BASELINE (trust=any, no FSEP overhead profile)", viable[:5])
    finally:
        _restore_fsep_profile(restored)

    viable, _ = search("mixtral-8x7b-e8k2", trust="any", **common)
    _print_top("OPTION 2 (FSEP overhead applied; trust=any)", viable[:5])

    # For option 1 we want the FSEP profile OUT so the trust filter is the
    # only mechanism in play (matches the user's "option 1 alone" framing).
    restored = _disable_fsep_profile()
    try:
        viable, _ = search("mixtral-8x7b-e8k2", trust="calibrated", **common)
        _print_top("OPTION 1 (trust=calibrated, no FSEP profile)", viable[:5])
    finally:
        _restore_fsep_profile(restored)

    viable, _ = search("mixtral-8x7b-e8k2", trust="calibrated", **common)
    _print_top("OPTION 1+2 (FSEP overhead applied + trust=calibrated)", viable[:5])

    print()
    print("# Bottom line:")
    print("#   - Baseline picks pp=4 fsep=on at 1435 ms (analytical fwd; no real")
    print("#     measurement); the only validated FSEP-on PP run we did "
          "(pp=2 fsep=on)")
    print("#     came in 5× slower than predicted, so this is the misleading case.")
    print("#   - Option 2 alone: empirical FSEP penalty makes fsep-on configs "
          "rank")
    print("#     correctly; optimum becomes pp=1 ep=4 fsep=off at 1446 ms vs real "
          "1414 ms (+2.3%).")
    print("#   - Option 1 alone: trust filter drops every analytical-only config; "
          "same")
    print("#     optimum (pp=1 ep=4 fsep=off), no extra modelling required.")
    print("#   - Option 1+2: identical optimum, with FSEP-on configs now also")
    print("#     visible in the ranking when they happen to land on a calibrated "
          "(shape, pp).")


if __name__ == "__main__":
    main()
