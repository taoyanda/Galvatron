"""Coverage-equivalence tests for the per-component step-8 aggregator.

The step-6 → step-8 merge replaced an analytical (``bwd_mult × forward``)
formula with a measured per-component fwd+bwd sweep. The two paths
*should* differ numerically — that's the point of the refactor — so
"equivalence" here is about **coverage**: every axis that step 6 exposed
must still be reachable from the new aggregator's outputs.

Axes asserted:
  - **Components:** every shape with mlp logs has both ``attention_fwd_bwd_ms``
    and ``mlp_fwd_bwd_ms`` (step 6's two components).
  - **fwd+bwd:** the per-component times come from ``[stage_time]`` (which
    measures fwd+bwd) — step 6's forward-only mode is no longer required.
  - **Shapes:** the (tp, ep, micro_bsz, pp) shape set covered by the
    per-component sweep is a *superset* of the shapes covered by the
    legacy "all" sweep — so step 8 picks up everything step 6 plus more.
  - **FSEP:** for shapes where FSEP applies, both fsep=on and fsep=off
    mlp measurements are present (so per-shape FSEP overhead is derivable
    end-to-end, not extrapolated from a single shape).
  - **DP:** ``unit_dp_modes`` is recorded under each shape's breakdown so
    sweeps that ran the same shape under multiple DP modes are visible.

Numeric checks are limited to *derived* quantities that follow a known
formula from the inputs (e.g. ``time_overhead_per_expert_layer_ms_from_mlp
== (mlp_on - mlp_off) / num_layers``) — never to compare the new
measurements against the analytical values they replaced.

Run via:
    python -m pytest galvatron/models/moe/scripts/test_per_component_aggregator.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


@pytest.fixture
def aggregator(monkeypatch, tmp_path):
    """Reload the aggregator with LOG_DIR/CONFIGS_DIR pinned to a tmpdir."""
    log_dir = tmp_path / "logs"
    configs_dir = tmp_path / "configs"
    log_dir.mkdir()
    configs_dir.mkdir()

    if "profile_cost_model_terms" in sys.modules:
        del sys.modules["profile_cost_model_terms"]
    module = importlib.import_module("profile_cost_model_terms")
    monkeypatch.setattr(module, "LOG_DIR", str(log_dir))
    monkeypatch.setattr(module, "CONFIGS_DIR", str(configs_dir))
    return module, log_dir, configs_dir


# ---------------------------------------------------------------------------
# Synthetic-log helpers.
# ---------------------------------------------------------------------------
_TEMPLATE_FULL = (
    "[real_measure] params_mb={params}\n"
    "[real_measure] optimizer_mb={opt} activation_peak_mb={act}"
    " cuda_peak_mb={peak}\n"
    "[stage_time] fwd_bwd_ms={fwd_bwd} opt_ms={opt_ms} window=[10,15)\n"
    "Average iteration time is: {iter_s}\n"
)
_TEMPLATE_NO_OPT = (
    "[stage_time] fwd_bwd_ms={fwd_bwd} opt_ms=0 window=[10,15)\n"
    "Average iteration time is: {iter_s}\n"
)


def _write_log(log_dir, name, **kwargs):
    body = (_TEMPLATE_FULL if "params" in kwargs else _TEMPLATE_NO_OPT).format(
        **kwargs
    )
    (log_dir / name).write_text(body)


# A small but axis-complete sweep emulating step 8's matrix:
#   - 3 (tp, ep) shapes: (1,4), (2,2), (1,2)
#   - 2 DP modes for the (1,4) shape (zero3 + zero2sdp) to assert DP coverage
#   - FSEP on/off for shapes with ep > 1
#   - all + attention + mlp for each shape (with the shell skip rule:
#     attention rows only at fsep=off — fsep is component-invariant for
#     attention so the on row would be redundant)
def _seed_step8_matrix(log_dir):
    base_full = dict(
        params=200.0, opt=400.0, act=1500.0, peak=8000.0,
        opt_ms=20.0, iter_s=0.40,
    )

    # Shape A: tp=1, ep=4, bsz=4 → micro_bsz=1; both DP modes.
    for dp_mode in ("zero3", "zero2sdp"):
        _write_log(log_dir,
                   f"cost_model_real_tp1_ep4_{dp_mode}_bsz4_fsepoff_nl4.log",
                   fwd_bwd=300.0, **base_full)
        _write_log(log_dir,
                   f"cost_model_real_tp1_ep4_{dp_mode}_bsz4_fsepon_nl4.log",
                   fwd_bwd=350.0, **base_full)
        _write_log(log_dir,
                   f"cost_model_real_tp1_ep4_{dp_mode}_bsz4_fsepoff_nl4_unitmlp.log",
                   fwd_bwd=180.0, **base_full)
        _write_log(log_dir,
                   f"cost_model_real_tp1_ep4_{dp_mode}_bsz4_fsepon_nl4_unitmlp.log",
                   fwd_bwd=230.0, **base_full)
        _write_log(log_dir,
                   f"cost_model_real_tp1_ep4_{dp_mode}_bsz4_fsepoff_nl4_unitattention.log",
                   fwd_bwd=120.0, iter_s=0.20)

    # Shape B: tp=2, ep=2, bsz=4 → micro_bsz=4 (data_ranks=2); single DP mode.
    _write_log(log_dir,
               "cost_model_real_tp2_ep2_zero3_bsz4_fsepoff_nl4.log",
               fwd_bwd=290.0, **base_full)
    _write_log(log_dir,
               "cost_model_real_tp2_ep2_zero3_bsz4_fsepon_nl4.log",
               fwd_bwd=340.0, **base_full)
    _write_log(log_dir,
               "cost_model_real_tp2_ep2_zero3_bsz4_fsepoff_nl4_unitmlp.log",
               fwd_bwd=170.0, **base_full)
    _write_log(log_dir,
               "cost_model_real_tp2_ep2_zero3_bsz4_fsepon_nl4_unitmlp.log",
               fwd_bwd=215.0, **base_full)
    _write_log(log_dir,
               "cost_model_real_tp2_ep2_zero3_bsz4_fsepoff_nl4_unitattention.log",
               fwd_bwd=115.0, iter_s=0.20)

    # Shape C: tp=1, ep=2, bsz=2 → micro_bsz=1 (data_ranks=4); FSEP-off only
    # (per the FSEP-on track exclusion when pp*tp would leave dp_of_ep=1 in
    # some setups; here it just exercises an FSEP-off-only shape).
    _write_log(log_dir,
               "cost_model_real_tp1_ep2_zero3_bsz2_fsepoff_nl4.log",
               fwd_bwd=260.0, **base_full)
    _write_log(log_dir,
               "cost_model_real_tp1_ep2_zero3_bsz2_fsepoff_nl4_unitmlp.log",
               fwd_bwd=160.0, **base_full)
    _write_log(log_dir,
               "cost_model_real_tp1_ep2_zero3_bsz2_fsepoff_nl4_unitattention.log",
               fwd_bwd=110.0, iter_s=0.20)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_log_regex_accepts_both_flavours(aggregator):
    """Filename grammar covers the legacy and the new ``_unit{X}`` axis."""
    module, _, _ = aggregator
    assert module._LOG_NAME_RE.match(
        "cost_model_real_tp1_ep4_zero3_bsz4_fsepoff_nl4.log"
    ), "legacy filename must still match"
    new_match = module._LOG_NAME_RE.match(
        "cost_model_real_tp1_ep4_zero3_bsz4_fsepoff_nl4_unitmlp.log"
    )
    assert new_match, "new _unit-suffixed filename must match"
    # Group order: tp(1), ep(2), dp_mode(3), bsz(4), fsep(5), nl(6),
    # pp(7), chunks(8 — added when chunks=2 calibration was wired in),
    # unit(9). Legacy logs don't carry chunks; the unit lands in group 9.
    assert new_match.group(9) == "mlp"


def test_breakdown_covers_attention_and_mlp_per_shape(aggregator):
    """Component coverage: every shape in the unit-row sweep exposes both
    attention and mlp fwd+bwd, isolated per dp_mode. Step 6 measured
    these forward-only — the new sweep must at minimum still measure
    both, just now end-to-end (fwd+bwd) and split by dp_mode."""
    module, log_dir, configs_dir = aggregator
    _seed_step8_matrix(log_dir)
    module.main()

    runtime = json.loads(
        (configs_dir / f"runtime_profiling_{module.PRECISION}_{module.MODEL}.json").read_text()
    )
    by_shape = runtime["by_shape"]

    shapes_with_unit = [
        (k, v) for k, v in by_shape.items() if v.get("unit_breakdown")
    ]
    assert shapes_with_unit, "no shapes carry a unit_breakdown — coverage gap"
    for shape_key, entry in shapes_with_unit:
        block = entry["unit_breakdown"]
        per_dp = block.get("per_dp_mode")
        assert per_dp, f"{shape_key}: per_dp_mode missing"
        for dp_mode, dp_block in per_dp.items():
            assert "attention_fwd_bwd_ms" in dp_block, (
                f"{shape_key}/{dp_mode}: missing attention component"
            )
            assert "mlp_fwd_bwd_ms" in dp_block, (
                f"{shape_key}/{dp_mode}: missing mlp component"
            )


def test_breakdown_axis_is_fwd_plus_bwd(aggregator):
    """The component times are pulled from ``[stage_time]`` lines, which
    instrument fwd+bwd (not just fwd). Step 6's forward-only mode is
    therefore subsumed: any backward-time contribution previously
    estimated via ``bwd_mult`` is now directly captured."""
    module, log_dir, _ = aggregator
    _seed_step8_matrix(log_dir)
    rows = module._enumerate_logs()

    unit_rows = [r for r in rows if r.get("profile_unit") in {"attention", "mlp"}]
    assert unit_rows, "no per-component rows parsed"
    for r in unit_rows:
        assert r["fwd_bwd_ms"] is not None, (
            f"{r['profile_unit']} row {r}: must report fwd_bwd_ms (fwd+bwd)"
        )


def test_unit_shape_set_supersets_legacy_shape_set(aggregator):
    """Step 8 is supposed to cover at least every (tp, ep, micro_bsz, pp)
    that step 6 covered — and ideally more. The old "step 6 forward-only"
    set maps to the legacy ``profile_unit=all`` rows here; the new
    per-component set must include every such tuple."""
    module, log_dir, _ = aggregator
    _seed_step8_matrix(log_dir)
    rows = module._enumerate_logs()

    def _shape_axes(rs):
        return {
            (r["tp"], r["ep"],
             module._micro_bsz_at_calibration(r["global_bsz"]), r["pp"])
            for r in rs
        }

    all_axes = _shape_axes(r for r in rows if r["profile_unit"] == "all")
    mlp_axes = _shape_axes(r for r in rows if r["profile_unit"] == "mlp")
    attn_axes = _shape_axes(r for r in rows if r["profile_unit"] == "attention")
    assert all_axes <= mlp_axes, (
        f"mlp coverage missing legacy shapes: {all_axes - mlp_axes}"
    )
    assert all_axes <= attn_axes, (
        f"attention coverage missing legacy shapes: {all_axes - attn_axes}"
    )


def test_fsep_on_and_off_both_present_for_mlp(aggregator):
    """FSEP coverage: any (tp, ep, micro_bsz, pp) that has *any* fsep=on
    mlp row must also have a matching fsep=off mlp row, so per-shape
    FSEP overhead can be derived end-to-end. (Attention rows are
    fsep-invariant, so we don't require an fsep=on attention row.)"""
    module, log_dir, _ = aggregator
    _seed_step8_matrix(log_dir)
    rows = module._enumerate_logs()

    mlp_rows = [r for r in rows if r["profile_unit"] == "mlp"]
    pairs: dict = {}
    for r in mlp_rows:
        key = (r["tp"], r["ep"],
               module._micro_bsz_at_calibration(r["global_bsz"]), r["pp"])
        pairs.setdefault(key, set()).add(r["fsep"])

    for key, fseps in pairs.items():
        if "on" in fseps:
            assert "off" in fseps, (
                f"shape {key} has fsep=on mlp row without a matching fsep=off"
            )


def test_fsep_overhead_derived_from_mlp_delta_per_dp_mode(aggregator):
    """Per-dp_mode FSEP overhead from mlp on/off equals
    ``(mlp_on - mlp_off) / num_layers`` for that mode (a pass-through
    identity from inputs, not a comparison against the analytical path
    step 8 replaces). The seeded matrix uses identical mlp values across
    DP modes so the per-mode overheads coincide — what we assert is the
    structural separation and value identity."""
    module, log_dir, configs_dir = aggregator
    _seed_step8_matrix(log_dir)
    module.main()

    fsep = json.loads(
        (configs_dir / f"fsep_overhead_profiling_{module.PRECISION}_{module.MODEL}.json").read_text()
    )
    a_entry = fsep["by_shape"]["tp1_ep4_micro_bsz4_seq4096"]
    per_dp = a_entry["time_overhead_per_expert_layer_ms_from_mlp_per_dp_mode"]
    assert set(per_dp.keys()) == {"zero3", "zero2sdp"}
    # mlp on=230, off=180, num_layers=4 → 12.5 ms/layer for each mode.
    assert per_dp["zero3"] == pytest.approx(12.5)
    assert per_dp["zero2sdp"] == pytest.approx(12.5)
    # Legacy scalar = mean across modes.
    assert a_entry["time_overhead_per_expert_layer_ms_from_mlp"] == pytest.approx(12.5)

    # Shape B was profiled under zero3 only.
    # gbsz=4 at chunks=1 → micro_bsz=4 (cost-model key uses micro_bsz
    # directly; DP/EP only enter as a feasibility guard).
    b_entry = fsep["by_shape"]["tp2_ep2_micro_bsz4_seq4096"]
    per_dp_b = b_entry["time_overhead_per_expert_layer_ms_from_mlp_per_dp_mode"]
    assert set(per_dp_b.keys()) == {"zero3"}
    # mlp on=215, off=170, num_layers=4 → 11.25.
    assert per_dp_b["zero3"] == pytest.approx(11.25)
    assert b_entry["time_overhead_per_expert_layer_ms_from_mlp"] == pytest.approx(11.25)


def test_dp_modes_isolated_in_breakdown(aggregator):
    """DP isolation: ``per_dp_mode`` records each DP mode separately and
    samples never average across modes. Step 6 didn't differentiate DP
    modes; step 8 keeps them as a real axis so the cost model can pick
    the matching block by ``(zero_stage, sdp)``."""
    module, log_dir, configs_dir = aggregator
    _seed_step8_matrix(log_dir)
    module.main()

    runtime = json.loads(
        (configs_dir / f"runtime_profiling_{module.PRECISION}_{module.MODEL}.json").read_text()
    )
    breakdown = runtime["by_shape"]["tp1_ep4_micro_bsz4_seq4096_fsepoff"]["unit_breakdown"]
    assert set(breakdown["dp_modes"]) == {"zero3", "zero2sdp"}
    per_dp = breakdown["per_dp_mode"]
    # One mlp row per dp_mode in the seed — proves they weren't combined.
    assert per_dp["zero3"]["mlp_n_samples"] == 1
    assert per_dp["zero2sdp"]["mlp_n_samples"] == 1

    # Shape C was profiled under zero3 only — only that mode appears.
    breakdown_c = runtime["by_shape"]["tp1_ep2_micro_bsz2_seq4096_fsepoff"]["unit_breakdown"]
    assert breakdown_c["dp_modes"] == ["zero3"]
