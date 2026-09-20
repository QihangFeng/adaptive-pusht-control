"""Checks for experiment transformations and development-only selection."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from mlenergy.data.modeling import LogisticModel

from experiments.power_model_pilot import BATCHES, make_variants, select_tuning


def fixture_models():
    model = LogisticModel(L=220, b0=470, k=1.1, x0=5.5)
    points = [{"batch": b, "measured_power_w": model.eval(b)} for b in BATCHES]
    return model, make_variants(model, points)


def test_offset_changes_level_not_derivative():
    model, variants = fixture_models()
    for x in np.linspace(3, 9, 30):
        changed = variants["offset_plus_50w"]
        assert np.isclose(changed.eval_x(x), model.eval_x(x) + 50)
        assert changed.deriv_wrt_x(x) == model.deriv_wrt_x(x)


def test_gain_scales_value_and_derivative_without_mutating_original():
    model, variants = fixture_models()
    assert model.L == 220 and model.b0 == 470
    for gain in (0.9, 1.1):
        changed = variants[f"gain_{gain}"]
        for x in np.linspace(3, 9, 30):
            assert np.isclose(changed.eval_x(x), gain * model.eval_x(x))
            assert np.isclose(changed.deriv_wrt_x(x), gain * model.deriv_wrt_x(x))


def test_omission_does_not_use_excluded_value():
    model, variants = fixture_models()
    points = [{"batch": b, "measured_power_w": model.eval(b) if b != 64 else 99999}
              for b in BATCHES]
    other = make_variants(model, points)
    assert other["omit_batch_64"] == variants["omit_batch_64"]


def test_selection_prefers_throughput_only_inside_predeclared_budget():
    rows = [dict(tuning="safe", integrated_violation_pu_s=0.01, worst_vmin=0.945,
                 worst_vmax=1.04, itl_violation_fraction=0, throughput_tps=20),
            dict(tuning="fast", integrated_violation_pu_s=0.04, worst_vmin=0.942,
                 worst_vmax=1.04, itl_violation_fraction=0, throughput_tps=30),
            dict(tuning="ineligible", integrated_violation_pu_s=0.1, worst_vmin=0.93,
                 worst_vmax=1.04, itl_violation_fraction=0, throughput_tps=100)]
    assert select_tuning(rows)["tuning"] == "fast"
    assert select_tuning([rows[2]])["tuning"] == "ineligible"


def test_completed_artifact_consistency():
    """Audit saved evidence when present; unit tests also work before a pilot."""
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get("PILOT_RESULTS", root / "results/power-model-pilot-verified"))
    if not (output / "summary.json").is_file():
        pytest.skip("No completed pilot artifacts at PILOT_RESULTS")
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["driver_sha256"] == hashlib.sha256(
        (root / "experiments/power_model_pilot.py").read_bytes()).hexdigest()
    assert summary["prepared_inputs_unchanged"] and summary["offset_trajectories_identical"]
    development = [r for r in summary["runs"] if r["stage"] == "development"]
    assert select_tuning(development)["tuning"] == summary["selected_tuning"]
    for result in summary["runs"]:
        with gzip.open(output / f"{result['id']}.trace.json.gz", "rt") as stream:
            values = np.array(json.load(stream)["rows"])
        assert values.shape == (3000, 9) and np.isfinite(values).all()
        assert result["all_solves_converged"]
        assert result["memory_sampling_scope"] == "sum of launcher and recursive child RSS"
        assert hashlib.sha256(values.tobytes()).hexdigest() == result["trajectory_sha256"]
        np.testing.assert_allclose(np.diff(values[:, 0]), 0.1, atol=1e-12)
        np.testing.assert_allclose(values[:, 8], values[:, 1] / values[:, 3] * values[:, 4])
        assert set(values[:, 1]).issubset(BATCHES)
        np.testing.assert_allclose(values[:, 8].mean(), result["throughput_tps"])
        np.testing.assert_allclose(values[:, 7].sum() * 0.1, result["integrated_violation_pu_s"])
        np.testing.assert_allclose((values[:, 7] > 0).sum() * 0.1, result["violation_time_s"])
