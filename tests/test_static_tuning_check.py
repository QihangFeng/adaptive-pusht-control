"""Selection and saved-trajectory checks; no simulations are rerun."""

import gzip
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def check_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("static_tuning_check")


def test_robust_selection_requires_both_fits_to_meet_budget(check_module, monkeypatch):
    monkeypatch.setattr(check_module.pilot, "TUNINGS", {"fragile": (0.1, 1), "balanced": (0.2, 1)})
    rows = []
    for tuning, violations, throughputs in (
        ("fragile", (0.01, 0.06), (100, 100)),
        ("balanced", (0.03, 0.04), (30, 25)),
    ):
        for variant, violation, throughput in zip(
            ("nominal", "omit_batch_64"), violations, throughputs
        ):
            rows.append(dict(tuning=tuning, variant=variant, integrated_violation_pu_s=violation,
                             throughput_tps=throughput, worst_vmin=0.945, worst_vmax=1.04,
                             itl_violation_fraction=0))
    aggregate = check_module.robust_rows(rows)
    assert aggregate[1]["throughput_tps"] == 25
    assert check_module.pilot.select_tuning(aggregate)["tuning"] == "balanced"


def test_saved_static_tuning_evidence(check_module):
    root = Path(__file__).resolve().parents[1]
    out = root / "results/static-tuning-check"
    if not (out / "summary.json").exists():
        pytest.skip("Follow-up not yet completed")
    summary = json.loads((out / "summary.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    assert summary["prepared_inputs_unchanged"]
    assert len(summary["new_runs"]) == 5 + 2 * 2 * len(selection["evaluated_tunings"]) * 2
    assert len(summary["new_runs"]) <= protocol["max_new_simulations"]
    for file, key in (("power_model_pilot.py", "driver_sha256"),
                      ("static_tuning_check.py", "followup_driver_sha256")):
        assert hashlib.sha256((root / "experiments" / file).read_bytes()).hexdigest() == protocol[key]
    aggregate = check_module.robust_rows(summary["development"])
    assert aggregate == selection["worst_case_development"]
    assert check_module.pilot.select_tuning(aggregate)["tuning"] == summary["selected_tuning"]
    for result in summary["new_runs"]:
        if result["stage"] == "evaluation":
            assert result["seed"] in protocol["evaluation_seeds"]
            assert result["scenario"] in protocol["evaluation_scenarios"]
            assert result["tuning"] in selection["evaluated_tunings"]
        assert result["power_parameters"] == protocol["variants"][result["variant"]]
        assert result["all_solves_converged"]
        with gzip.open(out / f"{result['id']}.trace.json.gz", "rt") as stream:
            values = np.array(json.load(stream)["rows"])
        assert values.shape == (3000, 9) and np.isfinite(values).all()
        assert hashlib.sha256(values.tobytes()).hexdigest() == result["trajectory_sha256"]
        np.testing.assert_allclose(values[:, 8], values[:, 1] / values[:, 3] * values[:, 4])
        np.testing.assert_allclose(values[:, 8].mean(), result["throughput_tps"])
        np.testing.assert_allclose(values[:, 7].sum() * 0.1, result["integrated_violation_pu_s"])
        np.testing.assert_allclose((values[:, 7] > 0).sum() * 0.1, result["violation_time_s"])
        assert (values[:, 6].max(), values[:, 5].min()) == (result["worst_vmax"], result["worst_vmin"])
