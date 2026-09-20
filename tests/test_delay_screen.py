"""Causality, information access, and saved evidence for the delay screen."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.delay_screen import ObservationView, SETTINGS, VoltageChannel, select_setting


def test_channel_waits_for_arrival_and_copies_inputs():
    channel = VoltageChannel()
    voltage, power = np.array([0.98]), np.array([1., 2., 3.])
    channel.publish(1.0, 0.5, voltage, power)
    voltage[0], power[0] = 99, 99
    assert channel.receive(1.49) is None
    packet = channel.receive(1.5)
    assert packet.measured == 1 and packet.arrives == 1.5
    np.testing.assert_array_equal(packet.voltage, [0.98])
    np.testing.assert_array_equal(packet.power_kw, [1, 2, 3])
    with pytest.raises(ValueError):
        packet.voltage[0] = 99


def test_older_late_packet_does_not_replace_newer_sample():
    channel = VoltageChannel()
    channel.publish(0, 3, [1], [1, 2, 3])
    channel.publish(1, 0.5, [2], [1, 2, 3])
    assert channel.receive(2).measured == 1
    assert channel.receive(3).measured == 1
    with pytest.raises(ValueError):
        channel.publish(4, -1, [1], [1, 2, 3])


def test_predictor_sign_units_and_zero_change():
    channel = VoltageChannel()
    channel.publish(0, 0, [0.94, 1.01], [100, 200, 300])
    view = ObservationView(predict=True)
    view.packet = channel.receive(0)
    view.sensitivity = np.array([[-0.001, 0, 0], [0, -0.001, 0]])
    view.current_power_kw = np.array([90, 200, 300])
    np.testing.assert_allclose(view.voltages_vector(), [0.95, 1.01])
    view.current_power_kw = view.packet.power_kw.copy()
    np.testing.assert_array_equal(view.voltages_vector(), view.packet.voltage)
    # No delegation of unknown reads to a real grid, including sensitivity's v0.
    assert not hasattr(view, "state") and not hasattr(view, "history")
    assert view.estimate_sensitivity()[1] is None
    view.packet = None
    with pytest.raises(RuntimeError):
        view.voltages_vector()


def test_selection_requires_all_delayed_development_cases():
    rows = []
    for i, setting in enumerate(SETTINGS):
        for j in range(4):
            rows.append(dict(setting=setting, delay_s=2, integrated_violation_pu_s=0.05,
                             worst_vmin=0.95, worst_vmax=1.05, itl_violation_fraction=0,
                             throughput_tps=100+i, batch_switches=10))
    rows[-1]["integrated_violation_pu_s"] = 0.11
    assert select_setting(rows)[0] == "half_primal_dual"


@pytest.mark.parametrize("directory", ["delay-screen", "delay-prototype"])
def test_saved_results_are_causal_and_metrics_match(directory):
    out = Path(__file__).resolve().parents[1] / "results" / directory
    if not (out / "summary.json").exists():
        pytest.skip("Stage not completed")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["prepared_inputs_unchanged"]
    for result in summary["runs"]:
        with gzip.open(out / f"{result['id']}.trace.json.gz", "rt") as stream:
            trace = json.load(stream)
        values = np.asarray(trace["rows"])
        assert values.shape == (3000, 9) and np.isfinite(values).all()
        assert hashlib.sha256(values.tobytes()).hexdigest() == result["trajectory_sha256"]
        np.testing.assert_allclose(values[:, 7].sum()*0.1, result["integrated_violation_pu_s"])
        np.testing.assert_allclose(values[:, 8].mean(), result["throughput_tps"])
        np.testing.assert_allclose((values[:, 7] > 0).sum()*0.1, result["violation_time_s"])
        assert result["all_solves_converged"]
        previous = -1
        for row in trace["observations"]:
            if not row["available"]:
                assert row["time_s"] < result["delay_s"]
                continue
            assert previous <= row["measured_s"] <= row["arrival_s"] <= row["time_s"]+1e-10
            previous = row["measured_s"]
            assert row["age_s"] == pytest.approx(result["delay_s"])
            sampled = values[round(row["measured_s"] / 0.1)]
            current = values[round(row["time_s"] / 0.1)]
            np.testing.assert_allclose(sum(row["packet_power_kw"])*1000, sampled[2])
            np.testing.assert_allclose(sum(row["current_power_kw"])*1000, current[2])
            # At t=0, upstream sensitivity estimation re-solves the same physical
            # operating point after GridState is emitted. The original OFO also
            # observes that re-solved value. Audit the small startup residual
            # separately; every later packet must match its physical snapshot.
            tolerance = 5e-5 if row["measured_s"] == 0 else 1e-8
            np.testing.assert_allclose([min(row["packet_voltage"]), max(row["packet_voltage"])],
                                       sampled[5:7], rtol=0, atol=tolerance)
            used = np.array(row["packet_voltage"])
            if result["method"] == "predict":
                used += np.array(trace["sensitivity"]) @ (
                    np.array(row["current_power_kw"])-row["packet_power_kw"])
            np.testing.assert_array_equal(used, row["voltage_used"])
    if directory == "delay-screen":
        check = json.loads((out / "zero-delay-check.json").read_text())
        assert check["matches"]
        assert select_setting(summary["runs"])[0] == summary["selected_setting"]
    else:
        baseline = json.loads((out.parent / "delay-screen" / "summary.json").read_text())
        for result in summary["runs"]:
            if result["id"].startswith("dev-") and result["delay_s"] == 0:
                original = next(r for r in baseline["runs"] if
                                r["setting"] == result["setting"] and
                                r["scenario"] == result["scenario"] and r["delay_s"] == 0)
                assert original["trajectory_sha256"] == result["trajectory_sha256"]
