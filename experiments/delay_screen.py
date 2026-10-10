"""Bounded delayed-voltage screen and optional power-history predictor.

Reuses pinned OpenG2G, measured traces and the earlier pilot's plant settings.
No upstream files are modified. Run `screen` before `prototype`; the latter is
an explicit second stage, not an automatic claim that delay requires a method.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
import heapq
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

try:
    from . import power_model_pilot as pilot
except ImportError:
    import power_model_pilot as pilot

import numpy as np

# Deliberately include the earlier robust tuning, reduced primal AND dual steps,
# and slower updates. The physical channel delay does not change with dt_control.
SETTINGS = {
    "reference": (0.1, 1e6, 1.0, 1),
    "double_step": (0.2, 1e6, 1.0, 1),
    "half_voltage_gain": (0.1, 0.5e6, 1.0, 1),
    "half_primal_dual": (0.05, 1e6, 0.5, 1),
    "slow_update": (0.1, 1e6, 1.0, 2),
}
DELAYS = (0.0, 0.5, 2.0)
DEVELOPMENT = ("development", "two_pulses")
SCENARIOS = dict(pilot.SCENARIOS,
                 heldout_ramps=[(3900, 45, 57), (2700, 110, 145),
                                (4700, 205, 220), (2400, 270, 280)])


@dataclass(frozen=True)
class Packet:
    measured: float
    arrives: float
    voltage: np.ndarray
    power_kw: np.ndarray


class VoltageChannel:
    """Causal, copied packets; late older arrivals never replace newer data."""

    def __init__(self):
        self.pending = []
        self.latest = None
        self.sequence = 0

    def publish(self, measured, delay, voltage, power_kw):
        if delay < 0 or not np.isfinite([measured, delay]).all():
            raise ValueError("Finite timestamp and nonnegative delay required")
        voltage, power_kw = np.array(voltage, copy=True), np.array(power_kw, copy=True)
        if not np.isfinite(voltage).all() or not np.isfinite(power_kw).all():
            raise ValueError("Packet values must be finite")
        voltage.setflags(write=False)
        power_kw.setflags(write=False)
        packet = Packet(measured, measured + delay, voltage, power_kw)
        heapq.heappush(self.pending, (packet.arrives, self.sequence, packet))
        self.sequence += 1

    def receive(self, now):
        # Small tolerance handles binary representation of 0.1-second ticks.
        while self.pending and self.pending[0][0] <= now + 1e-10:
            _, _, packet = heapq.heappop(self.pending)
            if self.latest is None or packet.measured > self.latest.measured:
                self.latest = packet
        return self.latest


class ObservationView:
    """Restricted OFO interface: contains no reference to the physical grid.

    Sensitivity is fixed using the initial operating point for every method.
    Current local power is allowed; current voltage is not exposed to OFO.
    """

    def __init__(self, predict=False):
        self.predict = predict
        self.v_index = ()
        self.sensitivity = None
        self.packet = None
        self.current_power_kw = None

    def estimate_sensitivity(self, **kwargs):
        if self.sensitivity is None:
            raise RuntimeError("Offline sensitivity not initialized")
        return self.sensitivity.copy(), None

    def voltages_vector(self):
        if self.packet is None:
            raise RuntimeError("No voltage packet has arrived")
        voltage = self.packet.voltage.copy()
        if self.predict:
            voltage += self.sensitivity @ (self.current_power_kw - self.packet.power_kw)
        return voltage


def select_setting(rows):
    """Freeze one setting across all delayed development cases, not per case."""
    aggregate = []
    for setting in SETTINGS:
        cases = [r for r in rows if r["setting"] == setting and r["delay_s"] > 0]
        assert len(cases) == 4
        aggregate.append(dict(
            setting=setting,
            eligible=all(r["integrated_violation_pu_s"] <= 0.10
                         and r["worst_vmin"] >= 0.94 and r["worst_vmax"] <= 1.06
                         and r["itl_violation_fraction"] <= 0.01 for r in cases),
            worst_violation=max(r["integrated_violation_pu_s"] for r in cases),
            min_throughput=min(r["throughput_tps"] for r in cases),
            max_switches=max(r["batch_switches"] for r in cases),
        ))
    eligible = [r for r in aggregate if r["eligible"]]
    if eligible:
        selected = max(eligible, key=lambda r: (r["min_throughput"], -r["max_switches"]))
    else:
        selected = min(aggregate, key=lambda r: (r["worst_violation"], -r["min_throughput"]))
    return selected["setting"], aggregate


def simulate(args):
    pilot.bootstrap(args.openg2g_root)
    from systems import ieee13
    from openg2g.controller.base import Controller
    from openg2g.controller.ofo import LogisticModelStore, OFOBatchSizeController, OFOConfig
    from openg2g.coordinator import Coordinator
    from openg2g.datacenter.config import DatacenterConfig, InferenceModelSpec, ReplicaSchedule
    from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
    from openg2g.datacenter.workloads.inference import InferenceData
    from openg2g.grid.opendss import OpenDSSGrid, dss
    from openg2g.metrics.performance import compute_performance_stats
    from openg2g.metrics.voltage import compute_allbus_voltage_stats

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    out = Path(args.output).resolve()  # OpenDSS changes cwd.
    robustness = job.get("robustness")
    if robustness is not None:
        try:
            from .robustness_check import external_power_kw, transport_delay
        except ImportError:
            from robustness_check import external_power_kw, transport_delay
    spec = InferenceModelSpec(
        model_label=pilot.LABEL, model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_model="H100", task="lm-arena-chat", precision="bfloat16",
        gpus_per_replica=1, tensor_parallel=1, itl_deadline_s=0.08,
        batch_sizes=pilot.BATCHES, feasible_batch_sizes=pilot.BATCHES)
    data = InferenceData.load(Path(args.prepared_data), (spec,),
                              duration_s=pilot.DURATION, dt_s=pilot.DT)
    schedule = ReplicaSchedule(initial=2400)
    for target, start, end in SCENARIOS[job["scenario"]]:
        schedule = schedule.ramp_to(target, t_start=start, t_end=end)
    dc = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500),
        OfflineWorkload(inference_data=data, replica_schedules={pilot.LABEL: schedule},
                        initial_batch_sizes={pilot.LABEL: 128}),
        name="pilot", dt_s=Fraction(1, 10), seed=job["seed"], total_gpu_capacity=4800)
    system = ieee13()
    grid = OpenDSSGrid(
        dss_case_dir=system["dss_case_dir"], dss_master_file=system["dss_master_file"],
        dt_s=Fraction(1, 10), source_pu=system["source_pu"],
        initial_tap_position=system["initial_taps"], exclude_buses=system["exclude_buses"])
    grid.attach_dc(dc, bus="671")
    external_attachment = None
    if robustness is not None and robustness["external_enabled"]:
        from openg2g.grid.load import ExternalLoad

        class IndependentLoad(ExternalLoad):
            def power_kw(self, t):
                # The pinned backend applies this value to EACH phase.
                return external_power_kw(float(t), robustness)

        grid.attach_load(IndependentLoad(), bus=robustness["external_bus"], power_factor=0.96)
        # Read-back for simulation diagnostics only; never exposed to OFO.
        external_attachment = grid._load_attachments[-1]
    fits = LogisticModelStore.load(Path(args.prepared_data), (spec,))
    primal, gain, dual, period = SETTINGS[job["setting"]]
    config = OFOConfig(
        primal_step_size=primal, w_throughput=0.001, w_switch=1,
        voltage_gradient_scale=gain, v_min=0.95, v_max=1.05,
        voltage_dual_step_size=dual, latency_dual_step_size=dual,
        sensitivity_update_interval=3600, sensitivity_perturbation_kw=100)
    view = ObservationView(predict=job["method"] == "predict")
    observations = []

    class Sensor(Controller[OfflineDatacenter, OpenDSSGrid]):
        """Simulation-only delivery plumbing, not accessible to the optimizer."""

        dt_s = Fraction(1, 10)

        def reset(self):
            self.channel = VoltageChannel()
            view.packet, view.sensitivity, view.current_power_kw = None, None, None
            view.v_index = ()

        def step(self, clock, events):
            t = float(clock.time_s)
            if view.sensitivity is None:
                # Common initial model information, not repeated online sensing.
                view.sensitivity, _ = grid.estimate_sensitivity(perturbation_kw=100, dc=dc)
                view.v_index = tuple(grid.v_index)
            power = dc.state.power_w
            view.current_power_kw = np.array([power.a, power.b, power.c]) / 1000
            delay = job["delay_s"] if robustness is None else transport_delay(t, robustness)
            self.channel.publish(t, delay, grid.voltages_vector(), view.current_power_kw)
            view.packet = self.channel.receive(t)
            return []

    class ObservedOFO(OFOBatchSizeController):
        def step(self, clock, events):
            t = float(clock.time_s)
            if view.packet is None:
                # Keep the initial batch until the first genuine packet arrives.
                observations.append(dict(time_s=t, available=False))
                return []
            packet = view.packet
            used = view.voltages_vector()
            observations.append(dict(time_s=t, available=True,
                measured_s=packet.measured, arrival_s=packet.arrives,
                age_s=t-packet.measured, voltage_used=used.tolist(),
                packet_voltage=packet.voltage.tolist(),
                packet_power_kw=packet.power_kw.tolist(),
                current_power_kw=view.current_power_kw.tolist(),
                # Diagnostic only; never passed to the controller.
                true_voltage=grid.voltages_vector().tolist()))
            return super().step(clock, events)

    controller = ObservedOFO((spec,), dc, view, fits, config=config,
                            dt_s=Fraction(period), initial_batch_sizes={pilot.LABEL: 128})
    coordinator = Coordinator(datacenters=[dc], grid=grid, controllers=[Sensor(), controller],
                              total_duration_s=pilot.DURATION)
    dc_states, grid_states, rows = [], [], []
    external_applied = []
    excluded = set(system["exclude_buses"])
    start = time.perf_counter()
    for tick in coordinator.run_iter():
        assert dss.Solution.Converged(), "Power-flow solve failed"
        state, gs = tick.dc_states[dc.name], tick.grid_state
        dc_states.append(state)
        grid_states.append(gs)
        volts = [v for bus in gs.voltages.buses() if bus not in excluded
                 for v in (gs.voltages[bus].a, gs.voltages[bus].b, gs.voltages[bus].c)
                 if np.isfinite(v)]
        b, itl = state.batch_size_by_model[pilot.LABEL], state.observed_itl_s_by_model[pilot.LABEL]
        n = state.active_replicas_by_model[pilot.LABEL]
        p = sum((state.power_w.a, state.power_w.b, state.power_w.c))
        rows.append([tick.t_s, b, p, itl, n, min(volts), max(volts),
                     sum(max(0.95-v, 0) + max(v-1.05, 0) for v in volts), b/itl*n])
        if robustness is not None:
            phase_kw = [0.0, 0.0, 0.0]
            if external_attachment is not None:
                for phase, name in enumerate(external_attachment.load_names):
                    dss.Loads.Name(name)
                    phase_kw[phase] = dss.Loads.kW()
            expected_kw = external_power_kw(float(tick.t_s), robustness)
            assert np.allclose(phase_kw, expected_kw, rtol=0, atol=1e-9), (
                "Applied external load disagrees with the declared per-phase profile")
            external_applied.append(phase_kw)
    values = np.asarray(rows)
    assert values.shape == (3000, 9) and np.isfinite(values).all()
    assert set(values[:, 1]).issubset(pilot.BATCHES)
    vs = compute_allbus_voltage_stats(grid_states, exclude_buses=tuple(excluded))
    perf = compute_performance_stats(dc_states, itl_deadline_s_by_model={pilot.LABEL: 0.08})
    integral = float(values[:, 7].sum() * pilot.DT)
    assert abs(integral-vs.integral_violation_pu_s) < 1e-8
    assert abs(values[:, 8].mean()-perf.mean_throughput_tps) < 1e-5
    delivered = [r for r in observations if r["available"]]
    error = np.array([np.array(r["voltage_used"])-r["true_voltage"] for r in delivered])
    raw_error = np.array([np.array(r["packet_voltage"])-r["true_voltage"] for r in delivered])
    result = dict(job, config=config.model_dump(), control_period_s=period,
        integrated_violation_pu_s=integral, violation_time_s=vs.violation_time_s,
        worst_vmin=vs.worst_vmin, worst_vmax=vs.worst_vmax,
        throughput_tps=perf.mean_throughput_tps, itl_violation_fraction=perf.itl_deadline_fraction,
        mean_modeled_itl_ms=float(values[:, 3].mean() * 1000),
        voltage_violation_fraction_percent=float(np.mean(values[:, 7] > 0) * 100),
        batch_switches=int(np.count_nonzero(np.diff(values[:, 1]))),
        batch_log2_total_variation=float(np.abs(np.diff(np.log2(values[:, 1]))).sum()),
        voltage_observation_rmse_pu=float(np.sqrt(np.mean(error**2))),
        raw_packet_rmse_pu=float(np.sqrt(np.mean(raw_error**2))),
        mean_age_s=float(np.mean([r["age_s"] for r in delivered])),
        max_age_s=max(r["age_s"] for r in delivered), control_updates=len(delivered),
        startup_skipped_updates=len(observations)-len(delivered),
        all_solves_converged=True, steps=len(rows), simulation_seconds=time.perf_counter()-start,
        trajectory_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    trace = dict(columns=["time_s", "batch", "power_w", "itl_s", "replicas", "vmin",
                          "vmax", "violation_sum_pu", "throughput_tps"], rows=rows,
                 observations=observations, sensitivity=view.sensitivity.tolist())
    if robustness is not None:
        trace["robustness"] = dict(
            robustness,
            external_power_kw_per_phase=[phase_kw[0] for phase_kw in external_applied],
            external_applied_kw_by_phase=external_applied)
        result["external_load_readback_matches_profile"] = True
    with gzip.open(out / f"{job['id']}.trace.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, allow_nan=False)
    pilot.write_json(out / f"{job['id']}.json", result)


def run_job(args, job):
    """Fresh process per OpenDSS run, with bounded time and process-tree RAM."""
    import psutil

    out = Path(args.output).resolve()
    job_path = out / f"{job['id']}.job.json"
    if job_path.exists():
        raise ValueError(f"Refusing to overwrite {job_path}")
    pilot.write_json(job_path, job)
    command = [sys.executable, str(Path(__file__).resolve()), "one", "--openg2g-root",
               args.openg2g_root, "--prepared-data", args.prepared_data,
               "--output", str(out), "--job", str(job_path)]
    start, peak = time.perf_counter(), 0
    with (out / f"{job['id']}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        monitor = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                rss = 0
                try:
                    members = [monitor, *monitor.children(recursive=True)]
                except psutil.NoSuchProcess:
                    members = []
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if rss > 3.5*1024**3 or time.perf_counter()-start > 180:
                    raise RuntimeError("Per-run resource cap exceeded")
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                for child in monitor.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
            process.wait()
    if process.returncode:
        raise RuntimeError(f"Simulation failed: {job['id']}; inspect its log")
    path = out / f"{job['id']}.json"
    result = json.loads(path.read_text())
    result.update(wall_seconds=time.perf_counter()-start, peak_sampled_rss_mib=peak/1024**2,
                  memory_sampling_scope="sum of launcher and recursive child RSS")
    pilot.write_json(path, result)
    print(f"{job['id']}: V={result['integrated_violation_pu_s']:.5f}, "
          f"R={result['throughput_tps']/1e6:.3f} M/s, switches={result['batch_switches']}", flush=True)
    return result


def suite(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new/empty output directory")
    root = Path(args.openg2g_root).resolve()
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    assert subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() == pilot.COMMIT
    assert not subprocess.check_output(git + ["status", "--porcelain"], text=True)
    before = pilot.fingerprint(args.prepared_data)
    prior = json.loads((Path(args.prior_results) / "manifest.json").read_text())
    assert before == prior["prepared_data_sha256"]
    protocol = dict(stage=args.action, openg2g_commit=pilot.COMMIT,
        dataset_revision=pilot.DATA_REVISION, prepared_data_sha256=before,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version, settings=SETTINGS, delays_s=DELAYS, scenarios=SCENARIOS,
        duration_s=300, sampling_period_s=0.1, development_seed=0,
        selection="One setting across four delayed development cases: V<=0.10, "
                  "vmin>=0.94, vmax<=1.06, ITL miss<=0.01 in each; maximize minimum "
                  "throughput, tie fewer switches. If none qualifies minimize worst V.",
        practical_scope="Assumed constant transport delay, not a measured field distribution",
        startup="Hold batch 128 until first genuine packet arrives; include all startup metrics",
        sensitivity="Common initial operating-point model; no online re-estimation",
        problem_screen="Flag V increase >=25% AND >=0.01 pu*s versus same setting fresh, "
                       "or throughput drop >=5% without lower V; inspect full tuning trade-offs",
        prototype_screen="Useful signal: V reduction >=20% AND >=0.01 pu*s at >=98% "
                         "baseline throughput and switches <=max(baseline+2,1.1*baseline). "
                         "Also inspect worst voltage, duration, ITL; not a safety certificate.")
    rows = []
    if args.action == "screen":
        protocol["max_runs"] = 30
        pilot.write_json(out / "protocol.json", protocol)
        old = json.loads((Path(args.prior_results) / "summary.json").read_text())
        reference = next(r for r in old["runs"] if r["id"] == "dev-reference")
        for scenario in DEVELOPMENT:
            for setting in SETTINGS:
                for delay in DELAYS:
                    job = dict(id=f"{scenario}-{setting}-d{delay:g}", scenario=scenario,
                               setting=setting, delay_s=delay, seed=0, method="hold")
                    result = run_job(args, job)
                    rows.append(result)
                    if len(rows) == 1:
                        assert result["trajectory_sha256"] == reference["trajectory_sha256"], (
                            "Zero-delay adapter changed the original trajectory; stop before screening")
                        pilot.write_json(out / "zero-delay-check.json", dict(
                            matches=True, prior_id=reference["id"], prior_results=args.prior_results,
                            trajectory_sha256=result["trajectory_sha256"]))
        selected, aggregate = select_setting(rows)
        pilot.write_json(out / "selection.json", dict(selected_setting=selected,
                                                     development_aggregate=aggregate))
    else:
        screen = Path(args.screen_results)
        selection = json.loads((screen / "selection.json").read_text())
        selected = selection["selected_setting"]
        protocol.update(selected_setting=selected, max_runs=18,
                        evaluation_seeds=[1, 2], evaluation_scenarios=["abrupt", "heldout_ramps"],
                        screen_selection_sha256=hashlib.sha256((screen / "selection.json").read_bytes()).hexdigest())
        pilot.write_json(out / "protocol.json", protocol)
        # No predictor parameters are fitted: all retain the same frozen H and OFO settings.
        for scenario in DEVELOPMENT:
            for delay in DELAYS:
                job = dict(id=f"dev-{scenario}-predict-d{delay:g}", scenario=scenario,
                           setting=selected, delay_s=delay, seed=0, method="predict")
                rows.append(run_job(args, job))
        for scenario in ("abrupt", "heldout_ramps"):
            for seed in (1, 2):
                for method, delay in (("hold", 0.0), ("hold", 2.0), ("predict", 2.0)):
                    job = dict(id=f"eval-{scenario}-s{seed}-{method}-d{delay:g}",
                               scenario=scenario, setting=selected, delay_s=delay,
                               seed=seed, method=method)
                    rows.append(run_job(args, job))
    assert pilot.fingerprint(args.prepared_data) == before
    assert len(rows) <= protocol["max_runs"]
    pilot.write_json(out / "summary.json", dict(selected_setting=selected, runs=rows,
                                              prepared_inputs_unchanged=True))
    print(f"Completed {len(rows)} runs; selected {selected}; inputs unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("screen", "prototype", "one"))
    parser.add_argument("--openg2g-root", required=True)
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-results")
    parser.add_argument("--screen-results")
    parser.add_argument("--job")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.action == "one":
        if not args.job:
            parser.error("one requires --job")
        simulate(args)
    else:
        if not args.prior_results or (args.action == "prototype" and not args.screen_results):
            parser.error("suite needs --prior-results; prototype also needs --screen-results")
        suite(args)


if __name__ == "__main__":
    main()
