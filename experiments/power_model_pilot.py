"""Small, CPU-only power-fit sensitivity pilot; no adaptive controller.

Uses an external, pinned OpenG2G checkout and previously prepared measured data.
All controller variants see the same plant. Each simulation is a fresh process.
See reports/2026-09-20-power-model-pilot.md for scope and reproduction.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import gzip
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

# Bound native worker counts before importing numerical packages.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

COMMIT = "c582a5115cfd49dcdfb140f1837224e7cbf0cfb1"
DATA_REVISION = "fea6cb99d037d11f86f9c45a6991b9e5764f4529"
LABEL = "Llama-3.1-8B"
BATCHES = (8, 16, 32, 64, 128, 256, 512)
DURATION = 300
DT = 0.1
SCENARIOS = {
    "development": [(4200, 60, 90), (2400, 180, 210)],
    "abrupt": [(4800, 60, 65), (2400, 180, 185)],
    "two_pulses": [(4500, 40, 60), (2400, 120, 140),
                   (4500, 180, 200), (2400, 250, 270)],
}
TUNINGS = {
    "reference": (0.1, 1e6),
    "half_step": (0.05, 1e6),
    "double_step": (0.2, 1e6),
    "half_voltage_gain": (0.1, 0.5e6),
    "double_voltage_gain": (0.1, 2e6),
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(root):
    """Hash input artifacts, not access times or absolute filenames."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def bootstrap(root):
    sys.path.insert(0, str(Path(root).resolve()))
    sys.path.insert(0, str(Path(root).resolve() / "examples" / "offline"))


def make_variants(nominal, measured):
    """Offset/gain diagnostics plus one preselected limited-profiling fit."""
    import numpy as np
    from mlenergy.data.modeling import LogisticModel

    training = [r for r in measured if r["batch"] != 64]
    omitted = LogisticModel.fit(
        np.log2([r["batch"] for r in training]),
        np.array([r["measured_power_w"] for r in training]),
    )
    return {
        "nominal": nominal,
        "offset_plus_50w": replace(nominal, b0=nominal.b0 + 50),
        "gain_0.9": replace(nominal, L=nominal.L * 0.9, b0=nominal.b0 * 0.9),
        "gain_1.1": replace(nominal, L=nominal.L * 1.1, b0=nominal.b0 * 1.1),
        "omit_batch_64": omitted,
    }


def describe_fit(model, nominal, measured):
    import numpy as np

    xs = np.linspace(3, 9, 121)
    gradients = np.array([nominal.deriv_wrt_x(x) for x in xs])
    difference = np.array([model.deriv_wrt_x(x) for x in xs]) - gradients
    predictions = [model.eval(r["batch"]) for r in measured]
    assert all(p > 0 for p in predictions)
    return {
        "parameters": model.to_dict(),
        "max_relative_error_at_measured_batches": max(
            abs(p / r["measured_power_w"] - 1) for p, r in zip(predictions, measured)),
        "relative_gradient_l2_difference_from_nominal": float(
            np.linalg.norm(difference) / np.linalg.norm(gradients)),
        "predicted_w_by_batch": dict(zip(map(str, BATCHES), predictions)),
        "gradient_w_per_log2_batch": [model.deriv_wrt_x(x) for x in range(3, 10)],
    }


def select_tuning(rows):
    """Select on development only; thresholds declared before execution."""
    eligible = [r for r in rows if r["integrated_violation_pu_s"] <= 0.05
                and r["worst_vmin"] >= 0.94 and r["worst_vmax"] <= 1.06
                and r["itl_violation_fraction"] <= 0.01]
    if eligible:
        return max(eligible, key=lambda r: (r["throughput_tps"],
                                           -r["integrated_violation_pu_s"]))
    return min(rows, key=lambda r: (r["integrated_violation_pu_s"],
                                   r["itl_violation_fraction"], -r["throughput_tps"]))


def simulate(args):
    bootstrap(args.openg2g_root)
    import numpy as np
    from mlenergy.data.modeling import LogisticModel
    from systems import ieee13
    from openg2g.controller.ofo import LogisticModelStore, OFOBatchSizeController, OFOConfig
    from openg2g.coordinator import Coordinator
    from openg2g.datacenter.config import DatacenterConfig, InferenceModelSpec, ReplicaSchedule
    from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
    from openg2g.datacenter.workloads.inference import InferenceData
    from openg2g.grid.opendss import OpenDSSGrid, dss
    from openg2g.metrics.performance import compute_performance_stats
    from openg2g.metrics.voltage import compute_allbus_voltage_stats

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    # OpenDSS may change process cwd while compiling its circuit.
    out = Path(args.output).resolve()
    spec = InferenceModelSpec(
        model_label=LABEL, model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_model="H100", task="lm-arena-chat", precision="bfloat16",
        gpus_per_replica=1, tensor_parallel=1, itl_deadline_s=0.08,
        batch_sizes=BATCHES, feasible_batch_sizes=BATCHES,
    )
    data = InferenceData.load(Path(args.prepared_data), (spec,), duration_s=DURATION, dt_s=DT)
    schedule = ReplicaSchedule(initial=2400)
    for target, start, end in SCENARIOS[job["scenario"]]:
        schedule = schedule.ramp_to(target, t_start=start, t_end=end)
    initial = job.get("fixed_batch", 128)
    dc = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500),
        OfflineWorkload(inference_data=data, replica_schedules={LABEL: schedule},
                        initial_batch_sizes={LABEL: initial}),
        name="pilot", dt_s=Fraction(1, 10), seed=job["seed"], total_gpu_capacity=4800,
    )
    system = ieee13()
    grid = OpenDSSGrid(
        dss_case_dir=system["dss_case_dir"], dss_master_file=system["dss_master_file"],
        dt_s=Fraction(1, 10), source_pu=system["source_pu"],
        initial_tap_position=system["initial_taps"], exclude_buses=system["exclude_buses"],
    )
    grid.attach_dc(dc, bus="671")
    controllers = []
    config = None
    if "fixed_batch" not in job:
        fits = LogisticModelStore.load(Path(args.prepared_data), (spec,))
        # New store changes ONLY controller knowledge, never data.power_templates.
        changed = LogisticModelStore(
            power={LABEL: LogisticModel.from_dict(job["power_parameters"])},
            latency=fits.latency_fits, throughput=fits.throughput_fits,
        )
        step, voltage_gain = TUNINGS[job["tuning"]]
        config = OFOConfig(
            primal_step_size=step, w_throughput=0.001, w_switch=1,
            voltage_gradient_scale=voltage_gain, v_min=0.95, v_max=1.05,
            voltage_dual_step_size=1, latency_dual_step_size=1,
            sensitivity_update_interval=3600, sensitivity_perturbation_kw=100,
        )
        controllers = [OFOBatchSizeController(
            (spec,), dc, grid, changed, config=config, dt_s=Fraction(1),
            initial_batch_sizes={LABEL: initial},
        )]
    coordinator = Coordinator(datacenters=[dc], grid=grid, controllers=controllers,
                              total_duration_s=DURATION)
    dc_states, grid_states, rows = [], [], []
    excluded = set(system["exclude_buses"])
    start = time.perf_counter()
    for tick in coordinator.run_iter():
        assert dss.Solution.Converged(), "Power-flow solve failed"
        state, gs = tick.dc_states[dc.name], tick.grid_state
        dc_states.append(state)
        grid_states.append(gs)
        voltages = [v for bus in gs.voltages.buses() if bus not in excluded
                    for v in (gs.voltages[bus].a, gs.voltages[bus].b, gs.voltages[bus].c)
                    if np.isfinite(v)]
        batch, itl = state.batch_size_by_model[LABEL], state.observed_itl_s_by_model[LABEL]
        replicas = state.active_replicas_by_model[LABEL]
        power = sum((state.power_w.a, state.power_w.b, state.power_w.c))
        assert batch in BATCHES and itl > 0 and power > 0
        rows.append([tick.t_s, batch, power, itl, replicas, min(voltages), max(voltages),
                     sum(max(0.95-v, 0) + max(v-1.05, 0) for v in voltages),
                     batch / itl * replicas])
    values = np.asarray(rows)
    assert len(rows) == round(DURATION / DT) and np.isfinite(values).all()
    vs = compute_allbus_voltage_stats(grid_states, exclude_buses=tuple(excluded))
    perf = compute_performance_stats(dc_states, itl_deadline_s_by_model={LABEL: 0.08})
    # Explicit equal-step integration includes all 3,000 0.1-second intervals.
    integral = float(np.sum(values[:, 7]) * DT)
    assert abs(integral - vs.integral_violation_pu_s) < 1e-8
    assert abs(float(values[:, 8].mean()) - perf.mean_throughput_tps) < 1e-5
    result = dict(job, config=None if config is None else config.model_dump(),
                  integrated_violation_pu_s=integral,
                  violation_time_s=vs.violation_time_s,
                  worst_vmin=vs.worst_vmin, worst_vmax=vs.worst_vmax,
                  throughput_tps=perf.mean_throughput_tps,
                  integrated_tokens=float(np.sum(values[:, 8]) * DT),
                  itl_violation_fraction=perf.itl_deadline_fraction,
                  batch_switches=int(np.count_nonzero(np.diff(values[:, 1]))),
                  batch_log2_total_variation=float(np.abs(np.diff(np.log2(values[:, 1]))).sum()),
                  mean_dc_power_kw=float(values[:, 2].mean() / 1000),
                  observed_batches=sorted(set(map(int, values[:, 1]))),
                  simulation_seconds=time.perf_counter()-start,
                  steps=len(rows), all_solves_converged=True,
                  trajectory_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    with gzip.open(out / f"{job['id']}.trace.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"columns": ["time_s", "batch", "power_w", "itl_s", "replicas",
                               "vmin", "vmax", "violation_sum_pu", "throughput_tps"],
                   "rows": rows}, stream, allow_nan=False)
    write_json(out / f"{job['id']}.json", result)


def bounded_run(args, job):
    import psutil

    out = Path(args.output).resolve()
    job_path = out / f"{job['id']}.job.json"
    write_json(job_path, job)
    command = [sys.executable, str(Path(__file__).resolve()), "one",
               "--openg2g-root", args.openg2g_root, "--prepared-data", args.prepared_data,
               "--output", str(out), "--job", str(job_path)]
    start, peak = time.perf_counter(), 0
    with (out / f"{job['id']}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        monitor = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:
                    members = [monitor, *monitor.children(recursive=True)]
                    rss = 0
                    for member in members:
                        try:
                            rss += member.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    break
                peak = max(peak, rss)
                if rss > 3.5 * 1024**3 or time.perf_counter() - start > 180:
                    raise RuntimeError(f"Resource limit exceeded: {job['id']}")
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                # Windows venv python.exe can be a launcher with a child interpreter.
                for child in monitor.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
            process.wait()
    if process.returncode:
        raise RuntimeError(f"Simulation failed: {job['id']}; see its .log file")
    result_path = out / f"{job['id']}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(wall_seconds=time.perf_counter()-start, peak_sampled_rss_mib=peak/1024**2,
                  memory_sampling_scope="sum of launcher and recursive child RSS")
    write_json(result_path, result)
    print(f"{job['id']}: V={result['integrated_violation_pu_s']:.5f} pu*s, "
          f"throughput={result['throughput_tps']/1e6:.3f} M tok/s, "
          f"wall={result['wall_seconds']:.1f}s", flush=True)
    return result


def suite(args):
    bootstrap(args.openg2g_root)
    from mlenergy.data.modeling import LogisticModel

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new/empty output directory; existing evidence is never overwritten")
    root = Path(args.openg2g_root).resolve()
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    assert commit == COMMIT and not subprocess.check_output(git + ["status", "--porcelain"], text=True)
    before = fingerprint(Path(args.prepared_data))
    fits = list(Path(args.prepared_data).glob("*/logistic_fit.json"))
    if len(fits) != 1:
        raise ValueError("This pilot requires exactly one prepared model profile")
    nominal = LogisticModel.from_dict(json.loads(fits[0].read_text())["power"])
    measured = json.loads(Path(args.fit_check).read_text())
    assert [r["batch"] for r in measured] == list(BATCHES)
    variants = make_variants(nominal, measured)
    descriptions = {name: describe_fit(model, nominal, measured) for name, model in variants.items()}
    # Describe all omissions without choosing one based on closed-loop performance.
    import numpy as np
    omissions = {}
    for omitted in BATCHES:
        subset = [r for r in measured if r["batch"] != omitted]
        fitted = LogisticModel.fit(np.log2([r["batch"] for r in subset]),
                                   np.array([r["measured_power_w"] for r in subset]))
        omissions[str(omitted)] = describe_fit(fitted, nominal, measured)
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    manifest = dict(openg2g_commit=commit, dataset_revision=DATA_REVISION,
                    python=sys.version, executable=sys.executable, package_versions=versions,
                    prepared_data_sha256=before,
                    fit_check_sha256=hashlib.sha256(Path(args.fit_check).read_bytes()).hexdigest(),
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scenarios=SCENARIOS, tunings=TUNINGS, model_variants=descriptions,
                    leave_one_batch_out_diagnostics=omissions, measured_power_points=measured,
                    duration_s=DURATION, dt_s=DT, development_seed=0, evaluation_seed=1,
                    base_kw_per_phase=500, initial_replicas=2400, initial_controlled_batch=128,
                    capacity=4800, control_interval_s=1,
                    selection_rule="Development nominal only: maximize throughput among V integral <=0.05, "
                    "vmin>=0.94, vmax<=1.06, ITL miss<=0.01; otherwise minimum V integral.",
                    max_planned_runs=29, calibration_implemented=False)
    write_json(out / "manifest.json", manifest)
    development = []
    for tuning in TUNINGS:
        job = dict(id=f"dev-{tuning}", stage="development", scenario="development", seed=0,
                   tuning=tuning, variant="nominal", power_parameters=nominal.to_dict())
        development.append(bounded_run(args, job))
    selected = select_tuning(development)["tuning"]
    # Persist selection before any held-out evaluation is run.
    write_json(out / "selection.json", {"selected_tuning": selected,
                                        "development_ids": [r["id"] for r in development]})
    print(f"Selected on development only: {selected}", flush=True)
    results = list(development)
    for scenario in ("abrupt", "two_pulses"):
        for tuning in dict.fromkeys(("reference", selected)):
            for name, model in variants.items():
                job = dict(id=f"eval-{scenario}-{tuning}-{name}", stage="evaluation",
                           scenario=scenario, seed=1, tuning=tuning, variant=name,
                           power_parameters=model.to_dict())
                results.append(bounded_run(args, job))
        for batch in (8, 512):
            job = dict(id=f"fixed-{scenario}-{batch}", stage="fixed-reference",
                       scenario=scenario, seed=1, fixed_batch=batch)
            results.append(bounded_run(args, job))
    assert fingerprint(Path(args.prepared_data)) == before, "Prepared plant inputs changed"
    for scenario in ("abrupt", "two_pulses"):
        for tuning in dict.fromkeys(("reference", selected)):
            paired = {r.get("variant"): r for r in results if r["scenario"] == scenario
                      and r.get("tuning") == tuning}
            assert paired["nominal"]["trajectory_sha256"] == paired["offset_plus_50w"]["trajectory_sha256"], (
                "Offset negative-control failed")
    write_json(out / "summary.json", {"selected_tuning": selected, "runs": results,
               "prepared_inputs_unchanged": True, "offset_trajectories_identical": True})
    print(f"Completed {len(results)} simulations; input hashes and offset controls passed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("suite", "one"))
    parser.add_argument("--openg2g-root", required=True)
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-check")
    parser.add_argument("--job")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.action == "suite":
        if not args.fit_check:
            parser.error("suite requires --fit-check")
        suite(args)
    else:
        if not args.job:
            parser.error("one requires --job")
        simulate(args)


if __name__ == "__main__":
    main()
