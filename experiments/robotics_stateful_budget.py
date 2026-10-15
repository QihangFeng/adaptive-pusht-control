"""Locked comparison of persistent 8/12/15 execution budgets, not generation length.

The model and milestone stay frozen. Only this candidate remembers a target
execution budget across predictions; dynamic simulator truth is scoring-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

import robotics_generation_length as gen


adaptive, base = gen.adaptive, gen.base
OUTPUT = Path("results/robotics-stateful-budget")
SCENES = list(range(90, 110))
DRAW = 30000
METHODS = ["fixed8", "adaptive12", "stateful15"]
BUDGETS = [8, 12, 15]
RULES = {"promotion_units": 2, "minimum_unit_progress": 0.0}
STATISTICS = {"bootstrap_samples": 10000, "bootstrap_seed": 20261014,
              "unit": "paired initial scene; one prediction sample per method"}


def smooth_future(features, commands, budget):
    points = np.asarray(commands, dtype=float)
    return bool(budget in BUDGETS and points.shape == (15, 2) and np.isfinite(points).all()
                and adaptive.extension_candidate(features) and adaptive._smooth(points[4:budget]))


def select_budget(policy, budget):
    """Crop execution queue, NOT generation; preserve observation history."""
    assert budget in BUDGETS
    queue = policy._queues["action"]
    assert queue.maxlen == 15 and len(queue) == 14
    retained = list(queue)[:budget - 1]
    queue.clear()
    queue.extend(retained)
    return 15 - budget


class StatefulBudgetController:
    def __init__(self):
        self.budget, self.streak = 8, 0
        self.start_coverage = None
        self.guard = adaptive.AdaptiveLengthController()

    def begin(self, features):
        self.start_coverage = features["coverage"] if features["quality"] else None
        return self.budget

    def _event(self, kind, reason, discarded, before):
        return {"kind": kind, "reason": reason, "discarded": discarded,
                "budget": before, "next_budget": self.budget, "streak": self.streak}

    def update(self, features, age, pending, step, commands, done=False):
        if done:
            return None
        before = self.budget
        assert before in BUDGETS and age + pending == before
        # Only scheduled actions count as pending. The original confirmation,
        # arming, cooldown and intervention-limit state machine is unchanged.
        event = self.guard.update(features, age, pending, step, commands[:before])
        if event and event["kind"] == "interrupt":
            self.budget, self.streak = 8, 0
            return self._event("interrupt", event["reason"], pending, before)
        if age == 8 and pending:
            if smooth_future(features, commands, before):
                return self._event("checkpoint_continue", "original_camera_and_path_eligibility", 0, before)
            self.budget, self.streak = 8, 0
            return self._event("checkpoint_replan", "original_checkpoint_not_eligible", pending, before)
        if pending == 0:
            good = (self.start_coverage is not None
                    and smooth_future(features, commands, before)
                    and features["coverage"] - self.start_coverage > RULES["minimum_unit_progress"])
            self.streak = self.streak + 1 if good else 0
            if not good:
                self.budget = 8
            elif self.streak >= RULES["promotion_units"]:
                self.budget = BUDGETS[min(BUDGETS.index(before) + 1, len(BUDGETS) - 1)]
                self.streak = 0
            return self._event("unit_end", "positive_eligible_progress" if good else "retain_feedback_frequency", 0, before)
        return None


def schedule():
    return [{"scene": scene, "draw": DRAW, "method": method}
            for index, scene in enumerate(SCENES)
            for method in METHODS[index % 3:] + METHODS[:index % 3]]


def check_output(output):
    assert Path(output).resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}


def protocol_document():
    inherited = gen.protocol_document()
    fingerprints = {**inherited["fingerprints"],
                    Path(__file__).resolve().relative_to(Path.cwd()).as_posix(): base.sha(Path(__file__))}
    return {"scope": "60 new attempts, three locked methods, twenty new scenes; one draw each",
            "milestone": gen.MILESTONE, "scenes": SCENES, "draw": DRAW, "methods": METHODS,
            "schedule": schedule(), "budgets": BUDGETS, "initial_budget": 8,
            "rules": RULES, "reused_milestone_rules": adaptive.RULES,
            "generation": {"horizon": 16, "usable_future_commands": 15, "observations": 2,
                           "denoising_steps": 100, "training": False},
            "max_movements": 300, "success": "true coverage strictly greater than .95",
            "runtime_truth_inputs": False, "prediction_seed": "scene plus 30000; paired by prediction-call order",
            "statistics": STATISTICS, "packages": inherited["packages"],
            "checkpoint": inherited["checkpoint"], "model_sources": inherited["model_sources"],
            "fingerprints": fingerprints,
            "smoke": {"scene": 0, "draw": 1234, "movements": 16,
                      "forced_eight": "exact baseline reproduction across two predictions",
                      "forced_fifteen": "all fifteen cached commands execute before next prediction"},
            "timing": inherited["timing"]}


def verify_protocol(output):
    check_output(output)
    document = base.read_json(output / "protocol.json")
    assert document == protocol_document(), "Frozen method, sources, packages or checkpoint changed."
    return document


def prepare(output):
    check_output(output)
    if (output / "protocol.json").exists():
        return verify_protocol(output)
    assert not any((output / name).exists() for name in ["smoke.json", "episodes", "episodes.json", "summary.json"])
    document = protocol_document()
    base.save_json(output / "protocol.json", document)
    return document


def configure_candidate(policy):
    assert policy.config.horizon == policy.diffusion.config.horizon == 16
    assert policy.config.n_obs_steps == policy.diffusion.config.n_obs_steps == 2
    for config in [policy.config, policy.diffusion.config]:
        config.n_action_steps = 15
    policy.reset()  # Episode boundary only, never between predictions.
    assert policy._queues["action"].maxlen == 15 and not policy._queues["action"]


def evaluate_episode(policy, scene, draw, method, directory, max_steps=300, force_budget=None):
    if method != "stateful15":
        assert force_budget is None
        return gen.evaluate_episode(policy, scene, draw, method, directory, max_steps)
    assert force_budget in [None, 8, 15]
    configure_candidate(policy)
    assert policy.diffusion.num_inference_steps == 100
    np.random.seed(scene + draw)
    torch.manual_seed(scene + draw)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=max_steps)
    observation, _ = env.reset(seed=scene)
    initial = adaptive.physical_state(env, observation)
    controller = StatefulBudgetController()
    estimator, rolling = adaptive.VisualAlignmentEstimator(), adaptive.RollingVisualFeatures()
    versions = tuple(parameter._version for parameter in policy.parameters())
    rows, units, events, prediction_seconds = [], [], [], []
    age, monitor_seconds, initial_features = 0, 0., None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        if force_budget is None:
            tick = time.perf_counter()
            features = rolling.update(estimator.estimate(observation["pixels"]), 0)
            initial_features = features
            monitor_seconds += time.perf_counter() - tick
        with gen.capture_generation(policy) as generated_shapes:
            for step in range(1, max_steps + 1):
                fresh = not policy._queues["action"]
                if fresh:
                    if age:
                        units.append(age)
                    age = 0
                    budget = force_budget if force_budget is not None else controller.begin(features)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                with torch.inference_mode():
                    command = policy.select_action(adaptive.observation_batch(observation, "cuda")).squeeze(0).cpu().numpy()
                if fresh:
                    torch.cuda.synchronize()
                    prediction_seconds.append(time.perf_counter() - tick)
                    commands = [command.tolist()] + [item.squeeze(0).detach().cpu().numpy().tolist()
                                                     for item in policy._queues["action"]]
                    assert len(commands) == 15
                    selection_discarded = select_budget(policy, budget)
                age += 1
                observation, _, terminated, truncated, _ = env.step(command)
                pending = len(policy._queues["action"])
                row = {"step": step, "action": command.tolist(), "new_prediction": bool(fresh),
                       "plan_age": age, "selected_budget": budget, "pending_before": pending,
                       "after": adaptive.physical_state(env, observation)}
                if fresh:
                    row.update(predicted_commands=commands, prediction_seconds=prediction_seconds[-1],
                               selection_discarded=selection_discarded)
                if force_budget is None and not (terminated or truncated):
                    tick = time.perf_counter()
                    features = rolling.update(estimator.estimate(observation["pixels"]), pending)
                    event = controller.update(features, age, pending, step, commands)
                    if event:
                        events.append({"step": step, "plan_age": age, **event, "features": features})
                        if event["discarded"]:
                            policy._queues["action"].clear()
                    monitor_seconds += time.perf_counter() - tick
                    row["features"] = features
                row["pending_after"] = len(policy._queues["action"])
                rows.append(row)
                if step % 80 == 0:
                    print(f"Running {method}, scene {scene}: movement {step}, calls {len(prediction_seconds)}, budget {budget}", flush=True)
                if terminated or truncated:
                    break
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        units.append(age)
        assert versions == tuple(parameter._version for parameter in policy.parameters())
        result = {"scene": scene, "draw": draw, "prediction_seed": scene + draw, "method": method,
                  "horizon": 16, "execution": 15, "force_budget": force_budget,
                  "success": bool(terminated), "truncated": bool(truncated), "steps": len(rows),
                  "model_queries": len(prediction_seconds), "initial": initial, "initial_features": initial_features,
                  "final_coverage": rows[-1]["after"]["coverage"], "wall_seconds": wall_seconds,
                  "monitor_seconds": monitor_seconds, "prediction_seconds": prediction_seconds,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "unit_lengths": units, "events": events, "generated_shapes": generated_shapes,
                  "final_budget": controller.budget, "final_streak": controller.streak,
                  "trace": gen.stem(scene, draw, method) + ".trace.json.gz"}
        directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(directory / result["trace"], "wt", encoding="utf-8") as stream:
            json.dump({"scene": scene, "draw": draw, "method": method, "initial": initial, "steps": rows}, stream, allow_nan=False)
        result["trace_sha256"] = base.sha(directory / result["trace"])
        base.save_json(directory / (gen.stem(scene, draw, method) + ".json"), result)
        return result
    finally:
        env.close()
        gen.configure(policy, "fixed8")


def verify_trial(trial, directory, allowed_scenes=SCENES, allowed_draws=(DRAW,), max_steps=300, allow_force=False):
    scene, draw, method = trial["scene"], trial["draw"], trial["method"]
    assert scene in allowed_scenes and draw in allowed_draws and method in METHODS
    if method != "stateful15":
        return gen.verify_trial(trial, directory, allowed_scenes, allowed_draws, max_steps)
    force = trial["force_budget"]
    assert force is None or (allow_force and force in [8, 15])
    assert trial["prediction_seed"] == scene + draw
    assert (trial["horizon"], trial["execution"]) == (16, 15)
    trace = gen.read_trace(trial, directory)
    assert (trace["scene"], trace["draw"], trace["method"]) == (scene, draw, method)
    assert trace["initial"] == trial["initial"]
    rows = trace["steps"]
    assert 1 <= len(rows) == trial["steps"] <= max_steps
    assert trial["final_coverage"] == rows[-1]["after"]["coverage"]
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert trial["success"] or (trial["truncated"] and len(rows) == max_steps)
    assert all(row["after"]["coverage"] <= .95 for row in rows[:-1])
    assert trial["wall_seconds"] >= trial["monitor_seconds"] >= 0
    assert all(math.isfinite(trial[key]) for key in ["wall_seconds", "monitor_seconds", "final_coverage"])
    assert trial["peak_gpu_allocated_bytes"] > 0
    controller, pending, age, units, seconds, events = StatefulBudgetController(), 0, 0, [], [], []
    features = trial["initial_features"]
    assert (features is None) == (force is not None)
    for index, row in enumerate(rows):
        assert row["step"] == index + 1 and row["new_prediction"] == (pending == 0)
        if pending == 0:
            if age:
                units.append(age)
            age = 0
            budget = force if force is not None else controller.begin(features)
            pending = budget
            commands = row["predicted_commands"]
            assert np.asarray(commands).shape == (15, 2) and np.isfinite(commands).all()
            assert row["selection_discarded"] == 15 - budget
            assert math.isfinite(row["prediction_seconds"]) and row["prediction_seconds"] > 0
            seconds.append(row["prediction_seconds"])
        else:
            assert "predicted_commands" not in row and "prediction_seconds" not in row and "selection_discarded" not in row
        age, pending = age + 1, pending - 1
        assert row["plan_age"] == age and row["selected_budget"] == budget
        assert row["pending_before"] == pending and row["action"] == commands[age - 1]
        if force is None and index < len(rows) - 1:
            features = row["features"]
            assert features["pending_actions"] == pending
            event = controller.update(features, age, pending, index + 1, commands)
            if event:
                events.append({"step": index + 1, "plan_age": age, **event, "features": features})
                pending -= event["discarded"]
        else:
            assert "features" not in row
        assert row["pending_after"] == pending
    units.append(age)
    assert trial["events"] == events and trial["unit_lengths"] == units
    assert trial["model_queries"] == len(units) == len(seconds) and sum(units) == len(rows)
    assert trial["prediction_seconds"] == seconds and sum(seconds) <= trial["wall_seconds"]
    assert trial["generated_shapes"] == [[1, 16, 2]] * len(seconds)
    assert (trial["final_budget"], trial["final_streak"]) == (controller.budget, controller.streak)
    return trace


def collect(policy, scene, draw, method, directory, max_steps=300, force_budget=None):
    path = directory / (gen.stem(scene, draw, method) + ".json")
    if path.exists():
        trial = base.read_json(path)
    else:
        assert not path.with_suffix(".trace.json.gz").exists() and not path.with_suffix(".json.tmp").exists(), "Inspect partial/orphan output."
        trial = evaluate_episode(policy, scene, draw, method, directory, max_steps, force_budget)
    assert (trial["scene"], trial["draw"], trial["method"]) == (scene, draw, method)
    if method == "stateful15":
        assert trial["force_budget"] == force_budget
    verify_trial(trial, directory, [scene], [draw], max_steps, force_budget is not None)
    return trial


def verify_prefixes(trials, directory):
    lengths = []
    for scene in sorted({row["scene"] for row in trials}):
        records = [row for row in trials if row["scene"] == scene]
        assert len(records) == 3 and {row["method"] for row in records} == set(METHODS)
        reference = next(row for row in records if row["method"] == "fixed8")
        traces = {row["method"]: gen.read_trace(row, directory)["steps"] for row in records}
        # Initial generated H16 predictions have matching first eight commands.
        # If a declared guard interrupts early, compare physical states only
        # through the common first-plan prefix, not after the intended divergence.
        limit = min(8, *(row["unit_lengths"][0] for row in records))
        for row in records:
            assert adaptive.states_match(row["initial"], reference["initial"])
            assert traces[row["method"]][0]["predicted_commands"][:8] == traces["fixed8"][0]["predicted_commands"][:8]
            for a, b in zip(traces[row["method"]][:limit], traces["fixed8"][:limit]):
                assert a["action"] == b["action"] and adaptive.states_match(a["after"], b["after"])
        lengths.append({"scene": scene, "common_executed_prefix": limit})
    return lengths


def verify_smoke(output):
    saved = base.read_json(output / "smoke.json")
    assert saved["passed"] and len(saved["trials"]) == 3
    for trial in saved["trials"]:
        assert base.read_json(output / "smoke" / (gen.stem(0, 1234, trial["method"]) + ".json")) == trial
        verify_trial(trial, output / "smoke", [0], [1234], 16)
    assert saved["prefixes"] == verify_prefixes(saved["trials"], output / "smoke")
    reference = next(row for row in saved["trials"] if row["method"] == "fixed8")
    baseline = gen.read_trace(reference, output / "smoke")["steps"]
    old = base.read_json(adaptive.OUTPUT / "prefix-smoke.json")["traces"]["fixed8"]
    for a, b in zip(baseline[:8], old[:8]):
        assert a["action"] == b["action"] and adaptive.states_match(a["after"], b["after"])
    for force in [8, 15]:
        trial = saved[f"forced{force}"]
        directory = output / f"forced{force}"
        assert base.read_json(directory / (gen.stem(0, 1234, "stateful15") + ".json")) == trial
        verify_trial(trial, directory, [0], [1234], 16, True)
        assert trial["force_budget"] == force
    forced8 = gen.read_trace(saved["forced8"], output / "forced8")["steps"]
    assert len(forced8) == len(baseline) == 16
    for a, b in zip(forced8, baseline):
        assert a["action"] == b["action"] and adaptive.states_match(a["after"], b["after"])
    assert saved["forced15"]["unit_lengths"] == [15, 1]
    full8 = forced8[0]["predicted_commands"]
    full15 = gen.read_trace(saved["forced15"], output / "forced15")["steps"][0]["predicted_commands"]
    assert full8 == full15 and len(full15) == 15
    return saved


def smoke(output):
    verify_protocol(output)
    if (output / "smoke.json").exists():
        return verify_smoke(output)
    assert torch.cuda.is_available(), "GPU required; no silent CPU fallback."
    policy, _ = adaptive.load_policy(base.MODEL)
    trials = [collect(policy, 0, 1234, method, output / "smoke", 16) for method in METHODS]
    forced = {f"forced{force}": collect(policy, 0, 1234, "stateful15", output / f"forced{force}", 16, force)
              for force in [8, 15]}
    base.save_json(output / "smoke.json", {"passed": True, "trials": trials,
                   "prefixes": verify_prefixes(trials, output / "smoke"), **forced})
    return verify_smoke(output)


def metric_table(trials):
    expected = {(scene, DRAW, method) for scene in SCENES for method in METHODS}
    assert len(trials) == 60 and {(r["scene"], r["draw"], r["method"]) for r in trials} == expected
    return {method: {"attempts": len(rows), "successes": sum(r["success"] for r in rows),
                     **{f"mean_{key}": float(np.mean([r[key] for r in rows]))
                        for key in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]},
                     "mean_prediction_seconds": float(np.mean([np.mean(r["prediction_seconds"]) for r in rows]))}
            for method in METHODS for rows in [[r for r in trials if r["method"] == method]]}


def paired(trials, reference):
    lookup = {(r["scene"], r["method"]): r for r in trials}
    assert len(lookup) == len(trials)
    rng = np.random.default_rng(STATISTICS["bootstrap_seed"])
    indices = rng.integers(0, len(SCENES), size=(STATISTICS["bootstrap_samples"], len(SCENES)))
    differences = {}
    for key in ["success", "final_coverage", "model_queries", "wall_seconds"]:
        values = np.asarray([float(lookup[s, "stateful15"][key]) - float(lookup[s, reference][key]) for s in SCENES])
        differences[key] = {"mean_difference": float(values.mean()),
                            "scene_pair_bootstrap_95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    return {"reference": reference, "method": "stateful15", "differences_method_minus_reference": differences,
            "gained_successes": [s for s in SCENES if lookup[s, "stateful15"]["success"] and not lookup[s, reference]["success"]],
            "lost_successes": [s for s in SCENES if lookup[s, reference]["success"] and not lookup[s, "stateful15"]["success"]]}


def summarize(output):
    verify_protocol(output)
    verify_smoke(output)
    document = base.read_json(output / "episodes.json")
    assert document["complete"]
    trials = document["episodes"]
    table = metric_table(trials)
    directory = output / "episodes"
    budgets, kinds = Counter(), Counter()
    for trial in trials:
        assert base.read_json(directory / (gen.stem(trial["scene"], trial["draw"], trial["method"]) + ".json")) == trial
        trace = verify_trial(trial, directory)
        if trial["method"] == "stateful15":
            budgets.update(str(row["selected_budget"]) for row in trace["steps"] if row["new_prediction"])
            kinds.update(event["kind"] for event in trial["events"])
    result = {"complete": True, "scope": "Twenty new initial scenes, one prediction sample; no generality or recovery guarantee",
              "methods": table, "comparisons": [paired(trials, reference) for reference in ["fixed8", "adaptive12"]],
              "candidate_selected_budgets": dict(budgets), "candidate_event_counts": dict(kinds),
              "common_prefixes": verify_prefixes(trials, directory)}
    path = output / "summary.json"
    if path.exists():
        assert base.read_json(path) == result, "Completed results changed."
    else:
        base.save_json(path, result)
    print(json.dumps(table, indent=2), flush=True)
    print(json.dumps({"candidate_selected_budgets": dict(budgets), "candidate_event_counts": dict(kinds)}), flush=True)
    return result


def run(output):
    verify_protocol(output)
    if (output / "summary.json").exists():
        return summarize(output)
    smoke(output)
    trials, policy = [], None
    for identity in schedule():
        scene, draw, method = identity["scene"], identity["draw"], identity["method"]
        if not (output / "episodes" / (gen.stem(scene, draw, method) + ".json")).exists() and policy is None:
            policy, _ = adaptive.load_policy(base.MODEL)
        trial = collect(policy, scene, draw, method, output / "episodes")
        trials.append(trial)
        base.save_json(output / "episodes.json", {"complete": len(trials) == 60, "episodes": trials})
        print(f"Progress {len(trials)}/60: {method} scene {scene}, success={trial['success']}, "
              f"calls={trial['model_queries']}, time={trial['wall_seconds']:.1f}s", flush=True)
    verify_protocol(output)
    return summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "run", "summarize"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    check_output(args.output_dir)
    if args.stage in ["smoke", "run"]:
        prepare(args.output_dir)
    tick = time.perf_counter()
    with adaptive.MemorySampler() as sampler:
        globals()[args.stage](args.output_dir)
    path = args.output_dir / f"{args.stage}-resources.json"
    if not path.exists():
        base.save_json(path, {"wall_seconds": time.perf_counter() - tick,
                            "peak_process_rss_bytes": sampler.peak_rss,
                            "minimum_system_available_bytes": sampler.minimum_available})


if __name__ == "__main__":
    main()
