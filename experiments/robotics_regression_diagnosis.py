"""Bounded diagnosis of milestone success regressions; not a new controller trial.

Old sources/results remain frozen. Full-scene repeats separate scene and policy
seeds. Local probes restore physics, policy queues, visual history, controller and
RNG, and use the original 300-action total budget, not 300 additional actions.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import robotics_adaptive_length as adaptive
import robotics_fresh_evaluation as fresh
from robotics_baseline_check import replay_prefix


OUTPUT = Path("results/robotics-regression-diagnosis")
MILESTONE = "0f4044966646df4cc38b0373fa52c0b733506e15"
CHECKPOINTS = {54: {"first_condition": 192, "actual_interruption": 193},
               59: {"first_condition": 150, "actual_interruption": 151}}
PREDICTION_SEEDS = {54: [54001, 54002], 59: [59001, 59002]}
FUTURE_SEEDS = [9100, 9101]
RESTORE_COVERAGE_ATOL = 1e-12
SOURCE = fresh.OUTPUT / "episodes"
base = adaptive.base


def trial_specs():
    specs = []
    for scene, points in CHECKPOINTS.items():
        first = points["first_condition"]
        specs.append({"id": f"control-{scene}-{first}", "kind": "control",
                      "scene_seed": scene, "method": "adaptive12", "checkpoint": first,
                      "prediction_seed": None, "arm": "original"})
    for scene in CHECKPOINTS:
        for index, seed in enumerate(PREDICTION_SEEDS[scene]):
            order = ["fixed8", "adaptive12"] if index % 2 == 0 else ["adaptive12", "fixed8"]
            for method in order:
                specs.append({"id": f"repeat-{scene}-{seed}-{method}", "kind": "repeat",
                              "scene_seed": scene, "prediction_seed": seed, "method": method})
    for scene, points in CHECKPOINTS.items():
        for name, step in points.items():
            changed = "force_replan_now" if name == "first_condition" else "defer_current_unit"
            for index, seed in enumerate(FUTURE_SEEDS):
                arms = ["original", changed] if index % 2 == 0 else [changed, "original"]
                for arm in arms:
                    specs.append({"id": f"probe-{scene}-{step}-{seed}-{arm}", "kind": "probe",
                                  "scene_seed": scene, "method": "adaptive12", "checkpoint": step,
                                  "prediction_seed": seed, "arm": arm})
    return specs


def check_output(output):
    fresh.check_output(output)
    target, old = Path(output).resolve(), fresh.OUTPUT.resolve()
    if target == old or old in target.parents or target in old.parents:
        raise ValueError("Diagnosis must not overlap the milestone evidence directory.")


def read_source(scene, method="adaptive12"):
    result = base.read_json(SOURCE / f"seed-{scene}-{method}.json")
    fresh.verify_trial(result, SOURCE)
    with gzip.open(SOURCE / result["trace"], "rt", encoding="utf-8") as stream:
        return result, json.load(stream)


def regression_condition(features):
    return bool(features["quality"]
                and features["warning_count"] >= adaptive.RULES["interrupt_warnings"]
                and features["drawdown"] >= adaptive.RULES["interrupt_drawdown"]
                and features["recent_peak"] >= adaptive.RULES["interrupt_peak"]
                and features["recovery"] < adaptive.RULES["interrupt_recovery"])


def saved_analysis():
    details = []
    for scene, points in CHECKPOINTS.items():
        episode, trace = read_source(scene)
        rows = trace["steps"]
        first, actual = points.values()
        assert regression_condition(rows[first - 1]["features"])
        assert not regression_condition(rows[first - 2]["features"])
        interrupt = next(e for e in episode["events"] if e["kind"] == "interrupt")
        assert interrupt["step"] == actual
        details.append({"scene_seed": scene, "checkpoints": points,
                        "baseline": {method: {key: read_source(scene, method)[0][key]
                                               for key in ["success", "steps", "final_coverage", "model_queries"]}
                                     for method in ["fixed8", "fixed12", "adaptive12"]},
                        "decline_window": rows[first - 4:actual],
                        "interpretation": "Contact persists during a large near-target decline in the first eight actions of a fresh prediction; loss of contact is not the trigger."})
    return {"scope": "Retrospective selection of two known losses, not new held-out evaluation.",
            "milestone": MILESTONE, "scenes": details,
            "caveat": "A decline is not a failure label: successful fixed12 scene59 also recovered from a large dip."}


def protocol_document(output):
    fresh.verify_protocol(fresh.OUTPUT)
    bindings = dict(fresh.protocol_document()["fingerprints"])
    bindings["experiments/robotics_regression_diagnosis.py"] = base.sha(Path(__file__))
    for name in ["protocol.json", "summary.json", "episodes.json"]:
        path = fresh.OUTPUT / name
        bindings[path.as_posix()] = base.sha(path)
    for scene in CHECKPOINTS:
        for method in ["fixed8", "fixed12", "adaptive12"]:
            episode, _ = read_source(scene, method)
            for path in [SOURCE / f"seed-{scene}-{method}.json", SOURCE / episode["trace"]]:
                bindings[path.as_posix()] = base.sha(path)
    return {"milestone": MILESTONE, "scope": "26 selected-scene diagnostic trials, not method adoption or general improvement",
            "trials": trial_specs(), "rules": dict(adaptive.RULES), "checkpoints": {str(k): v for k, v in CHECKPOINTS.items()},
            "max_total_actions": 300, "success": "simulator termination; coverage strictly >.95",
            "rng": "Full repeats separate scene seed from prediction seed. Local randomized futures use the same seed by prediction order, not action time. Controls restore original numpy/CPU/CUDA RNG exactly.",
            "interventions": {
                "force_replan_now": "At the first complete existing regression condition, consume one interruption and discard cached actions one movement earlier; retain visual/observation histories and unchanged future rule.",
                "defer_current_unit": "Suppress interruption only until the next prediction, without consuming budget/cooldown. Preserve normal eight-action trim/extension, then resume the unchanged rule."},
            "restore_gate": "Every regenerated prefix row and both unchanged original-RNG full suffixes must match actions, physical/image states, features, queues and events before randomized probes. Only simulator truth coverage uses the existing physical-state absolute tolerance; every other value remains exact.",
            "restore_truth_coverage_absolute_tolerance": RESTORE_COVERAGE_ATOL,
            "packages": {p: importlib.metadata.version(p) for p in adaptive.PACKAGES},
            "fingerprints": bindings, "analysis_sha256": base.sha(output / "saved-analysis.json")}


def prepare(output):
    check_output(output)
    path = output / "protocol.json"
    if not path.exists():
        assert not any((output / name).exists() for name in ["trials", "trials.json", "summary.json"])
        base.save_json(output / "saved-analysis.json", saved_analysis())
        base.save_json(path, protocol_document(output))
    return verify_protocol(output)


def verify_protocol(output):
    check_output(output)
    saved = base.read_json(output / "protocol.json")
    assert saved == protocol_document(output), "Frozen diagnosis sources, evidence or settings changed."
    return saved


def decide(controller, features, age, pending, step, commands, suppress_interrupt=False):
    before = dict(vars(controller))
    event = controller.update(features, age, pending, step, commands)
    blocked = bool(suppress_interrupt and event and event["kind"] == "interrupt")
    if blocked:
        # Keep confirmation history, but do not consume an unexecuted interruption.
        for key in ["interruptions", "last_interrupt", "armed"]:
            setattr(controller, key, before[key])
        event = None
        if age == adaptive.RULES["base_length"] and pending == 4:
            retain = adaptive.extension_candidate(features) and adaptive.plan_continuity(commands)
            event = {"kind": "extend" if retain else "trim",
                     "reason": "camera_and_plan_eligible" if retain else "retain_eight_action_interval",
                     "discarded": 0 if retain else pending}
    return event, blocked


def branch_decision(snapshot, arm, controller):
    controller.__dict__.update(copy.deepcopy(snapshot["controller_post"]))
    if arm == "original":
        return copy.deepcopy(snapshot["event"]), False
    if arm == "force_replan_now":
        assert snapshot["event"] is None, "Do not force a second interruption at an existing event."
        assert regression_condition(snapshot["features"])
        assert controller.armed and snapshot["age"] >= adaptive.RULES["minimum_plan_age"]
        assert snapshot["pending"] >= adaptive.RULES["minimum_pending"]
        assert controller.interruptions < adaptive.RULES["maximum_interruptions"]
        assert snapshot["step"] - controller.last_interrupt >= adaptive.RULES["cooldown_actions"]
        controller.interruptions += 1
        controller.last_interrupt, controller.armed = snapshot["step"], False
        return {"kind": "interrupt", "reason": "diagnostic_force_replan", "discarded": snapshot["pending"]}, False
    if arm == "defer_current_unit":
        assert snapshot["event"] and snapshot["event"]["kind"] == "interrupt"
        controller.__dict__.update(copy.deepcopy(snapshot["controller_pre"]))
        event, blocked = decide(controller, snapshot["features"], snapshot["age"], snapshot["pending"],
                                snapshot["step"], snapshot["commands"], suppress_interrupt=True)
        assert blocked
        return event, True
    raise ValueError(f"Unknown diagnostic arm: {arm}")


def clone_queues(policy):
    return {key: {"maxlen": queue.maxlen, "values": [value.detach().cpu().clone() for value in queue]}
            for key, queue in policy._queues.items()}


def restore_queues(policy, queues, device):
    policy._queues = {key: deque([value.to(device).clone() for value in data["values"]], maxlen=data["maxlen"])
                      for key, data in queues.items()}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def same_rows(actual, expected):
    assert len(actual) == len(expected), "Restored suffix length changed."
    for left, right in zip(actual, expected):
        left, right = dict(left), dict(right)
        left_state, right_state = left.pop("after"), right.pop("after")
        # Gym-PushT unions a shape set in iteration order: identical geometry can
        # change truth coverage's last bit. Reuse the earlier physical-state check.
        assert left == right and adaptive.states_match(left_state, right_state), \
            f"Restored trajectory differs beyond coverage rounding at action {right['step']}."


def collect(policy, spec, source=None, snapshot=None, capture_steps=(), stop=300):
    scene, method = spec["scene_seed"], spec["method"]
    base.configure_horizon(policy, 8 if method == "fixed8" else 12)
    controller = adaptive.AdaptiveLengthController()
    estimator, rolling = adaptive.VisualAlignmentEstimator(), adaptive.RollingVisualFeatures()
    rows, events, units, captures = [], [], [], {}
    age, queries, prefix_queries, prefix_step, unit_count, monitor_seconds = 0, 0, 0, 0, 0, 0.
    commands, suppressed = [], False
    if snapshot is None:
        np.random.seed(spec["prediction_seed"])
        torch.manual_seed(spec["prediction_seed"])
        env = adaptive.gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
        observation, _ = env.reset(seed=scene)
    else:
        prefix_step = snapshot["step"]
        env, observation, _ = replay_prefix(scene, source["steps"], prefix_step)
        restore_queues(policy, snapshot["queues"], "cuda")
        estimator.previous_pose = copy.deepcopy(snapshot["pose"])
        rolling.history = copy.deepcopy(snapshot["history"])
        age, commands, prefix_queries = snapshot["age"], copy.deepcopy(snapshot["commands"]), snapshot["queries"]
        if spec["prediction_seed"] is None:
            np.random.set_state(snapshot["numpy_rng"])
            torch.set_rng_state(snapshot["cpu_rng"])
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
        else:
            np.random.seed(spec["prediction_seed"])
            torch.manual_seed(spec["prediction_seed"])
    initial = adaptive.physical_state(env, observation)
    started = time.perf_counter()
    try:
        if snapshot is not None:
            tick = time.perf_counter()
            event, suppressed = branch_decision(snapshot, spec["arm"], controller)
            if event:
                events.append({"step": prefix_step, "plan_age": age, **event, "features": snapshot["features"]})
                if event["discarded"]:
                    policy._queues["action"].clear()
            monitor_seconds += time.perf_counter() - tick
        elif method == "adaptive12":
            tick = time.perf_counter()
            rolling.update(estimator.estimate(observation["pixels"]), 0)
            monitor_seconds += time.perf_counter() - tick
        for step in range(prefix_step + 1, stop + 1):
            fresh_prediction = not policy._queues["action"]
            if fresh_prediction:
                if unit_count:
                    units.append(unit_count)
                age, unit_count, queries = 0, 0, queries + 1
                suppressed = False
            with torch.inference_mode():
                command = policy.select_action(adaptive.observation_batch(observation, "cuda")).squeeze(0).cpu().numpy()
            if fresh_prediction:
                commands = [command.tolist()] + [a.squeeze(0).detach().cpu().numpy().tolist()
                                                for a in policy._queues["action"]]
            age, unit_count = age + 1, unit_count + 1
            observation, _, terminated, truncated, _ = env.step(command)
            pending = len(policy._queues["action"])
            row = {"step": step, "action": command.tolist(), "new_prediction": bool(fresh_prediction),
                   "plan_age": age, "pending_before": pending, "after": adaptive.physical_state(env, observation)}
            if method == "adaptive12" and not (terminated or truncated):
                tick = time.perf_counter()
                features = rolling.update(estimator.estimate(observation["pixels"]), pending)
                pre = dict(vars(controller))
                event, blocked = decide(controller, features, age, pending, step, commands, suppressed)
                if step in capture_steps:
                    captures[step] = {"step": step, "age": age, "pending": pending, "queries": queries,
                                      "commands": copy.deepcopy(commands), "features": copy.deepcopy(features),
                                      "event": copy.deepcopy(event), "controller_pre": pre,
                                      "controller_post": dict(vars(controller)), "queues": clone_queues(policy),
                                      "pose": copy.deepcopy(estimator.previous_pose), "history": copy.deepcopy(rolling.history),
                                      "numpy_rng": copy.deepcopy(np.random.get_state()), "cpu_rng": torch.get_rng_state().clone(),
                                      "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()]}
                if event:
                    events.append({"step": step, "plan_age": age, **event, "features": features})
                    if event["discarded"]:
                        policy._queues["action"].clear()
                monitor_seconds += time.perf_counter() - tick
                row["features"] = features
                if blocked:
                    row["diagnostic_interrupt_suppressed"] = True
            row["pending_after"] = len(policy._queues["action"])
            rows.append(row)
            if capture_steps:
                same_rows([row], [source["steps"][step - 1]])
            if step % 80 == 0:
                print(f"{spec['id']}: action {step}, new calls {queries}", flush=True)
            if terminated or truncated:
                break
        torch.cuda.synchronize()
        units.append(unit_count)
        result = {**spec, "prefix_step": prefix_step, "prefix_queries": prefix_queries,
                  "success": bool(terminated), "truncated": bool(truncated), "steps": len(rows),
                  "final_step": rows[-1]["step"], "model_queries": queries,
                  "total_model_queries": queries + prefix_queries, "initial": initial,
                  "final_coverage": rows[-1]["after"]["coverage"], "unit_lengths": units,
                  "wall_seconds": time.perf_counter() - started, "monitor_seconds": monitor_seconds,
                  "events": events, "initial_suppression": bool(snapshot and spec["arm"] == "defer_current_unit")}
        if spec["kind"] == "control":
            same_rows(rows, source["steps"][prefix_step:])
            episode, _ = read_source(scene)
            assert events == [e for e in episode["events"] if e["step"] >= prefix_step]
            assert result["total_model_queries"] == episode["model_queries"]
            result["exact_suffix_match"] = True
            result["truth_coverage_absolute_tolerance"] = RESTORE_COVERAGE_ATOL
            result["maximum_restored_coverage_difference"] = max(
                abs(a["after"]["coverage"] - b["after"]["coverage"])
                for a, b in zip(rows, source["steps"][prefix_step:]))
        return result, {"initial": initial, "steps": rows}, captures
    finally:
        env.close()


def snapshot_audit(snapshot):
    return {key: snapshot[key] for key in ["step", "age", "pending", "queries", "commands", "features", "event", "controller_pre", "controller_post"]} | {
        "visual_history": list(snapshot["history"]), "previous_pose": np.asarray(snapshot["pose"]).tolist(),
        "queues": {key: {"maxlen": data["maxlen"], "tensor_sha256": [tensor_hash(value) for value in data["values"]]}
                   for key, data in snapshot["queues"].items()},
        "cpu_rng_sha256": tensor_hash(snapshot["cpu_rng"]),
        "cuda_rng_sha256": [tensor_hash(state) for state in snapshot["cuda_rng"]],
        "numpy_rng_sha256": hashlib.sha256(snapshot["numpy_rng"][1].tobytes()).hexdigest(),
        "numpy_rng_metadata": [snapshot["numpy_rng"][0], *snapshot["numpy_rng"][2:]]}


def save_trial(result, trace, output):
    directory = output / "trials"
    directory.mkdir(exist_ok=True)
    result["trace"] = result["id"] + ".trace.json.gz"
    assert not (directory / result["trace"]).exists(), "Inspect a partial trace before resuming."
    with gzip.open(directory / result["trace"], "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, allow_nan=False)
    result["trace_sha256"] = base.sha(directory / result["trace"])
    base.save_json(directory / (result["id"] + ".json"), result)


def verify_trial(trial, output):
    specs = {s["id"]: s for s in trial_specs()}
    assert trial["id"] in specs
    assert all(trial.get(key) == value for key, value in specs[trial["id"]].items())
    assert trial["trace"] == trial["id"] + ".trace.json.gz"
    path = output / "trials" / trial["trace"]
    assert base.sha(path) == trial["trace_sha256"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    rows, prefix = trace["steps"], trial.get("checkpoint", 0)
    assert trial["prefix_step"] == prefix
    assert 1 <= len(rows) == trial["steps"] <= 300 - prefix
    assert [r["step"] for r in rows] == list(range(prefix + 1, prefix + len(rows) + 1))
    assert trial["final_step"] == prefix + len(rows) <= 300
    assert trial["initial"] == trace["initial"]
    assert trial["model_queries"] == sum(r["new_prediction"] for r in rows)
    assert trial["total_model_queries"] == trial["model_queries"] + trial["prefix_queries"]
    assert all(isinstance(n, int) and n > 0 for n in trial["unit_lengths"])
    assert sum(trial["unit_lengths"]) == len(rows)
    assert trial["final_coverage"] == rows[-1]["after"]["coverage"]
    assert trial["success"] == (trial["final_coverage"] > .95)
    assert trial["success"] or (trial["truncated"] and trial["final_step"] == 300)
    assert all(math.isfinite(trial[k]) for k in ["wall_seconds", "monitor_seconds", "final_coverage"])
    assert 0 <= trial["monitor_seconds"] <= trial["wall_seconds"]
    if trial["kind"] == "control":
        assert trial["exact_suffix_match"]
        episode, source = read_source(trial["scene_seed"])
        same_rows(rows, source["steps"][prefix:])
        assert trial["events"] == [e for e in episode["events"] if e["step"] >= prefix]
    return trial


def summarize(output):
    verify_protocol(output)
    document = base.read_json(output / "trials.json")
    trials = document["trials"]
    assert document["complete"] and len(trials) == 26
    assert {t["id"] for t in trials} == {t["id"] for t in trial_specs()}
    for trial in trials:
        verify_trial(trial, output)
    groups = []
    for scene in CHECKPOINTS:
        for kind, step, methods in [("repeat", None, ["fixed8", "adaptive12"]),
                                    ("probe", CHECKPOINTS[scene]["first_condition"], ["original", "force_replan_now"]),
                                    ("probe", CHECKPOINTS[scene]["actual_interruption"], ["original", "defer_current_unit"])]:
            table = {}
            for name in methods:
                rows = [t for t in trials if t["scene_seed"] == scene and t["kind"] == kind
                        and t.get("checkpoint") == step and t.get("arm", t["method"]) == name]
                assert len(rows) == 2
                table[name] = {"attempts": len(rows), "successes": sum(t["success"] for t in rows),
                               **{f"mean_{key}": float(np.mean([t[key] for t in rows]))
                                  for key in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]}}
            groups.append({"scene_seed": scene, "kind": kind, "checkpoint": step, "methods": table})
    summary = {"complete": True, "trials": 26, "exact_controls": 2, "groups": groups,
               "scope": "Two retrospectively selected baseline successes lost by the milestone; tiny conditional diagnostic samples, not an adopted new method or population estimate."}
    base.save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(output):
    verify_protocol(output)
    assert not (output / "summary.json").exists(), "Completed diagnosis: inspect saved results instead of rerunning."
    assert torch.cuda.is_available(), "CUDA inference is required; no silent CPU fallback."
    torch.cuda.reset_peak_memory_stats()
    policy, _ = adaptive.load_policy(base.MODEL)
    sampler = adaptive.MemorySampler()
    sampler.__enter__()
    started = time.perf_counter()
    captures, sources, audits = {}, {}, {}
    try:
        for scene, points in CHECKPOINTS.items():
            _, source = read_source(scene)
            sources[scene] = source
            spec = {"id": f"prefix-{scene}", "kind": "prefix", "scene_seed": scene,
                    "prediction_seed": scene, "method": "adaptive12"}
            print(f"Regenerating exact prefix: scene {scene}", flush=True)
            _, trace, snapshots = collect(policy, spec, source=source, capture_steps=list(points.values()), stop=max(points.values()))
            same_rows(trace["steps"], source["steps"][:max(points.values())])
            assert set(snapshots) == set(points.values())
            captures[scene] = snapshots
            audits[str(scene)] = {"exact_prefix_match": True, "actions": len(trace["steps"]),
                                 "checkpoints": {str(step): snapshot_audit(snap) for step, snap in snapshots.items()}}
        path = output / "restore-prefix-audit.json"
        if path.exists():
            assert base.read_json(path) == audits, "Regenerated hidden state changed on resume."
        else:
            base.save_json(path, audits)
        completed = []
        for spec in trial_specs():
            path = output / "trials" / (spec["id"] + ".json")
            if path.exists():
                result = base.read_json(path)
            else:
                assert not (output / "trials" / (spec["id"] + ".trace.json.gz")).exists(), \
                    "Inspect an orphan partial trace before restarting inference."
                print(f"Starting {spec['id']}", flush=True)
                snap = captures[spec["scene_seed"]][spec["checkpoint"]] if "checkpoint" in spec else None
                result, trace, _ = collect(policy, spec, source=sources[spec["scene_seed"]], snapshot=snap)
                save_trial(result, trace, output)
            verify_trial(result, output)
            completed.append(result)
            base.save_json(output / "trials.json", {"complete": len(completed) == 26, "trials": completed})
            print(f"Progress {len(completed)}/26: {spec['id']}, success={result['success']}, coverage={result['final_coverage']:.4f}, calls={result['model_queries']}, time={result['wall_seconds']:.1f}s", flush=True)
        verify_protocol(output)
        summarize(output)
    finally:
        sampler.__exit__()
        base.save_json(output / "resources.json", {"wall_seconds": time.perf_counter() - started,
                       "includes_prefix_regeneration": True,
                       "peak_process_rss_bytes": sampler.peak_rss,
                       "minimum_system_available_bytes": sampler.minimum_available,
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=["prepare", "run", "summarize"], required=True)
    args = parser.parse_args()
    check_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    {"prepare": prepare, "run": run, "summarize": summarize}[args.stage](args.output)
