"""Bounded, one-decision benefit-gate pilot around the frozen PushT policy.

Prepare uses existing camera-feature replays, labels use matched continuations,
and evaluation uses previously unused episodes. No runtime simulator state reaches
the controller. Run from the repository root using .venv-robotics.
"""

from __future__ import annotations

import argparse
from collections import deque
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from robotics_baseline_check import load_policy, physical_state, prime_history, states_match
from robotics_failure_investigation import continuation, load_records
from robotics_feasibility import MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256, observation_batch, MemorySampler
from robotics_visual_monitor import FEATURE_NAMES, MONITOR_RULES, RollingVisualFeatures, VisualAlignmentEstimator
from robotics_benefit_model import aggregate_pairs, choose_threshold, fit_tree, predict_tree


OUTPUT = Path("results/robotics-benefit-pilot")
DISCOVERY = Path("results/robotics-failure-patterns")
MODEL = Path("data/raw/robotics/diffusion_pusht")
LABEL_SEEDS = [3000, 3001, 3002]
EVALUATION_SEEDS = list(range(30, 50))
METHODS = ["fixed8", "fixed4", "warning_rule", "learned_gate"]
ELIGIBILITY = {
    "minimum_step": 32, "maximum_step": 284, "minimum_pending": 2,
    "warning_count": 2, "drawdown": 0.03, "recent_peak": 0.80,
    "stagnation_window": 24, "stagnation_coverage": 0.75, "stagnation_best_gain": 0.01,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    """Replace only derived pilot files; do not risk a partially written checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprints():
    summary = read_json(DISCOVERY / "summary.json")
    paths = [DISCOVERY / name for name in ["summary.json", "protocol.json", "analysis.json", "recovery.json"]]
    for row in summary["episodes"]:
        paths.append(Path(row["source_directory"]) / row["trace"])
    paths += [Path("results/robotics-baseline-check") / name for name in ["summary.json", "protocol.json"]]
    return {p.as_posix(): sha(p) for p in paths}


def code_fingerprints():
    names = ["robotics_benefit_pilot.py", "robotics_visual_monitor.py", "robotics_benefit_model.py",
             "robotics_baseline_check.py", "robotics_failure_investigation.py", "robotics_feasibility.py"]
    return {**{name: sha(Path(__file__).parent / name) for name in names},
            "requirements-robotics.txt": sha(Path("requirements-robotics.txt"))}


def vector(features):
    result = [float(features[name]) for name in FEATURE_NAMES]
    if not np.isfinite(result).all():
        raise ValueError("Controller features must be finite.")
    return result


def eligibility(features, history, step, pending):
    """Past-only candidate, not a failure label; no episode outcome is consulted."""
    if (not features["quality"] or step < ELIGIBILITY["minimum_step"]
            or step > ELIGIBILITY["maximum_step"] or pending < ELIGIBILITY["minimum_pending"]):
        return None
    regression = (features["warning_count"] >= ELIGIBILITY["warning_count"]
                  and features["drawdown"] >= ELIGIBILITY["drawdown"]
                  and max(history[-8:]) >= ELIGIBILITY["recent_peak"])
    width = ELIGIBILITY["stagnation_window"]
    stagnant = (len(history) >= width + 1
                and min(history[-width - 1], history[-1]) >= ELIGIBILITY["stagnation_coverage"]
                and max(history[-width:]) - history[-width - 1] <= ELIGIBILITY["stagnation_best_gain"])
    return "regression" if regression else ("stagnation" if stagnant else None)


def hand_decision(features, history):
    return bool(features["warning_count"] >= ELIGIBILITY["warning_count"]
                and features["drawdown"] >= ELIGIBILITY["drawdown"]
                and max(history[-8:]) >= ELIGIBILITY["recent_peak"])


def prepare(output):
    if (output / "protocol.json").exists():
        verify_protocol(output)
        print("Prepared pilot already exists; no replay or selection repeated.", flush=True)
        return
    validation = read_json(output / "visual-validation.json")
    assert validation["passed"], "Visual validation failed; do not proceed to expensive labels."
    near_target = read_json(output / "near-target-validation.json")
    assert near_target["passed"], "Near-target visual validation failed; do not collect labels."
    with gzip.open(output / "visual-features.json.gz", "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    cases = []
    for episode in sorted(episodes, key=lambda row: row["seed"]):
        history = []
        for features in episode["features"]:
            # At this recorded observation, done is available to a live controller.
            if features["step"] >= episode["steps"]:
                continue
            if not features["quality"]:
                history.clear()
                continue
            history.append(float(features["coverage"]))
            row = features
            reason = eligibility(features, history, row["step"], features["pending_actions"])
            if reason:
                cases.append({"checkpoint_id": f"seed-{episode['seed']}-step-{row['step']}",
                              "episode_id": episode["seed"], "branch_step": row["step"],
                              "features": vector(features), "feature_values": features, "reason": reason,
                              "hand_replan": hand_decision(features, history)})
                break
        if len(cases) == 20:
            break
    assert len(cases) >= 10, "Too few eligible independent situations for this bounded training pilot."
    # Interleave validation situations in seed order without looking at outcomes.
    validation_ids = [case["episode_id"] for i, case in enumerate(cases) if i % 3 == 2]
    train_ids = [case["episode_id"] for case in cases if case["episode_id"] not in validation_ids]
    assert len(train_ids) >= 6 and len(validation_ids) >= 3
    protocol = {
        "purpose": "preliminary one-decision benefit-gate evaluation, not publication proof",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "lerobot", "gym-pusht", "numpy", "scipy", "shapely", "diffusers"]},
        "config": read_json(MODEL / "config.json"), "feature_names": list(FEATURE_NAMES),
        "monitor_rules": MONITOR_RULES, "eligibility": ELIGIBILITY,
        "selection": "first eligible camera-derived checkpoint per seed, seeds 0-29 ascending, cap 20",
        "cases": cases, "train_episode_ids": train_ids, "validation_episode_ids": validation_ids,
        "label_seeds": LABEL_SEEDS, "evaluation_seeds": EVALUATION_SEEDS,
        "evaluation_rng": "torch/numpy reset to episode seed for each method; matched by prediction order not time",
        "methods": METHODS, "max_total_actions": 300, "success": "simulator termination: coverage strictly >0.95",
        "decision_scope": "learned gate: first eligible point only, then fixed8; warning baseline: first confirmed regression, then fixed8",
        "tree": {"max_depth": 2, "min_leaf": 3}, "thresholds": [0, 0.1, 0.34, 0.67, 1.1],
        "minimum_margin": 0.05, "threshold_selection": "validation success then fewer calls then higher threshold",
        "source_fingerprints": source_fingerprints(), "code_fingerprints": code_fingerprints(),
        "visual_evidence": {name: sha(output / name) for name in
                            ["visual-validation.json", "visual-features.json.gz", "near-target-validation.json"]},
    }
    save_json(output / "protocol.json", protocol)
    print(f"Frozen {len(cases)} checkpoints: {len(train_ids)} train / {len(validation_ids)} validation.", flush=True)


def verify_protocol(output):
    protocol = read_json(output / "protocol.json")
    assert protocol["source_fingerprints"] == source_fingerprints(), "Original evidence changed."
    assert protocol["code_fingerprints"] == code_fingerprints(), "Pilot source changed after protocol freeze."
    assert protocol["config"] == read_json(MODEL / "config.json")
    for name, digest in protocol["visual_evidence"].items():
        assert sha(output / name) == digest, "Visual evidence changed."
    return protocol


def verify_trial(trial, directory, branch=0):
    """Check completed sidecars against their trace before reuse or aggregation."""
    path = directory / trial["trace"]
    assert path.exists(), "Completed result is missing its trace."
    if "trace_sha256" in trial:
        assert sha(path) == trial["trace_sha256"], "Saved trace changed."
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    rows = trace["steps"]
    count = trial.get("steps", trial.get("remaining_actions_executed"))
    assert len(rows) == count and count > 0, "Recorded action count differs from trace."
    assert [row["step"] for row in rows] == list(range(branch + 1, branch + count + 1))
    assert branch + count <= 300 and (trial["success"] or branch + count == 300)
    assert trial["success"] == (rows[-1]["after"]["coverage"] > .95)
    assert abs(rows[-1]["after"]["coverage"] - trial["final_coverage"]) <= 1e-12
    queries = trial.get("model_queries", trial.get("remaining_model_queries"))
    assert sum(row["new_prediction"] for row in rows) == queries
    assert all(np.isfinite(row["action"]).all() for row in rows)
    return trace


def labels(output):
    protocol = verify_protocol(output)
    by_seed = {row["episode"]["seed"]: row for row in load_records(DISCOVERY)}
    target = output / "labels"
    target.mkdir(exist_ok=True)
    trials = []
    policy = None
    for case in protocol["cases"]:
        record = by_seed[case["episode_id"]]
        original_case = {"event": {"branch_step": case["branch_step"]}}
        for sample in LABEL_SEEDS:
            for arm in ["continue_cached", "replan_once"]:
                path = target / f"seed-{case['episode_id']}-sample-{sample}-{arm}.json"
                if path.exists():
                    trial = read_json(path)
                    assert (trial["seed"], trial["branch_step"], trial["diffusion_seed"], trial["arm"]) == (
                        case["episode_id"], case["branch_step"], sample, arm)
                    assert (target / trial["trace"]).exists(), "Saved label is missing its trace."
                else:
                    if policy is None:
                        policy, _ = load_policy(MODEL)
                    trial = continuation(policy, record, original_case, sample, arm, target)
                    trial["trace_sha256"] = sha(target / trial["trace"])
                    save_json(path, trial)
                trace = verify_trial(trial, target, case["branch_step"])
                assert states_match(trace["branch_state"], record["rows"][case["branch_step"] - 1]["after"])
                trials.append({"checkpoint_id": case["checkpoint_id"], "episode_id": case["episode_id"],
                               "sample_seed": sample, "arm": "continue" if arm == "continue_cached" else "replan",
                               "success": trial["success"], "model_queries": trial["remaining_model_queries"],
                               "features": case["features"], "final_coverage": trial["final_coverage"],
                               "wall_seconds": trial["wall_seconds"], "trace": trial["trace"]})
                save_json(output / "labels.json", {"complete": False, "records": trials})
                print(f"Label progress {len(trials)}/{len(protocol['cases']) * 6}", flush=True)
    save_json(output / "labels.json", {"complete": True, "records": trials})
    verify_protocol(output)


def train(output):
    protocol = verify_protocol(output)
    if (output / "model.json").exists():
        model = read_json(output / "model.json")
        assert model["labels_sha256"] == sha(output / "labels.json")
        assert model["protocol_sha256"] == sha(output / "protocol.json")
        print("Fitted model already frozen; no retraining.", flush=True)
        return
    document = read_json(output / "labels.json")
    assert document["complete"]
    cases = aggregate_pairs(document["records"], expected_repeats=3)
    expected = {case["checkpoint_id"]: case for case in protocol["cases"]}
    assert {case["checkpoint_id"] for case in cases} == set(expected), "Label checkpoint set changed."
    for case in cases:
        frozen = expected[case["checkpoint_id"]]
        assert (case["episode_id"], case["features"]) == (frozen["episode_id"], frozen["features"])
        assert [pair["sample_seed"] for pair in case["pairs"]] == LABEL_SEEDS, "Label random seeds changed."
    train_cases = [case for case in cases if case["episode_id"] in protocol["train_episode_ids"]]
    val_cases = [case for case in cases if case["episode_id"] in protocol["validation_episode_ids"]]
    tree = fit_tree([case["features"] for case in train_cases], [case["benefit"] for case in train_cases],
                    **protocol["tree"])
    selection = choose_threshold(tree, val_cases, thresholds=protocol["thresholds"], minimum_margin=0.05)
    model = {"tree": tree, "threshold": selection["threshold"], "minimum_margin": 0.05,
             "feature_names": list(FEATURE_NAMES), "label_target": "empirical paired success difference; not calibrated probability",
             "labels_sha256": sha(output / "labels.json"), "validation": selection,
             "protocol_sha256": sha(output / "protocol.json"),
             "train_cases": train_cases, "validation_cases": val_cases}
    save_json(output / "model.json", model)
    print(f"Frozen depth-two gate, threshold={model['threshold']}", flush=True)


def configure_horizon(policy, horizon):
    """Both configuration and queue capacity must agree before policy.reset()."""
    policy.config.n_action_steps = horizon
    policy.diffusion.config.n_action_steps = horizon
    policy.reset()
    assert policy._queues["action"].maxlen == horizon


def evaluate_episode(policy, seed, method, model, output):
    horizon = 4 if method == "fixed4" else 8
    configure_horizon(policy, horizon)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=300)
    observation, _ = env.reset(seed=seed)
    estimator, monitor = VisualAlignmentEstimator(), RollingVisualFeatures()
    rows, history = [], []
    initial = physical_state(env, observation)
    started = time.perf_counter()
    decision = None
    queries, monitor_seconds = 0, 0.0
    monitor_active = method in ["warning_rule", "learned_gate"]
    if monitor_active:
        tick = time.perf_counter()
        estimate = estimator.estimate(observation["pixels"])
        if estimate["quality"]:
            history.append(float(estimate["coverage"]))
        monitor.update(estimate, 0)
        monitor_seconds += time.perf_counter() - tick
    try:
        for step in range(1, 301):
            new_prediction = not policy._queues["action"]
            queries += bool(new_prediction)
            with torch.inference_mode():
                action = policy.select_action(observation_batch(observation, "cuda"))
            command = action.squeeze(0).cpu().numpy()
            observation, _, terminated, truncated, _ = env.step(command)
            pending = len(policy._queues["action"])
            row = {"step": step, "action": command.tolist(), "new_prediction": bool(new_prediction),
                   "pending_actions_before_decision": pending, "after": physical_state(env, observation)}
            if monitor_active and decision is None:
                tick = time.perf_counter()
                features = monitor.update(estimator.estimate(observation["pixels"]), pending)
                if features["quality"]:
                    history.append(float(features["coverage"]))
                else:
                    history.clear()
                if decision is None and not terminated and not truncated:
                    reason = eligibility(features, history, step, pending)
                    if reason and (method == "learned_gate" or hand_decision(features, history)):
                        score = predict_tree(model["tree"], vector(features)) if method == "learned_gate" else None
                        replan = (score > max(model["threshold"], model["minimum_margin"]) if score is not None
                                  else hand_decision(features, history))
                        decision = {"step": step, "reason": reason, "features": features,
                                    "score": score, "replan": bool(replan)}
                        if replan:
                            policy._queues["action"].clear()  # preserve observation history
                monitor_seconds += time.perf_counter() - tick
                row["visual_features"] = features
            row["pending_actions_after_decision"] = len(policy._queues["action"])
            rows.append(row)
            if step % 80 == 0:
                print(f"Evaluation {method}, seed {seed}, step {step}", flush=True)
            if terminated or truncated:
                break
        torch.cuda.synchronize()
        result = {"seed": seed, "method": method, "success": bool(terminated), "truncated": bool(truncated),
                  "steps": len(rows), "model_queries": queries, "initial_coverage": initial["coverage"],
                  "final_coverage": rows[-1]["after"]["coverage"],
                  "best_coverage": max(row["after"]["coverage"] for row in rows),
                  "wall_seconds": time.perf_counter() - started, "monitor_seconds": monitor_seconds,
                  "decision": decision, "trace": f"seed-{seed}-{method}.trace.json.gz"}
        with gzip.open(output / result["trace"], "wt", encoding="utf-8") as stream:
            json.dump({"initial": initial, "steps": rows}, stream, allow_nan=False)
        result["trace_sha256"] = sha(output / result["trace"])
        verify_trial(result, output)
        save_json(output / f"seed-{seed}-{method}.json", result)
        print(f"Complete {method}, seed {seed}: success={terminated}, calls={queries}", flush=True)
        return result
    finally:
        env.close()


def evaluate(output):
    protocol = verify_protocol(output)
    model = read_json(output / "model.json")
    assert model["labels_sha256"] == sha(output / "labels.json")
    assert model["protocol_sha256"] == sha(output / "protocol.json")
    target = output / "evaluation"
    target.mkdir(exist_ok=True)
    identity = {"protocol_sha256": sha(output / "protocol.json"), "model_sha256": sha(output / "model.json")}
    identity_path = output / "evaluation-protocol.json"
    if identity_path.exists():
        assert read_json(identity_path) == identity, "Evaluation settings/model changed."
    else:
        save_json(identity_path, identity)
    results, policy = [], None
    # Rotate arm order to avoid associating thermal/order effects with one method.
    for index, seed in enumerate(protocol["evaluation_seeds"]):
        methods = METHODS[index % len(METHODS):] + METHODS[:index % len(METHODS)]
        for method in methods:
            path = target / f"seed-{seed}-{method}.json"
            if path.exists():
                result = read_json(path)
                assert (result["seed"], result["method"]) == (seed, method)
                assert (target / result["trace"]).exists()
            else:
                if policy is None:
                    policy, _ = load_policy(MODEL)
                result = evaluate_episode(policy, seed, method, model, target)
            verify_trial(result, target)
            results.append(result)
            save_json(output / "evaluation.json", {"complete": False, "episodes": results})
            print(f"Evaluation progress {len(results)}/{len(EVALUATION_SEEDS) * len(METHODS)}", flush=True)
    save_json(output / "evaluation.json", {"complete": True, "episodes": results})
    verify_protocol(output)


def summarize(output):
    verify_protocol(output)
    document = read_json(output / "evaluation.json")
    assert document["complete"]
    table = {}
    by_key = {(row["seed"], row["method"]): row for row in document["episodes"]}
    assert len(document["episodes"]) == len(EVALUATION_SEEDS) * len(METHODS), "Unexpected duplicate/missing evaluation trials."
    assert len(by_key) == len(EVALUATION_SEEDS) * len(METHODS)
    for method in METHODS:
        episodes = [by_key[seed, method] for seed in EVALUATION_SEEDS]
        table[method] = {"attempts": len(episodes), "successes": sum(row["success"] for row in episodes),
                         **{f"mean_{name}": float(np.mean([row[name] for row in episodes]))
                            for name in ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]},
                         "eligible_decisions": sum(row["decision"] is not None for row in episodes),
                         "interruptions": sum(bool(row["decision"] and row["decision"]["replan"]) for row in episodes)}
    comparisons = {}
    for method in ["fixed8", "fixed4", "warning_rule"]:
        comparisons[method] = {"gate_only_success": 0, "comparator_only_success": 0, "both_success": 0, "both_fail": 0}
        for seed in EVALUATION_SEEDS:
            a, b = by_key[seed, "learned_gate"]["success"], by_key[seed, method]["success"]
            key = "both_success" if a and b else ("both_fail" if not a and not b else
                    ("gate_only_success" if a else "comparator_only_success"))
            comparisons[method][key] += 1
    result = {"complete": True, "methods": table, "paired_success_comparisons": comparisons,
              "scope": "20 new initial scenarios, one stochastic sample per method; preliminary, not statistical proof"}
    save_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "labels", "train", "evaluate", "summarize", "run"], required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    stages = [prepare, labels, train, evaluate, summarize] if args.stage == "run" else [globals()[args.stage]]
    for stage in stages:
        if stage in [labels, evaluate]:
            assert torch.cuda.is_available(), "Do not silently fall back to CPU inference."
        if stage in [labels, evaluate]:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with MemorySampler() as sampler:
            stage(args.output_dir)
        resources = {"wall_seconds": time.perf_counter() - started,
                     "peak_process_rss_bytes": sampler.peak_rss,
                     "minimum_system_available_bytes": sampler.minimum_available}
        if stage in [labels, evaluate]:
            resources.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        save_json(args.output_dir / f"{stage.__name__}-resources.json", resources)


if __name__ == "__main__":
    main()
