"""Three locked recovery-pilot conditions using the unchanged episode evaluator.

Only the existing six-tensor output block differs for recovery12. Inner episode
identities remain fixed8/adaptive12; separate outer directories identify weights.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
from pathlib import Path
import time

import numpy as np
import torch

import robotics_generation_length as gen
import robotics_recovery_training as training
import robotics_recovery_pilot as pilot


base, adaptive = gen.base, gen.adaptive
OUTPUT = Path("results/robotics-recovery-pilot")
SCENES = list(range(150, 170))
DRAW = 50000
ARMS = {"fixed8": "fixed8", "milestone12": "adaptive12", "recovery12": "adaptive12"}
STATISTICS = {"bootstrap_samples": 10000, "bootstrap_seed": 20261016,
              "unit": "paired initial scene; one prediction sample per condition"}


def schedule():
    names = list(ARMS)
    return [{"scene": scene, "draw": DRAW, "arm": arm}
            for index, scene in enumerate(SCENES)
            for arm in names[index % 3:] + names[:index % 3]]


def check_output(output):
    assert Path(output).resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}


def gpu_guard():
    training.gpu_guard()


def input_document(output):
    check_output(output)
    protocol = pilot.verify(output)
    expected = {"scenes": SCENES, "draw": DRAW, "arms": list(ARMS), "max_steps": 300,
                "rotation": "scene_index_mod_three", "success": "true_coverage_strictly_greater_than_0.95"}
    assert protocol["evaluation"] == expected, "Frozen evaluation conditions changed."
    assert protocol["milestone"] == gen.MILESTONE
    assert protocol["checkpoint"]["weights_sha256"] == base.WEIGHTS_SHA256
    for path, value in protocol["fingerprints"].items():
        assert base.sha(Path(path)) == value, f"Frozen source changed: {path}"
    report = base.read_json(output / "training.json")
    assert report["complete"] is True and report["final_head"]["optimizer_step"] == 600
    assert report["protocol_sha256"] == base.sha(output / "protocol.json")
    assert report["dataset_sha256"] == base.sha(output / "dataset.json")
    head = report["final_head"]
    head_path = Path(head["path"])
    assert head_path.resolve().is_relative_to(Path.cwd().resolve()), "Head must belong to this project."
    assert base.sha(head_path) == head["sha256"], "Trained head file changed."
    config_sha = base.sha(base.MODEL / "config.json")
    assert config_sha == protocol["checkpoint"]["config_sha256"]
    return {"protocol_sha256": base.sha(output / "protocol.json"),
            "dataset_sha256": base.sha(output / "dataset.json"),
            "training_sha256": base.sha(output / "training.json"),
            "final_head": head, "original_checkpoint_sha256": base.WEIGHTS_SHA256,
            "original_config_sha256": config_sha, "schedule": schedule(),
            "inner_methods": ARMS, "statistics": STATISTICS,
            "packages": {name: importlib.metadata.version(name) for name in
                         ["torch", "lerobot", "numpy", "gym-pusht", "diffusers"]}}


def verify_bindings(output):
    document = base.read_json(output / "eval-bindings.json")
    assert document["inputs"] == input_document(output), "Frozen evaluation inputs changed."
    assert document["original_head_digest"] != document["inputs"]["final_head"]["tensor_digest"]
    return document


def prepare_bindings(output, policy=None):
    """Freeze trained artifact identity before any test-scene prediction."""
    check_output(output)
    if (output / "eval-bindings.json").exists():
        saved = verify_bindings(output)
        if policy is not None:
            assert training.tail_digest(training.tail_state(policy)) == saved["original_head_digest"]
        return saved
    inputs = input_document(output)
    assert not any((output / name).exists() for name in
                   ["evaluation", "evaluation.json", "smoke.json", "smoke", "summary.json"])
    if policy is None:
        gpu_guard()
        policy, _ = adaptive.load_policy(base.MODEL)
    original = training.tail_digest(training.tail_state(policy))
    trained = training.read_tail(Path(inputs["final_head"]["path"]), inputs["final_head"]["sha256"])
    assert training.tail_digest(trained) == inputs["final_head"]["tensor_digest"] != original
    saved = {"inputs": inputs, "original_head_digest": original,
             "scope": "All three conditions collected after training; no reused evaluation episodes."}
    base.save_json(output / "eval-bindings.json", saved)
    return saved


def head_identity(bindings, arm):
    assert arm in ARMS
    return (bindings["inputs"]["final_head"]["tensor_digest"] if arm == "recovery12"
            else bindings["original_head_digest"])


def activate(policy, state, expected_digest):
    training.load_tail(policy, state)
    policy.eval()
    for parameter in policy.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    assert training.tail_digest(training.tail_state(policy)) == expected_digest


def condition_binding(output, arm, directory, bindings):
    expected = {"arm": arm, "inner_method": ARMS[arm],
                "head_digest": head_identity(bindings, arm),
                "eval_bindings_sha256": base.sha(output / "eval-bindings.json")}
    path = directory / "model-binding.json"
    if path.exists():
        assert base.read_json(path) == expected
    else:
        assert not directory.exists() or not any(directory.iterdir()), "Do not retroactively label episode weights."
        base.save_json(path, expected)
    return expected


def verify_record(record, output, bindings, max_steps=300, smoke=False):
    arm, trial, directory = record["arm"], record["trial"], Path(record["directory"])
    expected_dir = output / ("smoke" if smoke else "evaluation") / record.get("label", arm)
    assert directory == expected_dir and arm in ARMS and trial["method"] == ARMS[arm]
    assert record["head_digest"] == head_identity(bindings, arm)
    assert record["eval_bindings_sha256"] == base.sha(output / "eval-bindings.json")
    condition_binding(output, arm, directory, bindings)
    assert base.read_json(directory / (gen.stem(trial["scene"], trial["draw"], trial["method"]) + ".json")) == trial
    return gen.verify_trial(trial, directory, [0] if smoke else SCENES,
                            [1234] if smoke else [DRAW], max_steps)


def collect(policy, scene, draw, arm, directory, output, bindings, heads, max_steps=300, label=None):
    condition_binding(output, arm, directory, bindings)
    inner = ARMS[arm]
    path = directory / (gen.stem(scene, draw, inner) + ".json")
    if path.exists():
        trial = base.read_json(path)
    else:
        activate(policy, heads["trained" if arm == "recovery12" else "original"], head_identity(bindings, arm))
        assert not path.with_suffix(".trace.json.gz").exists() and not path.with_suffix(".json.tmp").exists(), \
            "Inspect orphan or partial evaluation output."
        trial = gen.evaluate_episode(policy, scene, draw, inner, directory, max_steps)
        assert training.tail_digest(training.tail_state(policy)) == head_identity(bindings, arm)
    assert (trial["scene"], trial["draw"], trial["method"]) == (scene, draw, inner)
    record = {"arm": arm, "directory": directory.as_posix(), "head_digest": head_identity(bindings, arm),
              "eval_bindings_sha256": base.sha(output / "eval-bindings.json"), "trial": trial}
    if label is not None:
        record["label"] = label
    verify_record(record, output, bindings, max_steps, label is not None)
    return record


def verify_smoke(output):
    bindings = verify_bindings(output)
    saved = base.read_json(output / "smoke.json")
    assert saved["passed"] and saved["original_backbone_unchanged"]
    records = saved["records"]
    assert [(r["label"], r["arm"]) for r in records] == [
        ("before", "fixed8"), ("milestone", "milestone12"),
        ("trained", "recovery12"), ("restored", "fixed8")]
    traces = {record["label"]: verify_record(record, output, bindings, 16, True) for record in records}
    before, restored = traces["before"]["steps"], traces["restored"]["steps"]
    assert len(before) == len(restored) == 16
    for left, right in zip(before, restored):
        assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])
    old = base.read_json(adaptive.OUTPUT / "prefix-smoke.json")["traces"]["fixed8"]
    for left, right in zip(before, old[:16]):
        assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])
    reference = records[0]["trial"]["initial"]
    for record in records:
        assert adaptive.states_match(reference, record["trial"]["initial"])
    # No action-equality requirement is imposed on the genuinely changed head.
    milestone = next(r["trial"] for r in records if r["arm"] == "milestone12")
    limit = min(8, milestone["unit_lengths"][0])
    for left, right in zip(traces["milestone"]["steps"][:limit], before[:limit]):
        assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])
    return saved


def smoke(output, policy=None):
    if (output / "smoke.json").exists():
        return verify_smoke(output)
    if policy is None:
        gpu_guard()
        policy, _ = adaptive.load_policy(base.MODEL)
    bindings = prepare_bindings(output, policy)
    heads = {"original": training.tail_state(policy),
             "trained": training.read_tail(Path(bindings["inputs"]["final_head"]["path"]),
                                           bindings["inputs"]["final_head"]["sha256"])}
    backbone = training.backbone_digest(policy)
    records = []
    try:
        for label, arm in [("before", "fixed8"), ("milestone", "milestone12"),
                           ("trained", "recovery12"), ("restored", "fixed8")]:
            records.append(collect(policy, 0, 1234, arm, output / "smoke" / label,
                                   output, bindings, heads, 16, label))
        assert training.backbone_digest(policy) == backbone
        base.save_json(output / "smoke.json", {"passed": True, "records": records,
                                               "original_backbone_unchanged": True})
    finally:
        activate(policy, heads["original"], bindings["original_head_digest"])
    return verify_smoke(output)


def metric_table(records):
    expected = {(scene, arm) for scene in SCENES for arm in ARMS}
    assert len(records) == len(expected)
    assert {(r["trial"]["scene"], r["arm"]) for r in records} == expected
    return {arm: {"attempts": len(rows), "successes": sum(row["success"] for row in rows),
                  **{f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in
                     ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]},
                  "mean_prediction_seconds": float(np.mean([np.mean(row["prediction_seconds"]) for row in rows])),
                  "mean_total_prediction_seconds": float(np.mean([sum(row["prediction_seconds"]) for row in rows]))}
            for arm in ARMS for rows in [[r["trial"] for r in records if r["arm"] == arm]]}


def paired(records, reference):
    assert reference in ["fixed8", "milestone12"]
    lookup = {(r["trial"]["scene"], r["arm"]): r["trial"] for r in records}
    assert len(lookup) == len(records) == 60
    indices = np.random.default_rng(STATISTICS["bootstrap_seed"]).integers(
        0, len(SCENES), size=(STATISTICS["bootstrap_samples"], len(SCENES)))
    differences = {}
    for key in ["success", "final_coverage", "model_queries", "wall_seconds"]:
        values = np.asarray([float(lookup[scene, "recovery12"][key]) - float(lookup[scene, reference][key])
                             for scene in SCENES])
        differences[key] = {"mean_difference": float(values.mean()),
                            "paired_scene_bootstrap_95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    gained = [scene for scene in SCENES if lookup[scene, "recovery12"]["success"] and not lookup[scene, reference]["success"]]
    lost = [scene for scene in SCENES if lookup[scene, reference]["success"] and not lookup[scene, "recovery12"]["success"]]
    count = len(gained) + len(lost)
    probability = min(1., 2 * sum(math.comb(count, k) for k in range(min(len(gained), len(lost)) + 1)) / 2**count) if count else 1.
    return {"method": "recovery12", "reference": reference, "gained_successes": gained,
            "lost_successes": lost, "differences_method_minus_reference": differences,
            "exact_two_sided_discordant_success_p": probability}


def summarize(output):
    bindings = verify_bindings(output)
    verify_smoke(output)
    document = base.read_json(output / "evaluation.json")
    assert document["complete"] and document["backbone_unchanged"]
    records = document["records"]
    assert [{"scene": r["trial"]["scene"], "draw": r["trial"]["draw"], "arm": r["arm"]}
            for r in records] == schedule()
    table = metric_table(records)
    traces = {(record["trial"]["scene"], record["arm"]): verify_record(record, output, bindings)
              for record in records}
    for scene in SCENES:
        initial = traces[scene, "fixed8"]["initial"]
        for arm in ARMS:
            assert adaptive.states_match(initial, traces[scene, arm]["initial"])
        # Original-weight controls share the first plan, not the trained arm.
        baseline, milestone = traces[scene, "fixed8"]["steps"], traces[scene, "milestone12"]["steps"]
        assert baseline[0]["predicted_commands"][:8] == milestone[0]["predicted_commands"][:8]
        first_length = next(r["trial"]["unit_lengths"][0] for r in records
                            if r["trial"]["scene"] == scene and r["arm"] == "milestone12")
        for left, right in zip(baseline[:min(8, first_length)], milestone[:min(8, first_length)]):
            assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])
    result = {"complete": True, "methods": table,
              "comparisons": [paired(records, reference) for reference in ["milestone12", "fixed8"]],
              "statistics": STATISTICS, "eval_bindings_sha256": base.sha(output / "eval-bindings.json"),
              "scope": "Twenty project-unseen initial scenes, one stochastic sample each; pilot, not proof of stable improvement."}
    path = output / "summary.json"
    if path.exists():
        assert base.read_json(path) == result, "Completed summary changed."
    else:
        base.save_json(path, result)
    return result


def _run(output):
    if (output / "summary.json").exists():
        return summarize(output)
    input_document(output)
    gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    bindings = prepare_bindings(output, policy)
    smoke(output, policy)
    heads = {"original": training.tail_state(policy),
             "trained": training.read_tail(Path(bindings["inputs"]["final_head"]["path"]),
                                           bindings["inputs"]["final_head"]["sha256"])}
    backbone = training.backbone_digest(policy)
    records = []
    try:
        for identity in schedule():
            arm = identity["arm"]
            records.append(collect(policy, identity["scene"], DRAW, arm,
                                   output / "evaluation" / arm, output, bindings, heads))
            base.save_json(output / "evaluation.json", {"complete": False, "records": records})
            print(f"Recovery evaluation {len(records)}/60: {arm}, scene {identity['scene']}, "
                  f"success={records[-1]['trial']['success']}", flush=True)
        assert training.backbone_digest(policy) == backbone
        base.save_json(output / "evaluation.json", {"complete": True, "backbone_unchanged": True, "records": records})
    finally:
        activate(policy, heads["original"], bindings["original_head_digest"])
    assert base.sha(base.MODEL / "model.safetensors") == base.WEIGHTS_SHA256
    return summarize(output)


def run(output):
    if (output / "summary.json").exists():
        return summarize(output)
    index_path = output / "evaluation.json"
    before = len(base.read_json(index_path)["records"]) if index_path.exists() else 0
    resource = output / f"evaluation-resources-from-{before:03}.json"
    retry = 0
    while resource.exists():
        retry += 1
        resource = output / f"evaluation-resources-from-{before:03}-retry-{retry}.json"
    tick = time.perf_counter()
    try:
        with adaptive.MemorySampler() as memory:
            result = _run(output)
        return result
    finally:
        after = len(base.read_json(index_path)["records"]) if index_path.exists() else 0
        base.save_json(resource, {"wall_seconds": time.perf_counter() - tick,
                                  "peak_process_rss_bytes": memory.peak_rss,
                                  "minimum_system_available_bytes": memory.minimum_available,
                                  "completed_attempts_before": before, "completed_attempts_after": after,
                                  "complete": (output / "summary.json").exists(),
                                  "scope": "This invocation, including loading, smoke and verification; episode timings exclude these."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=["prepare_bindings", "smoke", "run", "summarize"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.stage](args.output_dir)


if __name__ == "__main__":
    main()
