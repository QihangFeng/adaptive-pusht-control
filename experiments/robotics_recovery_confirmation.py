"""Frozen recovery-head confirmation: forty new scenes, two draws, three arms.

Only collection/bookkeeping is new. Prediction, physics, visual monitoring and
the milestone controller are reused without changing their frozen sources.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

import robotics_recovery_evaluation as old


base, gen, training, adaptive = old.base, old.gen, old.training, old.adaptive
OUTPUT = Path("results/robotics-recovery-confirmation")
SCENES = list(range(170, 210))
DRAWS = [60000, 70000]
ARMS = dict(old.ARMS)
SMOKE_ARMS = [("before", "fixed8"), ("milestone", "milestone12"),
              ("trained", "recovery12"), ("restored", "fixed8")]
STATISTICS = {"bootstrap_samples": 10000, "bootstrap_seed": 20261017,
              "unit": "paired initial scene, averaging its two draws before resampling",
              "independent_scenes": 40, "draws_per_scene": 2}
activate, head_identity = old.activate, old.head_identity


def schedule():
    names = list(ARMS)
    return [{"scene": scene, "draw": draw, "arm": arm}
            for index, scene in enumerate(SCENES) for di, draw in enumerate(DRAWS)
            for arm in names[(2 * index + di) % 3:] + names[:(2 * index + di) % 3]]


def check_output(output):
    assert Path(output).resolve() in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}, \
        "Use only the dedicated confirmation output or its -repeat directory."


def input_document():
    """Verify the completed pilot, not a fresh or selectively chosen checkpoint."""
    bindings = old.verify_bindings(old.OUTPUT)
    training_report = base.read_json(old.OUTPUT / "training.json")
    assert training_report["original_tail_digest"] == bindings["original_head_digest"]
    assert training_report["frozen_backbone_and_normalization_unchanged"] is True
    assert base.sha(base.MODEL / "model.safetensors") == base.WEIGHTS_SHA256
    return bindings


def protocol_document():
    previous = input_document()
    inherited = old.pilot.verify(old.OUTPUT)
    paths = [Path(__file__).resolve().relative_to(Path.cwd()), Path("tests/test_robotics_recovery_confirmation.py"),
             *[old.OUTPUT / name for name in ["protocol.json", "dataset.json", "training.json",
                                              "eval-bindings.json", "smoke.json", "summary.json", "independent-audit.json"]]]
    fingerprints = {**inherited["fingerprints"], **{path.as_posix(): base.sha(path) for path in paths}}
    return {"date_utc": "2026-10-17", "scope": "240 fresh attempts; no training or controller revision",
            "milestone": gen.MILESTONE, "scenes": SCENES, "draw_offsets": DRAWS,
            "arms": ARMS, "schedule": schedule(), "expected_attempts": 240,
            "max_steps": 300, "success": "true_coverage_strictly_greater_than_0.95",
            "rotation": "(2*scene_index+draw_index)_mod_three", "statistics": STATISTICS,
            "prediction_seed": "initial scene plus draw offset; paired by call order",
            "generation": {"horizon": 16, "observations": 2, "denoising_steps": 100},
            "training": False, "checkpoint_selection": "previous final600 only",
            "previous_inputs": previous["inputs"], "original_head_digest": previous["original_head_digest"],
            "packages": inherited["packages"], "model_sources": inherited["model_sources"],
            "fingerprints": dict(sorted(fingerprints.items())), "rules": adaptive.RULES,
            "smoke": {"scene": 0, "draw": 1234, "max_steps": 16, "order": SMOKE_ARMS},
            "primary_comparison": "recovery12 minus milestone12; fixed8 secondary reference",
            "timing": inherited["timing"],
            "limitations": ["One previously trained checkpoint; not repeated training seeds.",
                            "Forty initial scenes, not eighty independent scene samples.",
                            "No pooling with the earlier twenty-scene pilot for confirmation inference.",
                            "Paused clean simulation, not physical or real-time deployment."]}


def verify_protocol(output=OUTPUT):
    check_output(output)
    saved = base.read_json(output / "protocol.json")
    # JSON stores smoke tuples as lists.
    expected = protocol_document()
    expected["smoke"]["order"] = [list(item) for item in SMOKE_ARMS]
    assert saved == expected, "Frozen confirmation protocol, sources or artifacts changed."
    return saved


def prepare(output=OUTPUT):
    check_output(output)
    if not (output / "protocol.json").exists():
        assert not output.exists() or not any(output.iterdir()), "Do not retroactively freeze existing results."
        old.pilot.write_once(output / "protocol.json", protocol_document())
    protocol = verify_protocol(output)
    expected = {"inputs": {"protocol_sha256": base.sha(output / "protocol.json"),
                           "final_head": protocol["previous_inputs"]["final_head"]},
                "original_head_digest": protocol["original_head_digest"],
                "scope": "Same final600 head and original weights; all 240 attempts newly collected"}
    if (output / "eval-bindings.json").exists():
        assert base.read_json(output / "eval-bindings.json") == expected
    else:
        assert not any((output / name).exists() for name in ["evaluation", "evaluation.json", "smoke", "smoke.json"])
        old.pilot.write_once(output / "eval-bindings.json", expected)
    return protocol


def verify_bindings(output):
    protocol = verify_protocol(output)
    saved = base.read_json(output / "eval-bindings.json")
    assert saved == {"inputs": {"protocol_sha256": base.sha(output / "protocol.json"),
                                "final_head": protocol["previous_inputs"]["final_head"]},
                     "original_head_digest": protocol["original_head_digest"],
                     "scope": "Same final600 head and original weights; all 240 attempts newly collected"}
    assert head_identity(saved, "recovery12") != head_identity(saved, "fixed8")
    return saved


def condition_binding(output, arm, directory, bindings):
    return old.condition_binding(output, arm, directory, bindings)


def identity(record):
    return {"scene": record["trial"]["scene"], "draw": record["trial"]["draw"], "arm": record["arm"]}


def verify_record(record, output, bindings, max_steps=300, smoke=False):
    arm, trial = record["arm"], record["trial"]
    directory = output / ("smoke" if smoke else "evaluation") / record.get("label", arm)
    assert arm in ARMS and Path(record["directory"]) == directory and trial["method"] == ARMS[arm]
    assert (directory / "model-binding.json").exists(), "Missing recorded model identity."
    condition_binding(output, arm, directory, bindings)
    assert record["head_digest"] == head_identity(bindings, arm)
    assert record["eval_bindings_sha256"] == base.sha(output / "eval-bindings.json")
    assert base.read_json(directory / (gen.stem(trial["scene"], trial["draw"], trial["method"]) + ".json")) == trial
    return gen.verify_trial(trial, directory, [0] if smoke else SCENES, [1234] if smoke else DRAWS, max_steps)


def _record(trial, arm, directory, output, bindings, label=None):
    row = {"arm": arm, "directory": directory.as_posix(), "trial": trial,
           "head_digest": head_identity(bindings, arm),
           "eval_bindings_sha256": base.sha(output / "eval-bindings.json")}
    if label is not None:
        row["label"] = label
    return row


def scan_records(output, bindings, smoke=False):
    """Validate sidecars, orphan traces and the progress index before loading GPU."""
    folder = output / ("smoke" if smoke else "evaluation")
    expected = ([{"scene": 0, "draw": 1234, "arm": arm, "label": label} for label, arm in SMOKE_ARMS]
                if smoke else schedule())
    allowed = {row.get("label", row["arm"]) for row in expected}
    if folder.exists():
        assert all(path.is_dir() and path.name in allowed for path in folder.iterdir()), "Unexpected condition output."
    rows = []
    found = set()
    for item in expected:
        directory = folder / item.get("label", item["arm"])
        path = directory / (gen.stem(item["scene"], item["draw"], ARMS[item["arm"]]) + ".json")
        if path.exists():
            trial = base.read_json(path)
            assert (trial["scene"], trial["draw"], trial["method"]) == (item["scene"], item["draw"], ARMS[item["arm"]])
            row = _record(trial, item["arm"], directory, output, bindings, item.get("label"))
            verify_record(row, output, bindings, 16 if smoke else 300, smoke)
            rows.append(row)
            found.update([path, directory / trial["trace"], directory / "model-binding.json"])
        elif directory.exists() and (directory / "model-binding.json").exists():
            condition_binding(output, item["arm"], directory, bindings)
            found.add(directory / "model-binding.json")
    if folder.exists():
        assert all(path in found for directory in folder.iterdir() for path in directory.iterdir()), \
            "Orphan trace, partial sidecar or unexpected result; inspect without overwriting."
    keys = ["scene", "draw", "arm", "label"] if smoke else ["scene", "draw", "arm"]
    actual = [{**identity(row), **({"label": row["label"]} if smoke else {})} for row in rows]
    assert actual == [{key: item[key] for key in keys} for item in expected[:len(rows)]], \
        "Completed attempts must be a schedule prefix."
    index_path = output / ("smoke.json" if smoke else "evaluation.json")
    assert not index_path.with_suffix(".json.tmp").exists(), "Inspect partial progress index; do not overwrite."
    if index_path.exists():
        saved = base.read_json(index_path)
        assert saved["records"] == rows[:len(saved["records"])], "Progress index disagrees with recorded trials."
        if saved.get("complete") or saved.get("passed"):
            assert len(saved["records"]) == len(rows) == len(expected)
    return rows


def collect(policy, scene, draw, arm, directory, output, bindings, heads, max_steps=300, label=None):
    condition_binding(output, arm, directory, bindings)
    path = directory / (gen.stem(scene, draw, ARMS[arm]) + ".json")
    if path.exists():
        trial = base.read_json(path)
    else:
        assert not path.with_suffix(".trace.json.gz").exists() and not path.with_suffix(".json.tmp").exists(), \
            "Orphan or partial trial; do not overwrite."
        activate(policy, heads["trained" if arm == "recovery12" else "original"], head_identity(bindings, arm))
        trial = gen.evaluate_episode(policy, scene, draw, ARMS[arm], directory, max_steps)
        assert training.tail_digest(training.tail_state(policy)) == head_identity(bindings, arm)
    assert (trial["scene"], trial["draw"], trial["method"]) == (scene, draw, ARMS[arm])
    record = _record(trial, arm, directory, output, bindings, label)
    verify_record(record, output, bindings, max_steps, label is not None)
    return record


def verify_smoke(output, bindings=None):
    bindings = bindings or verify_bindings(output)
    saved = base.read_json(output / "smoke.json")
    assert saved["passed"] and saved["original_backbone_unchanged"]
    rows = scan_records(output, bindings, True)
    assert [(row["label"], row["arm"]) for row in rows] == SMOKE_ARMS
    traces = {row["label"]: verify_record(row, output, bindings, 16, True) for row in rows}
    reference = traces["before"]["initial"]
    for trace in traces.values():
        assert adaptive.states_match(reference, trace["initial"])
    before, restored = traces["before"]["steps"], traces["restored"]["steps"]
    assert len(before) == len(restored) == 16
    original = base.read_json(adaptive.OUTPUT / "prefix-smoke.json")["traces"]["fixed8"]
    for left, right, prior in zip(before, restored, original[:16]):
        assert left["action"] == right["action"] == prior["action"]
        assert adaptive.states_match(left["after"], right["after"]) and adaptive.states_match(left["after"], prior["after"])
    limit = min(8, rows[1]["trial"]["unit_lengths"][0])
    for left, right in zip(before[:limit], traces["milestone"]["steps"][:limit]):
        assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])
    return saved


def smoke(output, policy=None):
    bindings = verify_bindings(output)
    if (output / "smoke.json").exists() and base.read_json(output / "smoke.json").get("passed"):
        return verify_smoke(output, bindings)
    rows = scan_records(output, bindings, True)
    if policy is None:
        training.gpu_guard()
        policy, _ = adaptive.load_policy(base.MODEL)
    heads = {"original": training.tail_state(policy),
             "trained": training.read_tail(Path(bindings["inputs"]["final_head"]["path"]),
                                           bindings["inputs"]["final_head"]["sha256"])}
    assert training.tail_digest(heads["original"]) == bindings["original_head_digest"]
    backbone = training.backbone_digest(policy)
    try:
        for label, arm in SMOKE_ARMS[len(rows):]:
            rows.append(collect(policy, 0, 1234, arm, output / "smoke" / label, output, bindings, heads, 16, label))
            base.save_json(output / "smoke.json", {"passed": False, "records": rows})
        assert training.backbone_digest(policy) == backbone
        base.save_json(output / "smoke.json", {"passed": True, "records": rows,
                                               "original_backbone_unchanged": True})
    finally:
        activate(policy, heads["original"], bindings["original_head_digest"])
    return verify_smoke(output, bindings)


def metric_table(records, draw=None):
    draws = DRAWS if draw is None else [draw]
    assert all(value in DRAWS for value in draws)
    assert all(row["trial"]["draw"] in DRAWS for row in records), "Unknown prediction draw."
    rows = [row for row in records if row["trial"]["draw"] in draws]
    expected = {(scene, sample, arm) for scene in SCENES for sample in draws for arm in ARMS}
    assert len(rows) == len(expected) and {(row["trial"]["scene"], row["trial"]["draw"], row["arm"]) for row in rows} == expected
    return {arm: {"attempts": len(trials), "successes": sum(row["success"] for row in trials),
                  **{f"mean_{key}": float(np.mean([row[key] for row in trials])) for key in
                     ["final_coverage", "model_queries", "wall_seconds", "monitor_seconds"]},
                  "mean_prediction_seconds": float(np.mean([np.mean(row["prediction_seconds"]) for row in trials])),
                  "mean_total_prediction_seconds": float(np.mean([sum(row["prediction_seconds"]) for row in trials]))}
            for arm in ARMS for trials in [[row["trial"] for row in rows if row["arm"] == arm]]}


def paired(records, reference):
    assert reference in ["fixed8", "milestone12"]
    metric_table(records)
    lookup = {(row["trial"]["scene"], row["trial"]["draw"], row["arm"]): row["trial"] for row in records}
    indices = np.random.default_rng(STATISTICS["bootstrap_seed"]).integers(
        0, len(SCENES), size=(STATISTICS["bootstrap_samples"], len(SCENES)))
    differences = {}
    for key in ["success", "final_coverage", "model_queries", "wall_seconds"]:
        values = np.asarray([np.mean([float(lookup[scene, draw, "recovery12"][key]) -
                                     float(lookup[scene, draw, reference][key]) for draw in DRAWS]) for scene in SCENES])
        differences[key] = {"mean_difference": float(values.mean()), "scene_mean_differences": values.tolist(),
                            "paired_scene_bootstrap_95_interval": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    discordant = lambda first, second: [{"scene": scene, "draw": draw} for scene in SCENES for draw in DRAWS
                                       if lookup[scene, draw, first]["success"] and not lookup[scene, draw, second]["success"]]
    return {"method": "recovery12", "reference": reference,
            "gained_successes": discordant("recovery12", reference),
            "lost_successes": discordant(reference, "recovery12"),
            "differences_method_minus_reference": differences,
            "scope": "Scene-cluster bootstrap over forty paired scene means; no independent-attempt binomial test."}


def verify(output=OUTPUT):
    bindings = verify_bindings(output)
    if (output / "smoke.json").exists() and base.read_json(output / "smoke.json").get("passed"):
        verify_smoke(output, bindings)
    else:
        scan_records(output, bindings, True)
    rows = scan_records(output, bindings)
    return {"complete": len(rows) == len(schedule()), "completed_attempts": len(rows), "expected_attempts": 240}


def summarize(output):
    bindings = verify_bindings(output)
    verify_smoke(output, bindings)
    saved = base.read_json(output / "evaluation.json")
    assert saved["complete"] and saved["backbone_unchanged"]
    rows = scan_records(output, bindings)
    assert [identity(row) for row in rows] == schedule()
    lookup = {(row["trial"]["scene"], row["trial"]["draw"], row["arm"]): row for row in rows}
    for scene in SCENES:
        initial = lookup[scene, DRAWS[0], "fixed8"]["trial"]["initial"]
        for draw in DRAWS:
            traces = {arm: verify_record(lookup[scene, draw, arm], output, bindings) for arm in ARMS}
            assert all(adaptive.states_match(initial, trace["initial"]) for trace in traces.values())
            baseline, milestone = traces["fixed8"]["steps"], traces["milestone12"]["steps"]
            assert baseline[0]["predicted_commands"][:8] == milestone[0]["predicted_commands"][:8]
            limit = min(8, lookup[scene, draw, "milestone12"]["trial"]["unit_lengths"][0])
            for left, right in zip(baseline[:limit], milestone[:limit]):
                assert left["action"] == right["action"] and adaptive.states_match(left["after"], right["after"])
    result = {"complete": True, "methods": metric_table(rows),
              "per_draw": {str(draw): metric_table(rows, draw) for draw in DRAWS},
              "comparisons": [paired(rows, reference) for reference in ["milestone12", "fixed8"]],
              "statistics": STATISTICS, "eval_bindings_sha256": base.sha(output / "eval-bindings.json"),
              "scope": "Forty project-unseen scenes, two draws each; all 240 newly collected, no pilot pooling."}
    path = output / "summary.json"
    if path.exists():
        assert base.read_json(path) == result, "Completed summary changed."
    else:
        base.save_json(path, result)
    return result


def _run(output):
    if (output / "summary.json").exists():
        return summarize(output)
    bindings = verify_bindings(output)
    verify(output)  # Validate every existing record and orphan before loading the model.
    training.gpu_guard()
    policy, _ = adaptive.load_policy(base.MODEL)
    heads = {"original": training.tail_state(policy),
             "trained": training.read_tail(Path(bindings["inputs"]["final_head"]["path"]),
                                           bindings["inputs"]["final_head"]["sha256"])}
    assert training.tail_digest(heads["original"]) == bindings["original_head_digest"]
    assert training.tail_digest(heads["trained"]) == head_identity(bindings, "recovery12")
    backbone = training.backbone_digest(policy)
    smoke(output, policy)
    rows = scan_records(output, bindings)
    try:
        for item in schedule()[len(rows):]:
            rows.append(collect(policy, item["scene"], item["draw"], item["arm"],
                                output / "evaluation" / item["arm"], output, bindings, heads))
            base.save_json(output / "evaluation.json", {"complete": False, "records": rows})
            print(f"Recovery confirmation {len(rows)}/240: {item['arm']}, scene {item['scene']}, "
                  f"draw {item['draw']}, success={rows[-1]['trial']['success']}", flush=True)
        assert training.backbone_digest(policy) == backbone
        base.save_json(output / "evaluation.json", {"complete": True, "backbone_unchanged": True, "records": rows})
    finally:
        activate(policy, heads["original"], bindings["original_head_digest"])
    assert base.sha(base.MODEL / "model.safetensors") == base.WEIGHTS_SHA256
    return summarize(output)


def run(output=OUTPUT):
    check_output(output)
    if (output / "summary.json").exists():
        return summarize(output)
    bindings = verify_bindings(output)
    before = len(scan_records(output, bindings))
    resource = output / f"evaluation-resources-from-{before:03}.json"
    retry = 0
    while resource.exists():
        retry += 1
        resource = output / f"evaluation-resources-from-{before:03}-retry-{retry}.json"
    started = time.perf_counter()
    memory = adaptive.MemorySampler()
    try:
        with memory:
            return _run(output)
    finally:
        index_path = output / "evaluation.json"
        after = len(base.read_json(index_path)["records"]) if index_path.exists() else before
        base.save_json(resource, {"wall_seconds": time.perf_counter() - started,
                                  "peak_process_rss_bytes": memory.peak_rss,
                                  "minimum_system_available_bytes": memory.minimum_available,
                                  "completed_attempts_before": before, "completed_attempts_after": after,
                                  "complete": (output / "summary.json").exists(),
                                  "scope": "This invocation including load, smoke and verification; not episode timing."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=["prepare", "verify", "run", "summarize"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = globals()[args.stage](args.output_dir)
    print(result if args.stage == "verify" else f"Confirmation stage {args.stage} completed.", flush=True)


if __name__ == "__main__":
    main()
