"""CPU-only retrospective review of milestone extensions and plan transitions.

Reads saved commands/states; does not import the policy, replay physics, fit RGB,
label a decline unrecoverable, or change a controller. Future windows below are
diagnostic outcomes, never proposed runtime inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCE = Path("results/robotics-fresh-evaluation")
OUTPUT = Path("results/robotics-decision-review")
MILESTONE = "0f4044966646df4cc38b0373fa52c0b733506e15"
SEEDS = list(range(50, 70))
METHODS = ["fixed8", "fixed12", "adaptive12"]
LOSS_PLAN_STARTS = {54: 189, 59: 145}  # Previously recorded diagnostic windows.
LOSS_PRE_DROP_STEPS = {54: 190, 59: 148}
DROP_SCREEN = .03  # Descriptive three-percentage-point screen, not a gate.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def delta(a, b):
    return [float(x) - float(y) for x, y in zip(a, b)]


def norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else None


def before(trace, index):
    return trace["initial"] if index == 0 else trace["steps"][index - 1]["after"]


def window_stats(start, rows):
    """True coverage changes are offline outcomes, not camera-gate inputs."""
    values = [float(start["coverage"])] + [float(r["after"]["coverage"]) for r in rows]
    peak, largest_drawdown = values[0], 0.
    for value in values[1:]:
        peak = max(peak, value)
        largest_drawdown = max(largest_drawdown, peak - value)
    return {"observed_movements": len(rows), "start_coverage": values[0],
            "end_coverage": values[-1], "net_change": values[-1] - values[0],
            "minimum_change_from_start": min(values) - values[0],
            "maximum_drawdown": largest_drawdown,
            "largest_single_drop": max([0.] + [a - b for a, b in zip(values, values[1:])])}


def current_unit(rows, index, maximum):
    selected = []
    for row in rows[index:index + maximum]:
        if selected and row["new_prediction"]:
            break
        selected.append(row)
    return selected


def extensions(episode, trace):
    rows, result = trace["steps"], []
    for event in episode["events"]:
        if event["kind"] != "extend":
            continue
        index = event["step"] - 1
        row = rows[index]
        assert row["plan_age"] == 8 and row["pending_before"] == row["pending_after"] == 4
        suffix = current_unit(rows, index + 1, 4) if index + 1 < len(rows) and not rows[index + 1]["new_prediction"] else []
        assert all(not r["new_prediction"] for r in suffix)
        record = {"step_after_eight": event["step"], "camera": row["features"],
                  "full_four_executed": len(suffix) == 4,
                  "suffix_steps": [r["step"] for r in suffix],
                  "block_translation": norm(delta(suffix[-1]["after"]["block_pos"], row["after"]["block_pos"])) if suffix else 0.,
                  "block_rotation_change": abs((suffix[-1]["after"]["block_angle"] - row["after"]["block_angle"] + math.pi)
                                               % (2 * math.pi) - math.pi) if suffix else 0.,
                  "outcome": window_stats(row["after"], suffix)}
        result.append(record)
    return result


def transitions(episode, trace):
    rows, result = trace["steps"], []
    for index, row in enumerate(rows):
        if index == 0 or not row["new_prediction"]:
            continue
        previous = rows[index - 1]
        preceding = rows[index - 2] if index >= 2 else None
        jump = delta(row["action"], previous["action"])
        prior_direction = delta(previous["action"], preceding["action"]) if preceding else None
        prior_event = next((e["kind"] for e in episode["events"] if e["step"] == previous["step"]), None)
        prefix = current_unit(rows, index, 8)
        state = previous["after"]
        result.append({"new_plan_starts_at": row["step"], "previous_plan_end": previous["step"],
                       "previous_plan_age": previous["plan_age"], "previous_end_event": prior_event,
                       "command_jump": norm(jump),
                       "command_direction_cosine": cosine(prior_direction, jump) if prior_direction else None,
                       "command_target_distance_from_pusher": norm(delta(row["action"], state["pusher_pos"])),
                       "pusher_speed_before": norm(state["pusher_velocity"]),
                       "pusher_block_signed_gap_before": state["pusher_block_gap"],
                       "camera_before": previous.get("features"),
                       "first_physical_displacement": norm(delta(row["after"]["pusher_pos"], state["pusher_pos"])),
                       "prefix_steps": [r["step"] for r in prefix],
                       "full_eight_observed": len(prefix) == 8,
                       "prefix_outcome": window_stats(state, prefix)})
    return result


def summarize_records(records):
    extensions_all = [e for r in records for e in r["extensions"]]
    transitions_all = [t for r in records for t in r["transitions"]]
    return {"scenes": len(records), "successes": sum(r["success"] for r in records),
            "extensions": len(extensions_all),
            "complete_four_movement_suffixes": sum(e["full_four_executed"] for e in extensions_all),
            "suffix_net_drops_over_one_point_five_pp": sum(e["outcome"]["net_change"] < -.015 for e in extensions_all),
            "suffix_net_drops_over_three_pp": sum(e["outcome"]["net_change"] < -DROP_SCREEN for e in extensions_all),
            "suffix_drawdowns_over_three_pp": sum(e["outcome"]["maximum_drawdown"] > DROP_SCREEN for e in extensions_all),
            "transitions": len(transitions_all),
            "command_direction_reversals": sum(t["command_direction_cosine"] is not None and
                                              t["command_direction_cosine"] < 0 for t in transitions_all),
            "prefix_drawdowns_over_three_pp": sum(t["prefix_outcome"]["maximum_drawdown"] > DROP_SCREEN
                                                 for t in transitions_all),
            "median_command_jump": statistics.median([t["command_jump"] for t in transitions_all])
            if transitions_all else None}


def recovery_review(trace, pre_drop_step):
    """Recovery of old overlap is different from crossing the success threshold."""
    rows = trace["steps"]
    reference = rows[pre_drop_step - 1]
    remaining = rows[pre_drop_step:]
    minimum = min(remaining, key=lambda r: r["after"]["coverage"])
    regained = next((r for r in rows[minimum["step"]:] if r["after"]["coverage"] >= reference["after"]["coverage"]), None)
    peak = max(rows, key=lambda r: r["after"]["coverage"])
    camera_rows = [r for r in rows if r.get("features", {}).get("quality")]
    camera_peak = max(camera_rows, key=lambda r: r["features"]["coverage"])
    return {"pre_drop": {"step": reference["step"], "coverage": reference["after"]["coverage"]},
            "minimum_after_drop": {"step": minimum["step"], "coverage": minimum["after"]["coverage"]},
            "first_regained_after_minimum": {"step": regained["step"], "coverage": regained["after"]["coverage"]} if regained else None,
            "whole_attempt_peak": {"step": peak["step"], "coverage": peak["after"]["coverage"]},
            "final_coverage": rows[-1]["after"]["coverage"],
            "camera_peak": {"step": camera_peak["step"], "camera_coverage": camera_peak["features"]["coverage"],
                            "true_coverage_same_step": camera_peak["after"]["coverage"]}}


def read_episode(seed, method, bindings):
    path = SOURCE / "episodes" / f"seed-{seed}-{method}.json"
    episode = read(path)
    assert (episode["seed"], episode["method"]) == (seed, method)
    assert episode["trace"] == f"seed-{seed}-{method}.trace.json.gz"
    trace_path = path.parent / episode["trace"]
    assert sha(trace_path) == episode["trace_sha256"]
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    rows = trace["steps"]
    assert trace["initial"] == episode["initial"]
    assert len(rows) == episode["steps"] <= 300
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    assert sum(r["new_prediction"] for r in rows) == episode["model_queries"]
    assert rows[-1]["after"]["coverage"] == episode["final_coverage"]
    assert episode["success"] == (episode["final_coverage"] > .95)
    for source in [path, trace_path]:
        bindings[source.as_posix()] = sha(source)
    return episode, trace


def check_output(output):
    """Only a dedicated directory or ignored repeat may receive derived output."""
    output = Path(output).resolve()
    allowed = {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}
    if output not in allowed:
        raise ValueError("Use the dedicated decision-review output or its -repeat directory.")


def build_review():
    protocol, summary = read(SOURCE / "protocol.json"), read(SOURCE / "summary.json")
    inventory, audit = read(SOURCE / "episodes.json"), read(SOURCE / "independent-audit.json")
    assert protocol["seeds"] == SEEDS and protocol["methods"] == METHODS
    assert inventory["complete"] and summary["complete"] and audit["passed"]
    expected = {(s, m) for s in SEEDS for m in METHODS}
    assert len(inventory["episodes"]) == 60
    assert {(r["seed"], r["method"]) for r in inventory["episodes"]} == expected
    bindings = dict(protocol["fingerprints"])
    for relative, expected_sha in bindings.items():
        assert sha(relative) == expected_sha, relative
    for name in ["protocol.json", "episodes.json", "summary.json", "independent-audit.json"]:
        path = SOURCE / name
        bindings[path.as_posix()] = sha(path)
    source = Path(__file__).relative_to(Path.cwd())
    bindings[source.as_posix()] = sha(source)
    records = []
    loaded = {}
    for seed in SEEDS:
        for method in METHODS:
            episode, trace = read_episode(seed, method, bindings)
            indexed = next(r for r in inventory["episodes"] if (r["seed"], r["method"]) == (seed, method))
            assert episode == indexed
            loaded[seed, method] = (episode, trace)
            records.append({"seed": seed, "method": method, "success": episode["success"],
                            "final_coverage": episode["final_coverage"],
                            "extensions": extensions(episode, trace),
                            "transitions": transitions(episode, trace)})
    groups = {"lost_baseline_success": [], "rescued_baseline_failure": [], "retained_baseline_success": []}
    for seed in SEEDS:
        fixed, adaptive = loaded[seed, "fixed8"][0], loaded[seed, "adaptive12"][0]
        assert fixed["success"] or adaptive["success"], "Unexpected shared milestone failure."
        name = ("lost_baseline_success" if not adaptive["success"] else
                "retained_baseline_success" if fixed["success"] else "rescued_baseline_failure")
        groups[name].append(next(r for r in records if (r["seed"], r["method"]) == (seed, "adaptive12")))
    loss_details = []
    for seed, start in LOSS_PLAN_STARTS.items():
        episode, trace = loaded[seed, "adaptive12"]
        current = next(r for r in records if (r["seed"], r["method"]) == (seed, "adaptive12"))
        transition = next(t for t in current["transitions"] if t["new_plan_starts_at"] == start)
        preceding = [e for e in current["extensions"] if e["step_after_eight"] < start]
        loss_details.append({"seed": seed, "previously_studied_plan_start": start,
                             "handover": transition, "preceding_extensions": preceding,
                             "decline_recovery": recovery_review(trace, LOSS_PRE_DROP_STEPS[seed]),
                             "last_preceding_extension": preceding[-1] if preceding else None,
                             "strongest_extension_drawdown": max(preceding, key=lambda e: e["outcome"]["maximum_drawdown"])
                             if preceding else None,
                             "handover_command_jump_rank_within_scene": 1 + sum(t["command_jump"] > transition["command_jump"]
                                                                             for t in current["transitions"]),
                             "handover_command_jump_rank_all_adaptive": 1 + sum(t["command_jump"] > transition["command_jump"]
                                                                               for r in records if r["method"] == "adaptive12"
                                                                               for t in r["transitions"])})
    return {"scope": "Retrospective CPU-only review; no new trials, changed controller or causal recovery proof.",
            "milestone": MILESTONE, "seeds": SEEDS, "methods": METHODS,
            "definitions": {"command_jump": "Distance between the last executed command target of an old plan and the first executed target of the new plan; not physical speed or a discarded-path mismatch.",
                            "direction_cosine": "Previous executed-command difference versus new boundary command difference; negative means reversal of command-target direction.",
                            "prefix": "At most the first eight movements of the new plan; stops at an earlier prediction or terminal step.",
                            "suffix": "At most four retained movements after an extension; stops at an earlier prediction or terminal step.",
                            "coverage": "Simulator truth for retrospective outcome analysis; camera estimates kept separately.",
                            "drop_screen": "Strictly greater than three percentage points; descriptive post-hoc screen, not a controller threshold or failure label."},
            "groups": {name: {"seeds": [r["seed"] for r in rows], **summarize_records(rows)} for name, rows in groups.items()},
            "method_summaries": {method: summarize_records([r for r in records if r["method"] == method]) for method in METHODS},
            "loss_details": loss_details, "records": records, "fingerprints": bindings,
            "limitations": ["Outcome-defined groups and old inspected scenes are development evidence, not fresh validation.",
                            "No alternative action was tried from a matched state in this review; association does not identify cause.",
                            "Unexecuted discarded command paths, original RGB pixels and latent prediction history are unavailable here.",
                            "Different methods generally reach different states; comparisons after the common initial prefix are not matched-state experiments.",
                            "Command direction changes and transient coverage drops can be useful adjustments.",
                            "One stochastic prediction sample per scene/method does not establish repeat reliability."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    check_output(args.output_dir)
    document = build_review()
    destination = args.output_dir / "analysis.json"
    if destination.exists():
        assert read(destination) == document, "Completed review changed; do not overwrite it."
        print("Completed review verified; no trials or analysis rewritten.")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"groups": document["groups"], "method_summaries": document["method_summaries"],
                      "verified_bindings": len(document["fingerprints"])}, indent=2))


if __name__ == "__main__":
    main()
