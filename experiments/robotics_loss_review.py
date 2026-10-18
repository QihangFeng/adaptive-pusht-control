"""Review six lost successes using saved evidence only; never change control.

Standard-library CPU diagnostics. No policy imports, inference, physics replay,
RGB fitting, new trials or controller thresholds. All future/true-state windows
are retrospective outcomes, not proposed runtime inputs or causal labels.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import statistics

import robotics_decision_review as previous


OUTPUT = Path("results/robotics-loss-review")
COHORT = Path("results/robotics-cohort-completion")
GATE = Path("results/robotics-weighted-gate")
ORIGINAL = Path("results/robotics-fresh-evaluation")
METHODS = ["fixed8", "fixed12", "adaptive12"]
SEEDS = list(range(50, 90))
LOSSES = [54, 59, 78, 80, 81, 82]
WINDOW = 48
NEAR = .90
GOAL = (256., 256., math.pi / 4)
STATIONARY_EPSILON = 1e-9


def angle_distance(left, right):
    return abs((left - right + math.pi) % (2 * math.pi) - math.pi)


def distance(left, right):
    return math.dist(left, right)


def pose_error(state):
    angle = angle_distance(state["block_angle"], GOAL[2])
    return {"position_error": distance(state["block_pos"], GOAL[:2]),
            "orientation_error_radians": angle,
            "orientation_error_degrees": math.degrees(angle)}


def terminal_window(trace, width=WINDOW):
    """Include the state BEFORE the selected terminal-aligned movements."""
    assert isinstance(width, int) and width > 0
    rows = trace["steps"]
    assert rows
    index = max(0, len(rows) - width)
    selected = rows[index:]
    start = previous.before(trace, index)
    states = [start] + [r["after"] for r in selected]
    coverage = [s["coverage"] for s in states]
    commands = ([rows[index - 1]["action"]] if index else []) + [r["action"] for r in selected]
    return {"start_before_step": index, "end_after_step": rows[-1]["step"],
            "observed_movements": len(selected), "start_coverage": coverage[0],
            "end_coverage": coverage[-1], "minimum_coverage": min(coverage),
            "maximum_coverage": max(coverage), "coverage_range": max(coverage) - min(coverage),
            "net_change": coverage[-1] - coverage[0], "best_gain": max(coverage) - coverage[0],
            "model_queries": sum(r["new_prediction"] for r in selected),
            "block_path": sum(distance(a["block_pos"], b["block_pos"]) for a, b in zip(states, states[1:])),
            "block_endpoint_displacement": distance(states[0]["block_pos"], states[-1]["block_pos"]),
            "block_rotation_path_radians": sum(angle_distance(a["block_angle"], b["block_angle"])
                                              for a, b in zip(states, states[1:])),
            "pusher_path": sum(distance(a["pusher_pos"], b["pusher_pos"]) for a, b in zip(states, states[1:])),
            "command_path": sum(distance(a, b) for a, b in zip(commands, commands[1:])),
            "final_goal_error": pose_error(states[-1])}


def coverage_drawdown(trace):
    """Largest chronological peak-to-trough drop; recovery follows THAT trough."""
    states = [(0, trace["initial"]["coverage"])] + [(r["step"], r["after"]["coverage"]) for r in trace["steps"]]
    peak = states[0]
    best_peak = best_trough = peak
    drawdown = 0.
    for point in states[1:]:
        if point[1] > peak[1]:
            peak = point
        if peak[1] - point[1] > drawdown:
            drawdown, best_peak, best_trough = peak[1] - point[1], peak, point
    recovered = next((point for point in states if point[0] > best_trough[0]
                      and point[1] >= best_peak[1]), None) if drawdown else None
    return {"peak_step": best_peak[0], "peak_coverage": best_peak[1],
            "trough_step": best_trough[0], "trough_coverage": best_trough[1],
            "drawdown": drawdown, "first_regained_step": recovered[0] if recovered else None,
            "regained_coverage": recovered[1] if recovered else None}


def stationary_tail(trace):
    states = [trace["initial"]] + [r["after"] for r in trace["steps"]]
    index = len(states) - 1
    while index > 0 and distance(states[index]["block_pos"], states[index - 1]["block_pos"]) <= STATIONARY_EPSILON \
            and angle_distance(states[index]["block_angle"], states[index - 1]["block_angle"]) <= STATIONARY_EPSILON:
        index -= 1
    tail = trace["steps"][index:]
    window = terminal_window(trace, len(tail)) if tail else None
    return {"start_before_step": index, "end_after_step": len(trace["steps"]),
            "movements": len(tail), "model_queries": sum(r["new_prediction"] for r in tail),
            "coverage": states[-1]["coverage"],
            "pusher_path": window["pusher_path"] if window else 0.,
            "command_path": window["command_path"] if window else 0.}


def stationary_runs(trace, minimum_movements=24):
    """Include successful NONterminal plateaus; do not infer failure from them."""
    assert isinstance(minimum_movements, int) and minimum_movements > 0
    states = [trace["initial"]] + [r["after"] for r in trace["steps"]]
    runs, start = [], None
    for index in range(1, len(states) + 1):
        stationary = index < len(states) and \
            distance(states[index]["block_pos"], states[index - 1]["block_pos"]) <= STATIONARY_EPSILON and \
            angle_distance(states[index]["block_angle"], states[index - 1]["block_angle"]) <= STATIONARY_EPSILON
        if stationary and start is None:
            start = index - 1
        if not stationary and start is not None:
            end = index - 1
            if end - start >= minimum_movements and states[start]["coverage"] >= NEAR:
                runs.append({"start_before_step": start, "end_after_step": end,
                             "movements": end - start, "coverage": states[start]["coverage"],
                             "model_queries": sum(r["new_prediction"] for r in trace["steps"][start:end])})
            start = None
    return runs


def outcome_group(fixed_success, adaptive_success):
    if fixed_success:
        return "retained_baseline_success" if adaptive_success else "lost_baseline_success"
    return "rescued_baseline_failure" if adaptive_success else "shared_failure"


def episode_diagnostics(episode, trace):
    rows = trace["steps"]
    observations = [{"step": 0, "after": trace["initial"]}] + rows
    peak = max(observations, key=lambda r: r["after"]["coverage"])
    valid = [r for r in rows if r.get("features", {}).get("quality")
             and isinstance(r["features"].get("coverage"), (int, float))
             and math.isfinite(r["features"]["coverage"])]
    camera_peak = max(valid, key=lambda r: r["features"]["coverage"]) if valid else None
    first_near = next((r["step"] for r in observations if r["after"]["coverage"] >= NEAR), None)
    duration = 0
    for row in reversed(rows):
        if row["after"]["coverage"] < NEAR:
            break
        duration += 1
    extension_steps = [e["step"] for e in episode["events"] if e["kind"] == "extend"]
    interrupts = [e["step"] for e in episode["events"] if e["kind"] == "interrupt"]
    windows = []
    for step in interrupts:
        selected = previous.current_unit(rows, step, 8)
        assert selected and selected[0]["new_prediction"]
        windows.append({"interrupt_after_step": step, "new_plan_starts_at": step + 1,
                        "following_steps": [r["step"] for r in selected],
                        "outcome": previous.window_stats(rows[step - 1]["after"], selected)})
    return {"peak_step": peak["step"], "peak_coverage": peak["after"]["coverage"],
            "peak_goal_error": pose_error(peak["after"]), "first_near_step": first_near,
            "terminal": terminal_window(trace), "terminal_near_duration": duration,
            "stationary_tail": stationary_tail(trace), "near_stationary_runs": stationary_runs(trace),
            "drawdown": coverage_drawdown(trace),
            "camera": {"valid_rows": len(valid),
                       "false_completion_rows": sum(r["features"]["coverage"] > .95
                                                     and r["after"]["coverage"] <= .95 for r in valid),
                       "maximum_error": max((abs(r["features"]["coverage"] - r["after"]["coverage"])
                                             for r in valid), default=None),
                       "peak": {"step": camera_peak["step"], "camera_coverage": camera_peak["features"]["coverage"],
                                "true_coverage_same_step": camera_peak["after"]["coverage"]} if camera_peak else None},
            "extensions": len(extension_steps), "interruptions": interrupts,
            "last_extension_step": extension_steps[-1] if extension_steps else None,
            "event_windows": windows}


def source_sidecar(seed, method):
    root = ORIGINAL if seed < 70 else GATE if method == "adaptive12" else COHORT
    return root / "episodes" / f"seed-{seed}-{method}.json"


def read_record(record, bindings):
    seed, method = record["seed"], record["method"]
    assert seed in SEEDS and method in METHODS
    path = source_sidecar(seed, method)
    if "source_sidecar" in record:
        assert record["source_sidecar"] == path.as_posix() and record["trace_directory"] == path.parent.as_posix()
    episode = previous.read(path)
    assert episode == {k: v for k, v in record.items() if k not in ["source_sidecar", "trace_directory"]}
    assert episode["trace"] == f"seed-{seed}-{method}.trace.json.gz"
    trace_path = path.parent / episode["trace"]
    assert previous.sha(trace_path) == episode["trace_sha256"]
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    rows = trace["steps"]
    assert trace["initial"] == episode["initial"] and 1 <= len(rows) == episode["steps"] <= 300
    assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    assert sum(r["new_prediction"] for r in rows) == episode["model_queries"]
    assert sum(episode["unit_lengths"]) == len(rows)
    assert episode["final_coverage"] == rows[-1]["after"]["coverage"]
    assert episode["success"] == (episode["final_coverage"] > .95)
    assert episode["success"] or (len(rows) == 300 and episode["truncated"])
    assert all(math.isfinite(r["after"]["coverage"]) and 0 <= r["after"]["coverage"] <= 1 for r in rows)
    assert all(r["after"]["coverage"] <= .95 for r in rows[:-1])
    for source in [path, trace_path]:
        bindings[source.as_posix()] = previous.sha(source)
    return episode, trace


def same_state(left, right):
    return {k: v for k, v in left.items() if k != "coverage"} == {k: v for k, v in right.items() if k != "coverage"} \
        and abs(left["coverage"] - right["coverage"]) <= 1e-12


def common_prefix(left, right):
    assert same_state(left["initial"], right["initial"])
    count = 0
    for a, b in zip(left["steps"], right["steps"]):
        if a["action"] != b["action"] or not same_state(a["after"], b["after"]):
            break
        count += 1
    assert count >= 8
    return count


def summarize_group(records):
    return {"seeds": [r["seed"] for r in records], "scenes": len(records),
            "successes": sum(r["success"] for r in records),
            "median_peak_coverage": statistics.median(r["diagnostics"]["peak_coverage"] for r in records),
            "median_terminal_range": statistics.median(r["diagnostics"]["terminal"]["coverage_range"] for r in records),
            "median_terminal_block_path": statistics.median(r["diagnostics"]["terminal"]["block_path"] for r in records),
            "terminal_stationary_for_24_movements": [r["seed"] for r in records
                                                     if r["diagnostics"]["stationary_tail"]["movements"] >= 24],
            "near_stationary_for_24_movements": [r["seed"] for r in records
                                                 if r["diagnostics"]["near_stationary_runs"]],
            "any_camera_over_95_while_truth_not": [r["seed"] for r in records
                                                   if r["diagnostics"]["camera"]["false_completion_rows"] > 0],
            "largest_drawdown_over_10pp": [r["seed"] for r in records
                                           if r["diagnostics"]["drawdown"]["drawdown"] > .10],
            "interrupted_scenes": [r["seed"] for r in records if r["diagnostics"]["interruptions"]]}


def build_review():
    cohort = previous.read(COHORT / "episodes.json")
    old = previous.read(ORIGINAL / "episodes.json")
    assert cohort["complete"] and old["complete"]
    prior = previous.read(previous.OUTPUT / "analysis.json")
    prior_records = {(r["seed"], r["method"]): r for r in prior["records"]}
    bindings = dict(previous.read(COHORT / "protocol.json")["fingerprints"])
    for path, value in prior["fingerprints"].items():
        assert path not in bindings or bindings[path] == value
        bindings[path] = value
    for root in [COHORT, ORIGINAL, GATE]:
        assert previous.read(root / "independent-audit.json")["passed"]
        for name in ["protocol.json", "episodes.json", "summary.json", "independent-audit.json"]:
            path = root / name
            bindings[path.as_posix()] = previous.sha(path)
    for path in [previous.OUTPUT / "analysis.json", previous.OUTPUT / "independent-audit.json",
                 Path(__file__).relative_to(Path.cwd())]:
        bindings[path.as_posix()] = previous.sha(path)
    assert previous.read(previous.OUTPUT / "independent-audit.json")["passed"]
    for path, expected in bindings.items():
        assert previous.sha(path) == expected, path
    source = old["episodes"] + cohort["episodes"]
    assert len(source) == 120 and {(r["seed"], r["method"]) for r in source} == {(s, m) for s in SEEDS for m in METHODS}
    records, loaded = [], {}
    for item in sorted(source, key=lambda r: (r["seed"], METHODS.index(r["method"]))):
        episode, trace = read_record(item, bindings)
        key = (episode["seed"], episode["method"])
        loaded[key] = (episode, trace)
        suffixes = prior_records[key]["extensions"] if key in prior_records else previous.extensions(episode, trace)
        records.append({"seed": key[0], "method": key[1], "source_sidecar": source_sidecar(*key).as_posix(),
                        "success": episode["success"], "steps": episode["steps"],
                        "final_coverage": episode["final_coverage"], "model_queries": episode["model_queries"],
                        "diagnostics": episode_diagnostics(episode, trace), "extension_windows": suffixes})
    groups = {name: [] for name in ["lost_baseline_success", "rescued_baseline_failure", "retained_baseline_success", "shared_failure"]}
    pairs = []
    for seed in SEEDS:
        fixed, adaptive = loaded[seed, "fixed8"], loaded[seed, "adaptive12"]
        group = outcome_group(fixed[0]["success"], adaptive[0]["success"])
        adaptive_record = next(r for r in records if (r["seed"], r["method"]) == (seed, "adaptive12"))
        groups[group].append(adaptive_record)
        pairs.append({"seed": seed, "group": group, "exact_common_movements": common_prefix(fixed[1], adaptive[1])})
    assert [r["seed"] for r in groups["lost_baseline_success"]] == LOSSES
    return {"scope": "Saved-evidence review only; no new predictions, trials, controller changes or causal recovery proof.",
            "milestone": previous.MILESTONE, "seeds": SEEDS, "methods": METHODS,
            "definitions": {"terminal_window_movements": WINDOW, "near_coverage": NEAR,
                            "goal_body_origin": list(GOAL), "stationary_per_movement_epsilon": STATIONARY_EPSILON,
                            "units": "Distances are simulator/action coordinate units, not metres or 96x96 camera pixels; angles wrap modulo2pi.",
                            "terminal_window": "Last48 executed movements including the state before the window; terminal-aligned, not common physical time.",
                            "stationary_tail": "Consecutive final movements with block-origin displacement and wrapped angle change each<=1e-9; not pusher stillness or failure label.",
                            "drawdown": "Largest chronological true-coverage peak-to-trough drop, followed by first regained peak; regaining coverage is not task success.",
                            "camera": "Only saved finite quality=True coverage rows; no image refitting. Truth comparison is scoring-only.",
                            "screens": "90% near-target, 24-motion stationary and10pp drawdown bins are descriptive post-hoc summaries, not new runtime thresholds."},
            "groups": {name: summarize_group(rows) for name, rows in groups.items()},
            "pairs": pairs, "records": records, "fingerprints": bindings,
            "reuse": "Completed50-69 extension windows reused from hash-bound previous decision review; other diagnoses are newly derived. No old probe/trial repeated.",
            "limitations": ["Outcome-defined inspected scenes, not held-out evaluation or a learned decision rule.",
                            "One stochastic sample per method/scene; cannot separate controller effects from changed predictions causally.",
                            "Only initial/common prefix states match; later cross-method histories are not matched-state alternatives.",
                            "Simulator pose, contact gap and future outcomes remain diagnostics only, never controller inputs.",
                            "No alternative action, physics replay, image fitting or new model inference is performed.",
                            "Successful repositioning can lower coverage; low net displacement can hide out-and-back motion.",
                            "Reused earlier traces lack full discarded plans; complete new adaptive plans do not fix that older limitation."]}


def check_output(output):
    destination = Path(output).resolve()
    if destination not in {OUTPUT.resolve(), OUTPUT.with_name(OUTPUT.name + "-repeat").resolve()}:
        raise ValueError("Use the dedicated loss-review directory or its -repeat directory.")


def save_review(document, output):
    check_output(output)
    path = Path(output) / "analysis.json"
    if path.exists():
        assert previous.read(path) == document, "Completed loss review changed; do not overwrite it."
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    check_output(args.output_dir)
    document = build_review()
    save_review(document, args.output_dir)
    print(json.dumps({"groups": document["groups"], "records": len(document["records"]),
                      "verified_bindings": len(document["fingerprints"])}, indent=2))


if __name__ == "__main__":
    main()
