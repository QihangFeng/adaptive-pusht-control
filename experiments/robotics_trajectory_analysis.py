"""Frozen exploratory trace rules; patterns are not exclusive failure causes.

Coverage/contact come from simulator diagnostics, not a deployable visual monitor.
This module uses saved traces only and performs no policy inference or simulation.
"""

from __future__ import annotations

import math


RULES = {
    "near_target_coverage": 0.90,
    "cached_regression_drop": 0.05,
    "regression_minimum_pending_actions": 1,
    "low_progress_window_steps": 24,
    "low_progress_max_running_best_gain": 0.01,
    "low_progress_minimum_endpoint_coverage": 0.50,
}
REGRESSION = "near_target_cached_regression"
LOW_PROGRESS = "low_progress_window"
CONTACT_LOSS = "contact_loss_candidate"


def _event(row, label, **details):
    return {
        "label": label, "branch_step": row["step"], "chunk": row["chunk"],
        "pending_actions": row["pending_actions"],
        "new_prediction": bool(row["new_prediction"]),
        "coverage": float(row["after"]["coverage"]),
        "pair_contact_points": row["pair_contact_points"],
        **details,
    }


def _merge_windows(windows):
    """Merge overlapping/adjacent windows; they are not independent failures."""
    spans = []
    for window in windows:
        if spans and window["start_step"] <= spans[-1]["end_step"] + 1:
            spans[-1]["end_step"] = window["branch_step"]
            spans[-1]["qualifying_windows"] += 1
        else:
            spans.append({"start_step": window["start_step"],
                          "end_step": window["branch_step"], "qualifying_windows": 1})
    return spans


def analyze_trace(rows, episode):
    """Return features and all nonexclusive events for one completed episode.

    A: past-only peak within the current chunk, >=90% coverage and >=5 percentage
    point regression on a cached action with at least one pending action. Retain
    the first qualifying event per chunk. Chunk-start pre-action state is included.
    B: every trailing 24-action window with <=1 percentage point improvement in
    global running-best coverage and >=50% coverage at both endpoints. Exclude
    windows ending at episode completion; retain overlaps and report merged spans.
    """
    if not rows:
        raise ValueError("A completed episode must have at least one trace row.")
    if [row["step"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Trace must contain consecutive steps starting at one.")
    if episode["steps"] != len(rows):
        raise ValueError("Episode length does not match its trace.")
    values = [float(rows[0]["before"]["coverage"])]
    values.extend(float(row["after"]["coverage"]) for row in rows)
    if not all(math.isfinite(value) and -1e-12 <= value <= 1 + 1e-12 for value in values):
        raise ValueError("Coverage must be finite and between zero and one.")
    running_best = [values[0]]
    for value in values[1:]:
        running_best.append(max(running_best[-1], value))

    regressions, low_progress = [], []
    chunk, peak, peak_step, flagged = None, 0.0, 0, False
    largest_cached_drop = 0.0
    width = RULES["low_progress_window_steps"]
    for index, row in enumerate(rows):
        if row["chunk"] != chunk:
            chunk, peak, peak_step, flagged = (row["chunk"],
                                              float(row["before"]["coverage"]),
                                              row["step"] - 1, False)
        before, after = float(row["before"]["coverage"]), values[index + 1]
        if before > peak:
            peak, peak_step = before, row["step"] - 1
        drop = peak - after
        if not row["new_prediction"]:
            largest_cached_drop = max(largest_cached_drop, drop)
            if (not flagged and peak >= RULES["near_target_coverage"]
                    and drop >= RULES["cached_regression_drop"] - 1e-12
                    and row["pending_actions"] >= RULES["regression_minimum_pending_actions"]
                    and index + 1 < len(rows)):
                regressions.append(_event(row, REGRESSION, peak_coverage=peak,
                                          peak_step=peak_step, coverage_drop=drop))
                flagged = True
        if after > peak:
            peak, peak_step = after, row["step"]
        if index + 1 >= width and index + 1 < len(rows):
            start_index = index + 1 - width
            gain = running_best[index + 1] - running_best[start_index]
            start_coverage = float(rows[start_index]["before"]["coverage"])
            if (gain <= RULES["low_progress_max_running_best_gain"] + 1e-12
                    and min(start_coverage, after) >= RULES["low_progress_minimum_endpoint_coverage"]):
                window = rows[start_index:index + 1]
                low_progress.append(_event(
                    row, LOW_PROGRESS, start_step=window[0]["step"],
                    start_coverage=start_coverage, running_best_gain=gain,
                    running_best_start=running_best[start_index],
                    running_best_end=running_best[index + 1],
                    contact_steps=sum(item["pair_contact_points"] > 0 for item in window),
                    new_predictions=sum(bool(item["new_prediction"]) for item in window),
                ))

    candidates = episode.get("candidates", [])
    contacts = sum(row["pair_contact_points"] > 0 for row in rows)
    peak_index = max(range(len(values)), key=values.__getitem__)
    return {
        "seed": episode["seed"], "success": bool(episode["success"]),
        "steps": len(rows), "model_queries": episode.get("model_queries"),
        "initial_coverage": values[0], "final_coverage": values[-1],
        "max_coverage": running_best[-1], "peak_step": peak_index,
        "peak_to_final_drop": running_best[-1] - values[-1],
        "largest_cached_chunk_drop": largest_cached_drop,
        "contact_steps": contacts, "contact_step_fraction": contacts / len(rows),
        "contact_loss_candidates": [item for item in candidates if item["label"] == CONTACT_LOSS],
        "contact_stall_candidates": [item for item in candidates if item["label"] == "contact_stall_candidate"],
        "events": regressions + low_progress,
        "regression_events": regressions, "low_progress_windows": low_progress,
        "low_progress_spans": _merge_windows(low_progress),
    }


def analyze_collection(records):
    """Analyze records containing {'episode': summary, 'rows': saved trace rows}.

    Success/failure groups count attempts containing a pattern, not independent
    causal failures. Ranked regression events support human review only; recovery
    selection should be separately prespecified (for example, first event by seed).
    """
    episodes = [analyze_trace(record["rows"], record["episode"]) for record in records]
    seeds = [episode["seed"] for episode in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Do not double-count repeated seeds in the discovery collection.")
    groups = {}
    for outcome, success in [("success", True), ("failure", False)]:
        subset = [episode for episode in episodes if episode["success"] is success]
        groups[outcome] = {
            "attempts": len(subset),
            "with_cached_regression": sum(bool(item["regression_events"]) for item in subset),
            "cached_regression_events": sum(len(item["regression_events"]) for item in subset),
            "with_low_progress": sum(bool(item["low_progress_windows"]) for item in subset),
            "low_progress_overlapping_windows": sum(len(item["low_progress_windows"]) for item in subset),
            "low_progress_merged_spans": sum(len(item["low_progress_spans"]) for item in subset),
            "with_contact_loss_candidate": sum(bool(item["contact_loss_candidates"]) for item in subset),
            "with_contact_stall_candidate": sum(bool(item["contact_stall_candidates"]) for item in subset),
        }
    ranked = [{"seed": episode["seed"], "baseline_success": episode["success"], **event}
              for episode in episodes for event in episode["regression_events"]]
    ranked.sort(key=lambda item: (-item["coverage_drop"], item["seed"], item["branch_step"]))
    failures = sorted((item for item in episodes if not item["success"]),
                      key=lambda item: (-item["peak_to_final_drop"], item["seed"]))
    return {
        "purpose": "exploratory discovery; nonexclusive patterns, not causal labels",
        "rules": dict(RULES), "episodes": episodes, "groups": groups,
        "ranked_regression_events": ranked,
        "failure_review_order": [item["seed"] for item in failures],
        "limitations": ["Simulator coverage/contact are privileged diagnostics.",
                        "Low-progress windows overlap and are not independent samples.",
                        "Patterns may also occur in successful attempts.",
                        "Neither association nor ranking establishes recovery benefit."],
    }
