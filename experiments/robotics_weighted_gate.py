"""Pure camera-only gates and a small, explicitly optimistic development search.

Runtime callers apply a gate only to an already-confirmed milestone interruption.
The offline labels instead compare one interruption with finishing the whole unit;
they are a selection proxy, not execution of the proposed per-movement controller.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
from numbers import Integral, Real


NORMALIZATION = {"D_scale": .20, "W_scale": 5, "R_scale": .10, "R_movements": 2}
WEIGHTS = [(.50, .25, .25), (.25, .50, .25), (.25, .25, .50), (1 / 3, 1 / 3, 1 / 3)]


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _bounded(value, low, high):
    return _finite(value) and low <= value <= high


def signals(features, history):
    """Normalize drawdown, warning count and signed last-two-movement trend."""
    invalid = {"quality": False, "D": None, "W": None, "R": None}
    if features.get("quality") is not True:
        return invalid
    try:
        values = list(history)[-3:]
    except TypeError:
        return invalid
    coverage, drawdown, warnings = (features.get(key) for key in ["coverage", "drawdown", "warning_count"])
    if (len(values) != 3 or not all(_bounded(value, 0, 1) for value in values)
            or not _bounded(coverage, 0, 1) or values[-1] != coverage
            or not _bounded(drawdown, 0, 1)
            or isinstance(warnings, bool) or not isinstance(warnings, Integral) or not 0 <= warnings <= 5):
        return invalid
    return {"quality": True, "D": float(min(1., drawdown / NORMALIZATION["D_scale"])),
            "W": float(warnings / NORMALIZATION["W_scale"]),
            "R": float(max(-1., min(1., (values[-1] - values[-3]) / NORMALIZATION["R_scale"])))}


def validate_config(config):
    """Reject malformed/nonfinite configurations without changing their values."""
    if not isinstance(config, dict):
        raise ValueError("Gate configuration must be a dictionary.")
    family = config.get("family")
    if family == "weighted":
        if set(config) != {"family", "weights", "threshold"}:
            raise ValueError("Weighted gates require only family, weights and threshold.")
        weights = config["weights"]
        if (not isinstance(weights, (list, tuple)) or len(weights) != 3
                or not all(_bounded(weight, 0, 1) for weight in weights)
                or not math.isclose(math.fsum(weights), 1., rel_tol=0., abs_tol=1e-12)
                or not _bounded(config["threshold"], 0, 1)):
            raise ValueError("Weights must be nonnegative and sum to one; threshold must be in [0,1].")
        return {"family": family, "weights": [float(weight) for weight in weights],
                "threshold": float(config["threshold"])}
    if family == "threshold":
        if set(config) != {"family", "min_warnings", "min_drawdown"}:
            raise ValueError("Threshold gates require only family, min_warnings and min_drawdown.")
        warnings = config["min_warnings"]
        if (isinstance(warnings, bool) or not isinstance(warnings, Integral) or not 0 <= warnings <= 5
                or not _bounded(config["min_drawdown"], 0, 1)):
            raise ValueError("Warning threshold must be an integer in [0,5]; drawdown must be in [0,1].")
        return {"family": family, "min_warnings": int(warnings),
                "min_drawdown": float(config["min_drawdown"])}
    raise ValueError("Unknown gate family.")


def gate_decision(config, features, history):
    """Return an inclusive gate decision; invalid camera history means continue."""
    config = validate_config(config)
    normalized = signals(features, history)
    if not normalized["quality"]:
        return {"allowed": False, "signals": normalized, "score": None}
    if config["family"] == "weighted":
        w_d, w_w, w_r = config["weights"]
        score = w_d * normalized["D"] + w_w * normalized["W"] - w_r * normalized["R"]
        allowed = score >= config["threshold"]
    else:
        score = None
        allowed = features["warning_count"] >= config["min_warnings"] and features["drawdown"] >= config["min_drawdown"]
    return {"allowed": bool(allowed), "signals": normalized, "score": score}


def candidate_configs():
    """Stable order supplies the final deterministic selection tie-break."""
    return ([{"family": "weighted", "weights": list(weights), "threshold": threshold}
             for weights in WEIGHTS for threshold in [.50, .35, .65]]
            + [{"family": "threshold", "min_warnings": warnings, "min_drawdown": drawdown}
               for warnings in [3, 2] for drawdown in [.15, .10, .20]])


def extract_case_history(case):
    """Read only the three camera-feature rows ending at the saved checkpoint."""
    with gzip.open(case["source_trace"], "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["steps"]
    step = case["checkpoint"]
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 3 or step > len(rows):
        raise ValueError("Checkpoint must have three recorded camera rows.")
    selected = rows[step - 3:step]
    if [row["step"] for row in selected] != list(range(step - 2, step + 1)):
        raise ValueError("Checkpoint camera rows are not consecutive.")
    return [row.get("features", {}).get("coverage")
            if row.get("features", {}).get("quality") is True else None for row in selected]


def _paired_records(cases, records):
    """Require a complete two-arm panel, without consulting original outcomes."""
    identities = {(case["scene_seed"], case["checkpoint"]) for case in cases}
    if not cases or len(identities) != len(cases) or len({case["scene_seed"] for case in cases}) != len(cases):
        raise ValueError("Cases must contain distinct source scenes/checkpoints.")
    panel = {}
    for record in records:
        scene, checkpoint, seed = (record[key] for key in ["scene_seed", "checkpoint", "prediction_seed"])
        arm = record["arm"]
        key = (scene, checkpoint, seed, arm)
        if ((scene, checkpoint) not in identities or arm not in ["original", "defer_current_unit"] or key in panel
                or type(record["success"]) is not bool or isinstance(record["model_queries"], bool)
                or not isinstance(record["model_queries"], Integral) or record["model_queries"] < 0
                or not _bounded(record["final_coverage"], 0, 1)):
            raise ValueError("Duplicate, unexpected or invalid comparison record.")
        panel[key] = record
    seeds_by_case = {identity: sorted({key[2] for key in panel if key[:2] == identity}) for identity in identities}
    if not panel or any(not seeds for seeds in seeds_by_case.values()):
        raise ValueError("Every context requires paired future samples.")
    seed_sets = {tuple(seeds) for seeds in seeds_by_case.values()}
    if len(seed_sets) != 1:
        raise ValueError("Contexts must use the same prescribed prediction samples.")
    for identity, seeds in seeds_by_case.items():
        for seed in seeds:
            if any((*identity, seed, arm) not in panel for arm in ["original", "defer_current_unit"]):
                raise ValueError("Missing matched comparison arm.")
    return panel, seeds_by_case


def development_search(cases, records, histories):
    """Select each family on reused one-shot labels, not full-controller outcomes."""
    panel, seeds_by_case = _paired_records(cases, records)
    results = []
    for index, config in enumerate(candidate_configs()):
        decisions = [gate_decision(config, case["features"], histories[case["scene_seed"]]) for case in cases]
        chosen = [panel[(case["scene_seed"], case["checkpoint"], seed,
                         "original" if decision["allowed"] else "defer_current_unit")]
                  for case, decision in zip(cases, decisions) for seed in seeds_by_case[(case["scene_seed"], case["checkpoint"])]]
        results.append({"candidate_index": index, "config": validate_config(config),
                        "decision_mask": [decision["allowed"] for decision in decisions], "decisions": decisions,
                        "conditional_successes": sum(record["success"] for record in chosen),
                        "conditional_model_queries": sum(record["model_queries"] for record in chosen),
                        "conditional_mean_final_coverage": math.fsum(record["final_coverage"] for record in chosen) / len(chosen)})
    selected, details = {}, {}
    objective = lambda row: (-row["conditional_successes"], row["conditional_model_queries"],
                             -row["conditional_mean_final_coverage"])
    for family in ["weighted", "threshold"]:
        family_rows = [row for row in results if row["config"]["family"] == family]
        best = min(family_rows, key=lambda row: (*objective(row), row["candidate_index"]))
        tied = [row for row in family_rows if objective(row) == objective(best)]
        equivalent = [row["candidate_index"] for row in family_rows if row["decision_mask"] == best["decision_mask"]]
        selected[family] = copy.deepcopy(best["config"])
        details[family] = {"selected_index": best["candidate_index"],
                           "tied_best_indices": [row["candidate_index"] for row in tied],
                           "tied_best_configurations": [copy.deepcopy(row["config"]) for row in tied],
                           "equivalent_selected_mask_indices": equivalent,
                           "selected_mask_equivalence_count": len(equivalent),
                           "distinct_decision_mask_count": len({tuple(row["decision_mask"]) for row in family_rows})}
    return {"scope": "Offline development selection on reused selected-state continuation labels; optimistic proxy, not full-controller results.",
            "normalization": dict(NORMALIZATION), "formula": "wD*D + wW*W - wR*R >= threshold; R is signed two-movement camera change",
            "selection_order": ["conditional successes descending", "conditional predictions ascending",
                                "conditional final alignment descending", "candidate index ascending"],
            "context_order": [{"scene_seed": case["scene_seed"], "checkpoint": case["checkpoint"],
                               "prediction_seeds": seeds_by_case[(case["scene_seed"], case["checkpoint"])]} for case in cases],
            "paired_futures": sum(len(seeds) for seeds in seeds_by_case.values()),
            "search_results": results, "selected": selected, "selection_details": details,
            "limits": ["These contexts/outcomes already informed the proposal; search objective is optimistic development evidence, not held-out accuracy.",
                       "Offline continue labels suppress interruption for the current unit; a runtime gate is reconsidered every movement and can interrupt later in that unit.",
                       "Future trajectories, prediction timing and interruption budgets can change after any gate decision; an outcome-selected mixture is not a measured controller.",
                       "Correlated signals, few contexts and mask-equivalent/tied configurations do not uniquely identify useful weights or guarantee generalization."]}
