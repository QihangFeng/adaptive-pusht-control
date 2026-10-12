"""Small, deterministic benefit tree for the frozen-policy replanning pilot.

Labels compare complete, matched continuations, not instantaneous alignment.
The tree and threshold are selected on discovery/validation episodes only.
This NumPy implementation avoids adding a training dependency to the simulator.
"""

from __future__ import annotations

import math

import numpy as np


def _vector(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("Features must be a nonempty, finite one-dimensional vector.")
    return result


def fit_tree(X, y, max_depth=2, min_leaf=3):
    """Fit a CART-style squared-error tree; return only JSON-compatible values.

    Features and candidate thresholds are visited in ascending order. Equal
    improvements retain the first split, making training deterministic. Depth
    counts edges from the root; min_leaf constrains children, not a tiny root.
    """
    if (isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
            or isinstance(min_leaf, bool) or not isinstance(min_leaf, int) or min_leaf < 1):
        raise ValueError("max_depth must be a nonnegative integer; min_leaf a positive integer.")
    features, labels = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (features.ndim != 2 or not all(features.shape) or labels.ndim != 1
            or len(labels) != len(features) or not np.isfinite(features).all()
            or not np.isfinite(labels).all()):
        raise ValueError("X must be a finite nonempty matrix with one finite y per row.")

    def grow(indices, depth):
        subset = labels[indices]
        prediction = float(subset.mean())
        error = float(np.square(subset - prediction).sum())
        node = {"prediction": prediction, "n_samples": int(len(indices)),
                "squared_error": error}
        if depth == max_depth or len(indices) < 2 * min_leaf or error <= 1e-12:
            return node
        best = None
        best_error = error
        for feature in range(features.shape[1]):
            values = np.unique(features[indices, feature])
            for lower, upper in zip(values[:-1], values[1:]):
                # The divided form also avoids overflowing two large finite values.
                threshold = float(lower / 2 + upper / 2)
                left_mask = features[indices, feature] <= threshold
                left, right = indices[left_mask], indices[~left_mask]
                if min(len(left), len(right)) < min_leaf:
                    continue
                split_error = float(np.square(labels[left] - labels[left].mean()).sum()
                                    + np.square(labels[right] - labels[right].mean()).sum())
                if split_error < best_error - 1e-12:
                    best_error = split_error
                    best = feature, threshold, left, right
        if best is not None:
            feature, threshold, left, right = best
            node.update(feature=feature, threshold=threshold,
                        left=grow(left, depth + 1), right=grow(right, depth + 1))
        return node

    return {"format_version": 1, "n_features": int(features.shape[1]),
            "max_depth": max_depth, "min_leaf": min_leaf,
            "root": grow(np.arange(len(features)), 0)}


def predict_tree(tree, x):
    """Predict one feature vector using an in-memory or JSON-loaded tree."""
    features = _vector(x)
    if tree.get("format_version") != 1 or len(features) != tree.get("n_features"):
        raise ValueError("Tree format or feature dimension does not match the input.")
    node = tree["root"]
    while "feature" in node:
        node = node["left" if features[node["feature"]] <= node["threshold"] else "right"]
    prediction = float(node["prediction"])
    if not math.isfinite(prediction):
        raise ValueError("Tree contains a nonfinite prediction.")
    return prediction


def aggregate_pairs(records, expected_repeats=None):
    """Aggregate unordered arm records into one training case per checkpoint.

    Required record fields: checkpoint_id, episode_id, sample_seed, arm,
    features, success, model_queries. Arms are 'continue' and 'replan'. Pairing
    uses sample_seed, never record position. Output preserves checkpoint order
    of first appearance and sorts its paired samples by their integer seed.
    """
    if expected_repeats is not None and (isinstance(expected_repeats, bool)
                                        or not isinstance(expected_repeats, int)
                                        or expected_repeats < 1):
        raise ValueError("expected_repeats must be a positive integer.")
    checkpoints = {}
    for record in records:
        checkpoint_id, episode_id = record["checkpoint_id"], record["episode_id"]
        if (not isinstance(checkpoint_id, (str, int)) or isinstance(checkpoint_id, bool)
                or not isinstance(episode_id, (str, int)) or isinstance(episode_id, bool)):
            raise ValueError("Checkpoint and episode IDs must be strings or integers.")
        features = _vector(record["features"]).tolist()
        seed, arm = record["sample_seed"], record["arm"]
        queries, success = record["model_queries"], record["success"]
        if isinstance(seed, bool) or not isinstance(seed, int) or arm not in ("continue", "replan"):
            raise ValueError("Every record needs an integer sample_seed and a recognized arm.")
        if (not isinstance(queries, int) or isinstance(queries, bool) or queries < 0
                or not isinstance(success, (bool, int)) or success not in (False, True)):
            raise ValueError("Success must be binary; model_queries a nonnegative integer.")
        case = checkpoints.setdefault(checkpoint_id, {
            "checkpoint_id": checkpoint_id, "episode_id": episode_id,
            "features": features, "samples": {},
        })
        if case["episode_id"] != episode_id or case["features"] != features:
            raise ValueError("All records for a checkpoint must share its episode and features.")
        sample = case["samples"].setdefault(seed, {})
        if arm in sample:
            raise ValueError("Duplicate checkpoint/sample/arm record.")
        sample[arm] = {"success": bool(success), "model_queries": queries}
    cases = []
    for case in checkpoints.values():
        samples = case.pop("samples")
        if any(set(sample) != {"continue", "replan"} for sample in samples.values()):
            raise ValueError("Every sample must contain both matched continuation arms.")
        if expected_repeats is not None and len(samples) != expected_repeats:
            raise ValueError("Checkpoint has an unexpected number of matched repetitions.")
        pairs = [{"sample_seed": seed, **samples[seed]} for seed in sorted(samples)]
        case["pairs"] = pairs
        case["benefit"] = float(np.mean([int(pair["replan"]["success"])
                                         - int(pair["continue"]["success"]) for pair in pairs]))
        cases.append(case)
    return cases


def grouped_split(cases, train_episodes, validation_episodes):
    """Explicit episode-level split; reject overlap, duplicates, and unassigned cases."""
    train_ids, validation_ids = set(train_episodes), set(validation_episodes)
    if train_ids & validation_ids:
        raise ValueError("An original episode cannot appear in both train and validation.")
    seen, train, validation = set(), [], []
    for case in cases:
        if case["checkpoint_id"] in seen:
            raise ValueError("Duplicate checkpoint case.")
        seen.add(case["checkpoint_id"])
        if case["episode_id"] in train_ids:
            train.append(case)
        elif case["episode_id"] in validation_ids:
            validation.append(case)
        else:
            raise ValueError("Every case must belong to an explicitly assigned episode.")
    if not train or not validation:
        raise ValueError("Both episode splits must contain at least one checkpoint case.")
    return train, validation


def should_replan(tree, features, threshold, minimum_margin=0.05):
    """Gate on positive predicted benefit; never use current continuation outcomes."""
    if (not math.isfinite(float(threshold)) or not math.isfinite(float(minimum_margin))
            or minimum_margin <= 0):
        raise ValueError("Threshold must be finite and minimum_margin strictly positive.")
    return predict_tree(tree, features) > max(float(threshold), float(minimum_margin))


def choose_threshold(tree, validation_cases, thresholds=(0, 0.1, 0.34, 0.67, 1.1),
                     minimum_margin=0.05):
    """Select only on validation: success first, fewer queries, higher threshold.

    Each checkpoint is equally weighted after averaging its paired samples.
    The always-continue candidate (1.1 for labels bounded by [-1, 1]) supplies
    a conservative fallback. This small-sample estimate is not a significance
    test or a performance guarantee. Freeze the returned threshold before test.
    """
    if not validation_cases or not thresholds:
        raise ValueError("Validation cases and threshold candidates cannot be empty.")
    candidates = []
    for threshold in thresholds:
        case_outcomes, case_queries, baseline_outcomes, decisions = [], [], [], []
        for case in validation_cases:
            decision = should_replan(tree, case["features"], threshold, minimum_margin)
            arm = "replan" if decision else "continue"
            pairs = case["pairs"]
            if not pairs:
                raise ValueError("Every validation case needs matched continuation pairs.")
            case_outcomes.append(float(np.mean([pair[arm]["success"] for pair in pairs])))
            case_queries.append(float(np.mean([pair[arm]["model_queries"] for pair in pairs])))
            baseline_outcomes.append(float(np.mean([pair["continue"]["success"] for pair in pairs])))
            decisions.append(bool(decision))
        chosen_success = float(np.mean(case_outcomes))
        baseline_success = float(np.mean(baseline_outcomes))
        candidates.append({
            "threshold": float(threshold), "mean_success": chosen_success,
            "mean_success_gain": chosen_success - baseline_success,
            "mean_model_queries": float(np.mean(case_queries)),
            "interventions": int(sum(decisions)), "cases": int(len(validation_cases)),
            "decisions": decisions,
        })
    # Rounded comparison keys suppress floating-point tie-breaking artifacts.
    selected = max(candidates, key=lambda item: (
        round(item["mean_success_gain"], 12), -round(item["mean_model_queries"], 12),
        item["threshold"],
    ))
    return {"threshold": selected["threshold"], "minimum_margin": float(minimum_margin),
            "selected": selected, "candidates": candidates,
            "selection_scope": "validation checkpoints only; not held-out test outcomes"}
