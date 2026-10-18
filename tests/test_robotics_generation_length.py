"""CPU-only integrity checks for the frozen generation-length screen.

The policy and environment doubles make no GPU calls or model predictions.
"""

import copy
from collections import deque
from contextlib import ExitStack, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import robotics_generation_length as screen


def test_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="generation-length-cpu-", dir=parent)


class FakePolicy:
    """Mirror LeRobot's queue/history API without invoking a neural model."""

    def __init__(self):
        self.config = SimpleNamespace(horizon=16, n_action_steps=8,
                                      n_obs_steps=2, down_dims=(512, 1024, 2048))
        self.diffusion = SimpleNamespace(config=copy.copy(self.config), num_inference_steps=100,
                                         conditional_sample=self.conditional_sample)
        self.resets = 0
        self.all_seen = []
        self.all_histories = []
        self.weight = screen.torch.nn.Parameter(screen.torch.ones(1))

    def parameters(self):
        return iter([self.weight])

    def conditional_sample(self):
        return screen.torch.zeros((1, self.diffusion.config.horizon, 2))

    def reset(self):
        self.resets += 1
        self.queries = 0
        self.seen = []
        self._queues = {"action": deque(maxlen=self.config.n_action_steps),
                        "observation.state": deque(maxlen=2)}

    def select_action(self, observation):
        step = observation["step"]
        self.seen.append(step)
        self.all_seen.append(step)
        self._queues["observation.state"].append(step)
        self.all_histories.append(list(self._queues["observation.state"]))
        if not self._queues["action"]:
            self.diffusion.conditional_sample()
            self.queries += 1
            for index in range(self.config.n_action_steps):
                target = 100. + 100. * self.queries + step + 3. * index
                self._queues["action"].append(screen.torch.tensor(
                    [[target, 200.]], dtype=screen.torch.float64))
        return self._queues["action"].popleft()


class ConfigurationTest(unittest.TestCase):
    def test_four_conditions_and_short_horizon_usable_future(self):
        self.assertEqual(screen.CONDITIONS, {
            "h8_e4": {"horizon": 8, "execution": 4},
            "h16_e4": {"horizon": 16, "execution": 4},
            "fixed8": {"horizon": 16, "execution": 8},
            "adaptive12": {"horizon": 16, "execution": 12}})
        # The first generated position is history aligned, not a future command.
        self.assertEqual(8 - FakePolicy().config.n_obs_steps + 1, 7)

    def test_configure_updates_both_configurations_and_queue_capacity(self):
        policy = FakePolicy()
        for index, method in enumerate(screen.CONDITIONS, 1):
            with self.subTest(method=method):
                screen.configure(policy, method)
                expected = screen.CONDITIONS[method]
                for config in (policy.config, policy.diffusion.config):
                    self.assertEqual(config.horizon, expected["horizon"])
                    self.assertEqual(config.n_action_steps, expected["execution"])
                    self.assertEqual(config.n_obs_steps, 2)
                self.assertEqual(policy.resets, index)
                self.assertEqual(policy._queues["action"].maxlen, expected["execution"])
                self.assertEqual(policy._queues["observation.state"].maxlen, 2)

    def test_unknown_method_is_rejected(self):
        policy = FakePolicy()
        with self.assertRaises((AssertionError, KeyError, ValueError)):
            screen.configure(policy, "rejected_weighted_gate")
        self.assertEqual(policy.resets, 0)

    def test_eight_generated_positions_cannot_supply_eight_future_commands(self):
        policy = FakePolicy()
        with patch.dict(screen.CONDITIONS, {"invalid": {"horizon": 8, "execution": 8}}):
            with self.assertRaises((AssertionError, ValueError)):
                screen.configure(policy, "invalid")

    def test_unsupported_temporal_length_is_rejected(self):
        policy = FakePolicy()
        for horizon in (4, 12, 15):
            with self.subTest(horizon=horizon), patch.dict(screen.CONDITIONS, {
                    "invalid": {"horizon": horizon, "execution": 4}}):
                with self.assertRaises((AssertionError, ValueError)):
                    screen.configure(policy, "invalid")

    def test_queue_capacity_must_follow_reset_configuration(self):
        class WrongCapacityPolicy(FakePolicy):
            def reset(self):
                super().reset()
                self._queues["action"] = deque(maxlen=99)

        with self.assertRaises(AssertionError):
            screen.configure(WrongCapacityPolicy(), "h8_e4")


class ScheduleAndScopeTest(unittest.TestCase):
    def test_exact_scene_draw_method_factorial_without_duplicates(self):
        schedule = screen.schedule()
        self.assertEqual(screen.SCENES, list(range(8)))
        self.assertEqual(screen.DRAWS, [10000, 20000])
        expected = {(scene, draw, method) for scene in screen.SCENES
                    for draw in screen.DRAWS for method in screen.CONDITIONS}
        self.assertEqual(len(schedule), 64)
        self.assertEqual({(r["scene"], r["draw"], r["method"]) for r in schedule}, expected)

    def test_method_order_is_balanced_and_deterministic(self):
        schedule = screen.schedule()
        methods = list(screen.CONDITIONS)
        for scene in screen.SCENES:
            for index, draw in enumerate(screen.DRAWS):
                rotation = (2 * scene + index) % 4
                expected = methods[rotation:] + methods[:rotation]
                actual = [r["method"] for r in schedule
                          if r["scene"] == scene and r["draw"] == draw]
                self.assertEqual(actual, expected)
        self.assertEqual(screen.schedule(), schedule)

    def test_prediction_seed_is_shared_by_method_and_differs_by_draw(self):
        for scene in screen.SCENES:
            self.assertEqual([screen.prediction_seed(scene, draw) for draw in screen.DRAWS],
                             [10000 + scene, 20000 + scene])
        self.assertEqual(len({screen.prediction_seed(scene, draw)
                              for scene in screen.SCENES for draw in screen.DRAWS}), 16)

    def test_output_guard_never_accepts_old_or_broad_evidence_directories(self):
        screen.check_output(screen.OUTPUT)
        screen.check_output(screen.OUTPUT.with_name(screen.OUTPUT.name + "-repeat"))
        for output in (Path("results"), Path("results/robotics-fresh-evaluation"),
                       Path("results/robotics-adaptive-length"),
                       screen.OUTPUT / "nested", Path("../outside")):
            with self.subTest(output=output), self.assertRaises((AssertionError, ValueError)):
                screen.check_output(output)


def state(step=0, coverage=.60):
    return {"coverage": coverage, "block_position": [100., 200.],
            "block_angle": .2, "agent_position": [step, 10.],
            "image_sha256": f"image-{step}"}


def write_trial(directory, trial, trace):
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / trial["trace"], "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, allow_nan=False)
    trial["trace_sha256"] = screen.base.sha(directory / trial["trace"])
    screen.base.save_json(directory / (screen.stem(trial["scene"], trial["draw"],
                                                 trial["method"]) + ".json"), trial)


def trial_fixture(directory, method="h8_e4", scene=0, draw=10000,
                  steps=16, success=True):
    """Generate a queue-consistent trace, not simulated experimental evidence."""
    config = screen.CONDITIONS[method]
    execution = config["execution"]
    rows = []
    for index in range(steps):
        age = index % execution + 1
        after = state(index + 1, .96 if success and index == steps - 1 else .60)
        row = {"step": index + 1, "action": [float(index), 100.],
               "new_prediction": age == 1, "plan_age": age,
               "pending_before": execution - age, "pending_after": execution - age,
               "after": after}
        if age == 1:
            row.update(predicted_commands=[[float(index + offset), 100.]
                                            for offset in range(execution)],
                       prediction_seconds=.125)
        rows.append(row)
    lengths = [min(execution, steps - index) for index in range(0, steps, execution)]
    trial = {"scene": scene, "draw": draw, "method": method,
             "prediction_seed": screen.prediction_seed(scene, draw),
             **config, "initial": state(), "initial_features": None, "steps": steps,
             "success": success, "truncated": not success,
             "final_coverage": rows[-1]["after"]["coverage"],
             "model_queries": len(lengths), "unit_lengths": lengths,
             "events": [], "monitor_seconds": 0., "wall_seconds": 10.,
             "prediction_seconds": [.125] * len(lengths),
             "peak_gpu_allocated_bytes": 4096,
             "generated_shapes": [[1, config["horizon"], 2]] * len(lengths),
             "trace": screen.stem(scene, draw, method) + ".trace.json.gz"}
    trace = {"scene": scene, "draw": draw, "method": method,
             "initial": state(), "steps": rows}
    write_trial(directory, trial, trace)
    return trial, trace


class TrialIntegrityTest(unittest.TestCase):
    def verify(self, trial, directory, **kwargs):
        return screen.verify_trial(trial, directory, max_steps=16, **kwargs)

    def test_fixed_four_and_fixed_eight_queue_boundaries_and_raw_shapes(self):
        for method, expected in (("h8_e4", [4, 4, 4, 4]),
                                 ("h16_e4", [4, 4, 4, 4]), ("fixed8", [8, 8])):
            with self.subTest(method=method), test_directory() as name:
                trial, trace = trial_fixture(Path(name), method)
                decoded = self.verify(trial, Path(name))
                self.assertEqual(decoded, trace)
                self.assertEqual(trial["unit_lengths"], expected)
                self.assertEqual(len(trial["generated_shapes"]), len(expected))

    def test_identity_seed_configuration_and_filename_tampering_are_rejected(self):
        changes = ({"scene": 8}, {"draw": 999}, {"prediction_seed": 123},
                   {"method": "weighted"}, {"horizon": 16}, {"execution": 8},
                   {"trace": "../another.trace.json.gz"})
        with test_directory() as name:
            directory = Path(name)
            trial, _ = trial_fixture(directory)
            for changeset in changes:
                with self.subTest(changes=changeset), self.assertRaises((AssertionError, KeyError)):
                    self.verify({**trial, **changeset}, directory)

    def test_hash_tampering_cannot_resume(self):
        with test_directory() as name:
            directory = Path(name)
            trial, _ = trial_fixture(directory)
            with self.assertRaises(AssertionError):
                self.verify({**trial, "trace_sha256": "0" * 64}, directory)

    def test_queue_queries_units_commands_and_shapes_are_audited(self):
        mutations = (
            lambda t, r: r["steps"][4].update(plan_age=4),
            lambda t, r: r["steps"][4].update(new_prediction=False),
            lambda t, r: r["steps"][0].update(pending_before=0),
            lambda t, r: r["steps"][0].update(pending_after=0),
            lambda t, r: r["steps"][0].update(step=2),
            lambda t, r: t.update(model_queries=99),
            lambda t, r: t.update(unit_lengths=[3, 5, 4, 4]),
            lambda t, r: r["steps"][0].update(predicted_commands=[]),
            lambda t, r: r["steps"][0]["predicted_commands"][1].__setitem__(0, 999.),
            lambda t, r: r["steps"][1].update(predicted_commands=[[1., 2.]]),
            lambda t, r: t.update(generated_shapes=[[1, 16, 2]] * 4),
            lambda t, r: t.update(generated_shapes=[[1, 8, 3]] * 4),
            lambda t, r: t.update(generated_shapes=[[1, 8, 2]] * 3),
            lambda t, r: r.update(scene=1),
            lambda t, r: r["initial"].update(block_angle=.3))
        for mutation in mutations:
            with test_directory() as name:
                directory = Path(name)
                trial, trace = trial_fixture(directory)
                mutation(trial, trace)
                write_trial(directory, trial, trace)
                with self.assertRaises(AssertionError):
                    self.verify(trial, directory)

    def test_terminal_coverage_threshold_and_full_timeout_are_exact(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace = trial_fixture(directory, success=False)
            trace["steps"][-1]["after"]["coverage"] = .95
            trial["final_coverage"] = .95
            write_trial(directory, trial, trace)
            self.verify(trial, directory)
            for changes in ({"success": True}, {"truncated": False}, {"steps": 15}):
                with self.subTest(changes=changes), self.assertRaises(AssertionError):
                    self.verify({**trial, **changes}, directory)
            short, _ = trial_fixture(directory, steps=12, success=False)
            with self.assertRaises(AssertionError):
                self.verify(short, directory)

    def test_first_successful_observation_must_terminate(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace = trial_fixture(directory)
            trace["steps"][5]["after"]["coverage"] = .951
            write_trial(directory, trial, trace)
            with self.assertRaises(AssertionError):
                self.verify(trial, directory)


class FakeEnvironment:
    def __init__(self, fail_step=None, truncate_at=None):
        self.fail_step = fail_step
        self.truncate_at = truncate_at
        self.closed = False

    def reset(self, seed):
        self.scene_seed, self.step_index, self.last_command = seed, 0, [0., 0.]
        return {"step": 0, "pixels": None}, {}

    def step(self, command):
        self.step_index += 1
        if self.step_index == self.fail_step:
            raise RuntimeError("synthetic movement error")
        self.last_command = np.asarray(command).tolist()
        return ({"step": self.step_index, "pixels": None}, 0.,
                self.step_index == 16, self.step_index == self.truncate_at, {})

    def close(self):
        self.closed = True


class FakeEstimator:
    def __init__(self, coverage):
        self.coverage = coverage
        self.calls = 0

    def estimate(self, pixels):
        self.calls += 1
        return {"quality": True, "coverage": self.coverage}


class FakeRolling:
    def update(self, estimate, pending):
        return {**estimate, "history_steps": 8, "warning_count": 0,
                "drawdown": 0., "recovery": 0., "recent_peak": estimate["coverage"],
                "pending_actions": pending}


def fake_physical_state(env, observation):
    return {**state(env.step_index, .96 if env.step_index == 16 else .60),
            "agent_position": env.last_command}


def integration_patches(policy, env, estimator):
    stack = ExitStack()
    stack.enter_context(patch.object(screen.gym, "make", return_value=env))
    stack.enter_context(patch.object(screen.adaptive, "observation_batch", side_effect=lambda obs, device: obs))
    stack.enter_context(patch.object(screen.adaptive, "physical_state", side_effect=fake_physical_state))
    stack.enter_context(patch.object(screen.adaptive, "VisualAlignmentEstimator", return_value=estimator))
    stack.enter_context(patch.object(screen.adaptive, "RollingVisualFeatures", return_value=FakeRolling()))
    stack.enter_context(patch.object(screen.torch.cuda, "synchronize"))
    stack.enter_context(patch.object(screen.torch.cuda, "reset_peak_memory_stats"))
    stack.enter_context(patch.object(screen.torch.cuda, "max_memory_allocated", return_value=4096))
    return stack


class EpisodeIntegrationTest(unittest.TestCase):
    def collect(self, method, coverage=.60):
        policy, env, estimator = FakePolicy(), FakeEnvironment(), FakeEstimator(coverage)
        with test_directory() as name:
            directory = Path(name)
            with integration_patches(policy, env, estimator):
                trial = screen.evaluate_episode(policy, 0, 10000, method, directory, max_steps=16)
            trace = screen.verify_trial(trial, directory, max_steps=16)
        self.assertTrue(env.closed)
        self.assertEqual(env.scene_seed, 0)
        self.assertEqual(policy.resets, 2)  # Start once, restore default only after the attempt.
        self.assertEqual(policy.all_seen, list(range(16)))
        self.assertEqual(policy.all_histories[-1], [14, 15])
        self.assertEqual(policy.config.horizon, 16)
        self.assertEqual(policy.config.n_action_steps, 8)
        self.assertEqual(policy._queues["action"].maxlen, 8)
        return trial, trace, estimator

    def test_four_step_generation_changes_raw_shape_not_feedback_interval(self):
        short, short_trace, short_monitor = self.collect("h8_e4")
        long, long_trace, long_monitor = self.collect("h16_e4")
        self.assertEqual(short["unit_lengths"], long["unit_lengths"])
        self.assertEqual(short["unit_lengths"], [4, 4, 4, 4])
        self.assertEqual(short["model_queries"], long["model_queries"])
        self.assertEqual(short["generated_shapes"], [[1, 8, 2]] * 4)
        self.assertEqual(long["generated_shapes"], [[1, 16, 2]] * 4)
        # Our double intentionally gives the same commands for shape changes;
        # this is a queue contract, not a claim about the real diffusion model.
        self.assertEqual([r["action"] for r in short_trace["steps"]],
                         [r["action"] for r in long_trace["steps"]])
        self.assertEqual(short_monitor.calls, 0)
        self.assertEqual(long_monitor.calls, 0)

    def test_all_h16_arms_share_first_four_without_resetting_observation_history(self):
        traces = [self.collect(method)[1] for method in ("h16_e4", "fixed8", "adaptive12")]
        expected = traces[0]["steps"][:4]
        for trace in traces[1:]:
            for left, right in zip(expected, trace["steps"][:4]):
                self.assertEqual(left["action"], right["action"])
                self.assertEqual(left["after"], right["after"])

    def test_milestone_retains_twelve_with_existing_extension_rule(self):
        trial, trace, estimator = self.collect("adaptive12", .60)
        self.assertEqual(trial["unit_lengths"], [12, 4])
        self.assertEqual(trial["events"][0]["kind"], "extend")
        self.assertEqual(trace["steps"][7]["pending_after"], 4)
        self.assertFalse(trace["steps"][8]["new_prediction"])
        self.assertEqual(estimator.calls, 16)  # Initial plus 15 non-terminal observations.
        self.assertGreater(trial["monitor_seconds"], 0.)

    def test_milestone_trim_matches_fixed_eight_commands_and_query_count(self):
        fixed, left, _ = self.collect("fixed8")
        trimmed, right, _ = self.collect("adaptive12", .80)
        self.assertEqual(fixed["unit_lengths"], trimmed["unit_lengths"])
        self.assertEqual(trimmed["unit_lengths"], [8, 8])
        self.assertEqual(trimmed["events"][0]["kind"], "trim")
        self.assertEqual(fixed["model_queries"], trimmed["model_queries"])
        self.assertEqual([r["action"] for r in left["steps"]],
                         [r["action"] for r in right["steps"]])

    def test_execution_error_closes_environment_and_restores_default_configuration(self):
        policy, env, estimator = FakePolicy(), FakeEnvironment(fail_step=3), FakeEstimator(.60)
        with test_directory() as name:
            directory = Path(name)
            with integration_patches(policy, env, estimator):
                with self.assertRaisesRegex(RuntimeError, "synthetic movement error"):
                    screen.evaluate_episode(policy, 0, 10000, "h8_e4", directory, max_steps=16)
            self.assertEqual(list(directory.iterdir()), [])
        self.assertTrue(env.closed)
        self.assertEqual(policy.resets, 2)
        self.assertEqual(policy.config.horizon, 16)
        self.assertEqual(policy.diffusion.config.horizon, 16)
        self.assertEqual(policy.config.n_action_steps, 8)


class CaptureGenerationTest(unittest.TestCase):
    def test_records_actual_raw_sample_shape_and_restores_method(self):
        policy = FakePolicy()
        screen.configure(policy, "h8_e4")
        original = policy.diffusion.conditional_sample
        with screen.capture_generation(policy) as shapes:
            value = policy.diffusion.conditional_sample()
        self.assertEqual(shapes, [[1, 8, 2]])
        self.assertEqual(tuple(value.shape), (1, 8, 2))
        self.assertIs(policy.diffusion.conditional_sample, original)

    def test_wrong_shape_or_nonfinite_sample_is_rejected_and_hook_restored(self):
        for value in (screen.torch.zeros((1, 16, 2)),
                      screen.torch.full((1, 8, 2), float("nan"))):
            policy = FakePolicy()
            screen.configure(policy, "h8_e4")
            original = lambda: value
            policy.diffusion.conditional_sample = original
            with self.assertRaises(AssertionError):
                with screen.capture_generation(policy):
                    policy.diffusion.conditional_sample()
            self.assertIs(policy.diffusion.conditional_sample, original)


class ProtocolAndResumeTest(unittest.TestCase):
    def fixture(self, root):
        artifact = root / "frozen-source.txt"
        artifact.write_text("unchanged", encoding="utf-8")
        stack = ExitStack()
        stack.enter_context(patch.object(screen, "check_output"))
        stack.enter_context(patch.object(screen, "fingerprints", side_effect=lambda: {
            artifact.as_posix(): screen.base.sha(artifact)}))
        stack.enter_context(patch.object(screen, "model_sources", return_value={"test.module": "source-hash"}))
        stack.enter_context(patch.object(screen.importlib.metadata, "version", return_value="test-version"))
        return stack, artifact

    def test_protocol_freezes_all_conditions_checkpoint_history_rules_and_sources(self):
        with test_directory() as name:
            root = Path(name)
            stack, artifact = self.fixture(root)
            with stack:
                output = root / "new"
                first = screen.prepare(output)
                encoded = (output / "protocol.json").read_bytes()
                self.assertEqual(first["new_attempts"], 64)
                self.assertEqual(first["conditions"], screen.CONDITIONS)
                self.assertEqual(first["rules"], screen.adaptive.RULES)
                self.assertEqual(first["n_obs_steps"], 2)
                self.assertEqual(first["denoising_steps"], 100)
                self.assertFalse(first["training"])
                self.assertEqual(first["milestone"], screen.MILESTONE)
                self.assertEqual(first["model_sources"], {"test.module": "source-hash"})
                self.assertEqual(first["checkpoint"]["weights_sha256"], screen.base.WEIGHTS_SHA256)
                self.assertIn(artifact.as_posix(), first["fingerprints"])
                self.assertEqual(screen.prepare(output), first)
                self.assertEqual((output / "protocol.json").read_bytes(), encoded)

    def test_mutated_sources_settings_or_protocol_cannot_resume(self):
        for field in ("source", "scenes", "conditions", "statistics", "training", "checkpoint", "rules", "packages", "model_sources"):
            with self.subTest(field=field), test_directory() as name:
                root = Path(name)
                stack, artifact = self.fixture(root)
                with stack:
                    output = root / "new"
                    document = screen.prepare(output)
                    if field == "source":
                        artifact.write_text("changed", encoding="utf-8")
                    else:
                        document[field] = "changed"
                        screen.base.save_json(output / "protocol.json", document)
                    with self.assertRaises(AssertionError):
                        screen.verify_protocol(output)

    def test_unbound_existing_results_are_never_adopted(self):
        for existing in ("episodes", "episodes.json", "summary.json", "smoke.json"):
            with test_directory() as name:
                output = Path(name)
                if existing == "episodes":
                    (output / existing).mkdir()
                else:
                    screen.base.save_json(output / existing, {})
                with patch.object(screen, "check_output"), patch.object(screen, "protocol_document") as make:
                    with self.assertRaises(AssertionError):
                        screen.prepare(output)
                make.assert_not_called()

    def test_valid_sidecar_reuses_trial_without_policy_inference(self):
        with test_directory() as name:
            directory = Path(name)
            trial, _ = trial_fixture(directory)
            with patch.object(screen, "evaluate_episode") as evaluate:
                result = screen.collect(object(), 0, 10000, "h8_e4", directory, max_steps=16)
            evaluate.assert_not_called()
            self.assertEqual(result, trial)

    def test_orphan_trace_or_partial_sidecar_never_gets_overwritten(self):
        for suffix in (".trace.json.gz", ".json.tmp"):
            with test_directory() as name:
                directory = Path(name)
                orphan = directory / (screen.stem(0, 10000, "h8_e4") + suffix)
                orphan.write_bytes(b"incomplete")
                with patch.object(screen, "evaluate_episode") as evaluate:
                    with self.assertRaisesRegex(AssertionError, "Inspect"):
                        screen.collect(object(), 0, 10000, "h8_e4", directory, max_steps=16)
                evaluate.assert_not_called()
                self.assertEqual(orphan.read_bytes(), b"incomplete")

    def test_changed_saved_trial_never_triggers_recollection(self):
        with test_directory() as name:
            directory = Path(name)
            trial, _ = trial_fixture(directory)
            trial["horizon"] = 16
            screen.base.save_json(directory / (screen.stem(0, 10000, "h8_e4") + ".json"), trial)
            with patch.object(screen, "evaluate_episode") as evaluate:
                with self.assertRaises(AssertionError):
                    screen.collect(object(), 0, 10000, "h8_e4", directory, max_steps=16)
            evaluate.assert_not_called()

class PrefixAndSmokeTest(unittest.TestCase):
    def fixture(self, root):
        directory = root / "smoke-episodes"
        records = []
        for method in screen.CONDITIONS:
            policy, env = FakePolicy(), FakeEnvironment(truncate_at=8)
            with integration_patches(policy, env, FakeEstimator(.60)):
                records.append(screen.evaluate_episode(policy, 0, 1234, method, directory, max_steps=8))
        reference = root / "old-prefix"
        fixed = next(row for row in records if row["method"] == "fixed8")
        old_steps = screen.read_trace(fixed, directory)["steps"]
        screen.base.save_json(reference / "prefix-smoke.json", {"traces": {"fixed8": old_steps}})
        saved = {"passed": True, "h16_first_four_exact": True,
                 "baseline_milestone_prefix_matches_saved_smoke": True,
                 "parameter_versions_unchanged": True,
                 "warmups_per_condition": 2, "measured_calls_per_condition": 3,
                 "model_weights_sha256": screen.base.WEIGHTS_SHA256,
                 "prefix_trials": records,
                 "prediction_seconds": {method: [.1, .2, .3] for method in screen.CONDITIONS},
                 "generated_shapes": {method: [[1, spec["horizon"], 2]] * 3
                                      for method, spec in screen.CONDITIONS.items()},
                 "peak_gpu_allocated_bytes": {method: [4096] * 3 for method in screen.CONDITIONS}}
        return saved, reference

    def test_h16_prefix_is_exact_but_h8_prediction_need_not_match(self):
        with test_directory() as name:
            root = Path(name)
            saved, _ = self.fixture(root)
            records = saved["prefix_trials"]
            short = next(row for row in records if row["method"] == "h8_e4")
            trace = screen.read_trace(short, root / "smoke-episodes")
            trace["steps"][0]["action"] = [-1., -1.]
            write_trial(root / "smoke-episodes", short, trace)
            screen.verify_prefixes(records, root / "smoke-episodes")

    def test_h16_prefix_rejects_changed_commands_physics_or_rgb(self):
        changes = (lambda trace: trace["steps"][3]["action"].__setitem__(0, 999.),
                   lambda trace: trace["steps"][2]["after"].update(block_angle=.21),
                   lambda trace: trace["steps"][1]["after"].update(image_sha256="different"))
        for change in changes:
            with test_directory() as name:
                root = Path(name)
                saved, _ = self.fixture(root)
                records = saved["prefix_trials"]
                trial = next(row for row in records if row["method"] == "h16_e4")
                trace = screen.read_trace(trial, root / "smoke-episodes")
                change(trace)
                write_trial(root / "smoke-episodes", trial, trace)
                with self.assertRaises(AssertionError):
                    screen.verify_prefixes(records, root / "smoke-episodes")

    def test_smoke_validates_all_four_saved_trials_and_actual_timing_shapes(self):
        with test_directory() as name:
            root = Path(name)
            saved, reference = self.fixture(root)
            with patch.object(screen.adaptive, "OUTPUT", reference):
                self.assertEqual(screen.verify_smoke(saved, root), saved)

    def test_missing_repeated_timing_wrong_shape_or_mutated_identity_cannot_pass_smoke(self):
        mutations = (
            lambda saved: saved.update(prefix_trials=saved["prefix_trials"][:-1]),
            lambda saved: saved["prediction_seconds"]["h8_e4"].pop(),
            lambda saved: saved["prediction_seconds"]["h8_e4"].__setitem__(0, float("nan")),
            lambda saved: saved["generated_shapes"].update(h8_e4=[[1, 16, 2]] * 3),
            lambda saved: saved["peak_gpu_allocated_bytes"]["h8_e4"].__setitem__(0, 0),
            lambda saved: saved.update(parameter_versions_unchanged=False),
            lambda saved: saved.update(model_weights_sha256="changed"),
            lambda saved: saved.update(warmups_per_condition=0))
        with test_directory() as name:
            root = Path(name)
            original, reference = self.fixture(root)
            with patch.object(screen.adaptive, "OUTPUT", reference):
                for mutation in mutations:
                    changed = copy.deepcopy(original)
                    mutation(changed)
                    with self.assertRaises(AssertionError):
                        screen.verify_smoke(changed, root)

    def test_completed_smoke_is_verified_without_loading_model_or_touching_gpu(self):
        with test_directory() as name:
            root = Path(name)
            saved, reference = self.fixture(root)
            screen.base.save_json(root / "smoke.json", saved)
            with (patch.object(screen.adaptive, "OUTPUT", reference),
                  patch.object(screen, "verify_protocol") as verify,
                  patch.object(screen.adaptive, "load_policy") as load,
                  patch.object(screen.torch.cuda, "is_available") as available):
                self.assertEqual(screen.smoke(root), saved)
            verify.assert_called_once_with(root)
            load.assert_not_called()
            available.assert_not_called()

    def test_new_smoke_cannot_silently_fall_back_to_cpu(self):
        with test_directory() as name:
            with (patch.object(screen, "verify_protocol"),
                  patch.object(screen.torch.cuda, "is_available", return_value=False),
                  patch.object(screen.adaptive, "load_policy") as load):
                with self.assertRaisesRegex(AssertionError, "no silent CPU fallback"):
                    screen.smoke(Path(name))
            load.assert_not_called()


class TrialCostIntegrityTest(unittest.TestCase):
    def verify(self, trial, directory):
        return screen.verify_trial(trial, directory, max_steps=16)

    def test_baselines_cannot_hide_monitoring_or_controller_events(self):
        with test_directory() as name:
            directory = Path(name)
            trial, trace = trial_fixture(directory)
            for changes in ({"events": [{"kind": "extend"}]}, {"monitor_seconds": .1}):
                with self.subTest(changes=changes), self.assertRaises(AssertionError):
                    self.verify({**trial, **changes}, directory)
            trace["steps"][0]["features"] = {"coverage": .60}
            write_trial(directory, trial, trace)
            with self.assertRaises(AssertionError):
                self.verify(trial, directory)

    def test_timing_list_matches_fresh_calls_and_never_accepts_nonfinite_cost(self):
        changes = ({"prediction_seconds": [.125]}, {"prediction_seconds": [-1.] * 4},
                   {"prediction_seconds": [float("nan")] * 4},
                   {"wall_seconds": float("inf")}, {"wall_seconds": -.1},
                   {"monitor_seconds": float("nan")},
                   {"peak_gpu_allocated_bytes": -1})
        with test_directory() as name:
            directory = Path(name)
            trial, trace = trial_fixture(directory)
            for changeset in changes:
                with self.subTest(changes=changeset), self.assertRaises(AssertionError):
                    self.verify({**trial, **changeset}, directory)
            trace["steps"][0]["prediction_seconds"] = .25
            write_trial(directory, trial, trace)
            with self.assertRaises(AssertionError):
                self.verify(trial, directory)


class StatisticsAndRunTest(unittest.TestCase):
    def rows(self):
        rows = []
        for scene in screen.SCENES:
            for draw in screen.DRAWS:
                for method in screen.CONDITIONS:
                    offset = int(method == "h8_e4")
                    rows.append({"scene": scene, "draw": draw, "method": method,
                                 "success": scene >= 2 if offset == 0 else scene != 2,
                                 "final_coverage": .90 + .01 * offset,
                                 "model_queries": 20 - 4 * offset,
                                 "wall_seconds": 100. - 10. * offset,
                                 "monitor_seconds": 1. if method == "adaptive12" else 0.,
                                 "prediction_seconds": [2. - .5 * offset],
                                 "peak_gpu_allocated_bytes": 4096 - 256 * offset})
        return rows

    def test_pairing_reports_exact_scene_draw_gains_and_losses(self):
        rows = self.rows()
        result = screen.paired(rows, "h16_e4", "h8_e4")
        self.assertEqual(result["gained_successes"], [[0, 10000], [0, 20000],
                                                     [1, 10000], [1, 20000]])
        self.assertEqual(result["lost_successes"], [[2, 10000], [2, 20000]])
        differences = result["differences_method_minus_reference"]
        self.assertAlmostEqual(differences["success"]["mean_difference"], 1 / 8)
        self.assertEqual(differences["model_queries"]["scene_cluster_bootstrap_95_interval"], [-4., -4.])
        self.assertEqual(differences["wall_seconds"]["scene_cluster_bootstrap_95_interval"], [-10., -10.])
        self.assertEqual(screen.paired(list(reversed(rows)), "h16_e4", "h8_e4"), result)

    def test_bootstrap_averages_prediction_samples_within_scene_first(self):
        rows = self.rows()
        left = {(r["scene"], r["draw"]): r for r in rows if r["method"] == "h16_e4"}
        for row in rows:
            if row["method"] == "h8_e4":
                difference = 2. * row["scene"] + (4. if row["draw"] == 10000 else -4.)
                row["model_queries"] = left[row["scene"], row["draw"]]["model_queries"] + difference
        result = screen.paired(rows, "h16_e4", "h8_e4")
        values = np.asarray([2. * scene for scene in screen.SCENES])
        rng = np.random.default_rng(screen.STATISTICS["bootstrap_seed"])
        indices = rng.integers(0, len(screen.SCENES),
                              size=(screen.STATISTICS["bootstrap_samples"], len(screen.SCENES)))
        expected = np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
        actual = result["differences_method_minus_reference"]["model_queries"]
        self.assertEqual(actual["mean_difference"], 7.)
        self.assertEqual(actual["scene_cluster_bootstrap_95_interval"], expected)

    def test_missing_or_duplicate_paired_keys_and_incomplete_tables_are_rejected(self):
        rows = self.rows()
        for invalid in (rows[1:], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.subTest(length=len(invalid)), self.assertRaises((AssertionError, KeyError)):
                screen.paired(invalid, "h16_e4", "h8_e4")
        for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.subTest(length=len(invalid)), self.assertRaises(AssertionError):
                screen.metric_table(invalid)

    def test_four_method_table_keeps_per_draw_success_and_actual_prediction_time(self):
        result = screen.metric_table(self.rows())
        self.assertEqual(set(result), set(screen.CONDITIONS))
        self.assertTrue(all(r["attempts"] == 16 for r in result.values()))
        self.assertEqual(result["h8_e4"]["successes"], 14)
        self.assertEqual(result["h8_e4"]["per_draw_successes"], {"10000": 7, "20000": 7})
        self.assertEqual(result["h8_e4"]["mean_prediction_seconds"], 1.5)
        self.assertEqual(result["adaptive12"]["mean_monitor_seconds"], 1.)
        self.assertEqual(result["h16_e4"]["mean_prediction_seconds"], 2.)

    def test_run_uses_frozen_balanced_schedule_and_loads_model_only_once(self):
        with test_directory() as name:
            output = Path(name)
            lookup = {(r["scene"], r["draw"], r["method"]): r for r in self.rows()}

            def collect(policy, scene, draw, method, directory):
                self.assertEqual(directory, output / "episodes")
                return lookup[scene, draw, method]

            with (patch.object(screen, "verify_protocol"), patch.object(screen, "smoke") as smoke,
                  patch.object(screen.adaptive, "load_policy", return_value=(object(), {})) as load,
                  patch.object(screen, "collect", side_effect=collect) as collect_mock,
                  patch.object(screen, "summarize", return_value={"complete": True}) as summarize,
                  redirect_stdout(io.StringIO())):
                result = screen.run(output)
            expected = [(r["scene"], r["draw"], r["method"]) for r in screen.schedule()]
            actual = [(c.args[1], c.args[2], c.args[3]) for c in collect_mock.call_args_list]
            self.assertEqual(actual, expected)
            load.assert_called_once()
            smoke.assert_called_once_with(output)
            summarize.assert_called_once_with(output)
            self.assertTrue(result["complete"])
            saved = screen.base.read_json(output / "episodes.json")
            self.assertTrue(saved["complete"])
            self.assertEqual(len(saved["episodes"]), 64)

    def test_completed_run_rechecks_results_without_smoke_model_or_gpu(self):
        with test_directory() as name:
            output = Path(name)
            screen.base.save_json(output / "summary.json", {"complete": True})
            with (patch.object(screen, "verify_protocol") as verify,
                  patch.object(screen, "smoke") as smoke,
                  patch.object(screen.adaptive, "load_policy") as load,
                  patch.object(screen, "collect") as collect,
                  patch.object(screen, "summarize", return_value={"complete": True}) as summarize):
                result = screen.run(output)
            verify.assert_called_once_with(output)
            summarize.assert_called_once_with(output)
            smoke.assert_not_called()
            load.assert_not_called()
            collect.assert_not_called()
            self.assertTrue(result["complete"])


if __name__ == "__main__":
    unittest.main()
