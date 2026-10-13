"""Toy verification against independently evaluated analytic trajectories."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
import yaml

from ctmot.proto import ConstantAcceleration, ConstantVelocity, ODEPredictor
from ctmot.proto import predictor_from_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "ctmot/proto/config"


class PredictionTests(unittest.TestCase):
    t0 = 1542799711898267

    def test_cv_irregular_times_against_analytic_solution(self):
        initial = np.array([10., -3., 2., 2., -1., .5])
        predictor = ODEPredictor(ConstantVelocity())
        for offset in [1, 3001, 250000, 549659, 2500000, 10000000]:
            with self.subTest(offset_us=offset):
                result = predictor.predict(initial, self.t0, self.t0 + offset)
                expected = np.r_[initial[:3] + initial[3:] * offset / 1e6, initial[3:]]
                np.testing.assert_allclose(result.state, expected, rtol=0, atol=1e-9)

    def test_ca_irregular_times_against_analytic_solution(self):
        initial = np.array([10., -3., 2., 2., -1., .5])
        acceleration = np.array([1., .4, -.2])
        predictor = ODEPredictor(ConstantAcceleration(acceleration))
        for offset in [1, 3001, 250000, 549659, 2500000, 10000000]:
            with self.subTest(offset_us=offset):
                dt = offset / 1e6
                expected = np.r_[initial[:3] + initial[3:] * dt + .5 * acceleration * dt**2,
                                 initial[3:] + acceleration * dt]
                result = predictor.predict(initial, self.t0, self.t0 + offset)
                np.testing.assert_allclose(result.state, expected, rtol=0, atol=1e-9)

    def test_stationary_object(self):
        initial = [1., 2., 3., 0., 0., 0.]
        result = ODEPredictor(ConstantVelocity()).predict(initial, 0, 2000000)
        np.testing.assert_array_equal(result.state, initial)

    def test_zero_acceleration_equals_cv(self):
        initial = [1., 2., 3., -3., 4., .7]
        cv = ODEPredictor(ConstantVelocity()).predict(initial, 0, 549659)
        ca = ODEPredictor(ConstantAcceleration([0, 0, 0])).predict(initial, 0, 549659)
        np.testing.assert_allclose(cv.state, ca.state, rtol=0, atol=1e-12)

    def test_numeric_solver_is_used_for_both_models(self):
        with patch("ctmot.proto.predictor.solve_ivp", wraps=solve_ivp) as solver:
            for model in [ConstantVelocity(), ConstantAcceleration([1, 0, 0])]:
                result = ODEPredictor(model).predict([0, 0, 0, 1, 0, 0], 0, 2500000)
                self.assertGreater(result.solver_evaluations, 0)
            self.assertEqual(solver.call_count, 2)

    def test_same_timestamp_skips_solver(self):
        initial = [1., 2., 3., 4., 5., 6.]
        with patch("ctmot.proto.predictor.solve_ivp") as solver:
            result = ODEPredictor(ConstantVelocity()).predict(initial, self.t0, self.t0)
            solver.assert_not_called()
        np.testing.assert_array_equal(result.state, initial)
        self.assertEqual(result.solver_evaluations, 0)

    def test_microsecond_resolution_at_large_epoch(self):
        # Above float64's exact-integer range: subtracting floats would lose 1 us.
        t0 = 10**16
        result = ODEPredictor(ConstantVelocity()).predict([0, 0, 0, 1, 0, 0], t0, t0 + 1)
        self.assertEqual(result.delta_t_s, 1e-6)
        self.assertAlmostEqual(result.state[0], 1e-6, places=15)

    def test_query_order_does_not_mutate_input_or_predictor(self):
        initial = np.array([1., 2., 3., 4., 5., 6.]);original = initial.copy()
        acceleration = np.array([.1, -.2, .3])
        model = ConstantAcceleration(acceleration)
        acceleration[:] = 999  # Model owns an immutable copy of its parameters.
        predictor = ODEPredictor(model)
        first = predictor.predict(initial, 0, 750000)
        predictor.predict(initial, 0, 100000)
        again = predictor.predict(initial, 0, 750000)
        np.testing.assert_array_equal(initial, original)
        self.assertEqual(first.state, again.state)
        self.assertEqual(model.acceleration_mps2, (.1, -.2, .3))

    def test_split_integration_equals_direct_query(self):
        initial = [1., 2., 3., 4., 5., 6.]
        for model in [ConstantVelocity(), ConstantAcceleration([.1, -.2, .3])]:
            predictor = ODEPredictor(model)
            mid = predictor.predict(initial, 0, 250001)
            end = predictor.predict(mid.state, 250001, 799660)
            direct = predictor.predict(initial, 0, 799660)
            np.testing.assert_allclose(end.state, direct.state, rtol=0, atol=1e-10)

    def test_velocity_vector_and_speed_are_distinct(self):
        result = ODEPredictor(ConstantVelocity()).predict([0, 0, 0, -3, 4, 12], 0, 1).to_dict()
        self.assertEqual(result["velocity_mps"], [-3., 4., 12.])
        self.assertEqual(result["speed_xy_mps"], 5.)
        self.assertEqual(result["speed_3d_mps"], 13.)

    def test_invalid_state_and_times_are_rejected(self):
        predictor = ODEPredictor(ConstantVelocity())
        for state in [[0]*5, [[0]*6], [0, 0, 0, np.nan, 0, 0], [0, 0, 0, np.inf, 0, 0]]:
            with self.subTest(state=state), self.assertRaises(ValueError):
                predictor.predict(state, 0, 1)
        with self.assertRaises(ValueError):
            predictor.predict([0]*6, 10, 9)
        for timestamp in [1.5, True, "1000", np.bool_(True)]:
            for t0, t in [(timestamp, 100), (0, timestamp)]:
                with self.subTest(t0=t0, t=t), self.assertRaises(TypeError):
                    predictor.predict([0]*6, t0, t)

    def test_invalid_parameters_are_rejected(self):
        for acceleration in [[1, 2], [1, 2, np.inf]]:
            with self.assertRaises(ValueError):
                ConstantAcceleration(acceleration)
        for key in ["rtol", "atol", "max_step_s"]:
            for value in [0, -1, np.inf, np.nan, True]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ODEPredictor(ConstantVelocity(), **{key: value})

    def test_solver_failure_is_not_reported_as_success(self):
        failure = SimpleNamespace(success=False, message="toy failure")
        with patch("ctmot.proto.predictor.solve_ivp", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "toy failure"):
                ODEPredictor(ConstantVelocity()).predict([0]*6, 0, 1000)
        invalid = SimpleNamespace(success=True, y=np.full((6, 1), np.nan))
        with patch("ctmot.proto.predictor.solve_ivp", return_value=invalid):
            with self.assertRaisesRegex(RuntimeError, "invalid state"):
                ODEPredictor(ConstantVelocity()).predict([0]*6, 0, 1000)

    def test_unknown_or_unused_config_rejected(self):
        configs = [None, {}, {"dynamics": {"model": "typo"}},
                   {"dynamics": {"model": "ca"}},
                   {"dynamics": {"model": "cv", "acceleration_mps2": [1, 0, 0]}},
                   {"dynamics": {"model": "cv"}, "solver": {"rtoll": 1e-9}}]
        for config in configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                predictor_from_config(config)

    def test_yaml_configs_and_cli_match_hand_calculated_examples(self):
        for name, expected in [("cv", [5, -2.5, 1.25, 2, -1, .5]),
                               ("ca", [8.125, -1.25, .625, 4.5, 0, 0])]:
            with self.subTest(model=name):
                path = CONFIG / ("prediction_" + name + ".yaml")
                with path.open() as stream:
                    predictor = predictor_from_config(yaml.safe_load(stream))
                result = predictor.predict([0, 0, 0, 2, -1, .5], 1000000, 3500000)
                np.testing.assert_allclose(result.state, expected, rtol=0, atol=1e-10)
                output = subprocess.check_output([
                    sys.executable, "-m", "ctmot.proto.run_prediction", "--config", str(path),
                    "--state", "0", "0", "0", "2", "-1", "0.5",
                    "--state-time-us", "1000000", "--query-time-us", "3500000",
                ], cwd=str(ROOT), universal_newlines=True)
                np.testing.assert_allclose(json.loads(output)["state"], expected, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
