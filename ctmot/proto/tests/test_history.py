"""Causality, assignment selection, and regression tests independent of nuScenes."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ctmot.proto.history import build_snapshot, estimate_acceleration, predict_shadow, baseline_from_same_anchor
from ctmot.proto.validate_history import iter_json_array, load_config, select_evaluation_candidate, verify_captured_outputs


def fixture():
    config = {"motion_model": {"model": {"2": "CTRV"}}, "geometry_model": {"use": {"2": False}}}
    records = []
    for frame, timestamp in enumerate([0, 200000, 550000, 1000000, 1550000]):
        t = timestamp / 1e6
        det = [100., 200., 3., 2., 4., 1., 2 + 3*t, -1 + .5*t, 1., 0., 0., 0., .8, 2.]
        other = det.copy(); other[6:8] = [999., 999.]
        records.append({"frame_idx": frame, "is_keyframe": True, "timestamp": timestamp, "token": str(frame),
            "before": {} if frame == 0 else {"7": {}},
            "after": {"7": {"state": [100., 200., 3., 2., 4., 1., 10., np.pi/2, .1],
                              "updated": {"tracking_name": "car"}}},
            "stages": [{"stage": "mix", "track_ids": [7, 8], "detections": [other, det],
                        "assignment": [[1], [0], [], []], "thresholds": {"2": 1.1}}]})
    return {"config": config, "trace": records}


class HistoryTests(unittest.TestCase):
    def test_baseline_rejects_intermediate_correction_and_missing_steps(self):
        import pre_processing
        from motion_module.motion_model import CTRV
        model = CTRV(True, False, False, .25)
        state0 = [0., 0., 1., 2., 4., 1., 10., .4, .05]
        state = np.mat(state0).T
        records = [{"frame_idx": 19, "is_keyframe": True, "internal_frame_id": 39}]
        for index, key in [(40, False), (41, True)]:
            before = np.asarray(state).ravel().tolist()
            state = model.stateTransition(state); model.warpStateYawToPi(state)
            box = model.getOutputInfo(state).tolist() + [.8, 2., 7., 0., index]
            records.append({"frame_idx": 20, "is_keyframe": key, "internal_frame_id": index,
                            "before": {"7": {"state": before}},
                            "predicted": {"all_valid_ids": [7], "np_tras": [box]}})
        trace = {"trace": records, "config": {"basic": {"LiDAR_interval": .25}}}
        snapshot = {"anchor_frame": 19, "track_id": 7, "posterior_ctrv": state0}
        baseline = baseline_from_same_anchor(trace, snapshot, 20)
        self.assertTrue(baseline["same_anchor_verified"])
        self.assertEqual(baseline["nominal_elapsed_s"], .5)
        changed = copy.deepcopy(trace); changed["trace"][2]["before"]["7"]["state"][0] += .1
        with self.assertRaisesRegex(ValueError, "intervening correction"):
            baseline_from_same_anchor(changed, snapshot, 20)
        changed = copy.deepcopy(trace); del changed["trace"][1]
        with self.assertRaisesRegex(ValueError, "missing intermediate"):
            baseline_from_same_anchor(changed, snapshot, 20)

    def test_saved_outputs_have_per_box_sample_token(self):
        trace = fixture()
        trace["trace"][3]["output"] = [{"tracking_id": "7", "translation": [1, 2, 3]}]
        original = {"3": [{"tracking_id": "7", "translation": [1, 2, 3], "sample_token": "3"}]}
        verify_captured_outputs(trace, original, [3])
        original["3"][0]["translation"][0] = 999
        with self.assertRaises(ValueError): verify_captured_outputs(trace, original, [3])

    def test_posterior_velocity_and_actual_assignments(self):
        snapshot = build_snapshot(fixture(), 3, 7, 4)
        np.testing.assert_allclose(snapshot["state_pv"], [100, 200, 3, 0, 10, 0], atol=1e-12)
        self.assertEqual([x["frame_idx"] for x in snapshot["history"]], [1, 2, 3])
        self.assertEqual([x["detection_index"] for x in snapshot["history"]], [1, 1, 1])

    def test_acceleration_from_irregular_observation_times(self):
        snapshot = build_snapshot(fixture(), 3, 7, 4)
        estimate = estimate_acceleration(snapshot["history"], snapshot["timestamp_us"], 3)
        np.testing.assert_allclose(estimate["acceleration_mps2"], [3, .5, 0], atol=1e-12)

    def test_future_gt_observations_and_posteriors_do_not_affect_predictions(self):
        trace = fixture(); changed = copy.deepcopy(trace)
        changed["trace"][4]["after"] = {"7": {"state": [float("nan")]*9}}
        changed["trace"][4]["stages"] = []
        changed["trace"][4]["gt"] = {"translation": [-1e9]*3}
        query = [{"frame_idx": 4, "timestamp_us": 1550000, "sample_token": "4"}]
        a = predict_shadow(build_snapshot(trace, 3, 7, 4), query, {}, 3)
        b = predict_shadow(build_snapshot(changed, 3, 7, 4), query, {}, 3)
        self.assertEqual(a, b)
        self.assertEqual(a["snapshot"]["state_pv"], b["snapshot"]["state_pv"])

    def test_cv_ca_share_same_posterior_and_hold_z(self):
        snapshot = build_snapshot(fixture(), 3, 7, 4)
        original = copy.deepcopy(snapshot)
        result = predict_shadow(snapshot, [{"timestamp_us": 1550000}], {}, 3)
        cv = result["queries"][0]["models"]["ct_cv"]["state"]
        ca = result["queries"][0]["models"]["ct_ca"]["state"]
        np.testing.assert_allclose(cv, [100, 205.5, 3, 0, 10, 0], atol=1e-10)
        np.testing.assert_allclose(np.array(ca[:3])-cv[:3], .5*np.array([3, .5, 0])*.55**2, atol=1e-10)
        self.assertEqual(snapshot, original)

    def test_insufficient_history_skips_ca_without_zero_acceleration_fallback(self):
        snapshot = build_snapshot(fixture(), 2, 7, 4)
        result = predict_shadow(snapshot, [{"timestamp_us": 1000000}], {}, 3)
        self.assertEqual(result["acceleration_estimate"]["status"], "insufficient_history")
        self.assertEqual(set(result["queries"][0]["models"]), {"ct_cv"})

    def test_duplicate_and_future_history_are_rejected(self):
        snapshot = build_snapshot(fixture(), 3, 7, 4)
        history = snapshot["history"]
        with self.assertRaises(ValueError): estimate_acceleration(history, 999999, 3)
        history[1]["timestamp_us"] = history[0]["timestamp_us"]
        with self.assertRaises(ValueError): estimate_acceleration(history, 1000000, 3)

    def test_unmatched_anchor_is_rejected(self):
        trace = fixture(); trace["trace"][3]["after"]["7"]["updated"] = None
        with self.assertRaises(ValueError): build_snapshot(trace, 3, 7, 4)

    def test_gt_candidate_absence_is_not_reported_as_gate_success(self):
        step = fixture()["trace"][3]
        self.assertIsNone(select_evaluation_candidate(step, [0, 0, 0], 2.))
        chosen = select_evaluation_candidate(step, [100, 200, 3], 2.)
        self.assertEqual(chosen["gt_xy_distance_m"], 0.)

    def test_streaming_annotations_across_chunk_boundaries(self):
        values = [{"text": "x" * 1100000}, {"text": "한국어", "n": 1}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.json"
            path.write_text(json.dumps(values, ensure_ascii=False))
            self.assertEqual(list(iter_json_array(path)), values)

    def test_default_config_and_unsupported_modes(self):
        import yaml
        path = Path(__file__).resolve().parents[1] / "config/validation_scene1072.yaml"
        config = load_config(path)
        self.assertEqual(config["case"]["anchor_frame"], 19)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            for section, key, value in [("prediction", "vertical_policy", "estimate"),
                                        ("comparison", "run_hungarian", True),
                                        ("case", "query_kind", "mid"),
                                        ("acceleration", "minimum_observations", 1)]:
                changed = copy.deepcopy(config); changed[section][key] = value
                target.write_text(yaml.safe_dump(changed))
                with self.subTest(key=key), self.assertRaises(ValueError): load_config(target)


if __name__ == "__main__":
    unittest.main()
