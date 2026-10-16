"""Causal history extraction and shadow prediction. This module never reads GT."""

import math

import numpy as np

from .dynamics import ConstantAcceleration, ConstantVelocity
from .predictor import ODEPredictor


def key_step(trace, frame):
    matches = [s for s in trace["trace"] if s["is_keyframe"] and s["frame_idx"] == frame]
    if len(matches) != 1:
        raise ValueError("Expected exactly one captured keyframe {}".format(frame))
    return matches[0]


def ctrv_state_to_pv(state):
    state = np.asarray(state, dtype=float)
    if state.shape != (9,) or not np.all(np.isfinite(state)):
        raise ValueError("Expected finite CTRV state [xyz,wlh,speed,yaw,yaw_rate]")
    speed, yaw = state[6:8]
    return np.r_[state[:3], speed * math.cos(yaw), speed * math.sin(yaw), 0.]


def build_snapshot(trace, anchor_frame, track_id, history_count):
    """Use posterior and actual 3D assignment records through the anchor only.

    Birth initialization is excluded: it has no Hungarian assignment to recover.
    Mid predictions and unassigned keyframe predictions are not observations.
    """
    anchor = key_step(trace, anchor_frame)
    tid = str(track_id)
    posterior = anchor["after"].get(tid)
    if posterior is None or posterior["updated"] is None:
        raise ValueError("Anchor must have a corrected posterior for the requested track")
    label = posterior["updated"]["tracking_name"]
    if label != "car" or trace["config"]["motion_model"]["model"]["2"] != "CTRV":
        raise ValueError("This adapter supports car with CTRV only")
    if trace["config"]["geometry_model"]["use"]["2"]:
        raise ValueError("CTRV geometry-filter state is not supported by this adapter")
    t0 = int(anchor["timestamp"])
    history = []
    for step in trace["trace"]:
        if not step["is_keyframe"] or step["frame_idx"] > anchor_frame:
            continue
        if int(step["timestamp"]) > t0:
            raise ValueError("Historical frame has a timestamp after the anchor")
        if tid not in step["before"]:  # No guessed detection mapping at birth.
            continue
        matches = []
        for stage in step["stages"]:
            if track_id not in stage["track_ids"]:
                continue
            col = stage["track_ids"].index(track_id)
            for det_index, track_index in zip(*stage["assignment"][:2]):
                if track_index == col:
                    matches.append((stage, det_index))
        if len(matches) > 1:
            raise ValueError("Multiple 3D assignments for one track at one keyframe")
        if not matches:
            continue
        stage, det_index = matches[0]
        det = np.asarray(stage["detections"][det_index], dtype=float)
        if det.shape != (14,) or int(det[13]) != 2 or not np.all(np.isfinite(det)):
            raise ValueError("Invalid accepted car detection")
        history.append({
            "frame_idx": step["frame_idx"], "timestamp_us": int(step["timestamp"]),
            "sample_token": step["token"], "track_id": track_id,
            "stage": stage["stage"], "detection_index": det_index,
            "position_m": det[:3].tolist(), "velocity_xy_mps": det[6:8].tolist(),
            "detection_score": float(det[12]),
        })
    history.sort(key=lambda x: x["timestamp_us"])
    if any(b["timestamp_us"] <= a["timestamp_us"] for a, b in zip(history, history[1:])):
        raise ValueError("Accepted observations must have distinct increasing timestamps")
    return {
        "anchor_frame": anchor_frame, "track_id": track_id, "timestamp_us": t0,
        "sample_token": anchor["token"], "posterior_ctrv": list(posterior["state"]),
        "state_pv": ctrv_state_to_pv(posterior["state"]).tolist(),
        "velocity_source": "internal CTRV speed*cos(yaw), speed*sin(yaw); vz=0",
        "vertical_policy": "hold", "history": history[-history_count:],
        "history_note": "Actual 3D matches only; birth and unmatched/mid states excluded.",
    }


def estimate_acceleration(history, anchor_timestamp_us, minimum_observations):
    if any(x["timestamp_us"] > anchor_timestamp_us for x in history):
        raise ValueError("Future observations are forbidden in acceleration estimation")
    times = np.array([(x["timestamp_us"] - anchor_timestamp_us) / 1e6 for x in history])
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Acceleration history timestamps must be strictly increasing")
    if len(history) < minimum_observations:
        return {"status": "insufficient_history", "observations": len(history),
                "required": minimum_observations, "acceleration_mps2": None}
    velocity = np.array([x["velocity_xy_mps"] for x in history], dtype=float)
    if not np.all(np.isfinite(velocity)):
        raise ValueError("Non-finite historical detection velocity")
    design = np.column_stack((np.ones(len(times)), times))
    fit, _, rank, _ = np.linalg.lstsq(design, velocity, rcond=None)
    if rank != 2:
        raise ValueError("Acceleration estimation requires distinct observation times")
    return {"status": "ok", "observations": len(history),
            "acceleration_mps2": [float(fit[1, 0]), float(fit[1, 1]), 0.],
            "fit_intercept_velocity_xy_mps": fit[0].tolist(),
            "residual_rms_xy_mps": np.sqrt(np.mean((design @ fit - velocity)**2, axis=0)).tolist(),
            "note": "OLS detection velocity slope. Intercept does not replace posterior v0."}


def predict_shadow(snapshot, queries, solver, minimum_observations):
    """Only a past-only snapshot and query timestamps enter the CT predictors."""
    acceleration = estimate_acceleration(snapshot["history"], snapshot["timestamp_us"], minimum_observations)
    models = {"ct_cv": ConstantVelocity()}
    if acceleration["status"] == "ok":
        models["ct_ca"] = ConstantAcceleration(acceleration["acceleration_mps2"])
    predictions = []
    for query in queries:
        if query["timestamp_us"] <= snapshot["timestamp_us"]:
            raise ValueError("Validation queries must be later than the anchor")
        predictions.append({
            **query,
            "models": {name: ODEPredictor(model, **solver).predict(
                snapshot["state_pv"], snapshot["timestamp_us"], query["timestamp_us"]
            ).to_dict() for name, model in models.items()},
        })
    return {"snapshot": snapshot, "acceleration_estimate": acceleration, "queries": predictions}


def baseline_from_same_anchor(trace, snapshot, query_frame):
    """Verify the recorded B0 prediction is a pure rollout from the same anchor.

    Reject comparisons after an intervening correction, absent state, or missing
    trace step. Do not compare a corrected B0 with an open-loop CT rollout.
    """
    import pre_processing  # Initialize repository geometry dependencies in their supported order.
    from motion_module.motion_model import CTRV

    anchor = key_step(trace, snapshot["anchor_frame"])
    query = key_step(trace, query_frame)
    start, end = anchor["internal_frame_id"], query["internal_frame_id"]
    steps = sorted([s for s in trace["trace"] if start < s["internal_frame_id"] <= end],
                   key=lambda s: s["internal_frame_id"])
    if [s["internal_frame_id"] for s in steps] != list(range(start + 1, end + 1)):
        raise ValueError("Trace has missing intermediate steps")
    tid = str(snapshot["track_id"])
    model = CTRV(True, False, False, trace["config"]["basic"]["LiDAR_interval"])
    state = np.mat(snapshot["posterior_ctrv"]).T
    for step in steps:
        if tid not in step["before"]:
            raise ValueError("Baseline track no longer exists at query")
        if not np.allclose(np.asarray(state).ravel(), step["before"][tid]["state"], atol=1e-9, rtol=0):
            raise ValueError("Baseline contains an intervening correction; select an earlier query")
        state = model.stateTransition(state)
        model.warpStateYawToPi(state)
        col = step["predicted"]["all_valid_ids"].index(snapshot["track_id"])
        box = step["predicted"]["np_tras"][col][:14]
        if not np.allclose(model.getOutputInfo(state), box[:12], atol=1e-9, rtol=0):
            raise ValueError("Original CTRV rollout does not reproduce recorded baseline")
    pv = ctrv_state_to_pv(np.asarray(state).ravel())
    return {"position_m": pv[:3].tolist(), "velocity_mps": pv[3:].tolist(),
            "speed_xy_mps": float(np.linalg.norm(pv[3:5])), "box": list(box),
            "nominal_elapsed_s": len(steps) * model.dt, "same_anchor_verified": True}
