"""Read-only high replay -> shadow CV/CA prediction -> held-out GT evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .history import build_snapshot, key_step, predict_shadow, baseline_from_same_anchor

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def iter_json_array(path):
    """Stream the large official annotation array without retaining all objects."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer, eof, first = "", False, True
        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                chunk = stream.read(1024 * 1024)
                buffer += chunk
                eof = not chunk
                continue
            if first:
                if not buffer.startswith("["):
                    raise ValueError("Expected a JSON array: {}".format(path))
                first = False
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            try:
                item, end = decoder.raw_decode(buffer)
            except ValueError:
                if eof:
                    raise ValueError("Incomplete JSON array: {}".format(path))
                chunk = stream.read(1024 * 1024)
                buffer += chunk
                eof = not chunk
                continue
            yield item
            buffer = buffer[end:]


def resolve(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    fields = {
        "dataset": {"root", "version", "split"},
        "source": {"results_path", "trace_root"},
        "case": {"scene", "track_id", "anchor_frame", "query_frames", "query_kind", "gt_instance_id"},
        "prediction": {"models", "velocity_source", "vertical_policy"},
        "acceleration": {"source", "estimator", "history_observations", "minimum_observations", "insufficient_history"},
        "comparison": {"box_geometry", "run_hungarian", "detection_gt_max_xy_m"},
        "solver": {"rtol", "atol", "max_step_s"}, "output": {"root"},
    }
    if not isinstance(config, dict) or set(config) != set(fields):
        raise ValueError("Config must contain exactly: {}".format(", ".join(fields)))
    for section, keys in fields.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError("Unexpected/missing keys in config section {}".format(section))
    expected = [("dataset", "version", "v1.0-trainval"), ("dataset", "split", "val"),
                ("case", "query_kind", "keyframe"), ("prediction", "models", ["cv", "ca"]),
                ("prediction", "velocity_source", "posterior_state"), ("prediction", "vertical_policy", "hold"),
                ("acceleration", "source", "accepted_detection_velocity"),
                ("acceleration", "estimator", "linear_least_squares"),
                ("acceleration", "insufficient_history", "skip_ca"),
                ("comparison", "box_geometry", "baseline_prediction"), ("comparison", "run_hungarian", False)]
    for section, key, value in expected:
        if config[section][key] != value:
            raise ValueError("Supported {}.{}: {!r}".format(section, key, value))
    c, a = config["case"], config["acceleration"]
    for value in [c["track_id"], c["anchor_frame"], a["history_observations"], a["minimum_observations"]]:
        if type(value) is not int or value < 0:
            raise ValueError("Track/frame/history values must be nonnegative integers")
    if not 3 <= a["minimum_observations"] <= a["history_observations"]:
        raise ValueError("Require 3 <= minimum_observations <= history_observations")
    frames = c["query_frames"]
    if not isinstance(frames, list) or not frames or any(type(x) is not int or x <= c["anchor_frame"] for x in frames):
        raise ValueError("query_frames must be later than anchor_frame")
    if len(set(frames)) != len(frames):
        raise ValueError("Duplicate query_frames")
    if not isinstance(c["scene"], str) or not c["scene"].startswith("scene-") or not c["scene"][6:].isdigit():
        raise ValueError("Expected scene-NNNN")
    limit = float(config["comparison"]["detection_gt_max_xy_m"])
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("detection_gt_max_xy_m must be positive and finite")
    return config


def validate_inputs(config, trace, result_path):
    from nuscenes.utils.splits import create_splits_scenes

    case = config["case"]
    if case["scene"] not in create_splits_scenes()["val"]:
        raise ValueError("Requested scene is not in official nuScenes val split")
    if trace["scene"] != case["scene"] or trace["config"]["basic"]["freq"] != "high" or trace["config"]["basic"]["split"] != "val":
        raise ValueError("Trace scene/high/val provenance mismatch")
    if not trace.get("exact_match") or not all(x["ids_equal"] and x["max_abs_error"] == 0 for x in trace["output_comparisons"]):
        raise ValueError("Trace must have exact replay validation")
    if sha256(result_path) != trace["source_results_sha256"]:
        raise ValueError("Original high results hash does not match verified trace")
    # Verify relevant captured output records against the actual result file too.
    original = read_json(result_path)["results"]
    verify_captured_outputs(trace, original, [case["anchor_frame"]] + case["query_frames"])


def verify_captured_outputs(trace, original, frames):
    for frame in frames:
        step = key_step(trace, frame)
        sort_boxes = lambda boxes: sorted(boxes, key=lambda b: str(b["tracking_id"]))
        # The replay stores the sample token once per step, not in every box.
        captured = [{**box, "sample_token": step["token"]} for box in step["output"]]
        if sort_boxes(original[step["token"]]) != sort_boxes(captured):
            raise ValueError("Captured outputs differ from saved high results at frame {}".format(frame))


def evaluate_gt(dataset, case, snapshot, queries):
    """Load evaluation labels only after shadow predictions have been written."""
    base = resolve(dataset["root"]) / dataset["version"]
    scenes = read_json(base / "scene.json")
    scene = next(s for s in scenes if s["name"] == case["scene"])
    samples = {s["token"]: s for s in read_json(base / "sample.json")}
    ordered, token = [], scene["first_sample_token"]
    while token:
        ordered.append(samples[token]); token = samples[token]["next"]
    for frame, timestamp, token in [(snapshot["anchor_frame"], snapshot["timestamp_us"], snapshot["sample_token"])] + [
            (q["frame_idx"], q["timestamp_us"], q["sample_token"]) for q in queries]:
        if ordered[frame]["timestamp"] != timestamp or ordered[frame]["token"] != token:
            raise ValueError("Trace frame/timestamp does not match official sample metadata")
    wanted = {q["sample_token"] for q in queries}
    labels = {a["sample_token"]: a for a in iter_json_array(base / "sample_annotation.json")
              if a["instance_token"] == case["gt_instance_id"] and a["sample_token"] in wanted}
    if set(labels) != wanted:
        raise ValueError("Requested GT instance is absent at one or more queries")
    instances = {x["token"]: x for x in read_json(base / "instance.json")}
    categories = {x["token"]: x["name"] for x in read_json(base / "category.json")}
    if categories[instances[case["gt_instance_id"]]["category_token"]] != "vehicle.car":
        raise ValueError("Evaluation GT must be a car")
    return labels


def association_cost(prediction, detection):
    import pre_processing
    from geometry.nusc_box import NuscBox
    from geometry.nusc_distance import fast_giou_3d

    def geometry(row):
        box = NuscBox(row[:3], row[3:6], row[8:12], name="car")
        return {"np_dets": np.array([row[:14]], dtype=float),
                "np_dets_norm_corners": np.array([box.norm_corners_])}
    return float(1 - fast_giou_3d(geometry(prediction), geometry(detection))[1][0, 0])


def select_evaluation_candidate(step, gt, max_xy):
    # GT selects the evaluation correspondence ONLY; nothing flows back to prediction.
    candidates = [(stage, i, row) for stage in step["stages"] for i, row in enumerate(stage["detections"])
                  if int(row[13]) == 2]
    if not candidates:
        return None
    stage, index, det = min(candidates, key=lambda x: np.linalg.norm(np.array(x[2][:2]) - gt[:2]))
    error = float(np.linalg.norm(np.array(det[:2]) - gt[:2]))
    if error > max_xy:
        return None
    return {"stage": stage["stage"], "index": index, "box": det,
            "gt_xy_distance_m": error, "score": det[12], "cost_limit": stage["thresholds"]["2"],
            "correspondence_rule": "nearest same-class GT center within configured XY radius; evaluation only"}


def measure(prediction, baseline_box, gt_position, candidate, spatial_limit):
    delta = np.array(prediction["position_m"]) - gt_position
    result = {"gt_xy_error_m": float(np.linalg.norm(delta[:2])),
              "gt_z_error_m": float(delta[2]), "gt_xyz_error_m": float(np.linalg.norm(delta)),
              "association": None}
    if candidate is not None:
        box = list(baseline_box); box[:3] = prediction["position_m"]
        distance = float(np.linalg.norm(np.array(box[:3]) - candidate["box"][:3]))
        cost = association_cost(box, candidate["box"])
        spatial_pass = distance < spatial_limit
        result["association"] = {"cost": cost, "cost_limit": candidate["cost_limit"],
            "spatial_distance_xyz_m": distance, "spatial_limit_m": spatial_limit,
            "spatial_gate_pass": spatial_pass, "cost_gate_pass": cost <= candidate["cost_limit"],
            "gate_pass": spatial_pass and cost <= candidate["cost_limit"],
            "hungarian_run": False}
    return result


def run(config_path):
    config_path = Path(config_path).resolve()
    config = load_config(config_path); case = config["case"]
    trace_path = resolve(config["source"]["trace_root"]) / (case["scene"].replace("-", "") + "_quantitative") / "trace.json"
    if not trace_path.is_file():
        raise ValueError("Missing verified replay trace: {}. Export a verified trace for this scene first.".format(trace_path))
    result_path = resolve(config["source"]["results_path"])
    output = resolve(config["output"]["root"]) / ("{}_track{}_anchor{}".format(case["scene"], case["track_id"], case["anchor_frame"]))
    for protected in [resolve(config["dataset"]["root"]), result_path.parent, trace_path.parent]:
        if output == protected or protected in output.parents:
            raise ValueError("Output must be separate from original dataset/results/trace")
    hashes = {str(path): sha256(path) for path in [config_path, trace_path, result_path]}
    trace = read_json(trace_path)
    validate_inputs(config, trace, result_path)
    snapshot = build_snapshot(trace, case["anchor_frame"], case["track_id"], config["acceleration"]["history_observations"])
    queries = [{"frame_idx": frame, "timestamp_us": int(key_step(trace, frame)["timestamp"]),
                "sample_token": key_step(trace, frame)["token"]} for frame in case["query_frames"]]
    shadow = predict_shadow(snapshot, queries, config["solver"], config["acceleration"]["minimum_observations"])
    for query in shadow["queries"]:
        query["models"]["baseline"] = baseline_from_same_anchor(trace, snapshot, query["frame_idx"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "predictions.json", shadow)  # Freeze predictions BEFORE evaluation labels.
    print("Predictions saved; now loading evaluation GT.", flush=True)
    labels = evaluate_gt(config["dataset"], case, snapshot, shadow["queries"])
    evaluation = []
    for query in shadow["queries"]:
        gt = labels[query["sample_token"]]
        step = key_step(trace, query["frame_idx"])
        candidate = select_evaluation_candidate(step, gt["translation"], config["comparison"]["detection_gt_max_xy_m"])
        baseline = query["models"]["baseline"]
        limit = float(trace["config"]["association"]["voxel_mask_size"]["2"])
        if not trace["config"]["ablation"]["voxel_mask"]:
            raise ValueError("This evaluation currently requires the original spatial gate")
        measurements = {name: measure(pred, baseline["box"], gt["translation"], candidate, limit)
                        for name, pred in query["models"].items()}
        if candidate is not None:
            stage = next(s for s in step["stages"] if s["stage"] == candidate["stage"])
            if case["track_id"] in stage["track_ids"]:
                col = stage["track_ids"].index(case["track_id"])
                recorded = float(stage["cost"][2][candidate["index"]][col])
                actual = measurements["baseline"]["association"]
                expected = actual["cost"] if actual["spatial_gate_pass"] else float("inf")
                if not (recorded == expected or np.isclose(recorded, expected, atol=1e-10, rtol=0)):
                    raise ValueError("Recomputed B0 cost does not match original matrix")
                candidate["baseline_matrix_cost"] = recorded if np.isfinite(recorded) else "inf"
                candidate["baseline_assignment"] = any(a == candidate["index"] and b == col for a, b in zip(*stage["assignment"][:2]))
        evaluation.append({"frame_idx": query["frame_idx"], "timestamp_us": query["timestamp_us"],
                           "gt": gt, "candidate": candidate, "measurements": measurements})
    unchanged = all(sha256(path) == digest for path, digest in hashes.items())
    if not unchanged:
        raise RuntimeError("Input changed during validation")
    report = {"case": case, "config": config, "prediction": shadow, "evaluation": evaluation,
              "provenance": {"input_sha256": hashes, "inputs_unchanged": unchanged,
                "baseline_same_anchor_verified": True, "prediction_saved_before_gt_loading": True,
                "gt_source": str(resolve(config["dataset"]["root"]) / config["dataset"]["version"] / "sample_annotation.json"),
                "code_sha256": {str(ROOT / name): sha256(ROOT / name) for name in [
                    "ctmot/proto/history.py", "ctmot/proto/validate_history.py", "ctmot/proto/predictor.py",
                    "ctmot/proto/dynamics.py", "geometry/nusc_distance.py", "geometry/nusc_box.py",
                    "motion_module/motion_model.py"]}},
              "scope": "Open-loop keyframe position validation only. No Hungarian, tracking, or official metric rerun."}
    write_json(output / "report.json", report)
    from .validation_report import write_report
    write_report(report, output)
    print("Output: {}".format(output), flush=True)
    for entry in evaluation:
        for name, metrics in entry["measurements"].items():
            print("frame {} {}: GT XY={:.6f}m XYZ={:.6f}m association={}".format(
                entry["frame_idx"], name, metrics["gt_xy_error_m"], metrics["gt_xyz_error_m"], metrics["association"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/validation_scene1072.yaml")
    args = parser.parse_args()
    try:
        run(args.config)
    except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
        parser.exit(2, "Validation failed: {}\n".format(exc))


if __name__ == "__main__":
    main()
