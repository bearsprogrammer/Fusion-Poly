"""
Generate high-rate 2D detections for nuScenes camera frames.

The script selects non-key-frame camera sample_data tokens between consecutive
key frames, runs an MMDetection detector, and saves the raw format consumed by
concat_2d_detection.py.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


CAMERA_CHANNELS = [
    "CAM_BACK",
    "CAM_FRONT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

# mmdet/nuImages class ids:
# 0-car, 1-truck, 2-trailer, 3-bus, 5-motorcycle, 6-pedestrian, 7-bicycle
DETECTION_CLASS_IDS = (0, 1, 2, 3, 5, 6, 7)

DEFAULT_CONFIG_FILE = (
    "/home/wx/mmdetection3d/configs/nuimages/"
    "cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim.py"
)
DEFAULT_CHECKPOINT_FILE = (
    "/home/wx/mmdetection3d/checkpoint/"
    "cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim_20201024_135753-e0e49778.pth"
)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(data, path: str, indent: Optional[int] = 4) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_sample_tokens(path: str, json_key: Optional[str] = None) -> List[str]:
    """Load sample tokens from a list JSON or a result JSON."""
    data = load_json(path)

    if json_key:
        data = data[json_key]
    elif isinstance(data, Mapping) and "results" in data:
        data = data["results"]

    if isinstance(data, Mapping):
        return list(data.keys())
    if isinstance(data, list):
        return list(data)

    raise TypeError(f"Unsupported sample token JSON format in {path}")


def load_token_set(path: Optional[str]) -> set:
    if not path:
        return set()

    data = load_json(path)
    if isinstance(data, Mapping):
        return set(data.keys())
    if isinstance(data, list):
        return set(data)

    raise TypeError(f"Unsupported token JSON format in {path}")


def _closest_token(
    timestamps: Sequence[int],
    tokens: Sequence[str],
    target_timestamp: float,
) -> str:
    diffs = np.abs(np.asarray(timestamps, dtype=np.float64) - target_timestamp)
    return tokens[int(np.argmin(diffs))]


def select_intermediate_tokens(
    nusc,
    sample_token: str,
    cameras: Sequence[str] = CAMERA_CHANNELS,
    fractions: Sequence[float] = (0.5,),
) -> List:
    """Select non-key-frame sample_data tokens near requested time fractions."""
    key_sample = nusc.get("sample", sample_token)
    next_sample_token = key_sample["next"]
    if not next_sample_token:
        raise ValueError(f"Sample {sample_token} has no next sample")

    next_sample = nusc.get("sample", next_sample_token)
    start_time = key_sample["timestamp"]
    interval = next_sample["timestamp"] - start_time

    per_camera_tokens = []
    for camera in cameras:
        key_sample_data = nusc.get("sample_data", key_sample["data"][camera])
        current_sample_data = nusc.get("sample_data", key_sample_data["next"])

        timestamps = []
        tokens = []
        while not current_sample_data["is_key_frame"]:
            if current_sample_data["sample_token"] != next_sample_token:
                raise AssertionError(
                    f"Unexpected sample token for {current_sample_data['token']}: "
                    f"{current_sample_data['sample_token']} != {next_sample_token}"
                )

            timestamps.append(current_sample_data["timestamp"])
            tokens.append(current_sample_data["token"])
            current_sample_data = nusc.get("sample_data", current_sample_data["next"])

        if not tokens:
            raise ValueError(
                f"No non-key-frame camera data between {sample_token} and {next_sample_token} "
                f"for camera {camera}"
            )

        selected = [
            _closest_token(timestamps, tokens, start_time + interval * fraction)
            for fraction in fractions
        ]
        if len(set(selected)) != len(selected):
            raise ValueError(
                f"Duplicate intermediate token selected for {sample_token}, camera {camera}: "
                f"{selected}. Try fewer fractions or inspect the camera timestamps."
            )

        per_camera_tokens.append(selected)

    if len(fractions) == 1:
        return [selected[0] for selected in per_camera_tokens]

    return [list(frame_tokens) for frame_tokens in zip(*per_camera_tokens)]


def build_intermediate_token_table(
    nusc,
    sample_tokens: Iterable[str],
    end_tokens: Iterable[str],
    cameras: Sequence[str] = CAMERA_CHANNELS,
    fractions: Sequence[float] = (0.5,),
) -> Dict[str, List]:
    """Build the intermediate camera-token table keyed by next sample token."""
    end_token_set = set(end_tokens)
    results = {}

    for sample_token in sample_tokens:
        if sample_token in end_token_set:
            continue

        key_sample = nusc.get("sample", sample_token)
        next_sample_token = key_sample["next"]
        if not next_sample_token:
            continue

        results[next_sample_token] = select_intermediate_tokens(
            nusc=nusc,
            sample_token=sample_token,
            cameras=cameras,
            fractions=fractions,
        )

    return results


def filter_detection_result(
    bbox_result: Sequence[np.ndarray],
    class_ids: Sequence[int] = DETECTION_CLASS_IDS,
) -> Dict[str, List[List[float]]]:
    """Keep selected MMDetection classes and convert arrays to JSON lists."""
    filtered = {}
    for class_id in class_ids:
        if class_id >= len(bbox_result):
            continue

        boxes = bbox_result[class_id]
        if boxes is None or len(boxes) == 0:
            continue

        filtered[str(class_id)] = np.asarray(boxes).tolist()

    return filtered


def _extract_bbox_result(inference_result):
    if isinstance(inference_result, tuple):
        return inference_result[0]
    return inference_result


def init_mmdet_model(config_file: str, checkpoint_file: str, device: str):
    from mmdet.apis import init_detector

    return init_detector(config_file, checkpoint_file, device=device)


def run_detection(
    nusc,
    nusc_path: str,
    token_table: Mapping[str, Sequence],
    model,
    class_ids: Sequence[int] = DETECTION_CLASS_IDS,
) -> Dict[str, object]:
    from mmdet.apis import inference_detector

    def detect_frame(sample_data_tokens: Sequence[str]) -> Dict[str, Dict[str, List[List[float]]]]:
        frame_result = {}
        for sample_data_token in sample_data_tokens:
            sample_data = nusc.get("sample_data", sample_data_token)
            image_path = os.path.join(nusc_path, sample_data["filename"])
            inference_result = inference_detector(model, image_path)
            bbox_result = _extract_bbox_result(inference_result)
            frame_result[sample_data_token] = filter_detection_result(
                bbox_result,
                class_ids=class_ids,
            )
        return frame_result

    detected_results = {}
    for sample_token, frame_token_groups in token_table.items():
        if not frame_token_groups:
            detected_results[sample_token] = {}
        elif isinstance(frame_token_groups[0], str):
            detected_results[sample_token] = detect_frame(frame_token_groups)
        else:
            detected_results[sample_token] = [
                detect_frame(sample_data_tokens)
                for sample_data_tokens in frame_token_groups
            ]

    return detected_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select high-rate nuScenes camera frames and run 2D detection."
    )
    parser.add_argument(
        "--mode",
        choices=("select_tokens", "detect", "all"),
        default="all",
        help="Run token selection, detection, or both.",
    )
    parser.add_argument(
        "--nusc_path",
        type=str,
        default="/data1/wyt_dataset1/nuscenes/",
        help="Path to nuScenes dataset root.",
    )
    parser.add_argument(
        "--nusc_version",
        type=str,
        default="v1.0-test",
        help="NuScenes version, e.g. v1.0-trainval or v1.0-test.",
    )
    parser.add_argument(
        "--sample_token_path",
        type=str,
        default=None,
        help="JSON containing sample tokens for token selection.",
    )
    parser.add_argument(
        "--sample_token_json_key",
        type=str,
        default=None,
        help="Optional JSON key used to read sample tokens from --sample_token_path.",
    )
    parser.add_argument(
        "--end_token_path",
        type=str,
        default=None,
        help="JSON containing sequence-end sample tokens to skip.",
    )
    parser.add_argument(
        "--token_output_path",
        type=str,
        default=None,
        help="Where to save selected intermediate sample_data tokens.",
    )
    parser.add_argument(
        "--token_input_path",
        type=str,
        default=None,
        help="Selected intermediate sample_data token JSON used for detection.",
    )
    parser.add_argument(
        "--detection_output_path",
        type=str,
        default=None,
        help="Where to save raw high-rate 2D detection results.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="MMDetection config file.",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=DEFAULT_CHECKPOINT_FILE,
        help="MMDetection checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cuda:7.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=CAMERA_CHANNELS,
        help="Camera channels to process.",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.5],
        help="Intermediate timestamps as fractions of the key-frame interval.",
    )
    parser.add_argument(
        "--class_ids",
        nargs="+",
        type=int,
        default=list(DETECTION_CLASS_IDS),
        help="MMDetection class ids to keep.",
    )
    parser.add_argument(
        "--json_indent",
        type=int,
        default=4,
        help="Indentation for output JSON. Use -1 for compact JSON.",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.mode in ("select_tokens", "all"):
        if not args.sample_token_path:
            raise ValueError("--sample_token_path is required for token selection")
        if not args.token_output_path:
            raise ValueError("--token_output_path is required for token selection")

    if args.mode in ("detect", "all"):
        if not (args.token_input_path or args.token_output_path):
            raise ValueError("--token_input_path or --token_output_path is required for detection")
        if not args.detection_output_path:
            raise ValueError("--detection_output_path is required for detection")


def main():
    args = parse_args()
    validate_args(args)

    from nuscenes.nuscenes import NuScenes

    json_indent = None if args.json_indent < 0 else args.json_indent
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_path, verbose=True)

    token_table = None
    if args.mode in ("select_tokens", "all"):
        sample_tokens = load_sample_tokens(args.sample_token_path, args.sample_token_json_key)
        end_tokens = load_token_set(args.end_token_path)
        print(f"Selecting intermediate tokens from {len(sample_tokens)} samples...")
        token_table = build_intermediate_token_table(
            nusc=nusc,
            sample_tokens=sample_tokens,
            end_tokens=end_tokens,
            cameras=args.cameras,
            fractions=args.fractions,
        )
        dump_json(token_table, args.token_output_path, indent=json_indent)
        print(f"Selected tokens saved to {args.token_output_path}")
        print(f"Total target key frames: {len(token_table)}")

    if args.mode in ("detect", "all"):
        if token_table is None:
            token_input_path = args.token_input_path or args.token_output_path
            token_table = load_json(token_input_path)

        print("Initializing 2D detector...")
        model = init_mmdet_model(args.config_file, args.checkpoint_file, args.device)
        print(f"Running 2D detection for {len(token_table)} target key frames...")
        detected_results = run_detection(
            nusc=nusc,
            nusc_path=args.nusc_path,
            token_table=token_table,
            model=model,
            class_ids=args.class_ids,
        )
        dump_json(detected_results, args.detection_output_path, indent=json_indent)
        print(f"Detection results saved to {args.detection_output_path}")


if __name__ == "__main__":
    main()
