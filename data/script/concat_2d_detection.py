"""
Convert raw high-rate 2D detections to the tracking format and concatenate
them with key-frame 2D detections.
"""

import argparse
import json
import os
import pdb

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes


# mmdet: 0-car, 1-truck, 2-trailer, 3-bus, 5-motorcycle,
#         6-pedestrian, 7-bicycle
# track: 0-motorcycle, 1-bus, 2-car, 3-pedestrian, 4-bicycle,
#        5-trailer, 6-truck
CLASS_TRANSFER = {0: 2, 1: 6, 2: 5, 3: 1, 5: 0, 6: 3, 7: 4}


def draw_boxes(image_path, boxes, output_path):
    """Draw 2D bounding boxes on an image and save the result."""
    image = cv2.imread(image_path)
    boxes = boxes[np.argsort(boxes[:, 4])]
    for box in boxes:
        x1, y1, x2, y2, score, cls = box
        color = (cls * 255 / 10, 255 - cls * 255 / 10, cls * 255 / 10)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2),), color, 2)
        label = f"Class: {int(cls)}, Score: {score:.2f}"
        cv2.putText(
            image,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    try:
        cv2.imwrite(output_path, image)
    except Exception:
        pdb.set_trace()
    print(f"Image saved at {output_path}")


def visualize_detections(nusc, nusc_path, high_rate_det, vis_output_dir):
    """Visualize high-rate 2D detections on source images."""
    os.makedirs(vis_output_dir, exist_ok=True)
    for sample_token, dets in high_rate_det.items():
        for sample_data_token, bboxes in dets.items():
            np_bboxes = []
            for cls, boxes in bboxes.items():
                for box in boxes:
                    box_with_cls = box.copy()
                    box_with_cls.append(int(cls))
                    np_bboxes.append(box_with_cls)
            np_bboxes = np.array(np_bboxes)
            file_path = nusc.get("sample_data", sample_data_token)["filename"]
            img_output_path = os.path.join(vis_output_dir, sample_data_token + ".jpg")
            draw_boxes(os.path.join(nusc_path, file_path), np_bboxes, img_output_path)
    print(f"Visualization saved to {vis_output_dir}")


def sort_detection_format(nusc, high_rate_det):
    """Map detector classes and group detections by camera channel."""
    sorted_result = {}
    for sample_token, dets in high_rate_det.items():
        if isinstance(dets, dict):
            temp_dict = {}
            for sample_data_token, bboxes in dets.items():
                np_bboxes = []
                for cls, boxes in bboxes.items():
                    if int(cls) not in CLASS_TRANSFER:
                        continue
                    for box in boxes:
                        box_with_cls = box.copy()
                        box_with_cls.append(CLASS_TRANSFER[int(cls)])
                        np_bboxes.append(box_with_cls)
                cam_type = nusc.get("sample_data", sample_data_token)["channel"]
                temp_dict[cam_type] = {
                    "np_boxes": np_bboxes.copy(),
                    "sample_data_token": sample_data_token,
                }
            sorted_result[sample_token] = temp_dict.copy()

        elif isinstance(dets, list):
            temp_dict_list = []
            for det in dets:
                temp_dict = {}
                for sample_data_token, bboxes in det.items():
                    np_bboxes = []
                    for cls, boxes in bboxes.items():
                        if int(cls) not in CLASS_TRANSFER:
                            continue
                        for box in boxes:
                            box_with_cls = box.copy()
                            box_with_cls.append(CLASS_TRANSFER[int(cls)])
                            np_bboxes.append(box_with_cls)
                    cam_type = nusc.get("sample_data", sample_data_token)["channel"]
                    temp_dict[cam_type] = {
                        "np_boxes": np_bboxes.copy(),
                        "sample_data_token": sample_data_token,
                    }
                assert len(temp_dict) == 6, (
                    f"Expected 6 cameras, got {len(temp_dict)} for token {sample_token}"
                )
                temp_dict_list.append(temp_dict.copy())
            sorted_result[sample_token] = temp_dict_list.copy()
            assert len(temp_dict_list) == 2, (
                f"Expected 2 intermediate frames, got {len(temp_dict_list)} "
                f"for token {sample_token}"
            )

    return sorted_result


def concat_detections(high_freq_sorted_result, key_result, first_token_table):
    """Interleave intermediate detections before each key-frame detection."""
    concat_result = {}
    for sample_token, item in key_result.items():
        assert sample_token in first_token_table or sample_token in high_freq_sorted_result, (
            f"Token {sample_token} not found in first_token_table or high_freq results"
        )

        if sample_token in first_token_table:
            assert sample_token not in high_freq_sorted_result
            concat_result[sample_token] = item.copy()
        else:
            high_freq = high_freq_sorted_result[sample_token]
            if isinstance(high_freq, dict):
                concat_result[sample_token + "_mid"] = high_freq.copy()
                concat_result[sample_token] = item.copy()
            elif isinstance(high_freq, list):
                assert isinstance(high_freq[0], dict) and isinstance(high_freq[1], dict), (
                    f"Unexpected format for token {sample_token}"
                )
                concat_result[sample_token + "_0"] = high_freq[0].copy()
                concat_result[sample_token + "_1"] = high_freq[1].copy()
                concat_result[sample_token] = item.copy()

    return concat_result


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate high-frequency 2D detections with key-frame detections."
    )
    parser.add_argument(
        "--nusc_path",
        type=str,
        default="/data1/wyt_dataset1/nuscenes/",
        help="Path to NuScenes dataset root.",
    )
    parser.add_argument(
        "--nusc_version",
        type=str,
        default="v1.0-trainval",
        help="NuScenes dataset version.",
    )
    parser.add_argument(
        "--high_rate_det_path",
        type=str,
        required=True,
        help="Path to raw high-rate 2D detection result JSON.",
    )
    parser.add_argument(
        "--key_frame_det_path",
        type=str,
        required=True,
        help="Path to key-frame 2D detection result JSON.",
    )
    parser.add_argument(
        "--first_token_path",
        type=str,
        required=True,
        help="Path to first token table JSON.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the concatenated 2D detection result JSON.",
    )
    parser.add_argument(
        "--sorted_det_path",
        type=str,
        default=None,
        help="Optional path for the converted intermediate detection JSON.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize 2D detections on images.",
    )
    parser.add_argument(
        "--vis_output_dir",
        type=str,
        default="/data3/wx_dataset/2dvis/",
        help="Directory for visualization images.",
    )
    args = parser.parse_args()

    print("Loading NuScenes dataset...")
    nusc = NuScenes(args.nusc_version, args.nusc_path, verbose=True)
    print(f"Loading high-rate detection results from {args.high_rate_det_path}...")
    with open(args.high_rate_det_path, "r") as f:
        high_rate_det = json.load(f)

    if args.visualize:
        print("Visualizing 2D detections...")
        visualize_detections(nusc, args.nusc_path, high_rate_det, args.vis_output_dir)

    print("Converting detection format...")
    sorted_result = sort_detection_format(nusc, high_rate_det)
    if args.sorted_det_path:
        output_dir = os.path.dirname(os.path.abspath(args.sorted_det_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.sorted_det_path, "w") as f:
            json.dump(sorted_result, f, indent=5)
        print(f"Sorted detection saved to {args.sorted_det_path}")

    print(f"Loading key-frame detection results from {args.key_frame_det_path}...")
    with open(args.key_frame_det_path, "r") as f:
        key_result = json.load(f)
    print(f"Loading first token table from {args.first_token_path}...")
    with open(args.first_token_path, "r") as f:
        first_token_table = json.load(f)

    print("Concatenating detections...")
    concat_result = concat_detections(sorted_result, key_result, first_token_table)

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(concat_result, f, indent=4)
    print(f"Concatenated result saved to {args.output_path}")
    print(
        f"Total entries: {len(concat_result)} "
        f"(key-frame: {len(key_result)}, high-freq: {len(sorted_result)})"
    )


if __name__ == "__main__":
    main()
