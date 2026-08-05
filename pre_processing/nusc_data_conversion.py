"""
data format conversion and data concat on the NuScenes dataset
"""

import numpy as np
from typing import Tuple, List, Dict
from geometry import NuscBox
from data.script.NUSC_CONSTANT import *


def concat_box_attr(nuscbox: NuscBox, *attrs) -> List:
    res = []
    for attr in attrs:
        tmp_attr = getattr(nuscbox, attr)
        if isinstance(tmp_attr, list):
            res += getattr(nuscbox, attr)
        elif isinstance(tmp_attr, (float, int)):
            res += [tmp_attr]
        elif isinstance(tmp_attr, np.ndarray):
            res += tmp_attr.tolist()
        elif isinstance(tmp_attr, tuple):
            res += list(tmp_attr)
        elif attr == 'detection_name':
            res += [CLASS_SEG_TO_STR_CLASS[tmp_attr]]
        else: raise Exception("unsupport data format to concat")
    return res


def concat_dict_attr(dictbox: dict, *attrs) -> List:
    res = []
    for attr in attrs:
        if attr == 'detection_name':
            res += [CLASS_SEG_TO_STR_CLASS[dictbox[attr]]]
            continue
        elif attr == 'detection_score':
            res += [dictbox[attr]]
            continue
        res += dictbox[attr]
    return res


def dictdet2array(dets: List[dict], *attrs) -> Tuple[List, np.array]:
    listdets = [concat_dict_attr(det, *attrs) for det in dets if det['detection_name'] in CLASS_SEG_TO_STR_CLASS]
    return listdets, np.array(listdets)


def arraydet2box(dets: np.array, ids: np.array = None, init_geo: bool = True, sample_token: str = None):
    # det -> (x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    if dets.ndim == 1: dets = dets[None, :]
    assert dets.shape[1] == 14, "The number of observed states must satisfy 14"

    NuscBoxes, boxes_bottom_corners, boxes_norm_corners = [], [], []
    for idx, det in enumerate(dets):
        curr_box = NuscBox(center=det[0:3], size=det[3:6], rotation=det[8:12],
                            velocity=tuple(det[6:8].tolist() + [0.0]), score=det[12],
                            name=CLASS_STR_TO_SEG_CLASS[int(det[13])], init_geo=init_geo, token=sample_token)
        if ids is not None: curr_box.tracking_id = int(ids[idx])
        NuscBoxes.append(curr_box)
        boxes_bottom_corners.append(curr_box.bottom_corners_)
        boxes_norm_corners.append(curr_box.norm_corners_)
    return np.array(NuscBoxes), np.array(boxes_bottom_corners), np.array(boxes_norm_corners)

def box2outputdict(box: NuscBox, sample_token: str) -> Dict:
    assert box.score >= 0
    return {
        "sample_token": sample_token,
        "translation": [float(box.center[0]), float(box.center[1]), float(box.center[2])],
        "size": [float(box.wlh[0]), float(box.wlh[1]), float(box.wlh[2])],
        "rotation": [float(box.orientation[0]), float(box.orientation[1]),
                     float(box.orientation[2]), float(box.orientation[3])],
        "velocity": [float(box.velocity[0]), float(box.velocity[1])],
        "tracking_id": str(box.tracking_id),
        "tracking_name": box.name,
        "tracking_score": box.score,
    }

def box2detdict(box: NuscBox, sample_token: str) -> Dict:
    assert box.score >= 0
    return {
        "sample_token": sample_token,
        "translation": [float(box.center[0]), float(box.center[1]), float(box.center[2])],
        "size": [float(box.wlh[0]), float(box.wlh[1]), float(box.wlh[2])],
        "rotation": [float(box.orientation[0]), float(box.orientation[1]),
                     float(box.orientation[2]), float(box.orientation[3])],
        "velocity": [float(box.velocity[0]), float(box.velocity[1])],
        "detection_name": box.name,
        "detection_score": box.score,
        'attribute_name': ''
    }