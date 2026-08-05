import os
import pdb
import sys

import numpy as np
# from main import iou3d, convert_3dbox_to_8corner
from scipy.optimize import linear_sum_assignment as linear_assignment

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.loaders import create_tracks
from pyquaternion import Quaternion

from utils.io import load_file
import pre_processing.nusc_nms
import pre_processing.nusc_data_conversion

import argparse

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def rotation_to_positive_z_angle(rotation):
    q = Quaternion(rotation)
    angle = q.angle if q.axis[2] > 0 else -q.angle
    return angle

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def convert_3dbox_to_8corner(bbox3d_input, nuscenes_to_kitti=False):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
        Note: the output of this function will be passed to the funciton iou3d
            for calculating the 3D-IOU. But the function iou3d was written for
            kitti, so the caller needs to set nuscenes_to_kitti to True if
            the input bbox3d_input is in nuscenes format.
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    if nuscenes_to_kitti:
        # transform to kitti format first
        bbox3d_nuscenes = copy.copy(bbox3d)
        # kitti:    [x,  y,  z,  a, l, w, h]
        # nuscenes: [y, -z, -x, -a, w, l, h]
        bbox3d[0] = bbox3d_nuscenes[1]
        bbox3d[1] = -bbox3d_nuscenes[2]
        bbox3d[2] = -bbox3d_nuscenes[0]
        bbox3d[3] = -bbox3d_nuscenes[3]
        bbox3d[4] = bbox3d_nuscenes[5]
        bbox3d[5] = bbox3d_nuscenes[4]

    R = roty(bbox3d[3])

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)

def get_mean(tracks):
    '''
    Input:
      tracks: {scene_token:  {t: [TrackingBox]}}
    '''
    print('len(tracks.keys()): ', len(tracks.keys()))

    # gt_trajectory_map to compute residual or velocity
    # tracking_name: {scene_token -> {tracking_id: {t_idx -> det_data}}
    # [h, w, l, x, y, z, yaw] #x_dot, y_dot, z_dot, yaw_dot]
    gt_trajectory_map = {tracking_name: {scene_token: {} for scene_token in tracks.keys()} for tracking_name in
                         NUSCENES_TRACKING_NAMES}

    # store every detection data to compute mean and variance
    gt_box_data = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}

    # added by xiaoyu
    dt = 1 / 2

    for scene_token in tracks.keys():
        # print('scene_token: ', scene_token)
        # print('tracks[scene_token].keys(): ', tracks[scene_token].keys())
        for t_idx in range(len(tracks[scene_token].keys())):
            # print('t_idx: ', t_idx)
            t = sorted(tracks[scene_token].keys())[t_idx]
            for box_id in range(len(tracks[scene_token][t])):
                # print('box_id: ', box_id)
                box = tracks[scene_token][t][box_id]
                # print('box: ', box)

                if box.tracking_name not in NUSCENES_TRACKING_NAMES:
                    continue
                # box:  {'sample_token': '6a808b09e5f34d33ba1de76cc8dab423', 'translation': [2131.657, 1108.874, 3.453], 'size': [3.078, 6.558, 2.95], 'rotation': [0.8520240186812739, 0.0, 0.0, 0.5235026949216329], 'velocity': array([-0.01800415,  0.0100023 ]), 'ego_dist': 54.20556415873658, 'num_pts': 4, 'tracking_id': 'cbaabbf2a83a4177b2145ab1317e296e', 'tracking_name': 'truck', 'tracking_score': -1.0}
                # [h, w, l, x, y, z, ry,
                # x_t - x_{t-1}, ...,  for [x,y,z,ry]
                # (x_t - x_{t-1}) - (x_{t-1} - x_{t-2}), ..., for [x,y,z,ry]
                # s = sqrt(x^2 +y^2)
                # s_t - s_{t-1}, ..., for [s]
                # (s_t - s_{t-1}) / dt - (s_{t-1} - s_{t-2}) / dt
                box_data = np.array([
                    box.size[2], box.size[0], box.size[1],
                    box.translation[0], box.translation[1], box.translation[2],
                    rotation_to_positive_z_angle(box.rotation),
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    np.linalg.norm([box.translation[0], box.translation[1]]), 0, 0
                ])

                if box.tracking_id not in gt_trajectory_map[box.tracking_name][scene_token]:
                    gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id] = {t_idx: box_data}
                else:
                    gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data

                # if we can find the same object in the previous frame, get the velocity
                if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and t_idx - 1 in \
                        gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
                    residual_vel = box_data[3:7] - gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][
                                                       t_idx - 1][3:7]
                    box_data[7:11] = residual_vel

                    # xiaoyu, get diff_s
                    box_data[-2] = box_data[-3] - gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][
                                                       t_idx - 1][-3]

                    gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data

                    # back fill
                    if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 1][7] == 0:
                        gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 1][
                        7:11] = residual_vel
                        gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 1][
                        -2] = box_data[-2]


                    # if we can find the same object in the previous two frames, get the acceleration
                    if box.tracking_id in gt_trajectory_map[box.tracking_name][scene_token] and t_idx - 2 in \
                            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id]:
                        residual_a = residual_vel - (
                                    gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 1][3:7] -
                                    gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 2][3:7])
                        # added by xiaoyu
                        residual_a = residual_a / dt
                        box_data[-1] = (box_data[-2] - (
                                gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 1][-3] -
                                gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 2][-3]
                        )) / dt

                        box_data[11:15] = residual_a
                        gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx] = box_data
                        # back fill
                        if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 1][11] == 0:
                            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 1][
                            11:15] = residual_a
                            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 1][
                            -1] = box_data[-1]
                        if gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 2][11] == 0:
                            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 2][
                            11:15] = residual_a
                            gt_trajectory_map[box.tracking_name][scene_token][box.tracking_id][t_idx - 2][
                                -1] = box_data[-1]

                # print(det_data)
                gt_box_data[box.tracking_name].append(box_data)

    gt_box_data = {tracking_name: np.stack(gt_box_data[tracking_name], axis=0) for tracking_name in
                   NUSCENES_TRACKING_NAMES}

    mean = {tracking_name: np.mean(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    std = {tracking_name: np.std(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    var = {tracking_name: np.var(gt_box_data[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}

    return mean, std, var


def matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks_gt, matching_dist):
    '''
    For each sample token, find matches of pred_boxes and gt_boxes, then get stats.
    tracks_gt has the temporal order info for each sample_token
    '''

    diff = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}  # [h, w, l, x, y, z, a]
    diff_vel = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}  # [x_dot, y_dot, z_dot, a_dot]

    # similar to main.py class AB3DMOT update()
    reorder = [3, 4, 5, 6, 2, 1, 0]
    reorder_back = [6, 5, 4, 0, 1, 2, 3]

    mea_attr = ('translation', 'size', 'velocity', 'rotation', 'detection_score', 'detection_name')

    for scene_token in tracks_gt.keys():
        # print('scene_token: ', scene_token)
        # print('tracks[scene_token].keys(): ', tracks[scene_token].keys())
        # {tracking_name: t_idx: tracking_id: det(7) }
        match_diff_t_map = {tracking_name: {} for tracking_name in NUSCENES_TRACKING_NAMES}
        for t_idx in range(len(tracks_gt[scene_token].keys())):
            # print('t_idx: ', t_idx)
            t = sorted(tracks_gt[scene_token].keys())[t_idx]
            # print(len(tracks_gt[scene_token][t]))
            if len(tracks_gt[scene_token][t]) == 0:
                continue
            box = tracks_gt[scene_token][t][0]
            sample_token = box.sample_token

            for tracking_name in NUSCENES_TRACKING_NAMES:

                # print('t: ', t)
                gt_all = [box for box in gt_boxes.boxes[sample_token] if box.tracking_name == tracking_name]
                if len(gt_all) == 0:
                    continue
                gts = np.stack([np.array([
                    box.size[2], box.size[0], box.size[1],
                    box.translation[0], box.translation[1], box.translation[2],
                    rotation_to_positive_z_angle(box.rotation)
                ]) for box in gt_all], axis=0)
                gts_ids = [box.tracking_id for box in gt_all]

                det_all = [box for box in pred_boxes.boxes[sample_token] if box.detection_name == tracking_name]
                if len(det_all) == 0:
                    continue

                # xiaoyu, NMS processing
                # all categories are blended together and sorted by detection score
                pdb.set_trace()
                list_dets = [concat_box_attr(det, *mea_attr) for det in det_all]
                # Score Filter based on category-specific thresholds
                np_dets = np.array([det for det in list_dets if det[-2] > -1])
                box_dets, np_dets_bottom_corners, np_dets_norm_corners = arraydet2box(np_dets)
                assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners) == len(np_dets_norm_corners) == len(det_all)
                tmp_infos = {'np_dets': np_dets, 'np_dets_bottom_corners': np_dets_bottom_corners,
                             'np_dets_norm_corners': np_dets_norm_corners, 'box_dets': box_dets}
                keep = globals()[self.NMS_type](box_infos=tmp_infos, metrics=self.NMS_metric, thre=self.NMS_thre)
                if len(keep) == 0: continue

                det_all = [det for idx, det in enumerate(det_all) if idx in keep]
                pdb.set_trace()


                dets = np.stack([np.array([
                    box.size[2], box.size[0], box.size[1],
                    box.translation[0], box.translation[1], box.translation[2],
                    rotation_to_positive_z_angle(box.rotation)
                ]) for box in det_all], axis=0)

                dets = dets[:, reorder]
                gts = gts[:, reorder]

                if matching_dist == '3d_iou':
                    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
                    gts_8corner = [convert_3dbox_to_8corner(gt_tmp) for gt_tmp in gts]
                    iou_matrix = np.zeros((len(dets_8corner), len(gts_8corner)), dtype=np.float32)
                    for d, det in enumerate(dets_8corner):
                        for g, gt in enumerate(gts_8corner):
                            iou_matrix[d, g] = iou3d(det, gt)[0]
                    # print('iou_matrix: ', iou_matrix)
                    distance_matrix = -iou_matrix
                    threshold = -0.1
                elif matching_dist == '2d_center':
                    distance_matrix = np.zeros((dets.shape[0], gts.shape[0]), dtype=np.float32)
                    for d in range(dets.shape[0]):
                        for g in range(gts.shape[0]):
                            distance_matrix[d][g] = np.sqrt(
                                (dets[d][0] - gts[g][0]) ** 2 + (dets[d][1] - gts[g][1]) ** 2)
                    threshold = 2
                else:
                    assert (False)

                matched_indices = linear_assignment(distance_matrix)
                # print('matched_indices: ', matched_indices)
                dets = dets[:, reorder_back]
                gts = gts[:, reorder_back]
                for pair_id in range(matched_indices.shape[0]):
                    if distance_matrix[matched_indices[pair_id][0]][matched_indices[pair_id][1]] < threshold:
                        diff_value = dets[matched_indices[pair_id][0]] - gts[matched_indices[pair_id][1]]
                        diff[tracking_name].append(diff_value)
                        gt_track_id = gts_ids[matched_indices[pair_id][1]]
                        if t_idx not in match_diff_t_map[tracking_name]:
                            match_diff_t_map[tracking_name][t_idx] = {gt_track_id: diff_value}
                        else:
                            match_diff_t_map[tracking_name][t_idx][gt_track_id] = diff_value
                        # check if we have previous time_step's matching pair for current gt object
                        # print('t: ', t)
                        # print('len(match_diff_t_map): ', len(match_diff_t_map))
                        if t_idx > 0 and t_idx - 1 in match_diff_t_map[tracking_name] and gt_track_id in \
                                match_diff_t_map[tracking_name][t_idx - 1]:
                            diff_vel_value = diff_value - match_diff_t_map[tracking_name][t_idx - 1][gt_track_id]
                            diff_vel[tracking_name].append(diff_vel_value)

    diff = {tracking_name: np.stack(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    mean = {tracking_name: np.mean(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    std = {tracking_name: np.std(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    var = {tracking_name: np.var(diff[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}

    diff_vel = {tracking_name: np.stack(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    mean_vel = {tracking_name: np.mean(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    std_vel = {tracking_name: np.std(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}
    var_vel = {tracking_name: np.var(diff_vel[tracking_name], axis=0) for tracking_name in NUSCENES_TRACKING_NAMES}

    return mean, std, var, mean_vel, std_vel, var_vel


if __name__ == '__main__':
    # Settings.
    parser = argparse.ArgumentParser(description='Get nuScenes stats.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eval_set', type=str, default='train',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--matching_dist', type=str, default='2d_center',
                        help='Which distance function for matching, 3d_iou or 2d_center.')
    args = parser.parse_args()

    eval_set_ = args.eval_set
    config_path = args.config_path
    verbose_ = bool(args.verbose)
    matching_dist = args.matching_dist

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    if 'train' in eval_set_:
        detection_file = '/juno/u/hkchiu/dataset/nuscenes_new/megvii_train.json'
        data_root = '/data1/wyt_dataset1/nuscenes/trainval'
        version = 'v1.0-trainval'
    elif 'val' in eval_set_:
        detection_file = '/home/wyt/lxy/Fast-Poly/data/detector/val/nuscenes_val_centerpoint_3d.json'
        data_root = '/data1/wyt_dataset1/nuscenes/trainval'
        version = 'v1.0-trainval'
    elif 'test' in eval_set_:
        detection_file = '"/home/wyt/lxy/Fast-Poly/data/detector/train/infos_train_10sweeps_withvelo_filter_True.json"'
        data_root = '/data1/wyt_dataset1/nuscenes/test'
        version = 'v1.0-test'

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    pred_boxes, _ = load_prediction(detection_file, 10000, DetectionBox)
    gt_boxes = load_gt(nusc, eval_set_, TrackingBox)

    assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
        "Samples in split don't match samples in predicted tracks."

    # Add center distances.
    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)

    print('len(pred_boxes.sample_tokens): ', len(pred_boxes.sample_tokens))
    print('len(gt_boxes.sample_tokens): ', len(gt_boxes.sample_tokens))

    tracks_gt = create_tracks(gt_boxes, nusc, eval_set_, gt=True)

    mean, std, var = get_mean(tracks_gt)
    print('GT: Global coordinate system')
    print('h, w, l, x, y, z, a, x_dot, y_dot, z_dot, a_dot, x_dot_dot, y_dot_dot, z_dot_dot, a_dot_dot, s, s_dot, s_dot_dot')
    print('mean: ', mean)
    print('std: ', std)
    print('var: ', var)

    # for observation noise covariance
    # raw_det_file = load_file(detection_file)["results"]
    mean, std, var, mean_vel, std_vel, var_vel = matching_and_get_diff_stats(pred_boxes, gt_boxes, tracks_gt,
                                                                             matching_dist)
    print('Diff: Global coordinate system')
    print('h, w, l, x, y, z, a')
    print('mean: ', mean)
    print('std: ', std)
    print('var: ', var)
    print('h_dot, w_dot, l_dot, x_dot, y_dot, z_dot, a_dot')
    print('mean_vel: ', mean_vel)
    print('std_vel: ', std_vel)
    print('var_vel: ', var_vel)
