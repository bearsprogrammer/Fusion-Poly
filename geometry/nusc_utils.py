"""
utils for geometry calculations on the NuScenes dataset
"""
import numpy as np
from .nusc_box import NuscBox
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from typing import List, Tuple, Union
from data.script.NUSC_CONSTANT import *
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
import nuscenes.scripts.export_2d_annotations_as_json as export_2d
from utils import expand_dims
from copy import deepcopy


def PolyArea2D_s(pts: np.array) -> float:
    """
    Serial version for computing area of polygon surrounded by pts
    :param pts: np.array, a collection of xy coordinates of points, [pts_num, 2]
    :return: float, Area of polygon surrounded by pts
    """
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area


def PolyArea2D(pts: np.array) -> np.array:
    """
    Parallel version for computing areas of polygons surrounded by pts
    :param pts: np.array, a collection of xy coordinates of points, [poly_num, pts_num, 2]
    :return: float, Areas of polygons surrounded by pts, [poly_num,]
    """
    roll_pts = np.roll(pts, -1, axis=1)
    area = np.abs(np.sum((pts[:, :, 0] * roll_pts[:, :, 1] - pts[:, :, 1] * roll_pts[:, :, 0]), axis=1)) * 0.5
    return area


def get_yaw_diff_in_radians(rad1: float, rad2: float) -> float:
    """
    Get the difference between two angles (same axis) and unify it on the interval [0, pi]
    :param rad1: float, boxa angles, in radian
    :param rad2: float, boxb angles, in radian
    :return: float, difference between two angles, value interval -> [0, pi]
    """
    angle_diff = rad1 - rad2
    while angle_diff >= M_PI:
        angle_diff -= TWO_PI
    while angle_diff < -M_PI:
        angle_diff += TWO_PI
    return abs(angle_diff)


def yaw_punish_factor(box_a: NuscBox, box_b: NuscBox) -> float:
    """
    :param box_a: NuscBox
    :param box_b: NuscBox
    :return: float, penalty factor due to difference in yaw between two boxes, value interval -> [1, 3]
    """
    boxa_radians = box_a.abs_orientation_axisZ(box_a.orientation).radians
    boxb_radians = box_b.abs_orientation_axisZ(box_b.orientation).radians
    yaw_diff = get_yaw_diff_in_radians(boxa_radians, boxb_radians)
    assert 0 <= yaw_diff <= np.pi
    return 2 - np.cos(yaw_diff)


def mask_between_boxes(labels_a: np.array, labels_b: np.array) -> Union[np.array, np.array]:
    """
    :param labels_a: np.array, labels of a collection
    :param labels_b: np.array, labels of b collection
    :return: np.array[bool] np.array , mask matrix, 1 denotes different, 0 denotes same
    """
    mask = labels_a.reshape(-1, 1).repeat(len(labels_b), axis=1) != labels_b.reshape(1, -1).repeat(len(labels_a),
                                                                                                   axis=0)
    return mask, mask.reshape(-1)


def logical_or_mask(mask: np.array, seq_mask: np.array, boxes_a: dict, boxes_b: dict) -> np.array:
    """
    merge all mask which True means invalid
    :param mask: np.array, mask matrix
    :param seq_mask: np.array, 1-d mask matrix
    :param boxes_a: dict, a boxes infos, keys may include 'mask'
    :param boxes_b: dict, b boxes infos, keys may include 'mask'
    :return: np.array, mask matrix after merging(logical or) all mask
    """
    if 'mask' in boxes_b or 'mask' in boxes_a:
        if 'mask' in boxes_b and 'mask' in boxes_a:
            mask_ab = np.logical_or(boxes_a['mask'], boxes_b['mask'])
        elif 'mask' in boxes_b:
            mask_ab = boxes_b['mask']
        elif 'mask' in boxes_a:
            mask_ab = boxes_a['mask']
        else: raise Exception("cannot be happened")
        mask = np.logical_or(mask, mask_ab)
        return mask, mask.reshape(-1)
    else:
        return mask, seq_mask


def loop_inter(polys1: List[Polygon], polys2: List[Polygon], mask: np.array) -> np.array:
    """
    :param polys1: List[Polygon], collection of polygons
    :param polys2: List[Polygon], collection of polygons
    :param mask: np.array[bool], True denotes Invalid, False denotes valid
    :return: np.array, intersection area between two polygon collections
    """
    inters = np.zeros_like(mask, float)
    for i, reca in enumerate(polys1):
        for j, recb in enumerate(polys2):
            inters[i, j] = reca.intersection(recb).area if not mask[i, j] else 0
    return inters


def loop_convex(bottom_corners_a: np.array, bottom_corners_b: np.array, mask: np.array) -> np.array:
    """
    :param bottom_corners_a: np.array, bottom corners of a polygons, [a_num, b_num, 4, 2]
    :param bottom_corners_b: np.array, bottom corners of b polygons, [a_num, b_num, 4, 2]
    :param mask: np.array[bool], True denotes Invalid, False denotes valid, [a_num, b_num]
    :return: np.array, convexhull areas between two polygons, [a_num, b_num]
    """

    def init_convex(bcs: np.array, mask_: np.array) -> np.array:
        fake_convex = ConvexHull(bcs[0])
        return [ConvexHull(bc) if not mask_[i] else fake_convex for i, bc in enumerate(bcs)]

    all_bcs = np.concatenate((bottom_corners_a, bottom_corners_b), axis=2).reshape(-1, 8, 2)  # [numa * numb, 8, 2]

    # construct convexhull for every two boxes
    convexs = init_convex(all_bcs, mask)
    conv_cors = np.array([convex.vertices for convex in convexs], dtype=object)

    # 9 denotes 9 possible situations of len(conv_cor)
    conv_nums = np.array([len(conv_cor) for conv_cor in conv_cors]).reshape(1, -1).repeat(9, axis=0)  # [9, numa * numb]
    poss_idxs = np.arange(9).reshape(-1, 1).repeat(len(conv_cors), axis=1)

    # True in each row means that the number of convexHull corner points is the same as corresponding row index
    idx_masks = (conv_nums == poss_idxs)
    row_valid_idx = [np.where(idx_mask) for idx_mask in idx_masks]

    # Obtain convexhull area in order of the points number
    convex_areas = np.zeros(len(conv_cors))
    for conv_num, valid_idx in enumerate(row_valid_idx):
        if len(valid_idx[0]) == 0: continue
        b_idx = np.arange(len(valid_idx[0])).reshape(-1, 1).repeat(conv_num, axis=1)  # [len(valid_idx), conv_num]
        i_idx = np.stack((np.array(cor, dtype=int) for cor in conv_cors[valid_idx]))  # [len(valid_idx), conv_num]
        convex_areas[valid_idx] = PolyArea2D(all_bcs[valid_idx][b_idx, i_idx, :])

    return convex_areas.reshape(bottom_corners_a.shape[:2])

def norm_yaw_corners(yaw_corners: np.array) -> np.array:
    """
    Normalized corner of boxes with yaw angle.
    :param yaw_corners: np.array, corner of boxes with yaw angle in the nuscenes global frame, [box_num, 4, 2]
    :return: norm_corner, np.array, normlized corners in the 'pixel' frame, [box_num, 4]
    """
    assert yaw_corners.ndim == 3, "yaw corners dims must equal 3"

    # note the respentation of frame.
    norm_bms = np.concatenate([np.min(yaw_corners[:, :, 0], axis=1, keepdims=True),
                               np.min(yaw_corners[:, :, 1], axis=1, keepdims=True),
                               np.max(yaw_corners[:, :, 0], axis=1, keepdims=True),
                               np.max(yaw_corners[:, :, 1], axis=1, keepdims=True)], axis=1)

    return norm_bms


def project_dets_to_images(cam_meta_infos: dict, dets: np.array = None) -> dict:
    """Project the 3D bounding box to the pixel plane to get the corner points
    TODO: parallelize the projection process through matrix operations

    Args:
        nusc (NuScenes): nuscenes database
        sample_token (str): current frame sample token
        dets(np.array, optional): np.array[NuscBox]. Defaults to None.
    """
    if dets is None or len(dets) == 0: return {}
    
    # # fill invalid value with -1
    # sample_data = nusc.get('sample', sample_token)['data']
    # det_num, cam_num = len(dets), len(USED_ALLCAM)
    # corners3d, corners2d = -np.ones(shape=(det_num, cam_num, 8, 2)), -np.ones(shape=(det_num, cam_num, 4))
    # is_visible_cam, visible_area = [[] for _ in range(det_num)], [[] for _ in range(det_num)]
    
    # # project 3d bounding box to the image plane
    # for c_idx, cam_name in enumerate(USED_ALLCAM):
    #     sd_rec = nusc.get('sample_data', sample_data[cam_name])
    #     cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    #     pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    #     cam_infos = {
    #         'intrinsic': np.array(cs_rec['camera_intrinsic']),
    #         'pose_rec': pose_rec,
    #         'cs_rec': cs_rec
    #     }
    #     for d_idx, det in enumerate(dets):
    #         corners3d[d_idx, c_idx], corners2d[d_idx, c_idx] = project_3d_to_2d(det, cam_infos)
    #         area = box_2d_area(corners2d[d_idx, c_idx])
    #         if area > 0:
    #             is_visible_cam[d_idx].append(c_idx)
    #             visible_area[d_idx].append(area)
    
    # fill invalid value with -1
    det_num, cam_num = len(dets), len(USED_ALLCAM)
    corners3d, corners2d = -np.ones(shape=(det_num, cam_num, 8, 2)), -np.ones(shape=(det_num, cam_num, 4))
    is_visible_cam, visible_area = [[] for _ in range(det_num)], [[] for _ in range(det_num)]
    
    # project 3d bounding box to the image plane (1st loop: det, 2nd loop: cam)
    # this is the implementation of projection acceleration,
    # loop project 3d bounding box to the image plane
    for d_idx, det in enumerate(dets):
        possible_visible, visible_flag = np.ones(cam_num, dtype=bool), False
        for c_idx, cam_name in enumerate(USED_ALLCAM):
            corners3d[d_idx, c_idx], corners2d[d_idx, c_idx] = project_3d_to_2d(det=det,
                                                                                cam_infos=cam_meta_infos['cam_matrices'][c_idx],
                                                                                is_project=possible_visible[c_idx])
            area = box_2d_area(corners2d[d_idx, c_idx])
            if area > 0:
                if not visible_flag:
                    possible_visible, visible_flag = cam_meta_infos['common_visible_hash'][c_idx], True
                is_visible_cam[d_idx].append(c_idx)
                visible_area[d_idx].append(area)
    
    # get max project camera id, -1 refers to invalid
    max_visible_cam = -np.ones(det_num, dtype=int)
    for d_idx in range(det_num):
        all_visible_area = visible_area[d_idx]
        if len(all_visible_area) != 0:
            max_visible_cam[d_idx] = is_visible_cam[d_idx][np.argmax(all_visible_area)]
    
    # output appearance attributes
    appear_infos = {
        'np_dets_bbox2d': corners2d,
        'np_dets_bbox3d': corners3d,
        'maxarea_cam_id': max_visible_cam,
        'visible_cam_id': is_visible_cam,
        'visible_area': visible_area,
    }
    return appear_infos


def project_3d_to_2d(det: NuscBox, cam_infos: dict, is_project: bool = True) -> Tuple[np.array, np.array]:
    """project nuscbox from the global frame to the image plane

    Args:
        det (NuscBox): detetion under NuscBox data format
        cam_infos (dict): {
            'intrinsic': camera intrinsics
            'pose_rec': camera extrinsics, from global frame to ego frame
            'cs_rec': camera extrinsics, from ego frame to camera frame
        }
        
    Returns:
        Tuple(np.array, np.array): bbox2d image corners, [4(x1, y1, x2, y2)]
                                   bbox3d image corners, [8(corner number), 2(x, y)]
    """
    if not is_project:
        return np.array([[-1, -1] for _ in range(8)], dtype=int), np.array([-1 for _ in range(4)], dtype=int)
    
    box = det.copy()
    
    # global frame -> ego frame
    box.translate(-np.array(cam_infos['pose_rec']['translation']))
    box.rotate(Quaternion(cam_infos['pose_rec']['rotation']).inverse)
    
    # ego frame -> camera frame
    box.translate(-np.array(cam_infos['cs_rec']['translation']))
    box.rotate(Quaternion(cam_infos['cs_rec']['rotation']).inverse)
    
    # Filter out points after the camera optical center(z < 0)
    corners_3d = box.corners()
    in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
    corners_3d = corners_3d[:, in_front]

    # project 3d corners to image plane
    corner_coords = view_points(corners_3d, cam_infos['intrinsic'], True).T[:, :2].tolist()

    # get 2d bbox corners
    corner_2d = export_2d.post_process_coords(corner_coords) if len(corner_coords) == 8 else None
    if corner_2d is None:
        corner_2d, corner_coords = [-1 for _ in range(4)], [[-1, -1] for _ in range(8)]
        
    return np.array(corner_coords, dtype=int), np.array(corner_2d, dtype=int)


def compute_jacobian(det: NuscBox, cam_infos: dict, epsilon: float = 1e-5) -> np.ndarray:
    """
    Compute the Jacobian matrix of the 2D projection w.r.t. the 3D box center.

    Args:
        det (NuscBox): The 3D bounding box.
        cam_infos (dict): Camera intrinsic and extrinsic parameters.
        epsilon (float): Small perturbation for central differences.

    Returns:
        np.ndarray: Jacobian matrix (2x3) mapping xyz to image (u, v).
    """
    jacobian = np.zeros((2, 3))
    for i in range(3):
        perturbed_box_plus = det.copy()
        perturbed_box_minus = det.copy()
        perturbed_box_plus.center[i] += epsilon
        perturbed_box_minus.center[i] -= epsilon
        _, corner_coords_plus = project_3d_to_2d(perturbed_box_plus, cam_infos, is_project=True)
        _, corner_coords_minus = project_3d_to_2d(perturbed_box_minus, cam_infos, is_project=True)
        u_plus, v_plus = (corner_coords_plus[:2] + corner_coords_plus[2:4]) / 2
        u_minus, v_minus = (corner_coords_minus[:2] + corner_coords_minus[2:4]) / 2
        jacobian[0, i] = (u_plus - u_minus) / (2 * epsilon)
        jacobian[1, i] = (v_plus - v_minus) / (2 * epsilon)
    return jacobian


def box_2d_area(box) -> int:
    """get bbox2d area in the image plane

    Args:
        box (np.array): [4, ], (x0, y0(top-left), x1, y1(bottom-right))

    Returns:
        float: box area
    """
    return (box[2] - box[0]) * (box[3] - box[1]) if box[0] >= 0 else -1

def get_nusc_cam_meta_infos( nusc: NuScenes, sample_token: str= None, sample_data_tokens: dict= None) -> dict:
    """
    get all cameras meta infos on the nuscenes dataset
    :param sample_token: str, the token of each frame
    :param nusc: NuScenes, the instance of nuscenes dataset
    :return: dict, the meta infos of all cameras at specific frame
    """
    # basic infos of current frame
    cam_num = len(USED_ALLCAM)
    if sample_token is not None:
        sample_data = nusc.get('sample', sample_token)['data']
    cam_meta_infos = {'common_visible_hash': np.zeros((cam_num, cam_num), dtype=bool), 'cam_matrices': []}

    # consider that an object is observed by at most two adjacent cameras.
    # get camera infos
    for c_idx, cam_name in enumerate(USED_ALLCAM):
        # construct possible common visible camera id
        if c_idx == 0:
            cam_meta_infos['common_visible_hash'][c_idx, [-1, 0, 1]] = True
        elif c_idx == 5:
            cam_meta_infos['common_visible_hash'][c_idx, [0, c_idx - 1, c_idx]] = True
        else:
            cam_meta_infos['common_visible_hash'][c_idx, [c_idx - 1, c_idx, c_idx + 1]] = True
        # construct the extrinsic and intrinsic parameter matrices of each camera
        
        sample_data_token = sample_data[cam_name] if sample_token is not None else sample_data_tokens[cam_name]
        sd_rec = nusc.get('sample_data', sample_data_token)
        cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        cam_meta_infos['cam_matrices'].append({
            'intrinsic': np.array(cs_rec['camera_intrinsic']),
            'pose_rec': pose_rec,
            'cs_rec': cs_rec
        })
    return cam_meta_infos

def concatenate_cam_infos(cam_infos):
    # step 1, concate rotate and translate
    cam_num = len(USED_ALLCAM)
    trans_mat, trans_mat1, trans_mat2, trans_mat3 = [np.zeros((cam_num, 4, 4)) for _ in range(4)]
    Inv_trans_mat, Inv_trans_mat1, Inv_trans_mat2, Inv_trans_mat3 = [np.zeros((cam_num, 4, 4)) for _ in range(4)]
    for c_idx, cam_name in enumerate(USED_ALLCAM):
        trans_mat1[c_idx, 3, 3], trans_mat2[c_idx, 3, 3], trans_mat3[c_idx, 3, 3] = 1.0, 1.0, 1.0
        Inv_trans_mat1[c_idx, 3, 3], Inv_trans_mat2[c_idx, 3, 3], Inv_trans_mat3[c_idx, 3, 3] = 1.0, 1.0, 1.0
        # trans_mat1: global frame -> ego frame 
        # Inv_trans_mat1: ego frame -> global frame
        Inv_trans_mat1[c_idx, :3, :3] = Quaternion(cam_infos['cam_matrices'][c_idx]['pose_rec']['rotation']).rotation_matrix
        Inv_trans_mat1[c_idx, :3, 3] = np.array(cam_infos['cam_matrices'][c_idx]['pose_rec']['translation'])
        trans_mat1[c_idx, :3, :3] = Inv_trans_mat1[c_idx, :3, :3].T
        trans_mat1[c_idx, :3, 3] = trans_mat1[c_idx, :3, :3] @ -Inv_trans_mat1[c_idx, :3, 3]
        # mat2: ego frame -> camera frame
        # Inv_trans_mat2: camera frame -> ego frame
        Inv_trans_mat2[c_idx, :3, :3] = Quaternion(cam_infos['cam_matrices'][c_idx]['cs_rec']['rotation']).rotation_matrix
        Inv_trans_mat2[c_idx, :3, 3] = np.array(cam_infos['cam_matrices'][c_idx]['cs_rec']['translation'])
        trans_mat2[c_idx, :3, :3] = Inv_trans_mat2[c_idx, :3, :3].T
        trans_mat2[c_idx, :3, 3] = trans_mat2[c_idx, :3, :3] @ -Inv_trans_mat2[c_idx, :3, 3]
        # mat3: cam_intrinsic
        # Inv_trans_mat3: Inv_cam_intrinsic
        trans_mat3[c_idx, :3, :3] = cam_infos['cam_matrices'][c_idx]['intrinsic']
        Inv_trans_mat3[c_idx, :3, :3] = np.linalg.inv(cam_infos['cam_matrices'][c_idx]['intrinsic'])
    # trans_mat(6,4,4) global frame -> camera frame
    trans_mat = np.einsum('mij,mjk->mik',trans_mat2,trans_mat1)
    # Inv_trans_mat(6,4,4) camera frame -> global frame
    Inv_trans_mat = np.einsum('mij,mjk->mik',Inv_trans_mat1,Inv_trans_mat2)
    trans_mat_infos = {
            'glo_cam': trans_mat,
            'cam_glo': Inv_trans_mat,
            'cam_in': trans_mat3,
            'inv_cam_in': Inv_trans_mat3,
        }
    return trans_mat_infos

def fast_corners(dets_xyz:np.ndarray, dets_wlh:np.ndarray, dets_orien_mat:np.ndarray, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Transforms N bounding boxes according to the provided rotation matrices and scale factor.
        
        :param dets_xyz: A (6, N, 3) numpy array containing the center coordinates (xyz) for 6 camera perspectives of each box.
        :param dets_wlh: A (N, 3) numpy array containing the size (wlh) of each box.
        :param dets_orien_mat: A (6, N, 3, 3) numpy array of rotation matrices for 6 camera perspectives of each box.
        :param wlh_factor: A scalar to scale the size of the bounding boxes.
        :return: A (6, N, 3, 8) numpy array representing the transformed corner points of the boxes for each camera.
            First four corners are the ones facing forward.
                The last four are the ones facing backwards.
        """
        scaled_wlh = dets_wlh * wlh_factor
        
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        # Compute the original corner points lwh
        x_corners = scaled_wlh[:, 1, np.newaxis] / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = scaled_wlh[:, 0, np.newaxis] / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = scaled_wlh[:, 2, np.newaxis] / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.stack((x_corners, y_corners, z_corners), axis=-2)  # (N, 3, 8)

        # Rotate mat(6,N,3,8) <- mat(6, N, 3, 3) @ mat(N, 3, 8)
        corners_expanded = np.einsum('mnij,njk->mnik', dets_orien_mat, corners)
        
        # Expand the dimension of dets_xyz from (6, N, 3) to (6, N, 3, 1)
        dets_xyz_expanded = np.expand_dims(dets_xyz, axis=-1)
        # Translate
        corners_expanded = corners_expanded + dets_xyz_expanded

        return corners_expanded
    
def fast_view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
        points -> corners3d_expanded_filtered(6,N,3,8)
        view -> trans_mat3(6,4,4)
        
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[1] <= 4
    assert view.shape[2] <= 4
    assert points.shape[2] == 3
    
    # mat(6,N,4) <- mat(N,4)
    points_view = np.einsum('mij,mnjk->mnik', view[:,:3,:3], points)

    if normalize:
        points_view = points_view / expand_dims(points_view[:, :, 2, :], 3, 2)

    return points_view

def fast_export_2d_mat(corners3d_expanded):
    # assert corners3d_expanded.shape[0] is euqual to cam_num
    det_num, cam_num = corners3d_expanded.shape[1], len(USED_ALLCAM)
    assert corners3d_expanded.shape[0] == cam_num
    # fill invalid value with -1
    corners3d, corners2d = -np.ones(shape=(cam_num, det_num, 8, 2)), -np.ones(shape=(cam_num, det_num, 4))
    is_visible_cam, visible_area = [[] for _ in range(det_num)], [[] for _ in range(det_num)]

    # each camera view
    for c_idx in range(corners3d_expanded.shape[0]):
        # each bounding box
        for box_idx in range(corners3d_expanded.shape[1]):
            has_nan = np.isnan(corners3d_expanded[c_idx, box_idx]).any()
            corner_coords = corners3d_expanded[c_idx, box_idx].T[:, :2].tolist()

            # get 2d bbox corners
            corner_2d = export_2d.post_process_coords(corner_coords) if has_nan == False else None
            if corner_2d is None:
                corner_2d, corner_coords = [-1 for _ in range(4)], [[-1, -1] for _ in range(8)]
                
            # Add the filtered array of corner points to the corresponding list
            corners2d[c_idx][box_idx] = np.array(corner_2d, dtype=int)
            corners3d[c_idx][box_idx] = np.array(corner_coords, dtype=int)

            area = box_2d_area(corners2d[c_idx, box_idx])
            if area > 0:
                is_visible_cam[box_idx].append(c_idx)
                visible_area[box_idx].append(area)

    # get max project camera id, -1 refers to invalid
    max_visible_cam = -np.ones(det_num, dtype=int)
    for d_idx in range(det_num):
        all_visible_area = visible_area[d_idx]
        if len(all_visible_area) != 0:
            max_visible_cam[d_idx] = is_visible_cam[d_idx][np.argmax(all_visible_area)]
            
            # if len(all_visible_area) > 1:
            #     min_cam_index = deepcopy(is_visible_cam[d_idx])
            #     min_cam_index.remove(max_visible_cam[d_idx])
            #     corners2d[min_cam_index[0]][d_idx] = np.array([-1 for _ in range(4)])
            #     corners3d[min_cam_index[0]][d_idx] = np.array([[-1, -1] for _ in range(8)])
    
    # output appearance attributes
    appear_infos = {
        'np_dets_bbox2d': corners2d,
        'np_dets_bbox3d': corners3d,
        'maxarea_cam_id': max_visible_cam,
        'visible_cam_id': is_visible_cam,
        'visible_area': visible_area,
    }
    return appear_infos

def fast_project_dets_to_images(transmat_infos, np_dets):
    # np_det -> (x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
    dets_wlh = deepcopy(np_dets[:, 3:6])
    dets_xyz = deepcopy(np_dets[:, 0:4])
    dets_xyz[:, 3] = 1.0

    dets_orien = deepcopy(np_dets[:, 8:12])
    dets_orien_mat = np.zeros((len(dets_orien),3,3))
    for i in range (len(dets_orien)):
        dets_orien_mat[i] = Quaternion(dets_orien[i]).rotation_matrix

    # translation
    # mat(6,N,3) <- mat(N,3)
    dets_xyz_expanded = np.einsum('ijk,lk->ilj', transmat_infos['glo_cam'][:, :3, :3], dets_xyz[:, :3])
    dets_xyz_expanded +=  transmat_infos['glo_cam'][:, np.newaxis, :3, 3]
    # mat(6,N,3,3) <- mat(N,3,3)
    dets_orien_mat_expanded = np.einsum('mij,njk->mnik', transmat_infos['glo_cam'][:,:3,:3], dets_orien_mat)
    # mat(6,N,3,8)
    corners3d_expanded = fast_corners(dets_xyz_expanded[:,:,0:3], dets_wlh, dets_orien_mat_expanded)

    corners3d_expanded_filtered = deepcopy(corners3d_expanded)
    # Filter out points after the camera optical center(z < 0), mat(6,N,3,8)
    z_mask = corners3d_expanded_filtered[:,:,2] <= 0
    z_mask = expand_dims(z_mask, 3, 2)
    # Set the value of these points to NaN (or 0, or whatever marker value you want)
    corners3d_expanded_filtered[z_mask] = np.nan

    # project 3d corners to image plane
    corner_coords = fast_view_points(corners3d_expanded_filtered, transmat_infos['cam_in'], True)
    appear_infos = fast_export_2d_mat(corner_coords)
    return appear_infos