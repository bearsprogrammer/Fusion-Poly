import copy
import numpy as np
from data.script.NUSC_CONSTANT import *
from pre_processing import arraydet2box
from geometry.nusc_utils import project_dets_to_images
from scipy.optimize import least_squares
from geometry.nusc_distance import iou_2d, giou_2d, iou_3d_s, giou_3d_s


DEFAULT_SIZE_PRIOR_WEIGHT = 1e4


class NuscLocationCoordinator:
    def __init__(self, res_form, use_weight, multi_view, size_prior_weight=DEFAULT_SIZE_PRIOR_WEIGHT):
        self.res_form = res_form
        self.use_weight = use_weight
        self.multi_view = multi_view
        self.size_prior_weight = float(size_prior_weight)

    @property
    def enable_size_opt(self):
        # Sentinel DEFAULT keeps historical xyz-only behavior.
        return self.size_prior_weight < DEFAULT_SIZE_PRIOR_WEIGHT
        
    def coordinator_s(self, matched_dict: dict, camera_meta_infos: dict):
        '''
        Serial implementation of location coordinator
        '''
        compensated_matched_dict = {}
        
        for det_3d_tuple, np_2d_dict in matched_dict.items():
            '''
            np_det: tuple, (x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label), (14,)
            np_2d_dict: dict, {'cam_name': np_2d(x1, y1, x2, y2, det_score, class_label)}, (6,)
            '''
            origin_np_3d = np.array(det_3d_tuple)
            # organize optimize inputs
            dets_3d = copy.deepcopy(origin_np_3d)
            dets_2d = np.array([np_2d_dict[cam_name] for cam_name in np_2d_dict.keys()])
            cam_idxs = [USED_ALLCAM.index(cam_name) for cam_name in np_2d_dict.keys()]
            bboxs_scores = np.concatenate([[origin_np_3d[-2]], [det_2d[-2] for det_2d in dets_2d]], axis=0)
            # check the class_label of dets_3d and dets_2d
            assert dets_3d[-1] == dets_2d[0][-1], 'Class label not match'
            
            if not self.multi_view:
                # find the max box2d
                idx_2d, np_2d_det = self.find_largest_bbox_2d(dets_2d)
                # find the highest score box2d
                # idx_2d, np_2d_det = self.find_highest_score_bbox_2d(dets_2d)
                
                weights = bboxs_scores = np.array([origin_np_3d[-2], dets_2d[idx_2d][-2]])
                cam_idxs = [cam_idxs[idx_2d]]
                dets_2d = np_2d_det[np.newaxis, :]

            else:
                # weights = []
                # for box_score in bboxs_scores:
                #     weight = self.weight_function(score=box_score, alpha=self.alpha[det_label], beta=self.beta[det_label])
                #     weights.append(weight)
                # weights = np.array(weights)
                weights = bboxs_scores

            common_kwargs = dict(
                jac='2-point', method='trf',
                ftol=1e-6, xtol=1e-6, gtol=1e-5, x_scale=1.0, loss='linear',
                f_scale=1.0, diff_step=None, tr_solver=None, tr_options={},
                jac_sparsity=None, max_nfev=100000, verbose=0,
                args=(dets_3d, dets_2d, camera_meta_infos, weights, cam_idxs),
            )

            if self.enable_size_opt:
                initial_guess = copy.deepcopy(dets_3d[:6])
                op_res = least_squares(fun=self.residual, x0=initial_guess, **common_kwargs)
                origin_np_3d[:6] = op_res.x
            else:
                # Bit-compatible path with historical xyz-only coordinator.
                initial_guess = copy.deepcopy(dets_3d[:3])
                op_res = least_squares(fun=self.residual_xyz, x0=initial_guess, **common_kwargs)
                origin_np_3d[:3] = op_res.x

            compensated_matched_dict.update({tuple(origin_np_3d): np_2d_dict})
        
        return compensated_matched_dict

    def size_prior_residual(self, opt_size, ref_size):
        return np.abs(np.asarray(opt_size, dtype=float) - np.asarray(ref_size, dtype=float)) * self.size_prior_weight

    def _geometry_residual(self, np_det_center, np_det_size, np_3d_det, np_2d_dets, camera_meta_infos, weights, cam_idxs):
        assert np_det_center.shape[0] == 3
        assert np_det_size.shape[0] == 3
        assert np_3d_det.shape[0] == 14
        assert np_2d_dets.shape[1] == 6
        assert len(np_2d_dets) == len(cam_idxs) == (len(weights) - 1)

        # box_a: NuscBox, origin_det
        box_det_a, _, _ = arraydet2box(np_3d_det[None, :])
        nusc_box_a = copy.deepcopy(box_det_a[0])
        # box_b: NuscBox, op_det
        tmp_np_3d_det = np.concatenate((np_det_center, np_det_size, np_3d_det[6:]), axis=0)
        assert tmp_np_3d_det.shape[0] == 14
        box_det_b, _, _ = arraydet2box(tmp_np_3d_det[None, :])
        nusc_box_b = copy.deepcopy(box_det_b[0])
        
        if self.res_form == 'eu':
            residual_1 = np.abs(np_det_center - np_3d_det[:3])
        else:
            metric_func = iou_3d_s if self.res_form == 'iou' else giou_3d_s
            _, metric = metric_func(nusc_box_a, nusc_box_b)
            residual_1 = np.array([1 - metric])
        
        proj_info = project_dets_to_images(camera_meta_infos, box_det_b)
        all_proj_boxes = proj_info['np_dets_bbox2d'][0]
        
        res2_list = []
        
        for idx, (cam_idx, np_2d_det) in enumerate(zip(cam_idxs, np_2d_dets)):
            proj_box = all_proj_boxes[cam_idx]
            if self.box_2d_area(proj_box) <= 0:
                res2_list.append([1600, 900] if self.res_form == 'eu' else 2.0) # image size (1600, 900)
                continue
            
            if self.res_form == 'eu':
                proj_box_cxy = (proj_box[:2] + proj_box[2:4]) / 2.0
                np_2d_det_cxy = (np_2d_det[:2] + np_2d_det[2:4]) / 2.0
                res2 = np.abs(proj_box_cxy - np_2d_det_cxy)
                res2_list.append(res2)
            else:
                np_bboxs_a = np_2d_det
                np_bboxs_b = np.zeros(6)
                np_bboxs_b[:4] = proj_box[:4]
                np_bboxs_b[4:6] = copy.deepcopy(np_2d_det[4:6])
                
                metric_func = iou_2d if self.res_form == 'iou' else giou_2d
                metric, _ = metric_func(np_bboxs_a, np_bboxs_b)
                res2 = 1 - metric[0][0]
                res2_list.append(res2)
        
        if self.res_form == 'eu':
            residual_2 = np.concatenate(res2_list)
        else:
            residual_2 = np.array(res2_list)

        residual_geo = np.concatenate([residual_1, residual_2])
        
        if self.use_weight:
            if self.res_form == 'eu':
                for idx, weight in enumerate(weights):
                    if idx == 0:
                        residual_geo[:3] *= weight
                    else:
                        residual_geo[3 + 2 * (idx - 1) : 3 + 2 * idx] *= weight
            else:
                residual_geo = residual_geo * np.asarray(weights, dtype=float)
        return residual_geo

    def residual_xyz(self, np_det_center, np_3d_det, np_2d_dets, camera_meta_infos, weights, cam_idxs):
        """Historical xyz-only residual used by default for reproducibility."""
        return self._geometry_residual(
            np_det_center,
            np_3d_det[3:6],
            np_3d_det,
            np_2d_dets,
            camera_meta_infos,
            weights,
            cam_idxs,
        )

    def residual(self, np_det_state, np_3d_det, np_2d_dets, camera_meta_infos, weights, cam_idxs):
        """
        Joint xyz+wlh residual.
        Enabled when size_prior_weight < DEFAULT_SIZE_PRIOR_WEIGHT.
        """
        assert np_det_state.shape[0] == 6
        residual_geo = self._geometry_residual(
            np_det_state[:3],
            np_det_state[3:6],
            np_3d_det,
            np_2d_dets,
            camera_meta_infos,
            weights,
            cam_idxs,
        )
        residual_size = self.size_prior_residual(np_det_state[3:6], np_3d_det[3:6])
        return np.concatenate([residual_geo, residual_size])

    def weight_function(self, score, alpha, beta):
        # score represents the confidence score of detection
        return 1 / (1 + np.exp(-alpha * (score - beta)))

    def box_2d_area(self, box) -> int:
        """
        get bbox2d area in the image plane
        Args:
            box (np.array): [4, ], (x0, y0(top-left), x1, y1(bottom-right))
        Returns:
            float: box area
        """
        return (box[2] - box[0]) * (box[3] - box[1]) if box[0] >= 0 else -1

    def find_largest_bbox_2d(self, np_2d_dets):
        """
        Find the 2d bounding box with the largest area from the given 2D numpy array and return its index and the bounding box itself.
        parameters:
            np_2d_dets (numpy.ndarray): A numpy array with the shape of [N, 6]. Each row represents the information of a detection box,
                                    including x1, y1, x2, y2, det_score, and class_label in sequence.
        returns:
            tuple: A tuple containing the index (int) of the largest bounding box and the information (numpy.ndarray) of the largest bounding box.
        """
        assert np_2d_dets.ndim == 2 and np_2d_dets.shape[1] == 6, "np_2d_dets should be a 2D numpy array with shape [N, 6]"
        
        areas = []
        for det_2d in np_2d_dets:
            area = self.box_2d_area(det_2d[:4])
            areas.append(area)
        largest_index = np.argmax(np.array(areas))
        largest_bbox = np_2d_dets[largest_index]
        return largest_index, largest_bbox

    def find_highest_score_bbox_2d(self, np_2d_dets):
        """
        Find the bounding box with the highest det_score from the given 2D numpy array and return its index and the bounding box itself.

        Parameters:
        np_2d_dets (numpy.ndarray): A numpy array with the shape of [N, 6]. Each row represents the information of a detection box,
                                    including x1, y1, x2, y2, det_score, and class_label in sequence.

        Returns:
        tuple: A tuple containing the index (int) of the bounding box with the highest det_score and the information (numpy.ndarray) of that bounding box.
        """
        scores = np_2d_dets[:, 4]  # get the det_score column for all bboxes
        highest_score_index = np.argmax(scores)
        highest_score_bbox = np_2d_dets[highest_score_index]
        return highest_score_index, highest_score_bbox
