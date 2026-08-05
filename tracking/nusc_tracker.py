"""
Tracker, Core of Poly-MOT.
Tracklet prediction and punishment, cost matrix construction, tracking id assignment, tracklet update and init, and output file

TODO: delete debug log in the release version
"""

import copy
import time, json

import numpy as np
from pre_processing import blend_nms
from .nusc_trajectory import Trajectory
from data.script.NUSC_CONSTANT import *
from utils.matching import Hungarian
from geometry.nusc_distance import iou_bev, iou_3d, giou_bev, giou_3d, d_eucl, giou_3d_s, fast_giou_3d, fast_giou_3d_unique, fast_giou_bev
from utils.script import mask_tras_dets, fast_compute_check, reorder_metrics, spec_metric_mask, voxel_mask
from pre_processing.data_fusion import datafusion
from geometry.nusc_utils import project_dets_to_images, fast_project_dets_to_images
from utils.script import mask_tras_dets, Count_List


class Tracker:
    def __init__(self, config, nusc):
        self.cfg = config
        # Hyper parameters
        self.is_debug = self.cfg['debug']['is_debug']
        self.cls_num = self.cfg['basic']['CLASS_NUM']
        self.first_thre, self.second_thre = config['association']['first_thre'], config['association']['second_thre']
        self.third_thre = config['association']['third_thre']
        self.algorithm = config['association']['algorithm']
        self.assoc_stage = config['association'].get('stage', 3)  # 1=mix, 2=mix+3d, 3=mix+3d+2d
        self.punish_num, self.metrics = config['output']['punish_num'], config['association']['category_metrics']
        self.post_nms_cfg = config['output']
        self.re_metrics = reorder_metrics(self.metrics)
        # init, notice that no tentative trajectories in Poly-MOT best experiments.
        # self.xx_tras -> {tracking id(int): trajectory}
        self.active_tras, self.tentative_tras, self.dead_tras, self.valid_tras = {}, {}, {}, {}
        self.id_seed, self.frame_id, self.seq_id, self.det_infos, self.tra_infos = 0, None, None, None, None
        self.nusc = nusc

        # only for ablation
        self.use_voxel_mask = config['ablation']['voxel_mask']
        self.voxel_mask_size = config['association']['voxel_mask_size']

        # only for debug
        self.predict_time, self.update_time, self.data_time, self.punish_time = {}, {}, {}, {}


    def reset(self) -> None:
        """
        Initialize Tracker for each new seq
        """
        self.active_tras, self.tentative_tras, self.dead_tras, self.valid_tras = {}, {}, {}, {}
        self.id_seed, self.frame_id, self.seq_id, self.det_infos, self.tra_infos = 0, None, None, None, None

    def tracking(self, data_info: dict) -> None:
        """
        :param data_info: the observation information(detection) of each frame
        :return: update data_info: {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result'[optimal]: bool
        }
        """
        # step0. reset tracker for each seq
        if data_info['is_first_frame']: self.reset()
        self.det_infos, self.frame_id, self.seq_id = data_info, data_info['frame_id'], data_info['seq_id']
        self.sample_token, self.is_key_frame = data_info['sample_token'], data_info['is_key_frame']

        # step1. predict all valid trajectories
        st = time.time()
        self.tras_predict()
        self.predict_time[self.frame_id] = time.time() - st
        json.dump(self.predict_time, open(TIME_COST_ROOT + 'predict_time.json', "w"))
        # print(f'tracklet predicting spend time: {time.time() - st}')

        # step2. if there is no dets, we will punish all valid trajectories
        if self.det_infos['no_dets']:
            st = time.time()
            self.tras_punish(data_info)
            self.punish_time[self.frame_id] = time.time() - st
            json.dump(self.punish_time, open(TIME_COST_ROOT + 'punish_time.json', "w"))
            if self.post_nms_cfg['post_nms']: self.post_nms_tras(data_info)
            return

        # step3. associate current frame detections with existing trajectories
        st = time.time()
        associate_result = self.data_association()
        self.data_time[self.frame_id] = time.time() - st
        json.dump(self.data_time, open(TIME_COST_ROOT + 'data_time.json', "w"))
        # print(f'data association spend time: {time.time() - st}')

        # step4. use observations(detection) to update the corresponding trajectories
        # and output unmatch trajectories(up to punish_num) states, and output new trajectories states
        st = time.time()
        dict_track_res = self.tras_update(associate_result, data_info)
        self.update_time[self.frame_id] = time.time() - st
        json.dump(self.update_time, open(TIME_COST_ROOT + 'update_time.json', "w"))
        # print(f'tracklet updating spend time: {time.time() - st}')
        
        # corner case, all valid tracklets dead at the current frame
        if len(dict_track_res['np_track_res']) == 0: 
            data_info.update({'no_val_track_result': True})
            return 
            
        # step5. update and output tracking results
        data_info.update({
            'np_track_res': dict_track_res['np_track_res'],
            'box_track_res': dict_track_res['box_track_res'],
            'bm_track_res': dict_track_res['bm_track_res'],
        })
        
        # whether to use post-predict to reduce FP prediction
        if self.post_nms_cfg['post_nms']: self.post_nms_tras(data_info)
        
        # only for debug
        # assert len(self.tentative_tras) == 0, "no tentative tracklet in the best performance version."

    def tras_predict(self) -> None:
        """
        State Prediction for all trajectories
        get self.tra_infos: {
            'np_tras': np.array, [valid_tra_num, 14]
            'np_tras_bottom_corners': np.array[NuscBox], [valid_tra_num,]
            'all_valid_ids': np.array, [valid_tra_num,]
            'all_valid_boxes': np.array[NuscBox], [valid_tra_num,]
            'tra_num': len(all_valid_ids)
        }
        """
        # debug, Check for tracklets with duplicate states
        if self.is_debug: self.debug()
        pred_infos, pred_bms, pred_norm_bms, pred_boxes, all_valid_ids = [], [], [], [], []

        # corner case(such as first frame..), no valid tracklets
        if len(self.valid_tras) == 0: return

        # predict tracklet for data association
        for tra_id, tra in self.valid_tras.items():
            # only for debug
            if self.is_debug: assert tra_id not in self.dead_tras

            # predict each valid tracklet state
            tra.state_predict(self.frame_id, self.is_key_frame)
            pred_object = tra[self.frame_id]
            
            all_valid_ids.append(tra_id)
            pred_boxes.append(pred_object.predict_box)
            pred_infos.append(pred_object.predict_infos)
            pred_bms.append(pred_object.predict_bms)
            pred_norm_bms.append(pred_object.predict_norm_bms)
            
        # only for debug
        if self.is_debug:
            self.valid_tras = self.merge_valid_tras()
            assert len(all_valid_ids) == len(self.valid_tras)
        self.tra_infos = {
            'np_tras': np.array(pred_infos),  # info dim: 17, add 'tracking_id', 'seq_id', 'frame_id'
            'np_tras_bottom_corners': np.array(pred_bms),
            'np_tras_norm_corners': np.array(pred_norm_bms),
            'all_valid_ids': np.array(all_valid_ids),
            'all_valid_boxes': np.array(pred_boxes),
            'tra_num': len(all_valid_ids)
        }

    def tras_punish(self, data_info: dict) -> None:
        """
        handle the corner case where there is no detection at current frame
        also can be seen as "short-cut"
        :param data_info: dict, output file
        :return: update pure predict state, up to output punish_num frame
        update data_info: {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
        }
        """
        # no valid tras after predicting(valid ids is empty) or no valid tras at prev frame(None)
        if self.tra_infos is None or self.tra_infos['tra_num'] == 0: 
            data_info.update({'no_val_track_result': True})
            return
        
        # manage trajectory
        temp_associate_result = {'mix_dets_num': 0, 'single_3d_dets_num': 0, 'm_tras_idx3': [], 'm_asso_1': {}, 'm_asso_2': {}}
        dict_track_res = self.tras_update(temp_associate_result, data_info)
        if len(dict_track_res['np_track_res']) == 0: 
            data_info.update({'no_val_track_result': True})
            return 
            
        # output punishment tracking results
        data_info.update({
            'np_track_res': dict_track_res['np_track_res'],
            'box_track_res': dict_track_res['box_track_res'],
            'bm_track_res': dict_track_res['bm_track_res'],
            'norm_bm_track_res': dict_track_res['norm_bm_track_res'],
        })


    def tras_update(self, associate_result: dict, data_info: dict) -> dict:
        """
        update the corresponding trajectories with observations, init new tras,
        and punish unmatched tracklets
        :param tracking_ids: np.array, tracking id of each detection
        :param data_info: dict, dets infos at the current frame
        :return: dict, valid estimated states(updated tras, new tras, valid unmatched tras)
            {
            'np_track_res': np.array, [valid_tra_num, 17],
            'box_track_res': np.array[NuscBox], [valid_tra_num,]
            'bm_track_res': np.array, [valid_tra_num, 4, 2]
            }
        """
        np_res, box_res, bm_res, norm_bm_res = [], [], [], []
        new_tras, ten_tras, act_tras = {}, {}, {}
        
        all_valid_ids, mix_matched_ids = list(self.valid_tras.keys()), []
        single_3d_matched_ids, single_2d_matched_ids = [], []
        m_asso_1list, m_asso_2list = list(associate_result['m_asso_1'].keys()), list(associate_result['m_asso_2'].keys())
        for det_idx in range(self.det_infos['mix_dets_num']):
            dict_det = {
                'nusc_box': data_info['mix_np_box_3d'][det_idx],
                'np_array': data_info['mix_np_dets_3d'][det_idx],
                'has_velo': data_info['has_velo'],
                'seq_id': data_info['seq_id'],
                'matched_2d_info': data_info['mix_dict'][tuple(data_info['mix_np_dets_3d'][det_idx])],
                'is_key_frame': self.is_key_frame,
                'camera_meta_infos': data_info['camera_meta_infos'],
                }
            if det_idx in m_asso_1list:
                tra_idx = associate_result['m_asso_1'][det_idx]
                tra_id = all_valid_ids[tra_idx]
                mix_matched_ids.append(tra_id)
                tra = self.valid_tras[tra_id]
                tra.state_update(timestamp=self.frame_id, det=dict_det)
            else:
                tra = Trajectory(timestamp=self.frame_id,
                                 config=self.cfg,
                                 track_id=self.id_seed,
                                 det_infos=dict_det)
                new_tras[self.id_seed] = tra
                self.id_seed += 1

        for det_idx in range(self.det_infos['single_3d_dets_num']):
            dict_det = {
                'nusc_box': data_info['single_np_box_3d'][det_idx],
                'np_array': data_info['single_np_dets_3d'][det_idx],
                'has_velo': data_info['has_velo'],
                'seq_id': data_info['seq_id'],
                'matched_2d_info': {},
                'is_key_frame': self.is_key_frame
                }
            if det_idx in m_asso_2list:
                tra_idx = associate_result['m_asso_2'][det_idx]
                tra_id = all_valid_ids[tra_idx]
                single_3d_matched_ids.append(tra_id)
                tra = self.valid_tras[tra_id]
                tra.state_update(timestamp=self.frame_id, det=dict_det)
            else:
                tra = Trajectory(timestamp=self.frame_id,
                                 config=self.cfg,
                                 track_id=self.id_seed,
                                 det_infos=dict_det)
                new_tras[self.id_seed] = tra
                self.id_seed += 1
        
        # P2DA / async matches: camera-only. Motion KF uses 2D obs with huge R (Eq.3, n=1).
        for tra_idx in associate_result['m_tras_idx3']:
            dict_det = {
                'nusc_box': None,
                'np_array': None,
                'has_velo': data_info['has_velo'],
                'seq_id': data_info['seq_id'],
                'matched_2d_info': associate_result['m_2d_det_info'][tra_idx],
                'is_key_frame': self.is_key_frame,
                'camera_meta_infos': data_info['camera_meta_infos'],
                }
            tra_id = all_valid_ids[tra_idx]
            single_2d_matched_ids.append(tra_id)
            tra = self.valid_tras[tra_id]
            tra.state_update(timestamp=self.frame_id, det=dict_det)

        if self.is_key_frame == False: assert len(mix_matched_ids) == 0 and len(single_3d_matched_ids) == 0
        for unmatch_id in set(all_valid_ids) - set(mix_matched_ids) - set(single_3d_matched_ids) - set(single_2d_matched_ids):
            tra = self.valid_tras[unmatch_id]
            tra.state_update(timestamp=self.frame_id, det=None)

        # merge all tras, include exist trajectories and newly generated trajectory
        tmp_merge_tras = {**self.valid_tras, **new_tras}
        
        # iterative trajectories, punish and output
        for tra_id, tra in tmp_merge_tras.items():
            update_object = tra[self.frame_id]
            # only active tracklets' state are output to the result file
            if tra.life_management.state == 'active':
                act_tras[tra_id] = tra
                if update_object.update_infos is not None:
                    np_res.append(update_object.update_infos)
                    box_res.append(update_object.update_box)
                    bm_res.append(update_object.update_bms)
                    norm_bm_res.append(update_object.update_norm_bms)
                elif tra.life_management.time_since_update <= self.punish_num:
                    np_res.append(update_object.predict_infos)
                    box_res.append(update_object.predict_box)
                    bm_res.append(update_object.predict_bms)
                    norm_bm_res.append(update_object.predict_norm_bms)
            elif tra.life_management.state == 'tentative':
                ten_tras[tra_id] = tra
            elif tra.life_management.state == 'dead':
                assert tra_id not in self.dead_tras
                self.dead_tras[tra_id] = tra
            else: raise Exception('Tracjectory state only have three attributes')
            
        # reorganize active/dead/tentative trajectories
        self.active_tras, self.tentative_tras = act_tras, ten_tras
        self.valid_tras = {**self.active_tras, **self.tentative_tras}
        
        dict_track_res = {
            'np_track_res': np_res,
            'box_track_res': box_res,
            'bm_track_res': bm_res,
            'norm_bm_track_res': norm_bm_res
        }
        return dict_track_res   

    def data_association(self) -> np.array:
        """
        Associate the track and the detection, and assign a tracking id to each detection
        :return: np.array, tracking ids of each detection
        """
        # corner case, no valid trajectory. directly return associate result.
        if len(self.valid_tras) == 0:
            associate_result = {'m_asso_1': {},
                                'm_asso_2': {},
                                'm_tras_idx3': [],
                                'm_2d_det_info': {},
                                'um_mix_idx': list(range(self.det_infos['mix_dets_num'])),
                                'um_3d_idx': list(range(self.det_infos['single_3d_dets_num'])),}
        else:
            # 1st association, associate mix detections.
            mix_np_dets_3d = self.det_infos['mix_np_dets_3d']
            if len(mix_np_dets_3d) > 0:
                mix_bottom_corners = self.det_infos['mix_boxes_bottom_corners_3d'] 
                mix_norm_corners = self.det_infos['mix_boxes_norm_corners_3d']
                mix_info = {'np_dets': mix_np_dets_3d, 
                            'det_num': len(mix_np_dets_3d),
                            'np_dets_bottom_corners': mix_bottom_corners,
                            'np_dets_norm_corners': mix_norm_corners,}
                cost_matrices = self.compute_cost(self.tra_infos, mix_info)
                m_dets1, m_tras1, um_dets1, um_tras1 = self.matching_cost(cost_matrices, self.first_thre)
            else: m_dets1, m_tras1, um_dets1, um_tras1 = [], [], [], np.arange(self.tra_infos['tra_num'], dtype=int)

            # 2nd association, associate single 3d detections.
            if self.assoc_stage >= 2:
                um_np_tras1 = self.tra_infos['np_tras'][um_tras1]
                um_tra_info1 = {'np_tras': um_np_tras1, 
                                'tra_num': len(um_np_tras1),
                                'np_tras_bottom_corners': self.tra_infos['np_tras_bottom_corners'][um_tras1],
                                'np_tras_norm_corners': self.tra_infos['np_tras_norm_corners'][um_tras1],}
                if self.det_infos['single_3d_dets_num'] > 0 and um_tra_info1['tra_num'] > 0:
                    np_dets_3d = self.det_infos['single_np_dets_3d']
                    only_3d_bottom_corners = self.det_infos['single_boxes_bottom_corners_3d']
                    only_3d_norm_corners = self.det_infos['single_boxes_norm_corners_3d']
                    only_3d_info = {'np_dets': np_dets_3d, 
                                    'det_num': len(np_dets_3d),
                                    'np_dets_bottom_corners': only_3d_bottom_corners,
                                    'np_dets_norm_corners': only_3d_norm_corners,}
                    cost_matrices2 = self.compute_cost(um_tra_info1, only_3d_info)
                    m_dets2, m_tras2, um_dets2, um_tras2 = self.matching_cost(cost_matrices2, self.second_thre)
                    origin_m_tras2 = um_tras1[m_tras2]
                elif self.det_infos['single_3d_dets_num'] > 0:
                    np_dets_3d = self.det_infos['single_np_dets_3d']
                    m_dets2, origin_m_tras2, um_dets2, um_tras2 = [], [], np.arange(len(np_dets_3d)).tolist(), np.zeros(0, dtype=int)
                else: 
                    m_dets2, origin_m_tras2, um_dets2, um_tras2 = [], [], [], np.arange(len(um_np_tras1), dtype=int)
            else:
                m_dets2, origin_m_tras2, um_dets2 = [], [], list(range(self.det_infos['single_3d_dets_num']))
                um_np_tras1 = self.tra_infos['np_tras'][um_tras1]
                um_tras2 = np.arange(len(um_np_tras1), dtype=int)

            # 3rd association, associate single 2d detections.
            if self.assoc_stage >= 3:
                um_np_tras2 = um_np_tras1[um_tras2]
                dets_2d_info = self.det_infos['single_2d_info']
                m_tras3, match_dict = self.predict_2d_association(um_np_tras2[:, :-3], dets_2d_info)
                sorted_list = [match_dict[_] for _ in m_tras3]
                origin_m_tras3 = um_tras1[um_tras2[m_tras3]]
                origin_match_dict = {key: value for key, value in zip(origin_m_tras3, sorted_list)}
            else:
                origin_m_tras3 = np.array([], dtype=int)
                origin_match_dict = {}

            all_match_idx = np.concatenate((m_tras1, origin_m_tras2, origin_m_tras3))
            assert len(all_match_idx) == len(np.unique(all_match_idx)), "Duplicate match tarjectory index"
            associate_result = {'m_asso_1': {_[0]: _[1] for _ in zip(m_dets1, m_tras1)},
                                'm_asso_2': {_[0]: _[1] for _ in zip(m_dets2, origin_m_tras2)},
                                'm_tras_idx3': origin_m_tras3,
                                'm_2d_det_info': origin_match_dict,
                                'um_mix_idx': um_dets1,
                                'um_3d_idx': um_dets2,}

        return associate_result

    def compute_cost(self, tra_infos: dict, det_infos: dict) -> dict:
        """
        Construct the cost matrix between the trajectory and the detection
        :return: dict, a collection of cost matrices,
        one-stage: np.array, [cls_num, det_num, tra_num], two-stage: np.array, [det_num, tra_num]
        """
        assert tra_infos is not None and tra_infos['tra_num'] is not None
        det_num, tra_num = det_infos['det_num'], tra_infos['tra_num']
        det_labels, tra_labels = det_infos['np_dets'][:, -1], tra_infos['np_tras'][:, -4]

        # [det_num, tra_num], True denotes valid (in the same voxel)
        if self.use_voxel_mask:
            if not isinstance(self.voxel_mask_size, dict):
                fast_voxel_2d_mask = voxel_mask(det_infos['np_dets'], tra_infos['np_tras'][:, :-3], thre=self.voxel_mask_size)
                fast_voxel_3d_mask = fast_voxel_2d_mask[None, :, :].repeat(self.cls_num, axis=0)
            else:
                fast_voxel_3d_mask = np.array([voxel_mask(det_infos['np_dets'],
                                                          tra_infos['np_tras'][:, :-3],
                                                          thre=self.voxel_mask_size[cls_idx])
                                               for cls_idx in range(self.cls_num)])
                fast_voxel_2d_mask = np.max(fast_voxel_3d_mask, axis=0)
        else:
            fast_voxel_2d_mask = np.ones((det_num, tra_num), dtype=bool)
            fast_voxel_3d_mask = np.ones((self.cls_num, det_num, tra_num), dtype=bool)

        # [cls_num, det_num, tra_num], True denotes valid (det label == tra label == cls idx)
        cls_3d_mask, cls_2d_mask = mask_tras_dets(self.cls_num, det_labels, tra_labels)
        # [cls_num, det_num, tra_num], final valid mask, True denotes valid
        valid_3d_mask = np.logical_and(fast_voxel_3d_mask, cls_3d_mask)

        # Construct valid det/tra infos
        tra_cost_infos = {'np_dets': tra_infos['np_tras'][:, :-3],
                          'np_dets_bottom_corners': tra_infos['np_tras_bottom_corners'],
                          'np_dets_norm_corners': tra_infos['np_tras_norm_corners']}
        det_cost_infos = {'np_dets': det_infos['np_dets'],
                          'np_dets_bottom_corners': det_infos['np_dets_bottom_corners'],
                          'np_dets_norm_corners': det_infos['np_dets_norm_corners']}

        first_cost = np.zeros((self.cls_num, det_num, tra_num))
        for metric, cls_list in self.re_metrics.items():
            # True denotes invalid(the object's category is not specific)
            tra_cost_infos['mask'] = spec_metric_mask(cls_list, det_labels, tra_labels)
            if metric in METRIC:
                _, cost1 = globals()[metric](det_cost_infos, tra_cost_infos)
            else:
                cost1 = globals()[metric](det_cost_infos, tra_cost_infos)
            first_cost[cls_list] = cost1

        # mask invalid value
        first_cost[np.where(~valid_3d_mask)] = -np.inf

        # Due to the execution speed of python,
        # construct the two-stage cost matrix under half-parallel framework is very tricky, 
        # we strongly recommend only use giou_bev as two-stage metric to build the cost matrix
        return {'one_stage': 1 - first_cost}

    def matching_cost(self, cost_matrices: dict, match_thre: dict) -> np.array:
        """
        Solve the matching pair according to the cost matrix
        :param cost_matrices: cost matrices between dets and tras construct in the one/two stage
        :return: np.array, tracking id of each detection
        """
        cost1 = cost_matrices['one_stage']
        # m_tras_1 is not the tracking id, but is the index of tracklet in the all valid trajectories
        m_dets_1, m_tras_1, um_dets_1, um_tras_1 = globals()[self.algorithm](cost1, match_thre)

        assert len(m_dets_1) == len(m_tras_1), "as the pair, number of the matched tras and dets must be equal"

        return m_dets_1, m_tras_1, um_dets_1, um_tras_1

    def predict_2d_association(self, np_dets: np.array, det_2d_info: dict) -> np.array:
        assert np_dets.shape[1] == 14, "np_dets must be [x, y, z, h, w, l, ry, score, cls, frame_id, track_id, seq_id, cam_id, obj_id]"
        det_3d_num = len(np_dets)
        matched_dict, unique_list = {}, np.array([], dtype=int)
        if det_3d_num != 0:
            panoramic_view_2D_cam_box = det_2d_info
            all_match_index_3D = []
            
            # project 3D to 2D
            trans_infos = self.det_infos['trans_infos']
            det_vis_infos = fast_project_dets_to_images(trans_infos, np_dets)
            
            # Each camera loops separately
            for c_idx, cam_name in enumerate(USED_ALLCAM):
                det_2d_num = len(panoramic_view_2D_cam_box[cam_name])
                if det_2d_num == 0:     # coner case: no object in this cam
                    continue
                
                frame_data2D_cam_box = np.array(panoramic_view_2D_cam_box[cam_name])
                
                # cam_box -> (x1, y1, x2, y2, det_score, class_label)
                frame_data3D_cam_box = np.zeros((len(np_dets),6))
                frame_data3D_cam_box[:, 0:4] = det_vis_infos['np_dets_bbox2d'][c_idx, :]
                frame_data3D_cam_box[:, 4] = copy.deepcopy(np_dets[:,-2])
                frame_data3D_cam_box[:, 5] = copy.deepcopy(np_dets[:,-1])

                fusion_thre = self.cfg['association']['third_thre'] if self.is_key_frame else self.cfg['association']['high_thre']
                tmp_config = {'basic': {'CLASS_NUM': self.cls_num}, 'fusion': {'fusion_algorithm': self.cfg['association']['algorithm'], 
                                                                 'fusion_first_thre': fusion_thre, 
                                                                 'fusion_metrics': self.cfg['association']['third_metric']}}

                match_index1, match_index2, _ = datafusion(frame_data3D_cam_box, frame_data2D_cam_box, tmp_config).data_association()
                all_match_index_3D += match_index1
                for index_3d, index_2d in zip(match_index1, match_index2):
                    if index_3d not in matched_dict:
                        matched_dict[index_3d] = {cam_name: frame_data2D_cam_box[index_2d]}
                    else:
                        matched_dict[index_3d].update({cam_name: frame_data2D_cam_box[index_2d]})
            
            unique_list = np.unique(all_match_index_3D).astype(int)

        return unique_list, matched_dict

    def merge_valid_tras(self) -> dict:
        """
        Get all valid trajectories, 'valid' denotes that 'active' and 'tentative'
        :return: dict, merge active tracklets and tentative tracklets
        """
        return {**self.active_tras, **self.tentative_tras}

    def post_nms_tras(self, data_info) -> None:
        """
        use post-predict to reduce FP prediction
        :param data_info: the final tracking result at each frame
        :retrun: no return, but filter FP results in the data_info
        """
        if 'no_val_track_result' in data_info: return
        post_metric = self.post_nms_cfg['NMS_metric']
        post_thre = self.post_nms_cfg['NMS_thre']
        post_type = self.post_nms_cfg['NMS_type']
        tmp_tra_infos = {'np_dets': np.array(data_info['np_track_res'])[:, :-3],
                         'np_dets_bottom_corners': np.array(data_info['bm_track_res']),
                         'np_dets_norm_corners': np.array(data_info['norm_bm_track_res']),}
        keep = globals()[post_type](box_infos=tmp_tra_infos, metrics=post_metric, thre=post_thre)
        if len(keep) == 0:
            data_info.update({'no_val_track_result': True})
        else:
            data_info['np_track_res'] = np.array(data_info['np_track_res'])[keep].tolist()
            data_info['box_track_res'] = np.array(data_info['box_track_res'])[keep].tolist()

    def debug(self) -> None:
        """
        only for debug, check whether the trajectory status is repeated
        TODO: delete at public version
        """
        assert len(self.tentative_tras.keys() & self.active_tras.keys()) == 0
        assert len(self.active_tras.keys() & self.dead_tras.keys()) == 0
        assert len(self.tentative_tras.keys() & self.dead_tras.keys()) == 0
