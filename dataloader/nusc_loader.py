"""
dataloader of NuScenes dataset
Obtain the observation information(detection) of each frame iteratively
--------ATTENTION: Detector files must be in chronological order-------
"""

import time, json, os, copy
import numpy as np
from utils.io import load_file
from nuscenes.nuscenes import NuScenes
from data.script.NUSC_CONSTANT import *
from pre_processing import dictdet2array, arraydet2box, blend_nms, scale_nms
from pre_processing.data_fusion import datafusion
from geometry.nusc_utils import get_nusc_cam_meta_infos, fast_project_dets_to_images, concatenate_cam_infos
from geometry.nusc_location_coordinator import NuscLocationCoordinator


class NuScenesloader:
    def __init__(self, detection_3d_path, detection_2d_path, first_token_path, root_path, config):
        """
        :param detection_3d_path: path of order 3d detection file
        :param detection_2d_path: path of 2d detection file
        :param first_token_path: path of first frame token for each seq
        :param root_path: path of datasets
        :param config: dict, hyperparameter setting
        """
        # detector -> {sample_token:[{det1_info}, {det2_info}], ...}， check the detailed "det_info" at nuscenes.org
        self.detector_3d = load_file(detection_3d_path)["results"]
        self.detector_2d = load_file(detection_2d_path)
        self.nusc = self.load_dataset(root_path, config['basic']['split'])
        self.all_sample_token = list(self.detector_2d.keys())
        self.seq_first_token = load_file(first_token_path)
        self.config, self.data_info = config, {}
        self.fake_SF_thre = config['fusion']['fake_SF_thre']
        self.use_coordinator = config['fusion']['use_coordinator']
        if self.use_coordinator:
            self.res_form = config['fusion']['res_form']
            self.use_weight = config['fusion']['use_weight']
            self.multi_view = config['fusion']['multi_view']
            self.size_prior_weight = config['fusion'].get('size_prior_weight', 1e4)
            self.location_coordinator = NuscLocationCoordinator(
                res_form=self.res_form,
                use_weight=self.use_weight,
                multi_view=self.multi_view,
                size_prior_weight=self.size_prior_weight,
            )
        
        self.SF_thre_3d, self.NMS_thre = config['preprocessing']['SF_thre_3d'], config['preprocessing']['NMS_thre']
        self.NMS_type, self.NMS_metric = config['preprocessing']['NMS_type'], config['preprocessing']['NMS_metric']
        self.SCALE = self.config['preprocessing']['SCALE'] if self.NMS_type == 'scale_nms' else None
        self.seq_id = self.frame_id = 0


        # only for ablation
        self.asso = config.get('ablation', {}).get('asso', False)
        self.nms_voxel_mask = config['ablation']['voxel_mask']
        self.voxel_mask_size = config['preprocessing']['voxel_mask_size']

        # only for debug
        os.makedirs(TIME_COST_ROOT, exist_ok=True)
        self.pre_time = {}

    @staticmethod
    def load_dataset(root_path: str, split: str) -> NuScenes:
        """load NuScenes dataset

        Args:
            root_path (str): root path of database
            split (str): split of nuscenes

        Returns:
            NuScenes: instance of nuscenes
        """
        assert split in ['val', 'test'], "set split must be val or test."
        version = 'v1.0-{}'.format('trainval' if split == 'val' else split)
        
        # load dataset
        nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
        return nusc

    def __getitem__(self, item) -> dict:
        """
        data_info(dict): {
            'is_first_frame': bool
            'timestamp': int
            'sample_token': str
            'seq_id': int
            'frame_id': int
            'has_velo': bool
            'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
            'np_dets_bottom_corners': np.array, [det_num, 4, 2]
            'box_dets': np.array[NuscBox], [det_num]
            'no_dets': bool, corner case,
            'det_num': int,
        }
        """
        st = time.time()
        curr_token = self.all_sample_token[item]
        
        # non-key frame, only return 2d infos
        if curr_token not in self.detector_3d.keys():
            assert curr_token.split('_')[0] in self.detector_3d.keys()
            # if don't use high rate detection, don't change the frame_id
            if self.config['basic']['freq'] == 'high': self.frame_id += 1
            single_2d_info = {cam_name: _['np_boxes'] for cam_name, _ in self.detector_2d[curr_token].items()}
            sample_data_tokens = {cam_name: _['sample_data_token'] for cam_name, _ in self.detector_2d[curr_token].items()}
            if self.asso:
                single_2d_info = {cam_name: [] for cam_name in USED_ALLCAM}
            det_2d_num = sum([len(single_2d_info[cam_name]) for cam_name in USED_ALLCAM])
            camera_meta_infos = get_nusc_cam_meta_infos(self.nusc, sample_data_tokens=sample_data_tokens)
            trans_infos = concatenate_cam_infos(camera_meta_infos)
            print(f"sample token {curr_token} not in 3d detector.")
            return {'is_first_frame': curr_token in self.seq_first_token,
                    'timestamp': item,
                    'sample_token': curr_token,
                    'seq_id': self.seq_id,
                    'frame_id': self.frame_id,
                    'has_velo': self.config['basic']['has_velo'],
                    'is_key_frame': False,
                    'camera_meta_infos': camera_meta_infos,
                    'trans_infos': trans_infos,

                    'mix_dict': {},
                    'mix_np_dets_3d': np.zeros(0),
                    'mix_np_box_3d': np.zeros(0),
                    'mix_boxes_bottom_corners_3d': np.zeros(0),
                    'mix_boxes_norm_corners_3d': np.zeros(0),
                    'single_np_dets_3d': np.zeros(0),
                    'single_np_box_3d': np.zeros(0),
                    'single_boxes_bottom_corners_3d': np.zeros(0),
                    'single_boxes_norm_corners_3d': np.zeros(0),
                    'single_2d_info': single_2d_info,
                    'mix_dets_num': 0,
                    'single_3d_dets_num': 0,
                    'single_2d_dets_num': det_2d_num,
                    'sample_data_tokens': sample_data_tokens,
                    'no_dets': det_2d_num == 0}
        
        # assign seq and frame id
        if curr_token in self.seq_first_token:
            self.seq_id += 1
            self.frame_id = 1
        else: self.frame_id += 1

        ori_dets = self.detector_3d[curr_token]

        # all categories are blended together and sorted by detection score
        list_dets, np_dets = dictdet2array(ori_dets, 'translation', 'size', 'velocity', 'rotation',
                                           'detection_score', 'detection_name')
        # process raw 3D dets
        det_3d_num = len(np_dets)
        matched_dict, Inv_unique_list, unmatched_2d_info = {}, [], {_: [] for _ in USED_ALLCAM}
        camera_meta_infos = get_nusc_cam_meta_infos(self.nusc, sample_token=curr_token)
        trans_infos = concatenate_cam_infos(camera_meta_infos)
        
        if det_3d_num != 0:
            panoramic_view_2D_cam_box = self.detector_2d[curr_token]
            all_match_index_3D, all_match_index_2Dlist, all_match_index_3Dlist = [], [], []
            
            # project 3D to 2D
            det_vis_infos = fast_project_dets_to_images(trans_infos, np_dets)
            
            # Each camera loops separately
            for c_idx, cam_name in enumerate(USED_ALLCAM):
                det_2d_num = len(panoramic_view_2D_cam_box[cam_name])
                if det_2d_num == 0:     # coner case: no object in this cam
                    all_match_index_3Dlist.append([])
                    all_match_index_2Dlist.append([])
                    continue
                
                frame_data2D_cam_box = np.array(panoramic_view_2D_cam_box[cam_name])
                
                # cam_box -> (x1, y1, x2, y2, det_score, class_label)
                frame_data3D_cam_box = np.zeros((len(np_dets),6))
                frame_data3D_cam_box[:, 0:4] = det_vis_infos['np_dets_bbox2d'][c_idx, :]
                frame_data3D_cam_box[:, 4] = copy.deepcopy(np_dets[:,-2])
                frame_data3D_cam_box[:, 5] = copy.deepcopy(np_dets[:,-1])
                
                match_index3d, match_index2d, unmatch_index2d = datafusion(frame_data3D_cam_box, frame_data2D_cam_box, self.config).data_association()
                all_match_index_3D += match_index3d
                all_match_index_3Dlist.append(match_index3d)
                all_match_index_2Dlist.append(match_index2d)
                unmatched_2d_info[cam_name] = frame_data2D_cam_box[unmatch_index2d].copy()

                for index_3d, index_2d in zip(match_index3d, match_index2d):
                    tuple_box = tuple(np_dets[index_3d])
                    if tuple_box not in matched_dict.keys():
                        matched_dict[tuple_box] = {cam_name: frame_data2D_cam_box[index_2d]}
                    else:
                        matched_dict[tuple_box].update({cam_name: frame_data2D_cam_box[index_2d]})

            # coordinator
            if self.use_coordinator:
                compensate_matched_dict = self.location_coordinator.coordinator_s(matched_dict=matched_dict,
                                                                                camera_meta_infos=camera_meta_infos)
                matched_dict = compensate_matched_dict
                compensate_np_dets = np.array([det for det in compensate_matched_dict.keys()])
            else:
                compensate_np_dets = np.array([det for det in matched_dict.keys()])

            unique_list = np.unique(all_match_index_3D)
            # Inv_unique_list is the index of the detection that is not matched, 
            Inv_unique_list = [i for i in range(det_3d_num) if i not in unique_list]
            assert det_3d_num == len(Inv_unique_list) + len(unique_list)

            if len(unique_list) != 0: 
                assert len(compensate_np_dets) == len(unique_list)
                np_dets_insec = compensate_np_dets
            else: np_dets_insec = np.zeros(0) # no object matched
            
        # recall the  high score 3d-only detections
        list_dets = np_dets[Inv_unique_list].tolist()
        # Score Filter based on category-specific thresholds
        np_dets = np.array([det for det in list_dets if det[-2] > self.SF_thre_3d[det[-1]]])

        # NMS, "blend" ref to blend all categories together during NMS
        if len(np_dets) != 0:
            box_dets, np_dets_bottom_corners, np_dets_norm_corners = arraydet2box(np_dets)
            assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners) == len(np_dets_norm_corners)
            tmp_infos = {'np_dets': np_dets, 'np_dets_bottom_corners': np_dets_bottom_corners,
                         'np_dets_norm_corners': np_dets_norm_corners, 'box_dets':box_dets}
            if self.NMS_type != 'scale_nms':
                keep = globals()[self.NMS_type](box_infos=tmp_infos,
                                                metrics=self.NMS_metric,
                                                thre=self.NMS_thre,
                                                voxel_mask_size=self.voxel_mask_size,
                                                use_voxel_mask=self.nms_voxel_mask)
            else:
                keep = scale_nms(box_infos=tmp_infos,
                                 metrics=self.NMS_metric,
                                 thres=self.NMS_thre,
                                 factors=self.SCALE,
                                 voxel_mask_size=self.voxel_mask_size,
                                 use_voxel_mask=self.nms_voxel_mask)
            keep_num = len(keep)
        
        else:
            keep_num, keep = 0, []  # corner case, no det_3d left

        print(f"\n seq id {self.seq_id}, frame id {self.frame_id}, "
              f"Total frame id {item + 1}.")

        np_dets_3d = np_dets[keep] if keep_num != 0 else np.zeros(0)
        box_dets_3d, boxes_bottom_corners_3d, boxes_norm_corners_3d = arraydet2box(np_dets_3d) if keep_num != 0 else (np.zeros(0), np.zeros(0), np.zeros(0))
        mix_box_dets_3d, mix_boxes_bottom_corners_3d, mix_boxes_norm_corners_3d = arraydet2box(np_dets_insec) if len(np_dets_insec) != 0 else (np.zeros(0), np.zeros(0), np.zeros(0))

        if self.asso:
            if len(np_dets_3d) != 0:
                for det in np_dets_3d:
                    matched_dict[tuple(det)] = {}
                if len(np_dets_insec) != 0:
                    np_dets_insec = np.concatenate([np_dets_insec, np_dets_3d], axis=0)
                    mix_box_dets_3d = np.concatenate([mix_box_dets_3d, box_dets_3d], axis=0)
                    mix_boxes_bottom_corners_3d = np.concatenate([mix_boxes_bottom_corners_3d, boxes_bottom_corners_3d], axis=0)
                    mix_boxes_norm_corners_3d = np.concatenate([mix_boxes_norm_corners_3d, boxes_norm_corners_3d], axis=0)
                else:
                    np_dets_insec = np_dets_3d
                    mix_box_dets_3d = box_dets_3d
                    mix_boxes_bottom_corners_3d = boxes_bottom_corners_3d
                    mix_boxes_norm_corners_3d = boxes_norm_corners_3d
            np_dets_3d = np.zeros(0)
            box_dets_3d = np.zeros(0)
            boxes_bottom_corners_3d = np.zeros(0)
            boxes_norm_corners_3d = np.zeros(0)
            unmatched_2d_info = {cam_name: [] for cam_name in USED_ALLCAM}

        unmatched_2d_num = sum([len(unmatched_2d_info[cam_name]) for cam_name in USED_ALLCAM])
        
        # Available information for the current frame
        data_info = {
            # sample infos
            'is_first_frame': curr_token in self.seq_first_token,
            'timestamp': item,
            'sample_token': curr_token,
            'seq_id': self.seq_id,
            'frame_id': self.frame_id,
            'has_velo': self.config['basic']['has_velo'],
            'is_key_frame': True,
            'camera_meta_infos': camera_meta_infos,
            'trans_infos': trans_infos,

            # detection infos
            'mix_dict': matched_dict,
            'mix_np_dets_3d': np_dets_insec,
            'mix_np_box_3d': mix_box_dets_3d,
            'mix_boxes_bottom_corners_3d': mix_boxes_bottom_corners_3d,
            'mix_boxes_norm_corners_3d': mix_boxes_norm_corners_3d,
            'single_np_dets_3d': np_dets_3d,
            'single_np_box_3d': box_dets_3d,
            'single_boxes_bottom_corners_3d': boxes_bottom_corners_3d,
            'single_boxes_norm_corners_3d': boxes_norm_corners_3d,
            'single_2d_info': unmatched_2d_info,
            'mix_dets_num': len(np_dets_insec),
            'single_3d_dets_num': len(np_dets_3d),
            'single_2d_dets_num': unmatched_2d_num,
            'no_dets': len(np_dets_3d) + len(np_dets_insec) + unmatched_2d_num == 0,
        }

        self.pre_time[self.frame_id] = time.time() - st
        json.dump(self.pre_time, open(TIME_COST_ROOT + 'pre_time.json', "w"))
        return data_info

    def __len__(self) -> int:
        return len(self.all_sample_token)