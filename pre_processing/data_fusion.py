import numpy as np
from utils.script import mask_tras_dets
from geometry.nusc_distance import giou_2d,iou_2d
from utils.matching import Hungarian

class datafusion:
    def __init__(self, frame_data3D_cam_box: dict, frame_data2D_cam_box: dict, config: dict) -> None:
        # load data, [det_num/tra_num, 6], (x1, y1, x2, y2, det_score, class_label)
        self.cam_3D_infos = frame_data3D_cam_box
        self.cam_2D_infos = frame_data2D_cam_box
        # load config
        self.config = config
        self.cls_num = config['basic']['CLASS_NUM']
        self.algorithm = config['fusion']['fusion_algorithm']
        self.first_thre = config['fusion']['fusion_first_thre']
        self.cost_metric = config['fusion']['fusion_metrics']

    def data_association(self) -> np.array:
        """
        Associate the track and the detection, and assign a tracking id to each detection
        :return: np.array, tracking ids of each detection
        """
        # corner case, no valid trajectory. quickly assign each det tracking id
        if len(self.cam_3D_infos) == 0 or len(self.cam_2D_infos) == 0:
            return [], [], list(range(len(self.cam_2D_infos)))
        else:
            cost_matrices = self.compute_cost()
            match_index_3d, match_index_2d, unmatch_index_2d = self.matching_cost(cost_matrices)
            
        return match_index_3d, match_index_2d, unmatch_index_2d

    def compute_cost(self) -> dict:
        """
        Construct the cost matrix between the trajectory and the detection
        :return: dict, a collection of cost matrices,
        one-stage: np.array, [cls_num, det_num, tra_num], two-stage: np.array, [det_num, tra_num]
        """
        
        # cam_3D_infos, cam_2D_infos, [det_num/tra_num, 6]
        cam_3D_labels, cam_2D_labels = self.cam_3D_infos[:, -1], self.cam_2D_infos[:, -1]

        # [cls_num, det_num, tra_num], True denotes valid (det label == tra label == cls idx)
        valid_mask, _ = mask_tras_dets(self.cls_num, cam_3D_labels, cam_2D_labels)
        
        # construct cost matrix, [cls_num, det_num, tra_num]
        first_cost, _ = globals()[self.cost_metric](self.cam_3D_infos, self.cam_2D_infos)
        first_cost = first_cost[None, :, :].repeat(self.cls_num, axis=0)

        # mask invalid value
        first_cost[np.where(~valid_mask)] = -np.inf

        # Due to the execution speed of python,
        # construct the two-stage cost matrix under half-parallel framework is very tricky, 
        # we strongly recommend only use giou_bev as two-stage metric to build the cost matrix
        
        return 1 - first_cost

    def matching_cost(self, cost_matrices: np.ndarray) -> np.array:
        """
        Solve the matching pair according to the cost matrix
        :param cost_matrices: cost matrices between dets and tras construct in the one/two stage
        :return: np.array, tracking id of each detection
        """
        cost1 = cost_matrices
        # m_tras_1 is not the tracking id, but is the index of tracklet in the all valid trajectories
        m_det, m_tra, um_det, um_tra = globals()[self.algorithm](cost1, self.first_thre)
        
        assert len(m_det) == len(m_tra), "as the pair, number of the matched tras and dets must be equal"
        
        return m_det, m_tra, um_tra
