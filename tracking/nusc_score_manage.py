"""
assign tracklet confidence score 
predict, update, punish tracklet score under category-specific way.
"""
from utils.math import ray_Dis
from data.script.NUSC_CONSTANT import *
from motion_module.nusc_object import FrameObject

class ScoreObject:
    def __init__(self) -> None:
        self.raw_score = self.final_score = None
        self.predict_score = self.update_score = None
        
    def __repr__(self) -> str:
        repr_str = 'Raw score: {}, Predict score: {}, Update score: {}, Final score: {}.'
        return repr_str.format(self.raw_score, self.predict_score, self.update_score, self.final_score)

class ScoreManagement:
    def __init__(self, timestamp: int, cfg: dict, cls_label: int, det_infos: dict) -> None:
        self.initstamp, self.cfg, self.frame_objects, self.trk_avg_score = timestamp, cfg['life_cycle'], {}, None
        self.dr, self.predict_mode = self.cfg['basic']['decay_rate'][cls_label], self.cfg['score']['predict_mode']
        self.score_dc, self.update_mode = self.cfg['score']['score_decay'][cls_label], self.cfg['score']['update_mode']
        self.tp_thre_sq = self.cfg['score']['tp_thre'][cls_label] ** 2
        self.confidence_ratio = self.cfg['score']['confidence_ratio'][cls_label]
        self.multi_score = self.cfg['score']['multi_score']
        self.high_score_ratio = self.cfg['high']['high_score_ratio'][cls_label]
        self.high_dr = self.cfg['high']['high_decay_rate'][cls_label]
        assert self.predict_mode in SCORE_PREDICT and self.update_mode in SCORE_UPDATE
        self.initialize(det_infos)
    
    def initialize(self, det_infos: dict) -> None:
        """init tracklet confidence score, no inplace ops needed

        Args:
            det_infos (dict): dict, detection infos under different data format
            {
                'nusc_box': NuscBox,
                'np_array': np.array,
                'has_velo': bool, whether the detetor has velocity info
            }
        """
        score_obj = ScoreObject()
        if self.multi_score == 'max':
            score_2d = max([_[-2] for _ in det_infos['matched_2d_info'].values()]) if det_infos['matched_2d_info'] else 0
        elif self.multi_score == 'mean':
            score_2d = sum([_[-2] for _ in det_infos['matched_2d_info'].values()]) / len(det_infos['matched_2d_info']) if det_infos['matched_2d_info'] else 0
        else:
            raise Exception("unsupport multi score function")
        fusion_score = self.confidence_ratio * det_infos['nusc_box'].score + (1 - self.confidence_ratio) * score_2d
        score_obj.raw_score = score_obj.final_score = fusion_score
        score_obj.predict_score = score_obj.update_score = fusion_score
        self.frame_objects[self.initstamp] = score_obj

        # calu tracklet average score
        self.trk_avg_score = self.calu_trk_avg_score()
    
    def predict(self, timestamp: int, is_key_frame, pred_obj: FrameObject = None) -> None:
        """decay tracklet confidence score, change score in the predict infos inplace.

        Args:
            timestamp (int): current frame id
            pred_obj (FrameObject): nusc box/infos predicted by the filter
        """
        score_obj = ScoreObject()
        score_obj.raw_score = prev_score = self.frame_objects[timestamp - 1].final_score

        # Paper Eq.(2): s_{t,t-1} = sigma_tau * s_{t-1}, tau in {sync, async}.
        if not is_key_frame:
            score_obj.predict_score = prev_score * self.high_dr  # sigma_async = high_decay_rate
        elif self.predict_mode == 'Normal':
            # Consider cov convergence time
            score_obj.predict_score = prev_score * self.dr  # sigma_sync = decay_rate
        elif self.predict_mode == 'Minus':
            score_obj.predict_score = max(prev_score - self.score_dc, 0)
        elif self.predict_mode == 'Prob':
            score_obj.predict_score = ray_Dis(self.tp_thre_sq, pred_obj.predict_cov)
        else:
            raise Exception("unsupport score predict function")
        self.frame_objects[timestamp] = score_obj
        
        # assign tracklet score inplace
        pred_obj.predict_box.score = pred_obj.predict_infos[-5] = max(score_obj.predict_score, 0)
        
        
    def update(self, timestamp: int, update_obj: FrameObject, raw_det: dict = None) -> None:
        """Update trajectory confidence scores inplace directly using matched det

        Args:
            timestamp (int): current frame id
            update_obj (FrameObject): nusc box/infos updated by the filter
            raw_det (dict, optional): same as data format in the init function. Defaults to None.
        """
        score_obj = self.frame_objects[timestamp]
        if raw_det is None:
            score_obj.final_score = score_obj.predict_score
            # calu tracklet average score
            self.trk_avg_score = self.calu_trk_avg_score()
            return
        if self.multi_score == 'max':
            score_2d = max([_[-2] for _ in raw_det['matched_2d_info'].values()]) if raw_det['matched_2d_info'] else 0
        elif self.multi_score == 'mean':
            score_2d = sum([_[-2] for _ in raw_det['matched_2d_info'].values()]) / len(raw_det['matched_2d_info']) if raw_det['matched_2d_info'] else 0
        else:
            raise Exception("unsupport multi score function")
        nusc_box_3d_score = raw_det['nusc_box'].score if raw_det['nusc_box'] is not None else 0
        fusion_score = self.confidence_ratio * nusc_box_3d_score + (1 - self.confidence_ratio) * score_2d

        # Score update: sync uses Eq.(4)(5); async uses Eq.(6) with beta = high_score_ratio.
        if not raw_det['is_key_frame']:
            temp_score = max([_[-2] for _ in raw_det['matched_2d_info'].values()])
            # st = 1 - (1 - s_{t,t-1}) * (1 - beta * s_single)
            update_score =  1 - (1 - self.high_score_ratio * temp_score) * (1 - score_obj.predict_score)
        elif self.update_mode == 'Normal':
            # Consider cov convergence time
            update_score = raw_det['nusc_box'].score
        elif self.update_mode == 'Multi':
            # st = 1 - (1 - s_{t,t-1}) * (1 - s_fused), s_fused from Eq.(4)
            update_score = 1 - (1 - fusion_score) * (1 - score_obj.predict_score)
        elif self.update_mode == 'Multi_Model':
            update_score = 1 - (1 - nusc_box_3d_score) * (1 - score_obj.predict_score) * (1 - score_2d)
        elif self.update_mode == 'Parallel':
            update_score = (1 - (1 - raw_det['nusc_box'].score) * (1 - score_obj.predict_score) /
                            (2 - raw_det['nusc_box'].score - score_obj.predict_score))
        elif self.update_mode == 'Prob':
            update_score = ray_Dis(self.tp_thre_sq, update_obj.update_cov)
        elif self.update_mode == 'EMA':
            alpha = self.confidence_ratio
            update_score = alpha * fusion_score + (1 - alpha) * score_obj.predict_score
        elif self.update_mode == 'Max':
            update_score = max(fusion_score, score_obj.predict_score)
        elif self.update_mode == 'WeightedAvg':
            update_score = 0.5 * fusion_score + 0.5 * score_obj.predict_score
        else:
            raise Exception("unsupport score update function")

        # assign score objects and output scores
        score_obj.update_score = score_obj.final_score = update_score
        
        if raw_det['nusc_box'] is not None:
            update_obj.update_box.score = update_obj.update_infos[-5] = max(update_score, 0)
        else:
            update_obj.predict_box.score = update_obj.predict_infos[-5] = max(update_score, 0)

        # calu tracklet average score
        self.trk_avg_score = self.calu_trk_avg_score()

    def calu_trk_avg_score(self) -> float:
        return sum([score_obj.final_score for _, score_obj in self.frame_objects.items()]) / len(self.frame_objects)

    def __getitem__(self, item) -> ScoreObject:
        return self.frame_objects[item]

    def __len__(self) -> int:
        return len(self.frame_objects)
        
        
    
    