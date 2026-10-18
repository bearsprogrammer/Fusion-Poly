"""Prediction injection, full assignment competition, and source guards."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from ctmot.proto.association_replay import replace_prediction, verify_trace_step, load_config
from motion_module.motion_model import CTRV
from motion_module.nusc_object import FrameObject
from pre_processing import arraydet2box
from utils.matching import Hungarian


class MotionStub:
    def __init__(self):
        self.state=np.mat([0., 0., 1., 2., 4., 1.6, 5., .3, .01]).T
        self.P=np.mat(np.diag(np.arange(1.,10.)))
        self.model=CTRV(True,False,False,.25)
        frame=FrameObject()
        info=self.getOutputInfo(self.state)
        boxes,bottoms,normals=arraydet2box(info,np.array([102]),True)
        frame.predict_infos=np.r_[info,[102,1,41]]
        frame.predict_box=boxes[0];frame.predict_bms=bottoms[0];frame.predict_norm_bms=normals[0]
        frame.predict_state=self.state;frame.predict_cov=self.P[:2,:2]
        self.frame=frame

    def getOutputInfo(self,state):
        return np.r_[self.model.getOutputInfo(state), [.91,2.]]

    def __getitem__(self,key):
        return self.frame


def tracker_fixture():
    motion=MotionStub(); f=motion.frame
    return SimpleNamespace(frame_id=41,valid_tras={102:SimpleNamespace(motion_management=motion)},
        tra_infos={'all_valid_ids':np.array([102]),'np_tras':np.array([f.predict_infos]),
                   'np_tras_bottom_corners':np.array([f.predict_bms]),
                   'np_tras_norm_corners':np.array([f.predict_norm_bms]),
                   'all_valid_boxes':np.array([f.predict_box])})


class ReplayTests(unittest.TestCase):
    def test_center_injection_updates_kf_and_all_geometry_caches(self):
        tracker=tracker_fixture(); motion=tracker.valid_tras[102].motion_management
        state,p=motion.state.copy(),motion.P.copy()
        old_corners=tracker.tra_infos['np_tras_norm_corners'].copy()
        replace_prediction(tracker,102,dict(injection='center_only',position_m=[.4,-.2,.8]))
        np.testing.assert_array_equal(motion.P,p)
        np.testing.assert_array_equal(motion.state[3:],state[3:])
        np.testing.assert_allclose(motion.frame.predict_infos[:3],[.4,-.2,.8])
        np.testing.assert_allclose(tracker.tra_infos['all_valid_boxes'][0].center,[.4,-.2,.8])
        self.assertFalse(np.array_equal(old_corners,tracker.tra_infos['np_tras_norm_corners']))
        self.assertAlmostEqual(motion.frame.predict_box.score,.91)
        np.testing.assert_array_equal(motion.frame.predict_infos[14:],[102,1,41])
        # A later tracker prediction must start at the injected state, not an old hidden mean.
        next_mean=motion.model.stateTransition(motion.state)
        old_next=motion.model.stateTransition(state)
        np.testing.assert_allclose(np.asarray(next_mean-old_next).ravel()[:3],[.4,-.2,-.2],atol=1e-12)

    def test_actual_dt_injection_preserves_covariance_but_updates_yaw(self):
        tracker=tracker_fixture(); motion=tracker.valid_tras[102].motion_management
        state=np.asarray(motion.state).ravel();state[7]+=.2
        p=motion.P.copy()
        replace_prediction(tracker,102,dict(injection='ctrv_mean',ctrv_state=state.tolist()))
        np.testing.assert_array_equal(motion.P,p)
        self.assertAlmostEqual(motion.frame.predict_box.yaw,.5)

    def test_gate_pass_does_not_guarantee_hungarian_selection(self):
        # Both tracks pass a cost limit for det0. Global assignment gives it to track1.
        costs=np.array([[1.05,.2],[.1,np.inf]])
        matched_det,matched_track,_,_=Hungarian(costs,{0:1.1})
        assignment=dict(zip(matched_det,matched_track))
        self.assertLess(costs[0,0],1.1)
        self.assertEqual(assignment,{0:1,1:0})

    def test_trace_guard_rejects_matching_and_input_drift(self):
        stage={'stage':'mix','track_ids':[102],'detections':[[1.]],'predictions':[[2.]],
               'cost':[[[1.05]]],'assignment':[[0],[0],[],[]]}
        record={'frame_idx':20,'stages':[stage]}
        verify_trace_step(record,copy.deepcopy(record))
        for key,value in [('assignment',[[],[],[0],[0]]),('detections',[[99.]]),('cost',[[[.9]]])]:
            other=copy.deepcopy(record);other['stages'][0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                verify_trace_step(record,other)

    def test_replay_config_window_and_variants(self):
        config,_=load_config(Path(__file__).resolve().parents[1]/'config/replay_scene1072.yaml')
        self.assertEqual(config['window'],{'start_frame':18,'end_frame':22})
        self.assertEqual(len(config['variants']),5)


if __name__=='__main__':
    unittest.main()
