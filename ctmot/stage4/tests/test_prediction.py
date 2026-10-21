import copy
import unittest
import numpy as np
from ctmot.stage4.prediction import physical_pv,actual_transition,set_geometric_center
from motion_module.motion_model import CTRV,CV,BICYCLE
from ctmot.proto.predictor import ODEPredictor
from ctmot.proto.dynamics import ConstantVelocity
from ctmot.proto.history import estimate_acceleration


class PredictionTests(unittest.TestCase):
    def test_ctrv_true_cartesian_velocity(self):
        model=CTRV(True,False,False,.25)
        state=np.mat([1,2,3,2,4,1.5,10,np.pi/2,.1]).T
        np.testing.assert_allclose(physical_pv(model,state),[1,2,3,0,10,0],atol=1e-12)

    def test_actual_transition_preserves_nominal_covariance_clock(self):
        model=CTRV(True,False,False,.25)
        state=np.mat([0,0,1,2,4,1.5,10,0,0]).T
        result=actual_transition(model,state,.549659)
        self.assertAlmostEqual(result[0,0],5.49659)
        self.assertEqual(model.dt,.25)
        self.assertEqual(state[0,0],0)
        with self.assertRaises(ValueError): actual_transition(model,state,0)

    def test_cv_actual_dt_preserves_original_hold_z_policy(self):
        model=CV(True,False,False,.25)
        state=np.mat([1,2,3,2,4,1.5,4,5,-.2,.4]).T
        result=actual_transition(model,state,.55)
        np.testing.assert_allclose(physical_pv(model,result),[3.2,4.75,3.,4,5,0.],atol=1e-12)

    def test_bicycle_geometric_center_velocity_matches_finite_difference(self):
        model=BICYCLE(True,False,False,.25)
        for steering in [0.,.15,-.2]:
            state=np.mat([10,20,1,2,5,1.8,8,0,.8,steering]).T
            pv=physical_pv(model,state)
            eps=1e-6
            following=actual_transition(model,state,eps)
            numeric=(model.getOutputInfo(following)[:3]-pv[:3])/eps
            np.testing.assert_allclose(numeric,pv[3:],atol=1e-5)
            replaced=set_geometric_center(model,state,[30,40,2])
            np.testing.assert_allclose(model.getOutputInfo(replaced)[:3],[30,40,2],atol=1e-12)
            np.testing.assert_array_equal(replaced[3:],state[3:])

    def test_history_has_no_future_and_fallback_is_explicit(self):
        history=[dict(timestamp_us=1000000,velocity_xy_mps=[1,2]),dict(timestamp_us=1500000,velocity_xy_mps=[2,3])]
        self.assertEqual(estimate_acceleration(history,1500000,3)['status'],'insufficient_history')
        with self.assertRaises(ValueError): estimate_acceleration(history,1400000,3)

    def test_actual_mid_queries_do_not_reset_ct_anchor(self):
        predictor=ODEPredictor(ConstantVelocity())
        state=[1,2,3,4,5,0];t0=1542799711898267
        mid=predictor.predict(state,t0,t0+299156)
        key=predictor.predict(state,t0,t0+549659)
        np.testing.assert_allclose(key.state[:3],np.array(state[:3])+np.array(state[3:])*.549659)
        self.assertNotEqual(mid.query_timestamp_us,key.query_timestamp_us)
        self.assertEqual(state,[1,2,3,4,5,0])


if __name__=='__main__': unittest.main()
