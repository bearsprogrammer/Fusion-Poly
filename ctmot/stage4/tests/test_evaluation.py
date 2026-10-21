"""Exercise official custom metrics and per-scene totals without loading a dataset."""
import unittest
import copy
from types import SimpleNamespace
import numpy as np
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingBox,TrackingMetricData
from nuscenes.eval.tracking.algo import TrackingEvaluation
from ctmot.stage4 import evaluation
from ctmot.stage4.common import compare_outputs


def box(identity,timestamp,score=1.):
    return TrackingBox(sample_token=str(timestamp).zfill(32),translation=(0.,0.,0.),size=(2.,4.,1.5),
        rotation=(1.,0.,0.,0.),velocity=(0.,0.),ego_translation=(0.,0.,0.),num_pts=5,
        tracking_id=identity,tracking_name='car',tracking_score=score)


class EvaluationTests(unittest.TestCase):
    def test_baseline_tolerance_does_not_hide_identity_or_score_changes(self):
        a=dict(tracking_id='140',tracking_name='car',translation=[0.,0.,0.],
               size=[2.,4.,1.5],rotation=[1.,0.,0.,0.],velocity=[1.,0.],tracking_score=.8)
        b=copy.deepcopy(a);b['velocity'][0]+=2e-8
        self.assertTrue(compare_outputs([a],[b])['ids_equal'])
        for key,value in [('tracking_id','141'),('tracking_score',.800000001),
                          ('translation',[2e-8,0.,0.]),('velocity',[1.000001,0.])]:
            changed=copy.deepcopy(a);changed[key]=value
            with self.assertRaises(AssertionError): compare_outputs([a],[changed])

    def test_scene_counts_use_official_fragmentation_and_mota(self):
        cfg=config_factory('tracking_nips_2019')
        gt={'scene':{t:[box('object',t)] for t in [1,2,3,4]}}
        pred={'scene':{1:[box('first',1,.9)],2:[],3:[box('second',3,.8)],4:[box('second',4,.8)]}}
        previous=evaluation.EVALUATOR;thresholds=evaluation.BASE_THRESHOLDS
        try:
            evaluation.EVALUATOR=SimpleNamespace(cfg=cfg,tracks_gt=gt,tracks_pred=pred,output_dir=None)
            evaluation.BASE_THRESHOLDS={'car':.1}
            name,md,rows=evaluation.class_eval('car')
            direct=TrackingEvaluation(gt,pred,'car',cfg.dist_fcn_callable,cfg.dist_th_tp,cfg.min_recall,
                TrackingMetricData.nelem,cfg.metric_worst,verbose=False).accumulate()
            np.testing.assert_allclose(md.mota,direct.mota,equal_nan=True)
            best=int(np.nanargmax(md.mota));own=next(r for r in rows if r['policy']=='own_best_mota')
            self.assertEqual(own['num_switches'],md.ids[best])
            self.assertEqual(own['num_fragmentations'],md.frag[best])
            self.assertEqual(own['num_false_positives'],md.fp[best])
            self.assertEqual(own['num_misses'],md.fn[best])
            self.assertEqual(own['mota'],md.mota[best])
        finally:
            evaluation.EVALUATOR=previous;evaluation.BASE_THRESHOLDS=thresholds


if __name__=='__main__': unittest.main()
