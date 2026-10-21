"""Official devkit accumulation/aggregation with independent classes in processes.

No metric formulas are reimplemented. B0 must reproduce saved official metrics.
Per-scene diagnostics use both own best-MOTA and common B0 class thresholds.
"""
import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import nuscenes.eval.tracking.evaluate as official
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.data_classes import TrackingMetricData
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.utils import create_motmetrics
from .common import *

EVALUATOR=None
BASE_THRESHOLDS=None


def class_eval(name):
    e=EVALUATOR
    worker=TrackingEvaluation(e.tracks_gt,e.tracks_pred,name,e.cfg.dist_fcn_callable,
        e.cfg.dist_th_tp,e.cfg.min_recall,TrackingMetricData.nelem,e.cfg.metric_worst,
        verbose=False,output_dir=e.output_dir,render_classes=[])
    md=worker.accumulate()
    best=int(np.nanargmax(md.mota)) if not np.all(np.isnan(md.mota)) else None
    own=float(md.confidence[best]) if best is not None else None
    thresholds={'own_best_mota':own,'common_b0':BASE_THRESHOLDS[name]}
    scene_rows=[];mh=create_motmetrics()
    wanted=['num_switches','num_fragmentations_custom','num_false_positives','num_misses','num_objects','mota_custom']
    for scene in e.tracks_gt:
        per_scene=TrackingEvaluation({scene:e.tracks_gt[scene]},{scene:e.tracks_pred[scene]},name,
            e.cfg.dist_fcn_callable,e.cfg.dist_th_tp,e.cfg.min_recall,TrackingMetricData.nelem,
            e.cfg.metric_worst,verbose=False,render_classes=[])
        previous={}
        for policy,threshold in thresholds.items():
            if threshold is None or not np.isfinite(threshold): continue
            if threshold in previous: metrics=previous[threshold]
            else:
                accumulated,_=per_scene.accumulate_threshold(threshold)
                metrics=mh.compute(accumulated,metrics=wanted,name='scene').loc['scene'].to_dict()
                metrics['num_fragmentations']=metrics.pop('num_fragmentations_custom')
                metrics['mota']=metrics.pop('mota_custom')
                previous[threshold]=metrics
            scene_rows.append(dict(scene_token=scene,class_name=name,policy=policy,threshold=threshold,**metrics))
    return name,md,scene_rows


class ParallelOfficialEval(official.TrackingEval):
    def evaluate(self):
        global EVALUATOR
        EVALUATOR=self;t0=time.perf_counter();data={};scene_rows=[]
        with ProcessPoolExecutor(max_workers=self.stage4_workers,mp_context=mp.get_context('fork')) as pool:
            futures={pool.submit(class_eval,name):name for name in self.cfg.class_names}
            for future in as_completed(futures):
                name,md,rows=future.result();data[name]=md;scene_rows.extend(rows)
                print('OFFICIAL CLASS COMPLETE',name,flush=True)
        # Invoke the devkit's unchanged aggregation and threshold selection code.
        def cached(*args,**kwargs):
            name=args[2] if len(args)>2 else kwargs['class_name']
            return SimpleNamespace(accumulate=lambda:data[name])
        with patch.object(official,'TrackingEvaluation',cached):
            metrics,details=super().evaluate()
        metrics.add_runtime(time.perf_counter()-t0)
        self.scene_rows=scene_rows
        return metrics,details


def evaluate_variant(cfg,variant):
    global BASE_THRESHOLDS
    root=output_root(cfg);result=root/'runs'/'full'/variant/'results.json';out=root/'evaluation'/variant
    if not result.exists(): raise ValueError('Run full tracking first')
    base_details=read_json(resolve(cfg['original_metrics'])/'metrics_details.json')
    BASE_THRESHOLDS={name:float(d['confidence'][int(np.nanargmax(d['mota']))]) for name,d in base_details.items()}
    t0=time.perf_counter()
    evaluator=ParallelOfficialEval(config=config_factory('tracking_nips_2019'),result_path=str(result),
        eval_set='val',output_dir=str(out),nusc_version='v1.0-trainval',
        nusc_dataroot=str(resolve(cfg['dataset_root'])),verbose=False,render_classes=[])
    if len(evaluator.tracks_gt)!=150: raise AssertionError('Official evaluation must contain 150 scenes')
    evaluator.stage4_workers=cfg['eval_workers']
    evaluator.main(render_curves=False)
    summary=read_json(out/'metrics_summary.json')
    if variant=='B0':
        old=read_json(resolve(cfg['original_metrics'])/'metrics_summary.json')
        for metric in ('amota','amotp','mota','ids','frag','fp','fn'):
            np.testing.assert_allclose(summary[metric],old[metric],atol=1e-10,rtol=0)
            for name in summary['label_metrics'][metric]:
                np.testing.assert_allclose(summary['label_metrics'][metric][name],old['label_metrics'][metric][name],atol=1e-10,rtol=0)
    # Cross-check per-scene own-threshold counts against the official aggregate.
    count_keys={'ids':'num_switches','frag':'num_fragmentations','fp':'num_false_positives','fn':'num_misses'}
    for name in evaluator.cfg.class_names:
        rows=[r for r in evaluator.scene_rows if r['class_name']==name and r['policy']=='own_best_mota']
        for metric,key in count_keys.items():
            np.testing.assert_allclose(sum(r[key] for r in rows),summary['label_metrics'][metric][name],rtol=0,atol=0)
    write_json(out/'scene_metrics.json',clean(evaluator.scene_rows))
    write_json(out/'provenance.json',dict(result_sha256=sha256(result),workers=cfg['eval_workers'],
        official_scene_count=len(evaluator.tracks_gt),official_sample_count=len(evaluator.sample_tokens),
        seconds=time.perf_counter()-t0,devkit_evaluate_sha256=sha256(Path(official.__file__)),
        evaluation_wrapper_sha256=sha256(Path(__file__)),
        b0_reference_metrics_match=variant=='B0',per_scene_counts_match=True,
        note='Official TrackingEvaluation.accumulate and TrackingEval.evaluate/main. Classes parallel; no changes to formulas/thresholds.'))
    print('OFFICIAL FINISHED',variant,summary['amota'],summary['ids'],flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['B0','B1','B2','B3'],required=True)
    args=parser.parse_args();evaluate_variant(load_config(),args.variant)
