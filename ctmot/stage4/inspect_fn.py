"""Read-only B0/B1 FN diagnosis on the three scenes ranked by common-B0 FN.

No tracking outputs/configs are overwritten; diagnostic replay must reproduce
the saved scene outputs. Official single-threshold matching is used, not a new
full validation sweep. GT is used only after replay, for diagnostic attribution.
"""
import copy
from collections import Counter, defaultdict
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import yaml
from .common import *
from .prediction import AllTrackPredictor
from ctmot.proto.association_replay import ReplayTracker


class DiagnosticTracker(ReplayTracker, AllTrackPredictor):
    def __init__(self, tracker_cfg, variant, experiment):
        AllTrackPredictor.__init__(self, tracker_cfg, variant, experiment)
        self.target_id = self.intervention_frame = -1
        self.plan = None
        self.records, self.lineages = [], {}
        self.injection_count = 0

    def step(self, step):
        self.frame_idx = step['frame_idx']
        self.actual_timestamp_us = step['timestamp_us']
        self.camera_times = step['camera_times']
        data = copy.deepcopy(step['data'])
        AllTrackPredictor.advance(self, data, step['timestamp_us'])
        self.record['output'] = sorted_output(data)
        self.records.append(clean(self.record))
        return self.records[-1]


def selection(cfg):
    root = output_root(cfg)
    names = {s['token']: s['name'] for s in read_json(resolve(cfg['dataset_root'])/'v1.0-trainval/scene.json')}
    saved = {v: {(r['scene_token'], r['class_name']): r
                 for r in read_json(root/'evaluation'/v/'scene_metrics.json') if r['policy']=='common_b0'}
             for v in ('B0', 'B1')}
    deltas = defaultdict(float)
    for key, row in saved['B1'].items():
        deltas[key[0]] += row['num_misses'] - saved['B0'][key]['num_misses']
    ranked = sorted(deltas, key=lambda t: (-deltas[t], names[t]))[:3]
    return [dict(scene=names[t], scene_token=t, delta_fn=deltas[t],
                 classes={c:dict(B0=saved['B0'][(t,c)]['num_misses'],B1=r['num_misses'],threshold=r['threshold'])
                          for (s,c),r in saved['B1'].items() if s==t}) for t in ranked], saved


def collect(cfg, out, chosen):
    config = yaml.safe_load(resolve(cfg['tracker_config']).read_text())
    experiment = {k:cfg[k] for k in ('solver','history_observations','minimum_observations')}
    checks = []
    for scene in chosen:
        cache = load_pickle(output_root(cfg)/'inputs'/(scene['scene']+'.pkl'))
        for variant in ('B0','B1'):
            path = out/(scene['scene']+'_'+variant+'_trace.pkl')
            if path.exists():
                continue
            archived_path = output_root(cfg)/'tracking'/variant/(scene['scene']+'.json')
            archived = read_json(archived_path)['results']
            tracker_timing(cfg, 'fn_diagnosis_'+scene['scene']+'_'+variant)
            tracker = DiagnosticTracker(config, variant, experiment)
            maximum = 0.
            for step in cache['steps']:
                record = tracker.step(step)
                if record['is_keyframe']:
                    check = compare_outputs(archived[record['token']],record['output'])
                    maximum=max(maximum,check['max_abs_error'])
            value=dict(records=tracker.records,source_sha256=sha256(archived_path),
                       ids_and_scores_verified=True,max_error=maximum)
            save_pickle(path,value)
            checks.append(dict(scene=scene['scene'],variant=variant,max_error=maximum))
            print('TRACE VERIFIED',scene['scene'],variant,maximum,flush=True)
    return checks


def single_threshold(gt, pred, cls, threshold, config):
    from nuscenes.eval.tracking.algo import TrackingEvaluation
    from nuscenes.eval.tracking.data_classes import TrackingMetricData
    from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
    worker = TrackingEvaluation(gt,pred,cls,config.dist_fcn_callable,config.dist_th_tp,
        config.min_recall,TrackingMetricData.nelem,config.metric_worst,verbose=False,render_classes=[])
    # One scene only: keep actual GT/track IDs rather than renumbering them.
    # This is a pure relabeling; original matching and metric formulas are unchanged.
    with patch.object(MOTAccumulatorCustom,'merge_event_dataframes',staticmethod(lambda dfs:dfs[0].events)):
        events,_ = worker.accumulate_threshold(threshold)
    scene = next(iter(gt)); times=[]
    for t, boxes in gt[scene].items():
        if any(b.tracking_name==cls for b in boxes) or any(b.tracking_name==cls and b.tracking_score>=threshold for b in pred[scene][t]):
            times.append(t)
    rows=[]
    for (frame,event),r in events.iterrows():
        if r['Type'] in ('MATCH','SWITCH','MISS','FP'):
            rows.append(dict(timestamp=int(times[int(frame)]),type=str(r['Type']),
                gt_id=None if r['Type']=='FP' else str(r['OId']),
                track_id=None if r['Type']=='MISS' else str(r['HId']),
                distance=None if not np.isfinite(r['D']) else float(r['D'])))
    return rows


def evaluate_selected(cfg, out, chosen, saved):
    from nuscenes import NuScenes
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import load_gt,add_center_dist,filter_eval_boxes
    from nuscenes.eval.tracking.data_classes import TrackingBox
    from nuscenes.eval.tracking.loaders import create_tracks
    nusc=NuScenes(version='v1.0-trainval',dataroot=str(resolve(cfg['dataset_root'])),verbose=False)
    config=config_factory('tracking_nips_2019'); tokens={x['scene_token'] for x in chosen}
    proxy=SimpleNamespace(sample=[s for s in nusc.sample if s['scene_token'] in tokens],get=nusc.get,
        version=nusc.version,sample_annotation=nusc.sample_annotation,box_velocity=nusc.box_velocity)
    gt=load_gt(proxy,'val',TrackingBox,verbose=False)
    gt=filter_eval_boxes(nusc,add_center_dist(nusc,gt),config.class_range)
    gt_tracks=create_tracks(gt,nusc,'val',gt=True)
    predictions={}; events={}; validation=[]
    for variant in ('B0','B1'):
        raw={}
        for scene in chosen:
            raw.update(read_json(output_root(cfg)/'tracking'/variant/(scene['scene']+'.json'))['results'])
        boxes=EvalBoxes.deserialize(raw,TrackingBox)
        boxes=filter_eval_boxes(nusc,add_center_dist(nusc,boxes),config.class_range)
        predictions[variant]=create_tracks(boxes,nusc,'val',gt=False)
        events[variant]={}
        for scene in chosen:
            token=scene['scene_token']; events[variant][token]={}
            for cls, settings in scene['classes'].items():
                if not (settings['B0'] or settings['B1']): continue
                rows=single_threshold({token:gt_tracks[token]},{token:predictions[variant][token]},cls,settings['threshold'],config)
                counts=Counter(r['type'] for r in rows)
                original=saved[variant][(token,cls)]
                assert counts['MISS']==original['num_misses'],(variant,scene['scene'],cls,counts,original)
                assert counts['FP']==original['num_false_positives']
                assert counts['SWITCH']==original['num_switches']
                events[variant][token][cls]=rows
                validation.append(dict(scene=scene['scene'],variant=variant,cls=cls,counts=dict(counts)))
    unthresholded={}
    for scene in chosen:
        token=scene['scene_token']; unthresholded[token]={}
        for cls,s in scene['classes'].items():
            if not (s['B0'] or s['B1']):continue
            unthresholded[token][cls]=single_threshold({token:gt_tracks[token]},
                {token:predictions['B1'][token]},cls,-np.inf,config)
    cameras={}
    for sample in proxy.sample:
        cameras[sample['token']]={}
        for camera,sd_token in sample['data'].items():
            if not camera.startswith('CAM_'):continue
            sd=nusc.get('sample_data',sd_token)
            cameras[sample['token']][camera]=dict(filename=sd['filename'],timestamp=sd['timestamp'],
                sample_data_token=sd_token,pose=nusc.get('ego_pose',sd['ego_pose_token']),
                calibration=nusc.get('calibrated_sensor',sd['calibrated_sensor_token']))
    archive=dict(gt={s:dict(ts) for s,ts in gt_tracks.items()},
        predictions={v:{s:dict(ts) for s,ts in values.items()} for v,values in predictions.items()},
        events=events,unthresholded_B1=unthresholded,cameras=cameras)
    save_pickle(out/'official_selected.pkl',archive)
    write_json(out/'official_validation.json',validation)
    print('FIXED-THRESHOLD COUNTS VERIFIED',flush=True)
    return nusc, archive


def summarize(out, chosen, archive):
    result=[]
    for scene in chosen:
        token=scene['scene_token']; cache=load_pickle(out.parent/'inputs'/(scene['scene']+'.pkl'))
        index={s['timestamp']:i for i,s in enumerate(cache['samples'])}
        for cls in archive['events']['B0'][token]:
            maps={v:{(r['timestamp'],r['gt_id']):r for r in archive['events'][v][token][cls] if r['gt_id']}
                  for v in ('B0','B1')}
            no_score={(r['timestamp'],r['gt_id']):r for r in archive['unthresholded_B1'][token][cls] if r['gt_id']}
            lost=[k for k,r in maps['B1'].items() if r['type']=='MISS' and maps['B0'][k]['type']!='MISS']
            gained=[k for k,r in maps['B0'].items() if r['type']=='MISS' and maps['B1'][k]['type']!='MISS']
            assert len(lost)-len(gained)==scene['classes'][cls]['B1']-scene['classes'][cls]['B0']
            for key in lost:
                timestamp,identity=key
                result.append(dict(scene=scene['scene'],scene_token=token,cls=cls,gt_id=identity,
                    frame_idx=index[timestamp],timestamp=timestamp,threshold=scene['classes'][cls]['threshold'],
                    B0=maps['B0'][key],B1_no_score=no_score[key],
                    immediate_reason='score_filter' if no_score[key]['type']!='MISS' else 'geometry_or_missing_output'))
            print('FN DELTA',scene['scene'],cls,'lost',len(lost),'gained',len(gained),flush=True)
    write_json(out/'additional_misses.json',result)
    print('TARGETS',dict(Counter((r['scene'],r['cls'],r['gt_id'],r['immediate_reason']) for r in result)),flush=True)


def main():
    cfg=load_config();out=output_root(cfg)/'fn_inspection';out.mkdir(exist_ok=True)
    chosen,saved=selection(cfg);write_json(out/'selection.json',chosen)
    collect(cfg,out,chosen)
    if (out/'official_selected.pkl').exists():
        archive=load_pickle(out/'official_selected.pkl')
    else:
        _,archive=evaluate_selected(cfg,out,chosen,saved)
    summarize(out,chosen,archive)


if __name__=='__main__':main()
