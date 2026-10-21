"""Five predeclared local cases, including gaps and missing/dead target tracks."""
import copy
import json
import time

import numpy as np
import yaml
from ctmot.proto.association_replay import (ReplayTracker, VARIANTS, clean, verify_output, write_csv,
                                            replace_prediction)
from ctmot.proto.history import build_snapshot, predict_shadow
from ctmot.proto.predictor import ODEPredictor
from ctmot.proto.dynamics import ConstantVelocity
from ctmot.proto.validate_history import iter_json_array, association_cost
from .common import *
from motion_module.motion_model import CTRV


class CaseTracker(ReplayTracker):
    def tras_predict(self):
        saved = self.plan
        absent = self.target_id not in self.valid_tras
        if absent: self.plan = None
        super().tras_predict()
        self.plan = saved
        self.record['intervention_skipped_track_absent'] = bool(absent and self.variant!='baseline'
            and self.is_key_frame and self.frame_idx==self.intervention_frame)


def to_history_trace(records, config):
    steps=[]
    for r in records:
        after={tid:dict(v,updated={'tracking_name':'car'} if v['updated'] else None) for tid,v in r['after'].items()}
        steps.append(dict(frame_idx=r['frame_idx'],is_keyframe=r['is_keyframe'],timestamp=r['timestamp_us'],
            token=r['token'],internal_frame_id=r['internal_frame_id'],
            before={tid:{'state':s} for tid,s in r['before'].items()},after=after,stages=r['stages'],
            predicted={'np_tras':r['predictions'],'all_valid_ids':[int(p[14]) for p in r['predictions']]}))
    return {'config':clean(config),'trace':steps}


def plans_for_case(trace, case, cfg):
    snapshot=build_snapshot(trace,case['anchor'],case['track_id'],cfg['history_observations'])
    q=next(s for s in trace['trace'] if s['frame_idx']==case['query'] and s['is_keyframe'])
    query={'timestamp_us':q['timestamp'],'frame_idx':case['query']}
    shadow=predict_shadow(snapshot,[query],cfg['solver'],cfg['minimum_observations'])
    dt=(q['timestamp']-snapshot['timestamp_us'])/1e6
    actual=CTRV(True,False,False,dt).stateTransition(np.mat(snapshot['posterior_ctrv']).T)
    plans={'baseline':None,'actual_dt_ctrv':dict(injection='ctrv_mean',ctrv_state=np.asarray(actual).ravel().tolist())}
    for name in ('ct_cv','ct_ca'):
        models=shadow['queries'][0]['models'];fallback=name not in models
        pred=models['ct_cv'] if fallback else models[name]
        plans[name]=dict(injection='center_only',position_m=pred['position_m'],ode_prediction=pred,ca_fallback_cv=fallback)
    history=snapshot['history']
    if len(history)<2: raise ValueError('Past XYZ needs two accepted detections')
    a,b=history[-2:];vel=(np.array(b['position_m'])-a['position_m'])/((b['timestamp_us']-a['timestamp_us'])/1e6)
    pred=ODEPredictor(ConstantVelocity(),**cfg['solver']).predict(b['position_m']+vel.tolist(),b['timestamp_us'],q['timestamp']).to_dict()
    plans['past_xyz_cv']=dict(injection='center_only',position_m=pred['position_m'],ode_prediction=pred,
                              initial_state_pv=b['position_m']+vel.tolist())
    # No hidden correction can be ignored between the declared posterior and query.
    model=CTRV(True,False,False,trace['config']['basic']['LiDAR_interval'])
    state=np.mat(snapshot['posterior_ctrv']).T
    anchor=next(s for s in trace['trace'] if s['frame_idx']==case['anchor'] and s['is_keyframe'])
    for s in trace['trace']:
        if anchor['internal_frame_id']<s['internal_frame_id']<=q['internal_frame_id']:
            if str(case['track_id']) in s['before']:
                np.testing.assert_allclose(np.asarray(state).ravel(),s['before'][str(case['track_id'])]['state'],atol=1e-9,rtol=0)
            state=model.stateTransition(state);model.warpStateYawToPi(state)
    return plans,dict(snapshot=snapshot,shadow=shadow,actual_dt_s=dt,
                      nominal_dt_s=(q['internal_frame_id']-anchor['internal_frame_id'])*model.dt)


def run(cfg):
    start=time.perf_counter();root=output_root(cfg)/'multicase';root.mkdir(exist_ok=True)
    config=yaml.safe_load(resolve(cfg['tracker_config']).read_text())
    originals=read_json(resolve(cfg['original_results']))['results']
    events={e['event_id']:e for e in read_json(resolve(cfg['original_events']))['events']}
    cases=copy.deepcopy(cfg['cases']);all_runs={};baseline_checks={}
    for name in sorted({c['scene'] for c in cases}):
        cached=load_pickle(output_root(cfg)/'inputs'/(name+'.pkl'));steps=cached['steps']
        chosen=[c for c in cases if c['scene']==name]
        tracker_timing(cfg,'multi_'+name)
        baseline=CaseTracker(config,None,-1,-1);check=[];forks={}
        for s in steps:
            r=baseline.advance(s['data'],s['frame_idx'],s['timestamp_us'],s['camera_times'])
            if r['is_keyframe']: check.append(dict(frame_idx=r['frame_idx'],**compare_outputs(originals[r['token']],r['output'])))
            for case in chosen:
                if r['is_keyframe'] and r['frame_idx']==case['anchor']:
                    forks[case['event_id']]=copy.deepcopy(baseline)
        baseline_checks[name]=check
        trace=to_history_trace(baseline.records,config)
        for case in chosen:
            cid=case['event_id'];case['gt_instance_id']=events[cid]['gt_instance_id']
            plans,inputs=plans_for_case(trace,case,cfg)
            start_frame=max(0,case['anchor']-1)
            records={'baseline':[r for r in baseline.records if start_frame<=r['frame_idx']<=case['end']]}
            for variant in VARIANTS[1:]:
                tr=copy.deepcopy(forks[cid]);tr.target_id=case['track_id'];tr.intervention_frame=case['query']
                tr.variant=variant;tr.plan=plans[variant]
                for s in steps:
                    if s['frame_idx']>case['anchor'] and s['frame_idx']<=case['end']:
                        tr.advance(s['data'],s['frame_idx'],s['timestamp_us'],s['camera_times'])
                records[variant]=[r for r in tr.records if start_frame<=r['frame_idx']<=case['end']]
            folder=root/str(cid);folder.mkdir(exist_ok=True)
            write_json(folder/'records.json',records)
            write_json(folder/'initial_conditions.json',clean(dict(case=case,plans=plans,inputs=inputs)))
            all_runs[cid]=dict(case=case,records=records,plans=plans,inputs=inputs,steps=steps)
            print('MULTICASE replay',cid,name,flush=True)
    # Only now read target evaluation GT, after every tracking branch has finished.
    wanted={c['gt_instance_id'] for c in cases}
    gt={(a['sample_token'],a['instance_token']):a for a in iter_json_array(resolve(cfg['dataset_root'])/'v1.0-trainval/sample_annotation.json') if a['instance_token'] in wanted}
    rows=[];summaries=[];impacts=[]
    for cid,run in all_runs.items():
        case=run['case'];tid=case['track_id'];records=run['records']
        by_token={s['data']['sample_token']:s['data'] for s in run['steps']}
        base={r['token']:r for r in records['baseline']}
        anchor=next(r for r in records['baseline'] if r['is_keyframe'] and r['frame_idx']==case['anchor'])
        target_lineage=anchor['after'][str(tid)]['lineage']
        for variant,sequence in records.items():
            var_rows=[]
            for r in sequence:
                if not r['is_keyframe']: continue
                frame=r['frame_idx'];ann=gt[(r['token'],case['gt_instance_id'])];pos=ann['translation']
                data=by_token[r['token']]
                options=[(stage,i,d) for stage,key in [('mix','mix_np_dets_3d'),('pure3d','single_np_dets_3d')]
                         for i,d in enumerate(data[key]) if int(d[-1])==2]
                candidate=min(options,key=lambda x:np.linalg.norm(np.array(x[2][:2])-pos[:2])) if options else None
                if candidate and np.linalg.norm(np.array(candidate[2][:2])-pos[:2])>2: candidate=None
                pred=next((p for p in r['predictions'] if int(p[14])==tid),None)
                owner=None;matrix=None;cost=None;selected=None;limit=None;distance=None
                if candidate:
                    stage,i,d=candidate
                    owner=next(a for a in r['allocations'] if a['stage']==stage and a['detection_index']==i)
                    limit=config['association']['first_thre' if stage=='mix' else 'second_thre'][2]
                    st=next((s for s in r['stages'] if s['stage']==stage and tid in s['track_ids']),None)
                    if st:
                        col=st['track_ids'].index(tid);matrix=st['cost'][2][i][col]
                        selected=next((di for di,tj in zip(*st['assignment'][:2]) if tj==col),None)
                    if pred:
                        cost=association_cost(pred[:14],d)
                        distance=float(np.linalg.norm(np.array(pred[:3])-d[:3]))
                state=r['after'].get(str(tid))
                row=dict(event_id=cid,scene=case['scene'],mechanism=case['mechanism'],variant=variant,frame=frame,
                    prediction_xy_error=float(np.linalg.norm(np.array(pred[:2])-pos[:2])) if pred else None,
                    prediction_xyz_error=float(np.linalg.norm(np.array(pred[:3])-pos)) if pred else None,
                    cost=cost,matrix_cost=matrix,cost_limit=limit,gate_pass=distance<3 and cost<=limit if distance is not None else None,
                    selected_index=selected,correct_detection_index=int(candidate[1]) if candidate else None,
                    correct_owner_id=owner['track_id'] if owner else None,new_id=owner['kind']=='birth' if owner else None,
                    assigned_correct=owner is not None and owner['kind']=='matched' and owner['track_id']==tid,
                    target_life=state['life'] if state else 'absent',
                    target_in_output=any(b['tracking_id']==str(tid) for b in r['output']),
                    skip_track_absent=r.get('intervention_skipped_track_absent',False))
                rows.append(row);var_rows.append(row)
                b=base[r['token']];ba={(a['stage'],a['detection_index']):a for a in b['allocations']}
                changed=[dict(original=ba[(a['stage'],a['detection_index'])],current=a) for a in r['allocations']
                         if ba[(a['stage'],a['detection_index'])]['lineage']!=a['lineage']]
                unrelated=[c for c in changed if not candidate or (c['current']['stage'],c['current']['detection_index'])!=candidate[:2]]
                bs={v['lineage']:v for v in b['after'].values()};vs={v['lineage']:v for v in r['after'].values()}
                state_changes=[l for l in set(bs)&set(vs) if l!=target_lineage and bs[l]!=vs[l]]
                impacts.append(dict(event_id=cid,variant=variant,frame=frame,assignment_changes=changed,
                                     unrelated_assignment_changes=len(unrelated),other_state_changes=state_changes))
            q=next(x for x in var_rows if x['frame']==case['query'])
            tail=[x for x in var_rows if x['frame']>=case['query']]
            summaries.append(dict(q,continuity=all(x['assigned_correct'] and x['target_in_output'] for x in tail),
                owner_ids=[x['correct_owner_id'] for x in tail],
                ca_fallback_cv=bool(run['plans'][variant] and run['plans'][variant].get('ca_fallback_cv')),
                actual_dt_s=run['inputs']['actual_dt_s'],nominal_dt_s=run['inputs']['nominal_dt_s']))
    # Mandatory Stage3 reproduction, before proceeding to full-track testing.
    expected={'baseline':1.1473571838075898,'actual_dt_ctrv':1.037977863021939,
              'ct_cv':1.049495559974355,'ct_ca':1.0200270778069684,'past_xyz_cv':.3181759091247832}
    for row in summaries:
        if row['event_id']==419:
            np.testing.assert_allclose(row['cost'],expected[row['variant']],rtol=0,atol=1e-10)
            assert row['continuity']==(row['variant']!='baseline')
    report=dict(cases=cases,summary=summaries,rows=rows,impacts=impacts,baseline_checks=baseline_checks,
                stage3_reproduced=True,seconds=time.perf_counter()-start)
    write_json(root/'report.json',clean(report));write_csv(root/'comparison.csv',rows);write_csv(root/'summary.csv',summaries)
    print(json.dumps(clean(summaries),indent=2),flush=True)
    return report


if __name__=='__main__':
    run(load_config())
