"""Explain additional fixed-threshold misses using verified diagnostic traces."""
from collections import Counter
from types import SimpleNamespace
import numpy as np
from .common import *
from ctmot.proto.validate_history import association_cost
from data.script.NUSC_CONSTANT import CLASS_SEG_TO_STR_CLASS

LABEL=CLASS_SEG_TO_STR_CLASS


def at_frame(records,frame):
    return next(r for r in records if r['is_keyframe'] and r['frame_idx']==frame)


def target_evidence(record, tid, gt, cls):
    tid=str(tid) if tid is not None else None
    after=record['after'].get(tid)
    prediction=next((p for p in record['predictions'] if str(int(p[14]))==tid),None)
    out=next((b for b in record['output'] if b['tracking_id']==tid),None)
    best=None
    for stage in record['stages']:
        for i,d in enumerate(stage['detections']):
            if int(d[13])!=LABEL[cls]:continue
            dist=float(np.linalg.norm(np.array(d[:2])-np.array(gt.translation[:2])))
            if best is None or dist<best['detection_gt_xy_m']:
                best=dict(stage=stage['stage'],index=i,detection=d,detection_gt_xy_m=dist,
                    limit=stage['thresholds'][str(LABEL[cls])],prediction_in_stage=False)
                allocation=next(a for a in record['allocations'] if a['stage']==stage['stage'] and a['detection_index']==i)
                best['allocation']=allocation
                if prediction is not None:
                    best.update(distance_xyz_m=float(np.linalg.norm(np.array(prediction[:3])-d[:3])),
                                raw_giou3d_cost=association_cost(prediction,d))
                    best['spatial_pass']=best['distance_xyz_m']<3.
                    best['cost_pass']=best['raw_giou3d_cost']<=best['limit']
                    if int(tid) in stage['track_ids']:
                        j=stage['track_ids'].index(int(tid));best['prediction_in_stage']=True
                        best['matrix_cost']=stage['cost'][LABEL[cls]][i][j]
    return dict(track_id=tid,prediction=prediction,after=after,raw_output=out,candidate=best,
        assigned_3d=[a for a in record['allocations'] if str(a['track_id'])==tid],
        matched_2d=next((m for m in record['camera_matches'] if str(m['track_id'])==tid),None))


def main():
    cfg=load_config();root=output_root(cfg);out=root/'fn_inspection'
    archive=load_pickle(out/'official_selected.pkl');misses=read_json(out/'additional_misses.json')
    actual={(a['sample_token'],a['instance_token']):a for a in read_json(out/'actual_gt.json')}
    cases=[]
    for scene in read_json(out/'selection.json'):
        name,token=scene['scene'],scene['scene_token']
        trace={v:load_pickle(out/(name+'_'+v+'_trace.pkl'))['records'] for v in ('B0','B1')}
        cache=load_pickle(root/'inputs'/(name+'.pkl'))
        sample_tokens={s['timestamp']:s['token'] for s in cache['samples']}
        counts=Counter((r['cls'],r['gt_id']) for r in misses if r['scene']==name)
        print(name,'targets',counts,flush=True)
        groups=[]
        for (cls,gtid),count in counts.most_common():
            rows=[r for r in misses if r['scene']==name and r['gt_id']==gtid and r['cls']==cls]
            frames=sorted(r['frame_idx'] for r in rows)
            b0ids=Counter(r['B0']['track_id'] for r in rows)
            b0id=b0ids.most_common(1)[0][0]
            lineage=next(r['after'][b0id]['lineage'] for r in trace['B0'] if b0id in r['after'])
            b1id=next((tid for r in trace['B1'] for tid,a in r['after'].items() if a['lineage']==lineage),None)
            per_frame=[]
            for a in [r for r in trace['B0'] if r['is_keyframe']]:
                gt=next((g for g in archive['gt'][token][a['timestamp_us']] if g.tracking_id==gtid),None)
                if gt is None:continue
                b=at_frame(trace['B1'],a['frame_idx'])
                sample=sample_tokens[a['timestamp_us']]
                real=actual.get((sample,gtid))
                diagnostic_gt=SimpleNamespace(translation=real['translation']) if real else gt
                reference=next(g for g in archive['gt'][token][a['timestamp_us']] if g.num_pts>=0)
                ego_xy=np.array(reference.translation[:2])-reference.ego_translation[:2]
                entry=dict(frame_idx=a['frame_idx'],timestamp=a['timestamp_us'],gt=gt.serialize(),
                           sample_token=sample,actual_gt=real,eval_gt_interpolated=gt.num_pts<0,
                           ego_xy=ego_xy.tolist(),
                           B0=target_evidence(a,b0id,diagnostic_gt,cls),B1=target_evidence(b,b1id,diagnostic_gt,cls))
                for v,r in [('B0',a),('B1',b)]:
                    raw=entry[v]['raw_output']
                    entry[v]['raw_ego_distance']=float(np.linalg.norm(np.array(raw['translation'][:2])-ego_xy)) if raw else None
                    entry[v]['raw_actual_gt_distance']=float(np.linalg.norm(np.array(raw['translation'][:2])-diagnostic_gt.translation[:2])) if raw else None
                    near=sorted([p for p in archive['predictions'][v][token][r['timestamp_us']] if p.tracking_name==cls],
                                key=lambda p:np.linalg.norm(np.array(p.translation[:2])-gt.translation[:2]))[:3]
                    entry[v]['nearest_eval_boxes']=[dict(p.serialize(),gt_distance=float(np.linalg.norm(np.array(p.translation[:2])-gt.translation[:2]))) for p in near]
                per_frame.append(entry)
            group=dict(cls=cls,gt_id=gtid,lost_frames=frames,additional_fn=count,
                immediate_reasons=dict(Counter(r['immediate_reason'] for r in rows)),
                B0_track_id=b0id,B1_same_birth_track_id=b1id,lineage=lineage,frames=per_frame)
            groups.append(group)
            print(' ',cls,gtid,frames,'ids',b0id,b1id,group['immediate_reasons'],flush=True)
        cases.append(dict(scene=name,scene_token=token,delta_fn=scene['delta_fn'],groups=groups))
    write_json(out/'evidence.json',clean(cases))


if __name__=='__main__':main()
