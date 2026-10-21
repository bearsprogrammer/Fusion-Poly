"""Run all-track variants on cached val observations, preserving original results."""
import argparse
import copy
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor,as_completed

import numpy as np
import yaml
from .common import *
from .prediction import AllTrackPredictor


def scene_job(cfg,variant,scene):
    root=output_root(cfg);cached=load_pickle(root/'inputs'/(scene+'.pkl'))
    tracker_cfg=yaml.safe_load(resolve(cfg['tracker_config']).read_text())
    prediction_cfg={k:cfg[k] for k in ('solver','history_observations','minimum_observations')}
    tracker=AllTrackPredictor(tracker_cfg,variant,prediction_cfg);tracker_timing(cfg,variant+'_'+scene)
    result={}; seconds=0.;start=time.perf_counter()
    for step in cached['steps']:
        data=copy.deepcopy(step['data'])
        t0=time.perf_counter();tracker.advance(data,step['timestamp_us']);seconds+=time.perf_counter()-t0
        if data['is_key_frame']: result[data['sample_token']]=sorted_output(data)
    elapsed=time.perf_counter()-start
    out=root/'tracking'/variant;out.mkdir(parents=True,exist_ok=True)
    write_json(out/(scene+'.json'),dict(results=result,runtime=dict(tracking_seconds=seconds,
        scene_wall_seconds=elapsed,keyframes=len(result),steps=len(cached['steps']),
        preprocessing_seconds=cached['preprocessing_seconds'],counters=tracker.counters)))
    return scene,seconds


def run(cfg,scenes,variants=('B0','B1','B2','B3'),label='small'):
    root=output_root(cfg);t0=time.perf_counter();jobs=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('fork')) as pool:
        for v in variants:
            for scene in scenes: jobs.append(pool.submit(scene_job,cfg,v,scene))
        for f in as_completed(jobs):
            scene,seconds=f.result();print('TRACKED',scene,round(seconds,2),flush=True)
    elapsed=time.perf_counter()-t0
    return finalize(cfg,scenes,variants,label,elapsed)


def finalize(cfg,scenes,variants=('B0','B1','B2','B3'),label='full',elapsed=None):
    """Aggregate completed scene jobs without repeating tracking."""
    root=output_root(cfg)
    originals=read_json(resolve(cfg['original_results']))['results']
    summary={};validation={}
    for v in variants:
        results={};runtimes=[]
        for scene in scenes:
            data=read_json(root/'tracking'/v/(scene+'.json'));results.update(data['results']);runtimes.append(data['runtime'])
        if v=='B0':
            for token,rows in results.items():
                previous=sorted(originals[token],key=lambda x:x['tracking_id'])
                current=sorted(rows,key=lambda x:x['tracking_id'])
                validation[token]=compare_outputs(previous,current)
        destination=root/'runs'/label/v;destination.mkdir(parents=True,exist_ok=True)
        write_json(destination/'results.json',dict(results=results,meta=dict(use_camera=True,use_lidar=True,
            use_radar=False,use_map=False,use_external=False)))
        secs=sum(r['tracking_seconds'] for r in runtimes);steps=sum(r['steps'] for r in runtimes)
        preprocess=sum(r['preprocessing_seconds'] for r in runtimes)
        summary[v]=dict(scenes=len(scenes),keyframes=len(results),steps=steps,tracking_cpu_wall_sum_s=secs,
            preprocessing_sum_s=preprocess,keyframe_fps=len(results)/secs,event_fps=steps/secs,
            end_to_end_keyframe_fps=len(results)/(secs+preprocess),
            counters={k:sum(r['counters'][k] for r in runtimes) for k in runtimes[0]['counters']})
        write_json(destination/'runtime.json',summary[v])
        write_json(destination/'config.json',dict(experiment=cfg,tracker_config=yaml.safe_load(resolve(cfg['tracker_config']).read_text()),variant=v))
    write_json(root/('tracking_'+label+'.json'),dict(variants=summary,worker_count=cfg['workers'],
        batch_wall_seconds=elapsed,scene_names=scenes,b0_ids_exact_numeric_tolerance=1e-8,
        b0_velocity_tolerance=1e-7,
        batch_wall_note='Unavailable when finalizing an interrupted aggregation' if elapsed is None else 'Measured'))
    write_json(root/('baseline_validation_'+label+'.json'),validation)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true')
    parser.add_argument('--finalize-only',action='store_true');args=parser.parse_args()
    cfg=load_config()
    if args.full:
        from nuscenes.utils.splits import create_splits_scenes
        scenes=sorted(create_splits_scenes()['val']);label='full'
    else:
        scenes=sorted({c['scene'] for c in cfg['cases']});label='small'
    (finalize if args.finalize_only else run)(cfg,scenes,label=label)
