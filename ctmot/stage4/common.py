"""Configuration and reproducible scene-input cache (no GT in cached inputs)."""
import contextlib
import copy
import json
import os
import pickle
import time
from pathlib import Path

import yaml
from ctmot.proto.validate_history import resolve, read_json, write_json, sha256
from ctmot.proto.association_replay import clean, digest, box_record


def load_config(path='ctmot/stage4/config.yaml'):
    cfg = yaml.safe_load(resolve(path).read_text())
    for k,v in {'mid_timestamp':'latest_camera','covariance':'original_nominal_schedule',
                'yaw':'original_model_actual_dt','state_intervention':'geometric_center_only',
                'insufficient_history':'cv'}.items():
        if cfg[k] != v:
            raise ValueError('Unsupported {}: {}'.format(k,cfg[k]))
    cfg['config_path'] = str(resolve(path))
    return cfg


def output_root(cfg):
    root=resolve(cfg['output_root'])
    if not resolve('ctmot/proto/outputs') in root.parents:
        raise ValueError('Stage4 outputs must live below ctmot/proto/outputs')
    root.mkdir(parents=True,exist_ok=True)
    return root


def save_pickle(path, value):
    temp=Path(str(path)+'.tmp')
    with temp.open('wb') as f: pickle.dump(value,f,protocol=4)
    temp.replace(path)


def load_pickle(path):
    # Only our locally generated cache files are accepted by callers.
    with Path(path).open('rb') as f: return pickle.load(f)


def samples_for_scene(nusc, scene):
    values=[]; token=scene['first_sample_token']
    while token:
        s=nusc.get('sample',token);values.append(s);token=s['next']
    return values


def input_provenance(cfg):
    return {k:sha256(resolve(cfg[k])) for k in ('tracker_config','original_results','detection_3d','detection_2d','first_tokens')}


def prepare(cfg, scene_names=None):
    """Preprocess each scene once using the exact original high loader."""
    from dataloader import nusc_loader as lm
    from nuscenes.utils.splits import create_splits_scenes
    root=output_root(cfg);cache=root/'inputs';cache.mkdir(exist_ok=True)
    hashes=input_provenance(cfg)
    manifest_path=cache/'manifest.json'
    old=read_json(manifest_path) if manifest_path.exists() else {'input_hashes':hashes,'scenes':{}}
    if old['input_hashes']!=hashes: raise ValueError('Stale cache input hashes')
    names=sorted(scene_names or create_splits_scenes()['val'])
    if not set(names)<=set(create_splits_scenes()['val']): raise ValueError('val only')
    missing=[name for name in names if name not in old['scenes'] or not (cache/(name+'.pkl')).exists()]
    if not missing: return old
    timing=root/'timing'/'loader';timing.mkdir(parents=True,exist_ok=True)
    lm.TIME_COST_ROOT=str(timing)+'/'
    tracker_cfg=yaml.safe_load(resolve(cfg['tracker_config']).read_text())
    with (root/'prepare.log').open('a') as log,contextlib.redirect_stdout(log):
        loader=lm.NuScenesloader(str(resolve(cfg['detection_3d'])),str(resolve(cfg['detection_2d'])),
            str(resolve(cfg['first_tokens'])),str(resolve(cfg['dataset_root'])),tracker_cfg)
        global PREP_LOADER, PREP_CFG, PREP_TOKENS
        PREP_LOADER, PREP_CFG, PREP_TOKENS = loader, cfg, loader.all_sample_token[:]
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=mp.get_context('fork')) as pool:
            futures=[pool.submit(prepare_scene,name) for name in missing]
            for future in as_completed(futures):
                name,entry=future.result()
                old['scenes'][name]=entry
                write_json(manifest_path,old)
                print('Prepared',name,flush=True)
    return old


def tracker_timing(cfg, label):
    from tracking import nusc_tracker as tm
    path=output_root(cfg)/'timing'/label;path.mkdir(parents=True,exist_ok=True)
    tm.TIME_COST_ROOT=str(path)+'/'


def sorted_output(data):
    rows=[] if 'no_val_track_result' in data else [box_record(b) for b in data['box_track_res']]
    rows=[dict(r,sample_token=data['sample_token']) for r in rows]
    return sorted(rows,key=lambda b:-b['tracking_score'])[:500]


def compare_outputs(previous, current, atol=1e-8, velocity_atol=1e-7):
    import numpy as np
    a={x['tracking_id']:x for x in previous};b={x['tracking_id']:x for x in current}
    if set(a)!=set(b): raise AssertionError('B0 track IDs differ')
    error=0.; errors={}
    for tid in a:
        if a[tid]['tracking_name']!=b[tid]['tracking_name']: raise AssertionError('B0 class differs')
        for key in ('translation','size','rotation','velocity','tracking_score'):
            av,bv=np.asarray(a[tid][key]),np.asarray(b[tid][key])
            if not np.all(np.isfinite(bv)): raise AssertionError('Nonfinite output')
            delta=float(np.max(np.abs(av-bv)))
            errors[key]=max(errors.get(key,0.),delta)
            limit=velocity_atol if key=='velocity' else (0. if key in ('size','tracking_score') else atol)
            if delta>limit: raise AssertionError('B0 {} difference {} exceeds {}'.format(key,delta,limit))
            error=max(error,delta)
    return dict(ids_equal=True,max_abs_error=error,tolerance=atol,
                velocity_tolerance=velocity_atol,max_error_by_field=errors)


PREP_LOADER = PREP_CFG = PREP_TOKENS = None


def prepare_scene(name):
    from dataloader import nusc_loader as lm
    loader, cfg = PREP_LOADER, PREP_CFG
    root=output_root(cfg);cache=root/'inputs'
    timing=root/'timing'/('loader_'+name);timing.mkdir(parents=True,exist_ok=True)
    lm.TIME_COST_ROOT=str(timing)+'/'
    with (cache/(name+'.log')).open('w') as log,contextlib.redirect_stdout(log):
        scene=next(s for s in loader.nusc.scene if s['name']==name)
        samples=samples_for_scene(loader.nusc,scene)
        indices={s['token']:i for i,s in enumerate(samples)}
        loader.all_sample_token=[t for t in PREP_TOKENS if t.split('_')[0] in indices]
        loader.seq_id=loader.seq_first_token.index(scene['first_sample_token']);loader.frame_id=0
        values=[];t0=time.perf_counter()
        for i in range(len(loader.all_sample_token)):
            data=loader[i];idx=indices[data['sample_token'].split('_')[0]]
            tokens=data.get('sample_data_tokens') or {c:t for c,t in samples[idx]['data'].items() if c.startswith('CAM_')}
            times={c:loader.nusc.get('sample_data',t)['timestamp'] for c,t in tokens.items()}
            stamp=int(samples[idx]['timestamp']) if data['is_key_frame'] else max(times.values())
            if values and stamp<=values[-1]['timestamp_us']:
                raise ValueError('Nonmonotonic actual timestamps in '+name)
            values.append(dict(data=data,frame_idx=idx,timestamp_us=stamp,camera_times=times))
        elapsed=time.perf_counter()-t0
        save_pickle(cache/(name+'.pkl'),dict(scene=name,scene_token=scene['token'],steps=values,
            samples=[{k:s[k] for k in ('token','timestamp','next','prev')} for s in samples],preprocessing_seconds=elapsed))
    return name,dict(keyframes=len(samples),steps=len(values),preprocessing_seconds=elapsed,
                     cache_sha256=sha256(cache/(name+'.pkl')))
