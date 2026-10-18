"""One-shot prediction intervention followed by the unmodified Fusion-Poly tracker.

Run from the repository root with python -m ctmot.proto.association_replay.
GT is accessed for evaluation only, after all variant tracking has completed.
"""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

# Repository import order matters (geometry/pre_processing circular dependencies).
from dataloader import nusc_loader as loader_module
from tracking import nusc_tracker as tracker_module
from pre_processing import arraydet2box
from motion_module.motion_model import CTRV
from .history import (build_snapshot, key_step, predict_shadow,
                      baseline_from_same_anchor, ctrv_state_to_pv)
from .predictor import ODEPredictor
from .dynamics import ConstantVelocity
from .validate_history import (resolve, read_json, write_json, sha256, validate_inputs,
                               association_cost)

VARIANTS = ('baseline', 'actual_dt_ctrv', 'ct_cv', 'ct_ca', 'past_xyz_cv')


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return 'inf' if value > 0 else '-inf' if value < 0 else 'nan'
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def box_record(box):
    return dict(translation=box.center.tolist(), size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(), velocity=list(box.velocity[:2]),
                tracking_id=str(box.tracking_id), tracking_name=box.name,
                tracking_score=float(box.score))


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    expected = {'validation_config', 'tracker_config', 'detection_3d_path',
                'detection_2d_path', 'first_token_path', 'window', 'variants', 'output_root'}
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise ValueError('Unexpected/missing replay config fields')
    if cfg['variants'] != list(VARIANTS):
        raise ValueError('Replay requires the five declared control/CT variants')
    if set(cfg['window']) != {'start_frame', 'end_frame'}:
        raise ValueError('window requires start_frame and end_frame')
    from .validate_history import load_config as load_validation_config
    validation = load_validation_config(resolve(cfg['validation_config']))
    c = validation['case']
    if len(c['query_frames']) != 1 or c['query_frames'][0] != c['anchor_frame'] + 1:
        raise ValueError('One-shot replay requires the next keyframe after the anchor')
    start, end = cfg['window']['start_frame'], cfg['window']['end_frame']
    if type(start) is not int or type(end) is not int or not 0 <= start <= c['anchor_frame'] < c['query_frames'][0] <= end:
        raise ValueError('Window must contain the anchor and intervention')
    return cfg, validation


def make_interventions(trace, validation):
    """Pure, causal construction; GT never enters the prediction inputs."""
    case, acc = validation['case'], validation['acceleration']
    snapshot = build_snapshot(trace, case['anchor_frame'], case['track_id'], acc['history_observations'])
    target = key_step(trace, case['query_frames'][0])
    q = {'frame_idx': target['frame_idx'], 'timestamp_us': target['timestamp'], 'sample_token': target['token']}
    shadow = predict_shadow(snapshot, [q], validation['solver'], acc['minimum_observations'])
    if 'ct_ca' not in shadow['queries'][0]['models']:
        raise ValueError('CA replay requires sufficient accepted historical observations')
    baseline = baseline_from_same_anchor(trace, snapshot, target['frame_idx'])
    dt = (target['timestamp'] - snapshot['timestamp_us']) / 1e6
    model = CTRV(True, False, False, dt)
    actual_state = model.stateTransition(np.mat(snapshot['posterior_ctrv']).T)
    model.warpStateYawToPi(actual_state)
    observations = snapshot['history'][-2:]
    if len(observations) != 2 or [x['frame_idx'] for x in observations] != [case['anchor_frame']-1, case['anchor_frame']]:
        raise ValueError('Past XYZ control requires accepted detections at the last two keyframes')
    h0, h1 = observations
    velocity = (np.array(h1['position_m']) - h0['position_m']) / ((h1['timestamp_us'] - h0['timestamp_us']) / 1e6)
    past_state = h1['position_m'] + velocity.tolist()
    past = ODEPredictor(ConstantVelocity(), **validation['solver']).predict(
        past_state, h1['timestamp_us'], target['timestamp']).to_dict()
    common = dict(timestamp_us=snapshot['timestamp_us'], query_timestamp_us=target['timestamp'],
                  actual_dt_s=dt, initial_state_pv=snapshot['state_pv'],
                  initial_posterior_ctrv=snapshot['posterior_ctrv'],
                  covariance_policy='original nominal-dt predicted covariance',
                  continuation='one intervention at query; original tracker thereafter')
    plans = {}
    for variant in VARIANTS:
        p = copy.deepcopy(common)
        p.update(variant=variant, injection='center_only', initial_position_source='frame anchor KF posterior',
                 velocity_source=snapshot['velocity_source'], geometry='original predicted size and yaw',
                 z_policy='hold posterior Z', nominal_baseline_elapsed_s=baseline['nominal_elapsed_s'])
        if variant == 'baseline':
            p.update(injection='none', position_m=baseline['position_m'], velocity_mps=baseline['velocity_mps'])
        elif variant == 'actual_dt_ctrv':
            p.update(injection='ctrv_mean', ctrv_state=np.asarray(actual_state).ravel().tolist(),
                     position_m=np.asarray(actual_state).ravel()[:3].tolist(),
                     velocity_mps=ctrv_state_to_pv(np.asarray(actual_state).ravel())[3:].tolist(),
                     geometry='original size; CTRV yaw propagated with actual dt')
        elif variant in ('ct_cv', 'ct_ca'):
            pred = shadow['queries'][0]['models'][variant]
            p.update(position_m=pred['position_m'], velocity_mps=pred['velocity_mps'], ode_prediction=pred)
            if variant == 'ct_ca':
                p['acceleration_estimate'] = shadow['acceleration_estimate']
        else:
            p.update(initial_state_pv=past_state, initial_position_source='frame anchor accepted GAAM-aligned detection',
                     velocity_source='XYZ finite difference of accepted aligned detections at frames {} and {}'.format(h0['frame_idx'], h1['frame_idx']),
                     z_policy='propagate estimated vz', position_m=past['position_m'], velocity_mps=past['velocity_mps'],
                     ode_prediction=past, finite_difference_interval_s=(h1['timestamp_us']-h0['timestamp_us'])/1e6)
        predicted_box = np.array(baseline['box'][:14])
        predicted_box[:3] = p['position_m']
        if variant == 'actual_dt_ctrv':
            predicted_box[:12] = model.getOutputInfo(actual_state)
        p.update(predicted_box=predicted_box.tolist(),
                 prediction_elapsed_s=baseline['nominal_elapsed_s'] if variant == 'baseline' else dt,
                 geometry_cost_function='1 - fast_giou_3d(original aligned detection, predicted box)',
                 detection_source='unchanged original high GAAM-aligned mix / pure3D inputs',
                 kf_velocity_policy='original CTRV speed/yaw/rate retained for center-only injection')
        plans[variant] = p
    return plans, snapshot, shadow


def replace_prediction(tracker, tid, plan):
    """Change mean consistently in the KF, FrameObject, box and geometry caches.

    CV/CA/old-XYZ controls replace XYZ only. Speed/yaw/rate and covariance stay
    original: mapping free Cartesian velocity to CTRV would change box yaw too.
    """
    motion = tracker.valid_tras[tid].motion_management
    frame = motion[tracker.frame_id]
    before_state, before_p = motion.state.copy(), motion.P.copy()
    state = motion.state.copy()
    if plan['injection'] == 'ctrv_mean':
        state[:] = np.mat(plan['ctrv_state']).T
    else:
        state[:3] = np.mat(plan['position_m']).T
    motion.state = state
    frame.predict_state = state
    info = motion.getOutputInfo(state)
    info[12] = frame.predict_infos[12]  # Preserve original confidence decay.
    boxes, bottoms, normals = arraydet2box(info, np.array([tid]), True)
    frame.predict_infos = np.r_[info, frame.predict_infos[14:]]
    frame.predict_box, frame.predict_bms, frame.predict_norm_bms = boxes[0], bottoms[0], normals[0]
    idx = list(tracker.tra_infos['all_valid_ids']).index(tid)
    for key, value in [('np_tras', frame.predict_infos), ('np_tras_bottom_corners', bottoms[0]),
                       ('np_tras_norm_corners', normals[0]), ('all_valid_boxes', boxes[0])]:
        tracker.tra_infos[key][idx] = value
    if not np.array_equal(before_p, motion.P):
        raise AssertionError('Prediction intervention changed covariance')
    if plan['injection'] == 'center_only' and not np.array_equal(before_state[3:], state[3:]):
        raise AssertionError('Center-only intervention changed other internal state')
    return {'before_state': np.asarray(before_state).ravel(), 'after_state': np.asarray(state).ravel(),
            'P_sha256': digest(before_p), 'covariance_unchanged': True}


class ReplayTracker(tracker_module.Tracker):
    """Instrument existing methods, without reimplementing matching or lifecycle."""
    def __init__(self, config, nusc, target_id, intervention_frame):
        super().__init__(config, nusc)
        self.target_id, self.intervention_frame = target_id, intervention_frame
        self.variant, self.plan = 'baseline', None
        self.records, self.lineages = [], {}
        self.injection_count = 0

    def tras_predict(self):
        self.record = {'frame_idx': self.frame_idx, 'internal_frame_id': self.frame_id,
                       'token': self.sample_token, 'is_keyframe': self.is_key_frame,
                       'timestamp_us': self.actual_timestamp_us, 'camera_timestamps_us': self.camera_times,
                       'stages': [], 'injection': None,
                       'before': {tid: np.asarray(t.motion_management.state).ravel().tolist()
                                  for tid, t in self.valid_tras.items()}}
        super().tras_predict()
        if self.is_key_frame and self.frame_idx == self.intervention_frame and self.plan and self.variant != 'baseline':
            if self.injection_count:
                raise AssertionError('More than one intervention')
            self.record['injection'] = replace_prediction(self, self.target_id, self.plan)
            self.injection_count += 1
        self.record['predictions'] = copy.deepcopy(self.tra_infos['np_tras']) if self.tra_infos else []

    def compute_cost(self, tra_infos, det_infos):
        result = super().compute_cost(tra_infos, det_infos)
        self.pending = dict(track_ids=tra_infos['np_tras'][:, 14].astype(int).copy(),
                            predictions=tra_infos['np_tras'].copy(), detections=det_infos['np_dets'].copy(),
                            cost=result['one_stage'].copy())
        return result

    def matching_cost(self, matrices, thresholds):
        result = super().matching_cost(matrices, thresholds)
        record = copy.deepcopy(self.pending)
        record.update(stage='mix' if thresholds is self.first_thre else 'pure3d',
                      thresholds=thresholds, assignment=result)
        self.record['stages'].append(record)
        return result

    def data_association(self):
        result = super().data_association()
        self.record['association'] = copy.deepcopy(result)
        return result

    def tras_update(self, association, data):
        ids = list(self.valid_tras)
        allocated, seed = [], self.id_seed
        # Record the original allocation order, including births, without choosing matches.
        for stage, key, count in [('mix', 'm_asso_1', data['mix_dets_num']),
                                  ('pure3d', 'm_asso_2', data['single_3d_dets_num'])]:
            for i in range(count):
                matched = i in association[key]
                tid = ids[association[key][i]] if matched else seed
                if not matched:
                    self.lineages[tid] = '{}:{}:{}'.format(self.sample_token, stage, i)
                    seed += 1
                allocated.append(dict(stage=stage, detection_index=i, track_id=tid,
                                      lineage=self.lineages[tid], kind='matched' if matched else 'birth'))
        self.record['allocations'] = allocated
        self.record['camera_matches'] = [dict(track_id=ids[int(i)], lineage=self.lineages[ids[int(i)]],
                    cameras=list(association['m_2d_det_info'][i])) for i in association['m_tras_idx3']]
        result = super().tras_update(association, data)
        if self.id_seed != seed:
            raise AssertionError('Recorded birth order differs from tracker')
        self.record['after'] = {tid: dict(state=np.asarray(t.motion_management.state).ravel(),
                        life=t.life_management.state, time_since_update=t.life_management.time_since_update,
                        updated=t[self.frame_id].update_box is not None, lineage=self.lineages[tid],
                        score=t.score_management[self.frame_id].final_score,
                        covariance_sha256=digest(t.motion_management.P))
                    for tid, t in {**self.valid_tras, **self.dead_tras}.items() if t.timestamp == self.frame_id}
        return result

    def advance(self, data, idx, timestamp, camera_times):
        self.frame_idx, self.actual_timestamp_us, self.camera_times = idx, timestamp, camera_times
        self.tracking(copy.deepcopy(data))
        self.record['output'] = ([] if 'no_val_track_result' in self.det_infos else
                                 [box_record(b) for b in self.det_infos['box_track_res']])
        self.records.append(clean(self.record))
        return self.records[-1]


def verify_output(record, originals):
    old = {b['tracking_id']: b for b in originals[record['token']]}
    new = {b['tracking_id']: b for b in record['output']}
    if set(old) != set(new):
        raise AssertionError('Baseline IDs differ at frame {}'.format(record['frame_idx']))
    error = 0.
    for tid in old:
        for field in ('translation', 'size', 'rotation', 'velocity', 'tracking_score'):
            error = max(error, float(np.max(np.abs(np.asarray(old[tid][field])-new[tid][field]))))
    if error != 0:
        raise AssertionError('Baseline numerical mismatch: {}'.format(error))
    return dict(frame_idx=record['frame_idx'], ids_equal=True, max_abs_error=error)


def verify_trace_step(record, archived):
    if len(record['stages']) != len(archived['stages']):
        raise AssertionError('Baseline matching cascade differs from archive')
    for current, old in zip(record['stages'], archived['stages']):
        for key in ('stage', 'track_ids', 'detections', 'predictions', 'cost', 'assignment'):
            if current[key] != old[key]:
                raise AssertionError('Baseline trace differs: frame {} {}'.format(record['frame_idx'], key))


def verify_local_intervention(records, case):
    """Only the target column may change at the intervention, before correction."""
    frame = case['query_frames'][0]
    base = next(r for r in records['baseline'] if r['is_keyframe'] and r['frame_idx'] == frame)
    result = {}
    for name, sequence in records.items():
        current = next(r for r in sequence if r['is_keyframe'] and r['frame_idx'] == frame)
        bp, cp = np.array(base['predictions']), np.array(current['predictions'])
        np.testing.assert_array_equal(bp[:, 14], cp[:, 14])
        others = bp[:, 14] != case['track_id']
        np.testing.assert_array_equal(bp[others], cp[others])
        first_base, first_current = base['stages'][0], current['stages'][0]
        if first_base['stage'] != 'mix' or first_current['stage'] != 'mix':
            raise ValueError('Local intervention verifier expects the mix stage first')
        np.testing.assert_array_equal(first_base['detections'], first_current['detections'])
        np.testing.assert_array_equal(np.array(first_base['cost'], float)[:, :, others],
                                      np.array(first_current['cost'], float)[:, :, others])
        result[name] = dict(other_predictions_identical=True,
                            other_first_stage_cost_columns_identical=True,
                            unchanged_track_count=int(np.sum(others)),
                            detections_identical=True)
    return result


def evaluate(records, plans, nusc, samples, validation, original_trace):
    """Post-hoc candidate labelling. No return value is fed back to tracking."""
    case = validation['case']
    frames = sorted({r['frame_idx'] for r in records['baseline'] if r['is_keyframe']})
    labels, candidates = {}, {}
    for frame in frames:
        anns = [nusc.get('sample_annotation', t) for t in samples[frame]['anns']]
        gt = next(a for a in anns if a['instance_token'] == case['gt_instance_id'])
        labels[frame] = gt
        step = key_step(original_trace, frame)
        options = [(stage, i, row) for stage, rows in [('mix', step['mix']), ('pure3d', step['pure3d'])]
                   for i, row in enumerate(rows) if int(row[-1]) == 2]
        stage, i, row = min(options, key=lambda x: np.linalg.norm(np.array(x[2][:2])-gt['translation'][:2]))
        distance = float(np.linalg.norm(np.array(row[:2])-gt['translation'][:2]))
        if distance > validation['comparison']['detection_gt_max_xy_m']:
            raise ValueError('No supported evaluation correspondence at frame {}'.format(frame))
        candidates[frame] = dict(stage=stage, index=i, detection=row, gt_xy_error_m=distance)
    anchor_record = next(r for r in records['baseline']
                         if r['is_keyframe'] and r['frame_idx'] == case['anchor_frame'])
    target_lineage = anchor_record['after'][str(case['track_id'])]['lineage']
    rows, impacts = [], []
    for variant, sequence in records.items():
        for record in sequence:
            frame = record['frame_idx']
            baseline = next(x for x in records['baseline'] if x['token'] == record['token'])
            base_alloc = {(a['stage'], a['detection_index']): a for a in baseline.get('allocations', [])}
            changed = []
            for a in record.get('allocations', []):
                b = base_alloc[(a['stage'], a['detection_index'])]
                if a['lineage'] != b['lineage'] or a['track_id'] != b['track_id'] or a['kind'] != b['kind']:
                    changed.append(dict(stage=a['stage'], detection_index=a['detection_index'],
                        baseline=b, variant=a, lineage_changed=a['lineage'] != b['lineage']))
            impacts.append(dict(variant=variant, frame_idx=frame, is_keyframe=record['is_keyframe'],
                                allocations_changed=changed, camera_matches=record.get('camera_matches', []),
                                baseline_camera_matches=baseline.get('camera_matches', [])))
            # Compare physical track histories by birth observation, not shifted ID numbers.
            before_by_lineage = {v['lineage']: v for v in baseline.get('after', {}).values()}
            current_by_lineage = {v['lineage']: v for v in record.get('after', {}).values()}
            state_changes = []
            for lineage in set(before_by_lineage) & set(current_by_lineage):
                if lineage == target_lineage:
                    continue
                a, b = before_by_lineage[lineage], current_by_lineage[lineage]
                if a != b:
                    state_changes.append(dict(lineage=lineage, baseline=a, variant=b))
            impacts[-1]['other_shared_lineage_state_changes'] = state_changes
            if not record['is_keyframe']:
                continue
            c, gt = candidates[frame], labels[frame]['translation']
            owner = next(a for a in record['allocations'] if a['stage']==c['stage'] and a['detection_index']==c['index'])
            pred = next((p for p in record['predictions'] if int(p[14]) == case['track_id']), None)
            stage = next((s for s in record['stages'] if s['stage']==c['stage'] and case['track_id'] in s['track_ids']), None)
            matrix_cost, selected_index, contenders = None, None, []
            if stage:
                col = stage['track_ids'].index(case['track_id'])
                val = stage['cost'][2][c['index']][col]
                matrix_cost = float(val) if isinstance(val, (int, float)) else None
                selected_index = next((d for d, t in zip(*stage['assignment'][:2]) if t==col), None)
                for j, tid in enumerate(stage['track_ids']):
                    cost = stage['cost'][2][c['index']][j]
                    if isinstance(cost, (int, float)) and tid != case['track_id']:
                        contenders.append(dict(track_id=tid, cost=cost, below_cost_limit=cost <= stage['thresholds']['2']))
            raw_cost = association_cost(pred[:14], c['detection']) if pred else None
            if matrix_cost is not None and not np.isclose(matrix_cost, raw_cost, rtol=0, atol=1e-12):
                raise AssertionError('Reported geometry cost differs from actual Hungarian input')
            distance = float(np.linalg.norm(np.array(pred[:3])-c['detection'][:3])) if pred else None
            gate = original_trace['config']['association']['voxel_mask_size']['2']
            threshold = original_trace['config']['association']['first_thre' if c['stage']=='mix' else 'second_thre']['2']
            state = record['after'].get(str(case['track_id']))
            target_output = next((b for b in record['output'] if b['tracking_id']==str(case['track_id'])), None)
            rows.append(dict(variant=variant, frame_idx=frame, timestamp_us=record['timestamp_us'],
                correct_stage=c['stage'], correct_detection_index=c['index'], detection_gt_xy_m=c['gt_xy_error_m'],
                prediction_xyz=pred[:3] if pred else None, gt_xyz=gt,
                prediction_gt_xy_m=float(np.linalg.norm(np.array(pred[:2])-gt[:2])) if pred else None,
                raw_geometry_cost=raw_cost, hungarian_matrix_cost=matrix_cost,
                spatial_distance_m=distance, spatial_gate_m=gate, cost_limit=threshold,
                gate_pass=distance < gate and raw_cost <= threshold if pred else False,
                selected_index=selected_index, correct_owner_id=owner['track_id'], owner_kind=owner['kind'],
                correct_detection_assigned_to_target=(owner['track_id']==case['track_id'] and owner['kind']=='matched'),
                target_life=state['life'] if state else None,
                target_time_since_update=state['time_since_update'] if state else None,
                target_in_output=target_output is not None,
                output_gt_xy_m=float(np.linalg.norm(np.array(target_output['translation'][:2])-gt[:2])) if target_output else None,
                competing_tracks=contenders))
    summary = {}
    for variant in plans:
        seq = [r for r in rows if r['variant']==variant]
        event = next(r for r in seq if r['frame_idx']==case['query_frames'][0])
        tail = [r for r in seq if r['frame_idx']>=case['query_frames'][0]]
        variant_impacts = [i for i in impacts if i['variant'] == variant]
        unrelated_changes = []
        for impact in variant_impacts:
            if not impact['is_keyframe']:
                continue
            candidate = candidates[impact['frame_idx']]
            unrelated_changes.extend(a for a in impact['allocations_changed']
                if a['lineage_changed'] and (a['stage'], a['detection_index']) != (candidate['stage'], candidate['index']))
        summary[variant] = dict(intervention_cost=event['raw_geometry_cost'],
            intervention_selected_correct=event['correct_detection_assigned_to_target'],
            correct_owner_ids=[r['correct_owner_id'] for r in seq],
            id_continuity_through_window=all(r['correct_detection_assigned_to_target'] for r in tail),
            competing_tracks_at_intervention=event['competing_tracks'],
            unrelated_detection_lineage_changes=len(unrelated_changes),
            other_shared_lineage_state_changes=sum(len(i['other_shared_lineage_state_changes']) for i in variant_impacts))
    return dict(rows=rows, summary=summary, impacts=impacts, evaluation_candidates=candidates,
                gt=labels, correspondence_rule='nearest same-class aligned detection to GT XY, evaluation only; not official IDS metric')


def run(config_path):
    started = time.time()
    cfg, validation = load_config(config_path)
    case = validation['case']
    trace_path = resolve(validation['source']['trace_root']) / (case['scene'].replace('-', '')+'_quantitative') / 'trace.json'
    trace = read_json(trace_path)
    results_path = resolve(validation['source']['results_path'])
    validate_inputs(validation, trace, results_path)
    tracker_cfg = yaml.safe_load(resolve(cfg['tracker_config']).read_text())
    if clean(tracker_cfg) != trace['config']:
        raise ValueError('Tracker configuration differs from original high trace')
    plans, snapshot, shadow = make_interventions(trace, validation)
    output = resolve(cfg['output_root']) / '{}_track{}_frames{}-{}'.format(case['scene'],case['track_id'],cfg['window']['start_frame'],cfg['window']['end_frame'])
    protected = [results_path.parent, trace_path.parent, resolve(validation['dataset']['root'])]
    if any(p==output or p in output.parents for p in protected):
        raise ValueError('Output must not be inside original data/result/trace directories')
    output.mkdir(parents=True, exist_ok=True)
    paths = dict(results=results_path, trace=trace_path, tracker_config=resolve(cfg['tracker_config']),
                 replay_config=resolve(config_path), validation_config=resolve(cfg['validation_config']),
                 detection_3d=resolve(cfg['detection_3d_path']), detection_2d=resolve(cfg['detection_2d_path']),
                 first_tokens=resolve(cfg['first_token_path']))
    before = {k: sha256(p) for k,p in paths.items()}
    write_json(output/'initial_conditions.json', dict(plans=plans, snapshot=snapshot, shadow=shadow))
    timing = output/'timing'; timing.mkdir(exist_ok=True)
    tracker_module.TIME_COST_ROOT = loader_module.TIME_COST_ROOT = str(timing)+'/'
    print('Loading high inputs and nuScenes...', flush=True)
    loader = loader_module.NuScenesloader(str(paths['detection_3d']),str(paths['detection_2d']),
                str(paths['first_tokens']),str(resolve(validation['dataset']['root'])),tracker_cfg)
    nusc = loader.nusc
    scene = next(s for s in nusc.scene if s['name']==case['scene'])
    samples, token = [], scene['first_sample_token']
    while token:
        sample=nusc.get('sample', token); samples.append(sample); token=sample['next']
    indices={s['token']:i for i,s in enumerate(samples)}
    loader.all_sample_token=[t for t in loader.all_sample_token if t.split('_')[0] in indices and indices[t.split('_')[0]]<=cfg['window']['end_frame']]
    loader.seq_id=loader.seq_first_token.index(scene['first_sample_token'])
    originals=read_json(results_path)['results']
    base=ReplayTracker(tracker_cfg,nusc,case['track_id'],case['query_frames'][0])
    branches={'baseline':base}; checks=[]; input_checks=[]; timing_rows=[]
    for i in range(len(loader.all_sample_token)):
        data=loader[i]; idx=indices[data['sample_token'].split('_')[0]]
        tokens=data.get('sample_data_tokens') or {cam:t for cam,t in samples[idx]['data'].items() if cam.startswith('CAM_')}
        camera_times={cam:nusc.get('sample_data',t)['timestamp'] for cam,t in tokens.items()}
        timestamp=samples[idx]['timestamp'] if data['is_key_frame'] else None
        signature=digest({k:data[k] for k in ('sample_token','mix_np_dets_3d','single_np_dets_3d','single_2d_info','mix_dict')})
        per_variant={}
        for name, tracker in branches.items():
            per_variant[name]=signature
            record=tracker.advance(data,idx,timestamp,camera_times)
            if name=='baseline':
                if data['is_key_frame']: checks.append(verify_output(record, originals))
                captured=next((s for s in trace['trace'] if s['token']==record['token']),None)
                if captured: verify_trace_step(record,captured)
        if digest({k:data[k] for k in ('sample_token','mix_np_dets_3d','single_np_dets_3d','single_2d_info','mix_dict')}) != signature:
            raise AssertionError('Shared input changed across branches')
        input_checks.append(dict(token=data['sample_token'],variant_input_hashes=per_variant))
        if data['is_key_frame'] and idx==case['anchor_frame']:
            actual=np.asarray(base.valid_tras[case['track_id']].motion_management.state).ravel()
            np.testing.assert_array_equal(actual, snapshot['posterior_ctrv'])
            for name in VARIANTS[1:]:
                branch=copy.deepcopy(base, {id(nusc):nusc})
                branch.variant,branch.plan=name,plans[name]
                branches[name]=branch
        print('Replay frame {} {} | {} variants'.format(idx,'key' if data['is_key_frame'] else 'mid',len(branches)),flush=True)
    if set(branches)!=set(VARIANTS) or any(t.injection_count!=(0 if name=='baseline' else 1) for name,t in branches.items()):
        raise AssertionError('Missing variant/intervention')
    records={name:[r for r in tracker.records if cfg['window']['start_frame']<=r['frame_idx']<=cfg['window']['end_frame']] for name,tracker in branches.items()}
    # Persist tracker results before fetching any target GT labels.
    write_json(output/'replay_records.json',records)
    intervention_checks = verify_local_intervention(records, case)
    evaluation=evaluate(records,plans,nusc,samples,validation,trace)
    after={k:sha256(p) for k,p in paths.items()}
    if before!=after: raise AssertionError('Original inputs changed during replay')
    anchor_internal=key_step(trace,case['anchor_frame'])['internal_frame_id']
    query_internal=key_step(trace,case['query_frames'][0])['internal_frame_id']
    for r in base.records:
        if anchor_internal<r['internal_frame_id']<=query_internal:
            tid=str(case['track_id']); state=np.array(r['after'][tid]['state']); pred=next(p for p in r['predictions'] if int(p[14])==case['track_id'])
            timing_rows.append(dict(frame_idx=r['frame_idx'],is_keyframe=r['is_keyframe'],
                internal_frame_id=r['internal_frame_id'],nominal_prediction_dt_s=tracker_cfg['basic']['LiDAR_interval'],
                sample_timestamp_us=r['timestamp_us'],camera_timestamps_us=r['camera_timestamps_us'],
                camera_elapsed_from_anchor_s={cam:(ts-snapshot['timestamp_us'])/1e6 for cam,ts in r['camera_timestamps_us'].items()},
                target_camera_matches=[x for x in r.get('camera_matches',[]) if x['track_id']==case['track_id']],
                target_update_box_written=r['after'][tid]['updated'],
                target_center_change_after_association_m=float(np.linalg.norm(state[:3]-pred[:3]))))
    code_paths=list((Path(__file__).resolve().parent).glob('*.py'))
    for folder in ('tracking','motion_module','pre_processing','geometry','dataloader','utils'):
        code_paths.extend(resolve(folder).glob('*.py'))
    report=dict(case=case,window=cfg['window'],scope='one-shot prediction-mean intervention; original cascade/KF/lifecycle continuation',
                plans=plans,timing=timing_rows,baseline_verification=checks,input_hashes=before,
                local_intervention_verification=intervention_checks,
                source_paths={k:str(v) for k,v in paths.items()},original_inputs_unchanged=True,
                variant_input_checks=input_checks,tracker_config=tracker_cfg,
                code_hashes={str(p.relative_to(resolve('.'))):sha256(p) for p in code_paths},
                evaluation=evaluation,seconds=time.time()-started)
    write_json(output/'report.json',clean(report))
    write_csv(output/'comparison.csv',evaluation['rows'])
    write_csv(output/'initial_conditions.csv',[dict(variant=k,**{x:v for x,v in p.items() if x!='variant'}) for k,p in plans.items()])
    from .replay_report import render_report
    render_report(output,report,records)
    print(json.dumps(clean(evaluation['summary']),indent=2),flush=True)
    print('OUTPUT {} | {:.1f}s'.format(output,report['seconds']),flush=True)
    return report


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='ctmot/proto/config/replay_scene1072.yaml')
    args=parser.parse_args()
    run(resolve(args.config))


if __name__=='__main__':
    main()
