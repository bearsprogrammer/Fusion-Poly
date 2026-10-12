"""Verified 3D assignments for visualization; never infer links by proximity."""
import contextlib
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from utils import viz as rgb


def cache_path(cfg):
    directory = cfg['paths'].get('association_dir')
    directory = (rgb.resolve_path(directory) if directory else
                 rgb.resolve_path(cfg['paths']['results_json']).parent / 'associations')
    return directory / (cfg['selection']['scene_name'] + '.json')


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def assigned_observations(association, track_ids, frame_data):
    """Use the same index-to-ID mapping as Tracker.tras_update, before mutation.

    Births and camera-only updates intentionally have no association edge.
    Mixed detections are the actual post-GAAM boxes used by the tracker.
    """
    records = []
    for stage, key, array in [('mix', 'm_asso_1', 'mix_np_dets_3d'),
                              ('pure3d', 'm_asso_2', 'single_np_dets_3d')]:
        for det_index, track_index in association[key].items():
            row = frame_data[array][det_index]
            records.append(dict(track_id=str(track_ids[track_index]), stage=stage,
                                detection_index=int(det_index),
                                detection=dict(translation=row[:3].tolist(), size=row[3:6].tolist(),
                                               rotation=row[8:12].tolist(), score=float(row[12]),
                                               name=rgb.DETECTION_CLASSES[int(row[13])])))
    if len({r['track_id'] for r in records}) != len(records):
        raise ValueError('Duplicate 3D assignment for one track')
    return records


def edge_segments(records, tracks, transform):
    """Visible POST-update track centers -> assigned 3D centers, in view frame.

    Detection display filters do not modify the actual assignment. Track class
    and score filtering is already applied by SceneData.objects().
    """
    visible = {str(t['id']): t for t in tracks if not t['gt']}
    segments = []
    for record in records:
        track = visible.get(record['track_id'])
        if track is None:
            continue
        detection = record['detection']
        if track['name'] != detection['name']:
            raise ValueError('Association class differs from displayed track')
        points = np.asarray([track['translation'], detection['translation']], dtype=np.float64)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError('Invalid association center')
        if np.linalg.norm(points[1] - points[0]) < 1e-9:
            continue  # Coincident centers have no visible segment.
        segments.append(points @ transform[:3, :3].T + transform[:3, 3])
    return np.asarray(segments, dtype=np.float64).reshape((-1, 2, 3))


def load_cache(cfg, scene, samples):
    path = cache_path(cfg)
    if not path.is_file():
        raise FileNotFoundError('Missing association cache: {}. Run utils/viz_3D.py '
                                'with the same --config and --export-associations first.'.format(path))
    data = json.loads(path.read_text())
    if (data.get('schema_version') != 1 or data.get('verified') is not True or
            data.get('scene_token') != scene['token'] or
            data.get('sample_tokens') != [s['token'] for s in samples] or
            set(data.get('matches', {})) != {s['token'] for s in samples}):
        raise ValueError('Invalid/incomplete association cache; re-export: ' + str(path))
    if data['results_sha256'] != digest(rgb.resolve_path(cfg['paths']['results_json'])):
        raise ValueError('Association cache belongs to different tracking results; re-export: ' + str(path))
    for records in data['matches'].values():
        ids = set(); detections = set()
        for record in records:
            detection = record['detection']
            key = (record['stage'], record['detection_index'])
            center = np.asarray(detection['translation'], dtype=float)
            if (record['track_id'] in ids or key in detections or record['stage'] not in ('mix', 'pure3d') or
                    center.shape != (3,) or not np.isfinite(center).all() or detection['name'] not in rgb.CLASSES):
                raise ValueError('Invalid/duplicate assignment in cache: ' + str(path))
            ids.add(record['track_id']); detections.add(key)
    return data


def export_cache(cfg):
    """Replay ONE complete scene (including high mids), verify output, save links.

    No evaluation, no changes to tracker algorithms or existing results.json.
    The temporary timing paths restore the loader/tracker globals on exit.
    """
    from dataloader import nusc_loader as loader_module
    from tracking import nusc_tracker as tracker_module

    class RecordingTracker(tracker_module.Tracker):
        def tracking(self, frame_data):
            self.assignments = []
            return super().tracking(frame_data)

        def tras_update(self, association, frame_data):
            self.assignments = assigned_observations(association, list(self.valid_tras), frame_data)
            return super().tras_update(association, frame_data)

    started = time.perf_counter()
    config_path = cfg['paths'].get('association_tracker_config')
    if not config_path:
        raise ValueError('Set paths.association_tracker_config to the config used for these results')
    tracker_config = yaml.safe_load(rgb.resolve_path(config_path).read_text())
    if tracker_config['basic']['split'] != 'val':
        raise ValueError('Association export currently supports val only')
    paths = {key: rgb.resolve_path(cfg['paths'][key]) for key in
             ('results_json', 'detection_3d_json', 'detection_2d_json', 'dataset_root')}
    first_tokens = rgb.resolve_path(cfg['paths'].get('first_token_json',
                                   'data/utils/first_token_table/trainval/nuscenes_val_first_sample_tokens.json'))
    source_hash = digest(paths['results_json'])
    original = json.loads(paths['results_json'].read_text())['results']
    output = cache_path(cfg); output.parent.mkdir(parents=True, exist_ok=True)
    old_timing = loader_module.TIME_COST_ROOT, tracker_module.TIME_COST_ROOT
    log_path = output.with_suffix('.replay.log')
    print('Replaying full {} for actual assignments; log: {}'.format(cfg['selection']['scene_name'], log_path), flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix='fusionpoly-associations-') as temporary:
            loader_module.TIME_COST_ROOT = tracker_module.TIME_COST_ROOT = temporary + '/'
            with log_path.open('w') as log, contextlib.redirect_stdout(log):
                loader = loader_module.NuScenesloader(str(paths['detection_3d_json']), str(paths['detection_2d_json']),
                    str(first_tokens), str(paths['dataset_root']), tracker_config)
                scene = next((s for s in loader.nusc.scene if s['name'] == cfg['selection']['scene_name']), None)
                if scene is None or scene['first_sample_token'] not in loader.seq_first_token:
                    raise ValueError('Requested scene is not in the val input')
                tokens = []; token = scene['first_sample_token']
                while token:
                    tokens.append(token); token = loader.nusc.get('sample', token)['next']
                token_set = set(tokens)
                loader.all_sample_token = [t for t in loader.all_sample_token if t.split('_')[0] in token_set and
                    (tracker_config['basic']['freq'] == 'high' or '_' not in t)]
                loader.seq_id = loader.seq_first_token.index(scene['first_sample_token'])
                tracker = RecordingTracker(tracker_config, loader.nusc)
                matches = {}; maximum_error = 0.
                for index in range(len(loader.all_sample_token)):
                    frame = loader[index]; tracker.tracking(frame)
                    if not frame['is_key_frame']:
                        continue
                    token = frame['sample_token']
                    boxes = [] if 'no_val_track_result' in frame else frame['box_track_res']
                    boxes = sorted(boxes, key=lambda b: -b.score)[:500]
                    current = {str(b.tracking_id): b for b in boxes}
                    expected = {b['tracking_id']: b for b in original.get(token, [])}
                    if token not in original or set(current) != set(expected):
                        raise ValueError('Replay track IDs differ from saved results at ' + token)
                    for tid, box in current.items():
                        row = expected[tid]
                        actual = dict(translation=box.center, size=box.wlh, rotation=box.orientation.elements,
                                      velocity=box.velocity[:2], tracking_score=box.score)
                        if row['tracking_name'] != box.name:
                            raise ValueError('Replay class differs from saved results')
                        for key, value in actual.items():
                            error = np.abs(np.asarray(value) - row[key])
                            if not np.isfinite(error).all() or np.max(error) > 1e-8:
                                raise ValueError('Replay differs from saved results at {} / {} / {}'.format(token, tid, key))
                            maximum_error = max(maximum_error, float(np.max(error)))
                    matches[token] = [r for r in tracker.assignments if r['track_id'] in current]
                    print('Verified', token, 'edges', len(matches[token]), flush=True)
                if set(matches) != token_set:
                    raise ValueError('Replay does not cover every keyframe')
    finally:
        loader_module.TIME_COST_ROOT, tracker_module.TIME_COST_ROOT = old_timing
    if digest(paths['results_json']) != source_hash:
        raise ValueError('Tracking results changed during replay')
    report = dict(schema_version=1, verified=True, scene_name=scene['name'], scene_token=scene['token'],
                  results_sha256=source_hash, sample_tokens=tokens, matches=matches,
                  tracker_config=tracker_config, max_output_error=maximum_error,
                  replay_steps=len(loader.all_sample_token), seconds=time.perf_counter()-started,
                  semantics='post-update displayed track -> actual assigned 3D detection; no births or 2D-only updates')
    # Replace only after the entire replay has passed. Preserve a previous cache on failure.
    with tempfile.NamedTemporaryFile(mode='w', dir=str(output.parent), delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, allow_nan=False)
    try:
        temporary.replace(output)
    finally:
        if temporary.exists(): temporary.unlink()
    print('Saved {}: {} keyframes, {} assignments, max output error {:.3g}, {:.1f}s'.format(
        output, len(tokens), sum(map(len, matches.values())), maximum_error, report['seconds']), flush=True)
    return report
