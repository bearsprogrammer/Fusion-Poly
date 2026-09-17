"""Extract val IDS events at saved best-MOTA thresholds; no AMOTA sweep or RGB rendering."""
import os
import sys

if __name__ == '__main__' and not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from nuscenes.eval.tracking.data_classes import TrackingConfig
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom

ROOT = Path(__file__).resolve().parents[1]


def path_from_root(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def extract_scene_events(gt_frames, pred_frames, class_name, threshold, distance_threshold, frame_meta):
    """Same matching loop as devkit accumulate_threshold; retain original IDs and frame mapping."""
    acc = MOTAccumulatorCustom()
    frame_map = {}
    frame_id = 0
    for timestamp, all_gt in gt_frames.items():
        gt = [b for b in all_gt if b.tracking_name == class_name]
        pred = [b for b in pred_frames[timestamp]
                if b.tracking_name == class_name and b.tracking_score >= threshold]
        if not gt and not pred:
            continue
        distances = (euclidean_distances(np.array([b.translation[:2] for b in gt]),
                                        np.array([b.translation[:2] for b in pred]))
                     if gt and pred else np.ones((0, 0)))
        distances[distances >= distance_threshold] = np.nan
        acc.update([b.tracking_id for b in gt], [b.tracking_id for b in pred], distances, frameid=frame_id)
        frame_map[frame_id] = frame_meta[timestamp]
        frame_id += 1
    events, previous = [], {}
    counts = Counter()
    for (fid, _), row in acc.events.iterrows():
        event_type = row['Type']
        counts[event_type] += 1
        if event_type not in ('MATCH', 'SWITCH'):
            continue
        gt_id, pred_id = str(row['OId']), str(row['HId'])
        current = frame_map[fid]
        if event_type == 'SWITCH':
            if gt_id not in previous:
                raise RuntimeError('SWITCH without a previous GT match')
            old_id, old_frame = previous[gt_id]
            events.append(dict(current, class_name=class_name, gt_instance_id=gt_id,
                               previous_track_id=old_id, track_id=pred_id,
                               previous_match_frame_idx=old_frame['frame_idx'],
                               previous_match_sample_token=old_frame['sample_token'],
                               previous_match_timestamp=old_frame['timestamp'],
                               threshold=threshold, match_distance_m=float(row['D'])))
        previous[gt_id] = (pred_id, current)
    return events, counts


def group_ranges(events, scene_lengths, context):
    """Merge overlapping context windows for the same scene/class/GT instance."""
    grouped = defaultdict(list)
    for event in events:
        grouped[(event['scene_name'], event['class_name'], event['gt_instance_id'])].append(event)
    ranges = []
    for (scene, cls, gt_id), group in sorted(grouped.items()):
        current = None
        for event in sorted(group, key=lambda e: e['frame_idx']):
            start = max(0, min(event['frame_idx'], event['previous_match_frame_idx']) - context)
            end = min(scene_lengths[scene] - 1, event['frame_idx'] + context)
            if current is None or start > current['frame_end'] + 1:
                current = {'scene_name': scene, 'scene_idx': int(scene.split('-')[1]),
                           'class_name': cls, 'gt_instance_id': gt_id,
                           'frame_start': start, 'frame_end': end, 'event_ids': [],
                           'switch_frames': [], 'track_ids': [], 'ids_count': 0}
                ranges.append(current)
            current['frame_end'] = max(current['frame_end'], end)
            current['event_ids'].append(event['event_id'])
            current['switch_frames'].append(event['frame_idx'])
            current['ids_count'] += 1
            for track_id in (event['previous_track_id'], event['track_id']):
                if track_id not in current['track_ids']:
                    current['track_ids'].append(track_id)
    return sorted(ranges, key=lambda r: (-r['ids_count'], r['scene_name'], r['frame_start'], r['class_name']))


def run(args):
    start = time.perf_counter()
    result_path, eval_path = path_from_root(args.result_path), path_from_root(args.eval_path)
    output, dataroot = path_from_root(args.output_dir), path_from_root(args.nusc_path)
    summary = read_json(eval_path / 'metrics_summary.json')
    details = read_json(eval_path / 'metrics_details.json')
    cfg = TrackingConfig.deserialize(summary['cfg'])
    if cfg.dist_fcn != 'center_distance':
        raise ValueError('Only the official center_distance matching is supported')
    thresholds = {}
    for cls in cfg.class_names:
        values = np.asarray(details[cls]['mota'], dtype=float)
        if np.all(np.isnan(values)):
            raise ValueError('No valid best-MOTA threshold for ' + cls)
        index = int(np.nanargmax(values))  # Same first-maximum tie rule as official evaluate.py.
        threshold = float(details[cls]['confidence'][index])
        if not np.isfinite(threshold):
            raise ValueError('Invalid saved threshold for ' + cls)
        thresholds[cls] = threshold
    print('Loading official val evaluation inputs...', flush=True)
    evaluator = TrackingEval(config=cfg, result_path=str(result_path), eval_set='val',
                             output_dir=str(output / '_evaluation'), nusc_version='v1.0-trainval',
                             nusc_dataroot=str(dataroot), verbose=False)
    # Small metadata tables suffice to recover original zero-based keyframe indices.
    metadata = dataroot / 'v1.0-trainval'
    samples = {s['token']: s for s in read_json(metadata / 'sample.json')}
    scenes = {s['token']: s for s in read_json(metadata / 'scene.json')}
    frame_meta, scene_lengths = {}, {}
    for scene_token in evaluator.tracks_gt:
        scene = scenes[scene_token]
        frame_meta[scene_token] = {}
        token, index = scene['first_sample_token'], 0
        while token:
            sample = samples[token]
            frame_meta[scene_token][sample['timestamp']] = {
                'scene_name': scene['name'], 'scene_idx': int(scene['name'].split('-')[1]),
                'scene_token': scene_token, 'frame_idx': index,
                'timestamp': sample['timestamp'], 'sample_token': token}
            token, index = sample['next'], index + 1
        scene_lengths[scene['name']] = index
    load_seconds = time.perf_counter() - start
    all_events, validation = [], {}
    for cls in cfg.class_names:
        counts = Counter()
        for scene_token in evaluator.tracks_gt:
            events, scene_counts = extract_scene_events(
                evaluator.tracks_gt[scene_token], evaluator.tracks_pred[scene_token], cls,
                thresholds[cls], cfg.dist_th_tp, frame_meta[scene_token])
            all_events.extend(events)
            counts.update(scene_counts)
        # nuScenes maps the saved 'tp' field to motmetrics num_matches,
        # which excludes SWITCH events (unlike num_detections).
        measured = {'ids': counts['SWITCH'], 'fp': counts['FP'], 'fn': counts['MISS'],
                    'tp': counts['MATCH']}
        expected = {key: int(summary['label_metrics'][key][cls]) for key in measured}
        validation[cls] = {'threshold': thresholds[cls], 'measured': measured,
                           'expected': expected, 'matches': measured == expected}
        print('{}: threshold={:.6f}, IDS={}/{} {}'.format(
            cls, thresholds[cls], measured['ids'], expected['ids'],
            'PASS' if measured == expected else 'MISMATCH'), flush=True)
    all_events.sort(key=lambda e: (e['scene_name'], e['frame_idx'], e['class_name'], e['gt_instance_id']))
    for i, event in enumerate(all_events):
        event['event_id'] = i
    ranges = group_ranges(all_events, scene_lengths, args.context_frames)
    passed = all(v['matches'] for v in validation.values()) and len(all_events) == int(summary['ids'])
    report = {'split': 'val', 'version': 'v1.0-trainval', 'results_path': str(result_path),
              'results_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
              'eval_path': str(eval_path), 'dataset_root': str(dataroot),
              'frame_index_convention': 'zero-based keyframe index within scene; ranges inclusive',
              'score_policy': 'saved per-class best-MOTA threshold; official filtering, track mean scores and interpolation',
              'context_frames': args.context_frames, 'scene_count': len(scene_lengths),
              'sample_count': len(evaluator.sample_tokens), 'ids_count': len(all_events),
              'expected_ids': int(summary['ids']), 'validation_passed': passed, 'validation': validation,
              'load_seconds': load_seconds, 'total_seconds': time.perf_counter() - start}
    for filename, content in [('ids_events.json', dict(report, events=all_events)),
                              ('inspection_ranges.json', dict(report, ranges=ranges))]:
        (output / filename).write_text(json.dumps(content, indent=2) + '\n')
    lines = ['VAL IDS: {} | ranges: {} | validation: {}'.format(len(all_events), len(ranges), passed),
             'Frames are zero-based inclusive; track IDs are scene-local. Ranges include previous match and context.']
    for r in ranges:
        lines.append('{} | frames {:03d}-{:03d} | {} | IDS {} | track_id {} | GT {}'.format(
            r['scene_name'], r['frame_start'], r['frame_end'], r['class_name'], r['ids_count'],
            ','.join(r['track_ids']), r['gt_instance_id']))
    (output / 'inspection_ranges.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:2 + args.top]))
    print('Full report: {} | elapsed {:.1f}s'.format(output, report['total_seconds']))
    if not passed:
        raise RuntimeError('Extracted counts differ from saved metrics. Do not treat output as validated; '
                           'check results/metrics pairing and devkit version.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nusc_path', default='/home/allen-5070ti/dataset/nuScenes/full_trainval')
    parser.add_argument('--result_path', default='Fusion_Poly_EXP/result/nusc_config_high/results.json')
    parser.add_argument('--eval_path', default='Fusion_Poly_EXP/eval_results/nusc_config_high/tracking')
    parser.add_argument('--output_dir', default='Fusion_Poly_EXP/inspection/nusc_config_high')
    parser.add_argument('--context_frames', type=int, default=5)
    parser.add_argument('--top', type=int, default=20, help='Console range limit; files always contain all events/ranges')
    args = parser.parse_args()
    if args.context_frames < 0 or args.top < 0:
        parser.error('--context_frames and --top must be nonnegative')
    run(args)
