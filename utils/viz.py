"""Render saved nuScenes tracks on RGB keyframes, without tracking or evaluation.

Run from the repository: python -m utils.viz --config config/viz_rgb.yaml
GT: white dashed boxes. Predictions: solid boxes with stable ID colors.
All relative paths are resolved against the repository root.
"""
import os
import sys

# Direct script execution puts utils/ ahead of the standard library, causing
# utils/math.py to shadow math during datetime/NumPy initialization.
# Match the import root used by `python -m utils.viz` before other imports.
if __name__ == '__main__' and not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.tracking.utils import category_to_tracking_name
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck'}
CAMERAS = {'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'}
DETECTION_CLASSES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']
DET2D_COLOR = (255, 255, 0)  # OpenCV BGR: cyan.
DET3D_COLOR = (255, 0, 255)  # Magenta.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    with resolve_path(path).open() as stream:
        cfg = yaml.safe_load(stream)
    selection, render = cfg['selection'], cfg['render']
    render['info_level'].setdefault('text_detection_legend', True)
    render['cube_level'].setdefault('cube_tracking_overlay', True)
    detection_defaults = dict(cube_2d_overlay=False, cube_3d_overlay=False, cube_heading=True,
                              text_class=False, text_score=False, score_threshold_2d=0.0,
                              score_threshold_3d=0.0, include_mid_frames=False)
    det = render.setdefault('detection_level', {})
    if not isinstance(det, dict) or set(det) - set(detection_defaults):
        raise ValueError('Invalid render.detection_level options')
    for key, default in detection_defaults.items():
        det.setdefault(key, default)
        if type(default) is bool and type(det[key]) is not bool:
            raise ValueError('render.detection_level.' + key + ' must be a boolean')
        if key.startswith('score_threshold') and (type(det[key]) not in (int, float) or not 0 <= det[key] <= 1):
            raise ValueError('render.detection_level.' + key + ' must be in [0, 1]')
    if det['include_mid_frames'] and not det['cube_2d_overlay']:
        raise ValueError('include_mid_frames requires cube_2d_overlay')
    for dimension in ('2d', '3d'):
        if det['cube_' + dimension + '_overlay'] and not cfg['paths'].get('detection_' + dimension + '_json'):
            raise ValueError('Missing paths.detection_' + dimension + '_json')
    groups = {
        'info_level': {'text_camera_name', 'text_scene_idx', 'text_frame_index',
                       'text_threshold', 'text_sample_token', 'text_detection_legend'},
        'cube_level': {'cube_gt_overlay', 'cube_tracking_overlay', 'cube_heading', 'text_track_id',
                       'text_class', 'text_score', 'text_gt_id'},
    }
    legacy = [key for key in render if key.startswith('show_') or key == 'gt_overlay']
    if legacy:
        raise ValueError('Legacy render options {}: move to info_level/cube_level with text_/cube_ prefixes; '
                         'see config/viz_rgb.yaml'.format(', '.join(legacy)))
    for group, keys in groups.items():
        values = render.get(group)
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError('render.{} must contain exactly: {}'.format(group, ', '.join(sorted(keys))))
        if any(type(value) is not bool for value in values.values()):
            raise ValueError('render.{} options must be YAML booleans (true/false)'.format(group))
    classes = selection.get('classes')
    if classes is not None and (not isinstance(classes, list) or not classes
                                or any(c not in CLASSES for c in classes)):
        raise ValueError('selection.classes must be null or a nonempty list of: ' + ', '.join(sorted(CLASSES)))
    cameras = render['cameras']
    if not isinstance(cameras, list) or not cameras or any(c not in CAMERAS for c in cameras):
        raise ValueError('render.cameras must be a nonempty list of nuScenes camera names')
    if len(set(cameras)) != len(cameras):
        raise ValueError('Duplicate cameras')
    for key in ('max_frames', 'max_scenes'):
        value = selection.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('selection.' + key + ' must be null or a positive integer')
    if type(selection['start_frame']) is not int or selection['start_frame'] < 0:
        raise ValueError('start_frame must be a nonnegative integer')
    if not 0 <= render['score_threshold'] <= 1:
        raise ValueError('score_threshold must be in [0, 1]')
    if type(render['jpeg_quality']) is not int or not 1 <= render['jpeg_quality'] <= 100:
        raise ValueError('jpeg_quality must be an integer in [1, 100]')
    return cfg


def track_color(scene_token, track_id):
    digest = hashlib.sha256((scene_token + ':' + str(track_id)).encode()).digest()
    hsv = np.uint8([[[int.from_bytes(digest[:2], 'big') % 180, 210, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def project_segment(a, b, intrinsic, width, height, near=0.1):
    """Clip in camera space before perspective division, then in pixel space."""
    a, b = np.asarray(a, dtype=float).copy(), np.asarray(b, dtype=float).copy()
    if not np.isfinite([a, b]).all() or max(a[2], b[2]) < near:
        return None
    if a[2] < near:
        a += (b - a) * ((near - a[2]) / (b[2] - a[2]))
    if b[2] < near:
        b += (a - b) * ((near - b[2]) / (a[2] - b[2]))
    points = np.asarray(intrinsic) @ np.column_stack((a, b))
    pixels = (points[:2] / points[2]).T
    # Liang-Barsky clipping avoids overflowing OpenCV integer coordinates.
    p, delta = pixels[0], pixels[1] - pixels[0]
    lo, hi = 0.0, 1.0
    for axis, limit in ((0, width - 1), (1, height - 1)):
        if abs(delta[axis]) < 1e-12:
            if not 0 <= p[axis] <= limit:
                return None
        else:
            t0, t1 = sorted(((0 - p[axis]) / delta[axis], (limit - p[axis]) / delta[axis]))
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return None
    return tuple(np.rint(p + lo * delta).astype(int)), tuple(np.rint(p + hi * delta).astype(int))


def project_box_edges(box, pose, calibration, width, height, heading=True):
    camera_box = box.copy()
    camera_box.translate(-np.asarray(pose['translation']))
    camera_box.rotate(Quaternion(pose['rotation']).inverse)
    camera_box.translate(-np.asarray(calibration['translation']))
    camera_box.rotate(Quaternion(calibration['rotation']).inverse)
    corners = camera_box.corners()
    segments = [(corners[:, i], corners[:, j]) for i, j in EDGES]
    if heading:
        segments.append((corners[:, [2, 3, 7, 6]].mean(axis=1), corners[:, [2, 3]].mean(axis=1)))
    projected = [project_segment(a, b, calibration['camera_intrinsic'], width, height)
                 for a, b in segments]
    return [segment for segment in projected if segment is not None]


def draw_segment(image, a, b, color, dashed):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = np.linalg.norm(b - a)
    intervals = [(0, 1)] if not dashed else [(s / max(length, 1), min(s + 7, length) / max(length, 1))
                                            for s in np.arange(0, length, 12)]
    for start, end in intervals:
        p = tuple(np.rint(a + start * (b - a)).astype(int))
        q = tuple(np.rint(a + end * (b - a)).astype(int))
        cv2.line(image, p, q, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.line(image, p, q, color, 2, cv2.LINE_AA)


def draw_text(image, text, origin, color=(255, 255, 255)):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def render_camera_frame(image, objects, pose, calibration, scene_token, options):
    """Draw GT first and predictions second; mutate and return the RGB source (OpenCV BGR)."""
    height, width = image.shape[:2]
    cube = options['cube_level']
    for obj in objects:
        if obj['gt'] and not cube['cube_gt_overlay']:
            continue
        if not obj['gt'] and not cube.get('cube_tracking_overlay', True):
            continue
        box = Box(obj['translation'], obj['size'], Quaternion(obj['rotation']))
        segments = project_box_edges(box, pose, calibration, width, height, cube['cube_heading'])
        if not segments:
            continue
        gt = obj['gt']
        color = (255, 255, 255) if gt else track_color(scene_token, obj['id'])
        for a, b in segments:
            draw_segment(image, a, b, color, dashed=gt)
        label = []
        if gt and cube['text_gt_id']:
            label.append('GT:' + str(obj['id']))
        if not gt and cube['text_track_id']:
            label.append('ID:' + str(obj['id']))
        if cube['text_class']:
            label.append(obj['name'])
        if not gt and cube['text_score']:
            label.append('{:.2f}'.format(obj['score']))
        if label:
            points = np.asarray(segments).reshape(-1, 2)
            text = ' '.join(label)
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            x = int(np.clip(points[:, 0].min(), 0, max(0, width - text_width - 2)))
            y = int(np.clip(points[:, 1].min() - 5, 42, height - 5))
            draw_text(image, text, (x, y), color)
    return image


def load_detections(paths, options):
    """Load only enabled inputs. No detector rerun, NMS or fusion."""
    loaded = {}
    for dimension in ('2d', '3d'):
        if options['cube_' + dimension + '_overlay']:
            with resolve_path(paths['detection_' + dimension + '_json']).open() as stream:
                data = json.load(stream)
            loaded[dimension] = data['results'] if dimension == '3d' else data
    return loaded


def detection_label(image, prefix, name, score, anchor, color, options):
    parts = []
    if options['text_class']:
        parts.append(name)
    if options['text_score']:
        parts.append('{:.2f}'.format(score))
    if parts:
        text = prefix + ' ' + ' '.join(parts)
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)[0][0]
        x = int(np.clip(anchor[0], 0, max(0, image.shape[1] - width - 2)))
        y = int(np.clip(anchor[1] - 5, 65, image.shape[0] - 5))
        draw_text(image, text, (x, y), color)


def render_2d_detections(image, boxes, classes, options):
    """Draw packed [x1,y1,x2,y2,score,class_id] rows; return visible box count."""
    count = 0
    height, width = image.shape[:2]
    for row in boxes:
        if len(row) != 6 or not np.isfinite(row).all():
            raise ValueError('Invalid 2D detection row: ' + str(row))
        x1, y1, x2, y2, score, label = row
        if int(label) != label or not 0 <= label < len(DETECTION_CLASSES):
            raise ValueError('Unexpected packed 2D class ID: ' + str(label))
        name = DETECTION_CLASSES[int(label)]
        if name not in classes or score < options['score_threshold_2d']:
            continue
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Inverted/empty 2D detection box')
        if x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
            continue
        x1, x2 = np.clip([x1, x2], 0, width - 1).astype(int)
        y1, y2 = np.clip([y1, y2], 0, height - 1).astype(int)
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for i in range(4):
            draw_segment(image, points[i], points[(i + 1) % 4], DET2D_COLOR, False)
        detection_label(image, 'D2', name, score, (x1, y1), DET2D_COLOR, options)
        count += 1
    return count


def render_3d_detections(image, boxes, pose, calibration, classes, options):
    """Project raw global detector cuboids using the camera's own ego pose."""
    count = 0
    height, width = image.shape[:2]
    for det in boxes:
        name, score = det['detection_name'], det['detection_score']
        if name not in classes or score < options['score_threshold_3d']:
            continue
        box = Box(det['translation'], det['size'], Quaternion(det['rotation']))
        segments = project_box_edges(box, pose, calibration, width, height, options['cube_heading'])
        if not segments:
            continue
        for a, b in segments:
            draw_segment(image, a, b, DET3D_COLOR, True)
        anchor = np.asarray(segments).reshape(-1, 2).min(axis=0)
        detection_label(image, 'D3', name, score, anchor, DET3D_COLOR, options)
        count += 1
    return count


def detection_legend(options, mid=False):
    parts = []
    if options['cube_2d_overlay']:
        parts.append('D2 cyan >= {:g}'.format(options['score_threshold_2d']))
    if options['cube_3d_overlay'] and not mid:
        parts.append('D3 magenta dashed >= {:g}'.format(options['score_threshold_3d']))
    return ' | '.join(parts)


def frame_header(scene_name, camera, index, token, options):
    info = options['info_level']
    fields = [
        ('text_scene_idx', scene_name),
        ('text_camera_name', camera),
        ('text_frame_index', 'frame {:06d}'.format(index)),
        ('text_threshold', 'track score >= {:g}'.format(options['score_threshold'])),
        ('text_sample_token', token),
    ]
    return ' | '.join(value for key, value in fields if info[key])


def select_scenes(nusc, predictions, selection, results_path):
    scenes_with_results = {nusc.get('sample', token)['scene_token'] for token in predictions}
    available = sorted((s for s in nusc.scene if s['token'] in scenes_with_results),
                       key=lambda s: s['name'])
    requested = selection.get('scene_name')
    if requested:
        scene = next((s for s in nusc.scene if s['name'] == requested), None)
        if scene is None:
            raise ValueError('Scene {} does not exist in the loaded dataset.'.format(requested))
        if scene['token'] not in scenes_with_results:
            from nuscenes.utils.splits import create_splits_scenes
            splits = create_splits_scenes()
            split = next((name for name in ('train', 'val', 'test') if requested in splits[name]), 'unknown')
            examples = ', '.join(s['name'] for s in available[:5])
            raise ValueError(
                'Scene {} exists in the dataset (split: {}), but has no sample entries in {}. '
                'Render requires tracking results for this scene. Select a scene present in results '
                '(examples: {}), set selection.scene_name to null, or provide results for {}.'
                .format(requested, split, results_path, examples, requested))
        available = [scene]
    return available[:selection.get('max_scenes')]


def render_tracking_results(cfg):
    started = time.perf_counter()
    paths, selection, options = cfg['paths'], cfg['selection'], cfg['render']
    dataset_root = resolve_path(paths['dataset_root'])
    results_path = resolve_path(paths['results_json'])
    output = resolve_path(paths['output_dir'])
    with results_path.open() as stream:
        predictions = json.load(stream)['results']
    if not predictions:
        raise ValueError('No sample tokens in results.json')
    nusc = NuScenes(version=cfg['dataset']['version'], dataroot=str(dataset_root), verbose=False)
    scenes = select_scenes(nusc, predictions, selection, results_path)
    if not scenes:
        raise ValueError('No requested scene found in results')
    classes = set(selection.get('classes') or CLASSES)
    det_options = options['detection_level']
    detections = load_detections(paths, det_options)
    load_seconds = time.perf_counter() - started
    manifest = {'config': cfg, 'results_path': str(results_path), 'dataset_root': str(dataset_root),
                'notes': ['White dashed GT, solid ID-colored predictions.',
                          'Raw 2D cyan, raw 3D magenta dashed; detection confidence thresholds are separate.',
                          'Async images contain only actual mid-frame 2D detections; no GT/track/3D propagation.',
                          'Keyframe poses projected with each camera ego pose; no object motion interpolation.',
                          'No evaluation matching, filtering or track interpolation. Occlusion is not inferred.'],
                'scenes': [], 'images': [], 'missing_prediction_tokens': [], 'missing_mid_tokens': []}
    output.mkdir(parents=True, exist_ok=True)
    render_started = time.perf_counter()
    frame_count = 0
    for scene in scenes:
        token, index, count = scene['first_sample_token'], 0, 0
        print('Rendering ' + scene['name'], flush=True)
        while token:
            sample = nusc.get('sample', token)
            if index >= selection['start_frame']:
                if selection.get('max_frames') is not None and count >= selection['max_frames']:
                    break
                if token not in predictions:
                    manifest['missing_prediction_tokens'].append(token)
                for dimension, data in detections.items():
                    if token not in data:
                        raise ValueError('Missing {} detection keyframe {}; missing input is not an empty detection.'.format(dimension, token))
                objects = []
                if options['cube_level']['cube_gt_overlay']:
                    for ann_token in sample['anns']:
                        ann = nusc.get('sample_annotation', ann_token)
                        name = category_to_tracking_name(ann['category_name'])
                        if name in classes:
                            objects.append(dict(ann, name=name, id=ann['instance_token'], gt=True))
                for pred in predictions.get(token, []):
                    if pred['tracking_name'] in classes and pred['tracking_score'] >= options['score_threshold']:
                        objects.append(dict(pred, name=pred['tracking_name'], id=pred['tracking_id'],
                                            score=pred['tracking_score'], gt=False))
                for camera in options['cameras']:
                    sd = nusc.get('sample_data', sample['data'][camera])
                    source = dataset_root / sd['filename']
                    image = cv2.imread(str(source))
                    if image is None:
                        raise FileNotFoundError('Cannot read camera image: ' + str(source))
                    pose = nusc.get('ego_pose', sd['ego_pose_token'])
                    calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                    render_camera_frame(image, objects, pose, calibration, scene['token'], options)
                    detection_counts = {}
                    if '3d' in detections:
                        detection_counts['3d'] = render_3d_detections(
                            image, detections['3d'][token], pose, calibration, classes, det_options)
                    if '2d' in detections:
                        detection_counts['2d'] = render_2d_detections(
                            image, detections['2d'][token][camera], classes, det_options)
                    header = frame_header(scene['name'], camera, index, token, options)
                    if header:
                        draw_text(image, header, (12, 25))
                    if options['info_level']['text_detection_legend'] and detections:
                        draw_text(image, detection_legend(det_options), (12, 49))
                    target = output / scene['name'] / camera / ('{:06d}_{}.jpg'.format(index, token))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, options['jpeg_quality']]):
                        raise OSError('Image write failed: ' + str(target))
                    manifest['images'].append({'path': str(target.relative_to(output)), 'sample_token': token,
                                               'sample_data_token': sd['token'], 'source': str(source),
                                               'timestamp': sd['timestamp'], 'bytes': target.stat().st_size,
                                               'frame_idx': index, 'frame_type': 'keyframe',
                                               'visible_detection_counts': detection_counts})
                    # Packed *_mid entries belong to the interval BEFORE this keyframe.
                    if det_options['include_mid_frames'] and sample['prev']:
                        mid_token = token + '_mid'
                        if mid_token not in detections['2d']:
                            if mid_token not in manifest['missing_mid_tokens']:
                                manifest['missing_mid_tokens'].append(mid_token)
                            continue
                        entry = detections['2d'][mid_token][camera]
                        mid_sd = nusc.get('sample_data', entry['sample_data_token'])
                        prev_sd = nusc.get('sample_data', nusc.get('sample', sample['prev'])['data'][camera])
                        if mid_sd['channel'] != camera or not prev_sd['timestamp'] < mid_sd['timestamp'] < sd['timestamp']:
                            raise ValueError('Mid-frame camera/timestamp mismatch: ' + mid_token)
                        mid_source = dataset_root / mid_sd['filename']
                        mid_image = cv2.imread(str(mid_source))
                        if mid_image is None:
                            raise FileNotFoundError('Cannot read mid-frame image: ' + str(mid_source))
                        visible = render_2d_detections(mid_image, entry['np_boxes'], classes, det_options)
                        # Header must not suggest that a track score is applied to these observations.
                        mid_options = dict(options, info_level=dict(options['info_level'], text_threshold=False))
                        mid_header = frame_header(scene['name'], camera, index, mid_token, mid_options)
                        draw_text(mid_image, mid_header + ' | MID before keyframe (2D only)', (12, 25))
                        if options['info_level']['text_detection_legend']:
                            draw_text(mid_image, detection_legend(det_options, mid=True), (12, 49))
                        mid_target = output / scene['name'] / (camera + '_mid') / ('{:06d}_{}.jpg'.format(index, mid_token))
                        mid_target.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(mid_target), mid_image, [cv2.IMWRITE_JPEG_QUALITY, options['jpeg_quality']]):
                            raise OSError('Image write failed: ' + str(mid_target))
                        manifest['images'].append({'path': str(mid_target.relative_to(output)), 'sample_token': token,
                                                   'detection_token': mid_token, 'sample_data_token': mid_sd['token'],
                                                   'source': str(mid_source), 'timestamp': mid_sd['timestamp'],
                                                   'bytes': mid_target.stat().st_size, 'frame_idx': index,
                                                   'frame_type': 'mid_before_keyframe',
                                                   'visible_detection_counts': {'2d': visible}})
                count += 1
                frame_count += 1
                if count % 10 == 0:
                    print('  {}: {} frames'.format(scene['name'], count), flush=True)
            token, index = sample['next'], index + 1
        manifest['scenes'].append({'name': scene['name'], 'token': scene['token'], 'frames': count})
    if not frame_count:
        raise ValueError('Selection produced no frames; check start_frame')
    render_seconds = time.perf_counter() - render_started
    manifest['statistics'] = {'load_seconds': load_seconds, 'render_seconds': render_seconds,
                              'total_seconds': time.perf_counter() - started, 'frames': frame_count,
                              'image_count': len(manifest['images']),
                              'image_bytes': sum(x['bytes'] for x in manifest['images']),
                              'input_sample_count': len(predictions)}
    with (output / 'manifest.json').open('w') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest['statistics'], indent=2), flush=True)
    print('Manifest: ' + str(output / 'manifest.json'))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/viz_rgb.yaml')
    args = parser.parse_args()
    render_tracking_results(load_config(args.config))
