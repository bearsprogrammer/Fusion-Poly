"""Open3D keyframe inspection GUI. One scene, saved observations/results only.

python utils/viz_3D.py --config config/viz_3D.yaml
python utils/viz_3D.py --validate-only
python utils/viz_3D.py --smoke-test
"""
import os
import sys
if __name__ == '__main__' and not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from nuscenes.eval.tracking.utils import category_to_tracking_name
from utils import viz as rgb
from utils import association_edges


def rigid_transform(record):
    """T_parent_child: nuScenes stores child pose IN parent (not its inverse)."""
    mat = np.eye(4)
    mat[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    mat[:3, 3] = record['translation']
    return mat


def transform_points(points, matrix):
    """N x 3 points; use float64 until creating render geometry."""
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def read_config(path):
    cfg = yaml.safe_load(rgb.resolve_path(path).read_text())
    if cfg['dataset']['version'] != 'v1.0-trainval':
        raise ValueError('Base GUI supports v1.0-trainval keyframes only')
    sel, ui, ren = cfg['selection'], cfg['gui'], cfg['render']
    edge = ren.setdefault('association_level', {})
    edge.setdefault('edge_overlay', False)
    edge.setdefault('edge_line_width', 4.0)
    if type(edge['edge_overlay']) is not bool:
        raise ValueError('association_level.edge_overlay must be a boolean')
    if (type(edge['edge_line_width']) not in (int, float) or
            not np.isfinite(edge['edge_line_width']) or edge['edge_line_width'] <= 0):
        raise ValueError('association_level.edge_line_width must be positive')
    if not isinstance(sel['scene_name'], str):
        raise ValueError('Select exactly one scene_name')
    if type(sel['start_frame']) is not int or sel['start_frame'] < 0:
        raise ValueError('start_frame must be a nonnegative integer')
    if sel['max_frames'] is not None and (type(sel['max_frames']) is not int or sel['max_frames'] < 1):
        raise ValueError('max_frames must be null or a positive integer')
    if sel['classes'] is not None and (not isinstance(sel['classes'], list) or not sel['classes'] or
                                      any(c not in rgb.CLASSES for c in sel['classes'])):
        raise ValueError('Invalid selection.classes')
    if cfg['coordinates']['frame'] not in ('ego', 'world_first', 'global'):
        raise ValueError('coordinates.frame must be ego, world_first or global')
    if not ren['cameras'] or any(c not in rgb.CAMERAS for c in ren['cameras']) or ui['camera'] not in ren['cameras']:
        raise ValueError('Invalid camera selection')
    for key in ('width', 'height', 'sidebar_width', 'lidar_stride'):
        if type(ui[key]) is not int or ui[key] < 1:
            raise ValueError('gui.' + key + ' must be a positive integer')
    if ui['sidebar_width'] >= ui['width'] - 200 or ui['height'] < 600:
        raise ValueError('Window needs a >=200px canvas and >=600px height')
    for key in ('playback_fps', 'point_size', 'line_width', 'lidar_radius_m'):
        if type(ui[key]) not in (int, float) or not np.isfinite(ui[key]) or ui[key] <= 0:
            raise ValueError('Invalid gui.' + key)
    for key in ('require_hardware_gpu', 'camera_panel'):
        if type(ui[key]) is not bool:
            raise ValueError('gui.' + key + ' must be a boolean')
    for group in ('info_level', 'cube_level', 'detection_level'):
        for key, value in ren[group].items():
            if key.startswith(('text_', 'cube_')) and type(value) is not bool:
                raise ValueError(group + '.' + key + ' must be a boolean')
    for value in (ren['score_threshold'], ren['detection_level']['score_threshold_2d'],
                  ren['detection_level']['score_threshold_3d']):
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Score thresholds must be in [0, 1]')
    return cfg


def gpu_diagnostics(required):
    """Verify the graphics device, not just CUDA availability."""
    try:
        proc = subprocess.run(['glxinfo', '-B'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('Cannot inspect OpenGL; run glxinfo -B in the desktop session: ' + str(exc))
    info = proc.stdout
    renderer = next((line.split(': ', 1)[1] for line in info.splitlines()
                     if line.startswith('OpenGL renderer string:')), '')
    software = any(name in renderer.lower() for name in ('llvmpipe', 'softpipe', 'swrast', 'software'))
    forced = any(os.environ.get(k, '').lower() in ('1', 'true')
                 for k in ('OPEN3D_CPU_RENDERING', 'LIBGL_ALWAYS_SOFTWARE'))
    if proc.returncode or not renderer or (required and (software or forced or 'direct rendering: Yes' not in info)):
        raise RuntimeError('Hardware OpenGL unavailable. Run in an authorized desktop session.\n' + info + proc.stderr)
    print('OpenGL renderer: ' + renderer, flush=True)
    return {'display': os.environ.get('DISPLAY'), 'renderer': renderer,
            'hardware': not software and not forced, 'glxinfo': info}


class SceneData:
    def __init__(self, cfg):
        started = time.perf_counter()
        self.cfg, self.root = cfg, rgb.resolve_path(cfg['paths']['dataset_root'])
        self.nusc = NuScenes(version=cfg['dataset']['version'], dataroot=str(self.root), verbose=False)
        self.scene = next((s for s in self.nusc.scene if s['name'] == cfg['selection']['scene_name']), None)
        if self.scene is None:
            raise ValueError('Scene does not exist: ' + cfg['selection']['scene_name'])
        self.classes = set(cfg['selection']['classes'] or rgb.CLASSES)
        self.tracks = json.loads(rgb.resolve_path(cfg['paths']['results_json']).read_text())['results']
        self.samples = []
        token = self.scene['first_sample_token']
        while token:
            s = self.nusc.get('sample', token)
            self.samples.append(s)
            token = s['next']
        start = cfg['selection']['start_frame']
        end = len(self.samples) if cfg['selection']['max_frames'] is None else start + cfg['selection']['max_frames']
        self.indices = list(range(start, min(end, len(self.samples))))
        if not self.indices:
            raise ValueError('Selected frame range is empty')
        if any(self.samples[i]['token'] not in self.tracks for i in self.indices):
            raise ValueError('Selected scene/range has no complete tracking results; check train/val split')
        self.det = rgb.load_detections(cfg['paths'], cfg['render']['detection_level'])
        for dimension, detections in self.det.items():
            if any(self.samples[i]['token'] not in detections for i in self.indices):
                raise ValueError('Missing ' + dimension + ' keyframe detections')
        self.first_ego = self.ego_pose(self.samples[0])
        self.associations = None
        if cfg['render']['association_level']['edge_overlay']:
            try:
                self.load_associations()
            except FileNotFoundError as exc:
                print(str(exc), flush=True)
        self.load_seconds = time.perf_counter() - started
        print('{}: {} selected keyframes, loaded in {:.1f}s'.format(self.scene['name'], len(self.indices), self.load_seconds), flush=True)

    def load_associations(self):
        if self.associations is None:
            self.associations = association_edges.load_cache(self.cfg, self.scene, self.samples)

    def sensor_records(self, sample, channel):
        sd = self.nusc.get('sample_data', sample['data'][channel])
        return sd, self.nusc.get('ego_pose', sd['ego_pose_token']), self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

    def ego_pose(self, sample):
        return rigid_transform(self.sensor_records(sample, 'LIDAR_TOP')[1])

    def view_from_global(self, sample, frame):
        if frame == 'ego':
            return np.linalg.inv(self.ego_pose(sample))
        if frame == 'world_first':
            return np.linalg.inv(self.first_ego)
        if frame == 'global':
            return np.eye(4)
        raise ValueError('Invalid coordinate frame')

    def points_global(self, sample):
        sd, pose, cal = self.sensor_records(sample, 'LIDAR_TOP')
        local = LidarPointCloud.from_file(str(self.root / sd['filename'])).points[:3].T.astype(np.float64)
        # lidar -> ego at lidar timestamp -> dataset global
        return transform_points(local, rigid_transform(pose) @ rigid_transform(cal))

    def objects(self, sample):
        objects = []
        for token in sample['anns']:
            ann = self.nusc.get('sample_annotation', token)
            name = category_to_tracking_name(ann['category_name'])
            if name in self.classes:
                objects.append(dict(ann, name=name, id=ann['instance_token'], gt=True))
        for track in self.tracks[sample['token']]:
            if track['tracking_name'] in self.classes and track['tracking_score'] >= self.cfg['render']['score_threshold']:
                objects.append(dict(track, name=track['tracking_name'], id=track['tracking_id'],
                                    score=track['tracking_score'], gt=False))
        return objects

    def validate(self):
        """Compare full transforms with devkit GT boxes in lidar and all camera frames."""
        errors = []; pixel_errors = []; checked = 0; point_count = 0
        for index in self.indices:
            sample = self.samples[index]
            points = self.points_global(sample)
            if not np.isfinite(points).all() or not len(points):
                raise ValueError('Invalid point cloud at frame ' + str(index))
            point_count += len(points)
            for frame in ('ego', 'world_first', 'global'):
                transform = self.view_from_global(sample, frame)
                restored = transform_points(transform_points(points[::100], transform), np.linalg.inv(transform))
                errors.append(float(np.max(np.abs(restored - points[::100]))))
            ego = transform_points(self.ego_pose(sample)[None, :3, 3], self.view_from_global(sample, 'ego'))
            assert np.max(np.abs(ego)) < 1e-8
            if index not in {self.indices[0], self.indices[len(self.indices)//2], self.indices[-1]}:
                continue
            for channel in ['LIDAR_TOP'] + self.cfg['render']['cameras']:
                sd, pose, cal = self.sensor_records(sample, channel)
                sensor_from_global = np.linalg.inv(rigid_transform(pose) @ rigid_transform(cal))
                _, reference_boxes, intrinsic = self.nusc.get_sample_data(sd['token'], box_vis_level=BoxVisibility.NONE)
                for reference in reference_boxes:
                    ann = self.nusc.get('sample_annotation', reference.token)
                    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                    actual = transform_points(box.corners().T, sensor_from_global).T
                    errors.append(float(np.max(np.abs(actual - reference.corners()))))
                    if intrinsic is not None and np.all(actual[2] > .1):
                        uv = intrinsic @ actual; uv = uv[:2] / uv[2]
                        ref_uv = view_points(reference.corners(), intrinsic, normalize=True)[:2]
                        pixel_errors.append(float(np.max(np.abs(uv - ref_uv))))
                    checked += 1
        maximum = max(errors)
        max_pixels = max(pixel_errors, default=0.)
        if maximum > 1e-6 or max_pixels > 1e-6:
            raise AssertionError('Coordinate/projection mismatch: ' + str(maximum))
        result = dict(scene=self.scene['name'], frames=len(self.indices), lidar_points_checked=point_count,
                      sensor_box_checks=checked, max_coordinate_error_m=maximum,
                      projection_checks=len(pixel_errors), max_projection_error_px=max_pixels,
                      coordinate_conventions={'ego': 'current lidar ego, inverse rotation AND translation',
                                              'world_first': 'fixed first scene ego rotation AND translation',
                                              'global': 'dataset global'}, passed=True)
        return result


class Viewer:
    def __init__(self, data, cfg, diagnostics, smoke=False):
        import open3d as o3d
        from open3d.visualization import gui, rendering
        self.o3d, self.gui, self.rendering = o3d, gui, rendering
        self.data, self.cfg, self.diagnostics = data, cfg, diagnostics
        self.options = copy.deepcopy(cfg['render'])
        self.options['association_level']['edge_overlay'] = bool(
            self.options['association_level']['edge_overlay'] and data.associations is not None)
        self.edge_counts = []
        self.coordinate = cfg['coordinates']['frame']
        self.frame_pos, self.camera = 0, cfg['gui']['camera']
        self.playing = False; self.show_points = True; self.show_axes = True
        self.last_tick = time.monotonic(); self.smoke = smoke; self.tick_step = 0
        self.visited = []; self.times = []; self.capturing = False; self.labels = []
        self.tested_controls = []
        self.app = gui.Application.instance
        self.window = self.app.create_window('Fusion-Poly | 3D keyframe inspection', cfg['gui']['width'], cfg['gui']['height'])
        # Request an OS-coordinate rectangle; the desktop window manager may
        # maximize it. The layout below always uses the actual content rectangle.
        self.window.os_frame = gui.Rect(20, 40, cfg['gui']['width'], cfg['gui']['height'])
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([.035, .045, .06, 1.])
        self.widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.window.add_child(self.widget)
        em = self.window.theme.font_size
        self.panel = gui.Vert(.4 * em, gui.Margins(em, em, em, em))
        self.window.add_child(self.panel)
        self.info = gui.Label('Loading frame...'); self.panel.add_child(self.info)
        self.coordinate_label = gui.Label(''); self.panel.add_child(self.coordinate_label)
        self.panel.add_child(gui.Label('Left drag: orbit | wheel: zoom\nCtrl+drag: pan | R: reset | Space: play'))
        buttons = gui.Horiz(.3 * em)
        for text, callback in [('Previous', lambda: self.step(-1)), ('Play / Pause', self.toggle_play),
                               ('Next', lambda: self.step(1)), ('Reset view', self.reset_view)]:
            button = gui.Button(text); button.set_on_clicked(callback); buttons.add_child(button)
        self.panel.add_child(buttons)
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, max(1, len(data.indices)-1))
        self.slider.set_on_value_changed(lambda value: self.set_position(int(value)))
        self.panel.add_child(self.slider)
        self.panel.add_child(gui.Label('Coordinates (origin AND orientation)'))
        coords = gui.Combobox()
        for name in ('ego', 'world_first', 'global'): coords.add_item(name)
        coords.selected_text = self.coordinate
        coords.set_on_selection_changed(lambda name, _: self.change_coordinates(name))
        self.panel.add_child(coords)
        self.add_check('LiDAR points', True, lambda v: self.set_points(v))
        self.add_check('Ego axes: X red / Y green / Z blue', True, lambda v: self.set_axes(v))
        for label, group, key in [('GT: white', 'cube_level', 'cube_gt_overlay'),
                                  ('Tracks: ID colors', 'cube_level', 'cube_tracking_overlay'),
                                  ('3D detections: magenta', 'detection_level', 'cube_3d_overlay'),
                                  ('2D detections: cyan (RGB panel)', 'detection_level', 'cube_2d_overlay')]:
            self.add_check(label, self.options[group][key], lambda v, g=group, k=key: self.set_layer(g,k,v))
        self.association_checkbox = self.add_check('Association edges: red',
            self.options['association_level']['edge_overlay'], self.set_associations)
        camera = gui.Combobox()
        for name in cfg['render']['cameras']: camera.add_item(name)
        camera.selected_text = self.camera
        camera.set_on_selection_changed(lambda name, _: self.change_camera(name))
        self.panel.add_child(camera)
        self.image_widget = gui.ImageWidget(); self.panel.add_child(self.image_widget)
        self.caption = gui.Label(''); self.panel.add_child(self.caption)
        self.panel.add_child(gui.Label('Keyframes only. 2D has no metric depth.\nEdges: verified 3D assignments; no 2D/birth links.'))
        self.status = gui.Label(diagnostics['renderer']); self.panel.add_child(self.status)
        save = gui.Button('Save 3D canvas PNG'); save.set_on_clicked(self.snapshot); self.panel.add_child(save)
        self.window.set_on_layout(self.layout)
        self.window.set_on_tick_event(self.on_tick)
        self.widget.set_on_key(self.on_key)
        self.show_frame(reset=True)

    def add_check(self, label, checked, callback):
        box = self.gui.Checkbox(label); box.checked = checked
        box.set_on_checked(callback); self.panel.add_child(box)
        return box

    def set_associations(self, value):
        if value:
            try:
                self.data.load_associations()
            except (OSError, ValueError) as exc:
                self.association_checkbox.checked = False
                self.options['association_level']['edge_overlay'] = False
                self.window.show_message_box('Association cache unavailable', str(exc))
                return
        self.options['association_level']['edge_overlay'] = value
        self.association_checkbox.checked = value
        self.show_frame()

    def layout(self, context):
        rect = self.window.content_rect; width = min(self.cfg['gui']['sidebar_width'], rect.width//2)
        self.widget.frame = self.gui.Rect(rect.x, rect.y, rect.width-width, rect.height)
        self.panel.frame = self.gui.Rect(rect.get_right()-width, rect.y, width, rect.height)

    def set_points(self, value): self.show_points=value; self.show_frame()
    def set_axes(self, value): self.show_axes=value; self.show_frame()
    def set_layer(self, group, key, value):
        dimension = '2d' if key == 'cube_2d_overlay' else '3d' if key == 'cube_3d_overlay' else None
        if value and dimension and dimension not in self.data.det:
            path = rgb.resolve_path(self.cfg['paths']['detection_' + dimension + '_json'])
            content = json.loads(path.read_text())
            self.data.det[dimension] = content['results'] if dimension == '3d' else content
        self.options[group][key]=value; self.show_frame()

    def change_camera(self, name): self.camera=name; self.show_frame()
    def change_coordinates(self, name): self.coordinate=name; self.show_frame(reset=True)
    def set_position(self, pos):
        pos = max(0, min(pos, len(self.data.indices)-1))
        if pos != self.frame_pos:
            self.frame_pos=pos; self.show_frame()
    def step(self, delta): self.set_position(self.frame_pos+delta)
    def toggle_play(self): self.playing = not self.playing; self.last_tick=time.monotonic()

    def on_key(self, event):
        gui = self.gui
        if event.type == gui.KeyEvent.DOWN:
            if event.key == gui.KeyName.SPACE: self.toggle_play()
            elif event.key == gui.KeyName.RIGHT: self.step(1)
            elif event.key == gui.KeyName.LEFT: self.step(-1)
            elif event.key == gui.KeyName.R: self.reset_view()
            else: return gui.Widget.EventCallbackResult.IGNORED
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def line_material(self, width=None):
        material = self.rendering.MaterialRecord(); material.shader='unlitLine'
        material.line_width=float(self.cfg['gui']['line_width'] if width is None else width)
        return material

    def add_association_edges(self, sample, objects, transform):
        segments = np.empty((0, 2, 3))
        edge = self.options['association_level']
        if edge['edge_overlay'] and self.options['cube_level']['cube_tracking_overlay']:
            records = self.data.associations['matches'][sample['token']]
            segments = association_edges.edge_segments(records, objects, transform)
        if len(segments):
            geometry = self.o3d.geometry.LineSet(
                self.o3d.utility.Vector3dVector(segments.reshape(-1, 3)),
                self.o3d.utility.Vector2iVector(np.arange(2 * len(segments)).reshape(-1, 2)))
            geometry.paint_uniform_color([0., 1., 0.])
            self.widget.scene.add_geometry('association_edges', geometry,
                                           self.line_material(edge['edge_line_width']))
        self.edge_counts.append(dict(frame=self.data.indices[self.frame_pos], count=len(segments),
                                     enabled=edge['edge_overlay'], coordinate=self.coordinate))

    def add_boxes(self, layer, boxes, transform, color_fn, heading, labels):
        points=[]; lines=[]; colors=[]
        for obj in boxes:
            box=Box(obj['translation'],obj['size'],Quaternion(obj['rotation']))
            corners=transform_points(box.corners().T,transform)
            offset=len(points); points.extend(corners)
            color=color_fn(obj)
            lines.extend([(offset+a,offset+b) for a,b in rgb.EDGES]); colors.extend([color]*12)
            if heading:
                a=corners[[2,3,7,6]].mean(axis=0); b=corners[[2,3]].mean(axis=0)
                offset=len(points); points.extend([a,b]); lines.append((offset,offset+1)); colors.append(color)
            text=labels(obj)
            if text:
                label=self.widget.add_3d_label(corners[np.argmax(corners[:,2])],text)
                label.color=self.gui.Color(*color)
                self.labels.append(label)
        if lines:
            geometry=self.o3d.geometry.LineSet(self.o3d.utility.Vector3dVector(points),self.o3d.utility.Vector2iVector(lines))
            geometry.colors=self.o3d.utility.Vector3dVector(colors)
            self.widget.scene.add_geometry(layer,geometry,self.line_material())

    def box_label(self, obj):
        cube=self.options['cube_level']; result=[]
        if obj['gt'] and cube['text_gt_id']: result.append('GT:'+obj['id'])
        if not obj['gt'] and cube['text_track_id']: result.append('ID:'+str(obj['id']))
        if cube['text_class']: result.append(obj['name'])
        if not obj['gt'] and cube['text_score']: result.append('{:.2f}'.format(obj['score']))
        return ' '.join(result)

    def show_frame(self, reset=False):
        started=time.perf_counter(); index=self.data.indices[self.frame_pos]; sample=self.data.samples[index]
        transform=self.data.view_from_global(sample,self.coordinate)
        self.widget.scene.clear_geometry()
        for label in self.labels:
            self.widget.remove_3d_label(label)
        self.labels.clear()
        if self.show_points:
            points=self.data.points_global(sample)[::self.cfg['gui']['lidar_stride']]
            ego_points=transform_points(points,np.linalg.inv(self.data.ego_pose(sample)))
            keep=np.linalg.norm(ego_points[:,:2],axis=1)<=self.cfg['gui']['lidar_radius_m']
            points=transform_points(points[keep],transform)
            cloud=self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(points))
            cloud.paint_uniform_color([.42,.49,.55])
            material=self.rendering.MaterialRecord(); material.shader='defaultUnlit'; material.point_size=float(self.cfg['gui']['point_size'])
            self.widget.scene.add_geometry('lidar',cloud,material)
        self.ego_view=transform @ self.data.ego_pose(sample)
        if self.show_axes:
            axes=self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.)
            axes.transform(self.ego_view)
            material=self.rendering.MaterialRecord();material.shader='defaultUnlit'
            self.widget.scene.add_geometry('ego_axes',axes,material)
        objects=self.data.objects(sample); cube=self.options['cube_level']; det=self.options['detection_level']
        if cube['cube_gt_overlay']:
            self.add_boxes('gt',[b for b in objects if b['gt']],transform,lambda b:[.95,.95,.95],cube['cube_heading'],self.box_label)
        if cube['cube_tracking_overlay']:
            self.add_boxes('tracks',[b for b in objects if not b['gt']],transform,
                           lambda b:[v/255 for v in rgb.track_color(self.data.scene['token'],b['id'])[::-1]],cube['cube_heading'],self.box_label)
        if det['cube_3d_overlay']:
            boxes=[b for b in self.data.det['3d'][sample['token']] if b['detection_name'] in self.data.classes and b['detection_score']>=det['score_threshold_3d']]
            def label(b):
                return ' '.join((['D3'] if det['text_class'] or det['text_score'] else [])+
                                ([b['detection_name']] if det['text_class'] else [])+
                                (['{:.2f}'.format(b['detection_score'])] if det['text_score'] else []))
            self.add_boxes('detections',boxes,transform,lambda b:[1.,0.,1.],det['cube_heading'],label)
        self.add_association_edges(sample, objects, transform)
        self.slider.int_value=self.frame_pos
        self.info.text=rgb.frame_header(self.data.scene['name'],self.camera,index,sample['token'],self.options).replace(' | ','\n')
        self.coordinate_label.text='Frame: {} | meters\nEgo axes: +X forward, +Y left, +Z up'.format(self.coordinate)
        self.update_image(sample,objects)
        self.window.set_needs_layout()
        if reset:self.reset_view()
        self.widget.force_redraw()
        self.visited.append(index); self.times.append(time.perf_counter()-started)

    def update_image(self, sample, objects):
        if not self.cfg['gui']['camera_panel']:
            self.image_widget.visible=False; return
        sd,pose,cal=self.data.sensor_records(sample,self.camera)
        image=cv2.imread(str(self.data.root/sd['filename']))
        if image is None:raise FileNotFoundError(sd['filename'])
        rgb.render_camera_frame(image,objects,pose,cal,self.data.scene['token'],self.options)
        det=self.options['detection_level']
        if det['cube_3d_overlay']:
            rgb.render_3d_detections(image,self.data.det['3d'][sample['token']],pose,cal,self.data.classes,det)
        if det['cube_2d_overlay']:
            rgb.render_2d_detections(image,self.data.det['2d'][sample['token']][self.camera],self.data.classes,det)
        width=self.cfg['gui']['sidebar_width']-40
        image=cv2.resize(image,(width,int(image.shape[0]*width/image.shape[1])))
        self.image_widget.update_image(self.o3d.geometry.Image(np.ascontiguousarray(image[:,:,::-1])))
        self.caption.text=(rgb.detection_legend(det).replace(' | ','\n') if self.options['info_level']['text_detection_legend'] else '')

    def reset_view(self):
        if not hasattr(self,'ego_view'):return
        center=self.ego_view[:3,3]
        bounds=self.o3d.geometry.AxisAlignedBoundingBox(center-50,center+50)
        self.widget.setup_camera(60.,bounds,center)
        # Initial perspective behind and above ego; axes come from the SAME view transform.
        eye=transform_points(np.array([[-25.,-35.,28.]]),self.ego_view)[0]
        self.widget.scene.camera.look_at(center,eye,self.ego_view[:3,2])
        self.widget.force_redraw()

    def snapshot(self):
        if self.capturing:return
        self.capturing=True
        output=rgb.resolve_path(self.cfg['paths']['output_dir']);output.mkdir(parents=True,exist_ok=True)
        index=self.data.indices[self.frame_pos]
        target=output/'{}_frame{:03d}_{}.png'.format(self.data.scene['name'],index,self.coordinate)
        def done(image):
            ok=self.o3d.io.write_image(str(target),image)
            def finish():
                self.capturing=False
                if not ok:raise RuntimeError('Screenshot failed: '+str(target))
                print('Saved '+str(target),flush=True)
                if self.smoke:
                    report=dict(self.diagnostics,scene=self.data.scene['name'],visited_frames=self.visited,
                                unique_frame_count=len(set(self.visited)),frame_update_seconds=self.times,
                                data_load_seconds=self.data.load_seconds, tested_controls=self.tested_controls,
                                association_edge_counts=self.edge_counts,
                                screenshot=str(target),open3d_version=self.o3d.__version__,passed=True)
                    (output/'gui_smoke_test.json').write_text(json.dumps(report,indent=2))
                    self.app.quit()
            self.app.post_to_main_thread(self.window,finish)
        self.widget.scene.scene.render_to_image(done)

    def on_tick(self):
        now=time.monotonic()
        if self.smoke and not self.capturing and now-self.last_tick>.25:
            self.last_tick=now
            if self.tick_step<len(self.data.indices):
                self.set_position(self.tick_step);self.tick_step+=1
            elif self.tick_step==len(self.data.indices):
                self.change_coordinates('world_first');self.tick_step+=1
            elif self.tick_step==len(self.data.indices)+1:
                self.change_coordinates('global');self.tick_step+=1
            elif self.tick_step==len(self.data.indices)+2:
                self.change_coordinates('ego')
                for setter in (self.set_points, self.set_axes):
                    setter(False); setter(True)
                for group, key in [('cube_level','cube_gt_overlay'), ('cube_level','cube_tracking_overlay'),
                                   ('detection_level','cube_3d_overlay'), ('detection_level','cube_2d_overlay')]:
                    original = self.options[group][key]
                    self.set_layer(group,key,not original); self.set_layer(group,key,original)
                    self.tested_controls.append(key)
                for camera in self.cfg['render']['cameras']:
                    self.change_camera(camera)
                    self.tested_controls.append(camera)
                self.change_camera(self.cfg['gui']['camera'])
                if self.data.associations is not None:
                    original = self.options['association_level']['edge_overlay']
                    self.set_associations(False)
                    assert not self.widget.scene.has_geometry('association_edges')
                    self.set_associations(True)
                    self.set_associations(original)
                    self.tested_controls.append('association_edges')
                self.toggle_play(); assert self.playing; self.toggle_play(); assert not self.playing
                self.step(-1); self.step(1)
                self.tested_controls.extend(['ego','world_first','global','lidar','ego_axes','play_pause','previous_next'])
                self.tick_step+=1
            else:self.set_position(len(self.data.indices)//2);self.snapshot()
            return True
        if self.playing and now-self.last_tick>=1/self.cfg['gui']['playback_fps']:
            self.last_tick=now
            if self.frame_pos==len(self.data.indices)-1:self.playing=False
            else:self.step(1)
            return True
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/viz_3D.yaml')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only',action='store_true',help='Check one scene against devkit coordinate/projection references; no GUI')
    mode.add_argument('--smoke-test',action='store_true',help='Visit selected scene frames in real GPU GUI, test controls, save canvas and close')
    mode.add_argument('--export-associations',action='store_true',help='Replay one complete scene, verify saved results and cache actual 3D assignments; no evaluation/GUI')
    args=parser.parse_args();cfg=read_config(args.config)
    if args.export_associations:
        association_edges.export_cache(cfg)
        return
    diagnostics=None if args.validate_only else gpu_diagnostics(cfg['gui']['require_hardware_gpu'])
    data=SceneData(cfg)
    output=rgb.resolve_path(cfg['paths']['output_dir']);output.mkdir(parents=True,exist_ok=True)
    if args.validate_only:
        report=data.validate();(output/'coordinate_validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2));return
    from open3d.visualization import gui
    gui.Application.instance.initialize()
    viewer=Viewer(data,cfg,diagnostics,args.smoke_test)
    gui.Application.instance.run()
    return viewer


if __name__=='__main__':
    main()
