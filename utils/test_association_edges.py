"""Assignment identity, coordinate transforms and cache safety regression tests.

Run: python -m unittest utils.test_association_edges
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.association_edges import assigned_observations, cache_path, digest, edge_segments, load_cache
from utils.viz_3D import Viewer


class AssociationEdgesTest(unittest.TestCase):
    def setUp(self):
        row = [4., 5., 6., 2., 4., 1., 0., 0., 1., 0., 0., 0., .8, 2.]
        self.frame = dict(mix_np_dets_3d=np.array([row, row]), single_np_dets_3d=np.array([row]))
        self.association = dict(m_asso_1={1: 2}, m_asso_2={0: 0}, m_tras_idx3=[1])
        self.records = assigned_observations(self.association, [41, 73, 152], self.frame)
        self.track = dict(id='152', name='car', translation=[1., 2., 3.], gt=False)

    def test_actual_assignment_indices_not_nearest_or_birth(self):
        self.assertEqual([(r['track_id'], r['stage'], r['detection_index']) for r in self.records],
                         [('152', 'mix', 1), ('41', 'pure3d', 0)])
        # mix[0] is unassigned/birth; ID73 was camera-only. Neither yields a link.
        self.assertNotIn('73', [r['track_id'] for r in self.records])
        self.assertEqual(self.records[0]['detection']['translation'], [4., 5., 6.])

    def test_duplicate_assignment_rejected(self):
        self.association['m_asso_2'] = {0: 2}
        with self.assertRaises(ValueError):
            assigned_observations(self.association, [41, 73, 152], self.frame)

    def test_both_endpoints_use_same_full_transform(self):
        transform = np.array([[0., -1., 0., 10.], [1., 0., 0., 20.],
                              [0., 0., 1., 30.], [0., 0., 0., 1.]])
        result = edge_segments(self.records, [self.track], transform)
        np.testing.assert_allclose(result, [[[8., 21., 33.], [5., 24., 36.]]])
        np.testing.assert_allclose(np.linalg.norm(result[:, 1]-result[:, 0], axis=1), [np.sqrt(27)])

    def test_hidden_tracks_gt_missing_observations_and_zero_length(self):
        self.assertEqual(edge_segments(self.records, [], np.eye(4)).shape, (0, 2, 3))
        self.assertEqual(len(edge_segments([], [self.track], np.eye(4))), 0)
        self.assertEqual(len(edge_segments(self.records, [dict(self.track, gt=True)], np.eye(4))), 0)
        self.assertEqual(len(edge_segments(self.records, [dict(self.track, translation=[4., 5., 6.])], np.eye(4))), 0)

    def test_wrong_class_or_nonfinite_center_rejected(self):
        with self.assertRaises(ValueError):
            edge_segments(self.records, [dict(self.track, name='pedestrian')], np.eye(4))
        with self.assertRaises(ValueError):
            edge_segments(self.records, [dict(self.track, translation=[np.nan, 2., 3.])], np.eye(4))

    def test_real_open3d_geometry_and_layer_gates(self):
        import open3d as o3d
        from open3d.visualization import rendering
        geometries = {}
        viewer = Viewer.__new__(Viewer)
        viewer.o3d, viewer.rendering = o3d, rendering
        viewer.widget = SimpleNamespace(scene=SimpleNamespace(add_geometry=lambda k, g, m: geometries.update({k: (g, m)})))
        viewer.data = SimpleNamespace(indices=[19], associations={'matches': {'sample': self.records}})
        viewer.options = dict(association_level=dict(edge_overlay=True, edge_line_width=4.),
                              cube_level=dict(cube_tracking_overlay=True))
        viewer.cfg = {'gui': {'line_width': 2.}}
        viewer.edge_counts = []; viewer.frame_pos = 0; viewer.coordinate = 'global'
        viewer.add_association_edges({'token': 'sample'}, [self.track], np.eye(4))
        geometry, material = geometries['association_edges']
        np.testing.assert_allclose(np.asarray(geometry.points), [[1., 2., 3.], [4., 5., 6.]])
        np.testing.assert_allclose(np.asarray(geometry.colors), [[1., 0., 0.]])
        self.assertEqual(material.line_width, 4.)
        self.assertEqual(material.shader, 'unlitLine')
        for group, key in [('association_level', 'edge_overlay'), ('cube_level', 'cube_tracking_overlay')]:
            geometries.clear(); viewer.options[group][key] = False
            viewer.add_association_edges({'token': 'sample'}, [self.track], np.eye(4))
            self.assertFalse(geometries)
            viewer.options[group][key] = True

    def test_cache_rejects_missing_stale_incomplete_and_duplicate_records(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / 'results.json'; result.write_text('{}')
            cfg = dict(paths=dict(results_json=str(result), association_dir=directory),
                       selection=dict(scene_name='scene-test'))
            scene = {'token': 'scene-token'}; samples = [{'token': 'sample'}]
            path = cache_path(cfg)
            with self.assertRaises(FileNotFoundError): load_cache(cfg, scene, samples)
            valid = dict(schema_version=1, verified=True, scene_token='scene-token', results_sha256=digest(result),
                         sample_tokens=['sample'], matches={'sample': self.records})
            path.write_text(json.dumps(valid))
            self.assertEqual(len(load_cache(cfg, scene, samples)['matches']['sample']), 2)
            result.write_text('{"different": true}')
            with self.assertRaises(ValueError): load_cache(cfg, scene, samples)
            result.write_text('{}')
            bad = copy.deepcopy(valid); bad['matches'] = {}
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load_cache(cfg, scene, samples)
            bad = copy.deepcopy(valid); bad['matches']['sample'] *= 2
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load_cache(cfg, scene, samples)


if __name__ == '__main__':
    unittest.main()
