# Actual 3D association edges

`Association edges: red`, below the Open3D GUI's LiDAR/ego/box layer checkboxes,
connects each displayed track center (the saved, post-update state) to the center
of its **actual assigned 3D detection**. No nearest-neighbor or IoU matching is
performed by the renderer.

```yaml
render:
  association_level:
    edge_overlay: true
    edge_line_width: 4.0  # red lines; pixel width, default box width is 2.0
```

Mixed detections use the box after fusion/GAAM correction, so the endpoint may
differ slightly from the raw magenta detection overlay. Pure-3D assignments are
also included. Unmatched tracks, camera-only updates and newly initialized
tracks have no edge. Coincident centers have no visible segment. Hiding tracks
hides their edges; the raw-detection layer and its display score filter do not
change the recorded assignments. Track class/score display filters still apply.
Both endpoints use the same full global/ego/world-first coordinate transform.

## First export, then render

The standard `results.json` has no detection-assignment fields. Generate a
verified cache **once per scene/result** before enabling this layer:

```bash
python utils/viz_3D.py --config config/viz_3D.yaml --export-associations
python utils/viz_3D.py --config config/viz_3D.yaml
```

Export replays the selected **complete val scene from its first frame**, including
async mid steps for high, regardless of the display's start/max-frame selection.
It uses `paths.association_tracker_config` (the saved run config), detector paths
and `paths.first_token_json`. It compares every keyframe's IDs, classes, boxes,
velocities and scores with the existing result (absolute tolerance 1e-8) and only
saves if all comparisons pass. No evaluation runs and results.json is unchanged.
`paths.association_dir/<scene-name>.json` stores the assignments and results SHA256;
a different result, wrong scene or incomplete cache is rejected. Subsequent GUI
runs only read this cache and do not replay tracking.

Missing caches leave edges off; clicking the checkbox shows the export
instruction. Older local configs without `association_level` default to edges
off and width 4. Set `paths.association_tracker_config` before exporting with
such a config. If `paths.association_dir` is absent, the cache defaults to an
`associations/` directory next to results.json. The default first-token input is
`data/utils/first_token_table/trainval/nuscenes_val_first_sample_tokens.json`.

For Mac/Docker, copy the matching cache together with the saved results, or export
inside the container using its local config and the original run config/detector
inputs. The cache does not embed host absolute paths. If transferring results
changes the JSON file bytes, regenerate the cache because its SHA256 will differ.

## Validation

```bash
python -m unittest utils.test_association_edges -v
python utils/viz_3D.py --config config/viz_3D.yaml --smoke-test
```

Tests cover actual assignment index/ID mapping, exclusion of unmatched/2D/birth
cases, rigid transforms of both endpoints, invalid caches, and real Open3D line
geometry/material properties. The GUI smoke test visits selected frames, switches
all coordinate modes, toggles layers including associations, saves a canvas PNG
and writes per-frame edge counts to `gui_smoke_test.json` under `paths.output_dir`.

The scene-0099 high cache was verified against all 39 saved keyframes with zero
output error: 77 tracker steps and 1,152 3D matches across all classes. Frame19
ID152 and frame21 ID184 have edges; frame20 ID152 is unmatched and frame20 ID184
is newly initialized, so both frame20 cases correctly have no association edge.
