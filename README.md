# Fusion-Poly

**Official Repository for the IROS 2026 Accepted Paper**

> [**Fusion-Poly: A Polyhedral Framework Based on Spatial-Temporal Fusion for 3D Multi-Object Tracking**](https://arxiv.org/abs/2603.08199),  
> Xian Wu*, Yitao Wu*, Xiaoyu Li*, Zijia Li, Lijun Zhao, Lining Sun  
> *arXiv technical report ([arXiv 2603.08199](https://arxiv.org/abs/2603.08199))*

![framework](docs/framework.jpg)

Fusion-Poly is a learning-free LiDAR–Camera 3D multi-object tracking (MOT) framework that jointly performs cross-modal fusion and cross-frequency integration on nuScenes. It associates trajectories with multi-modal observations at synchronized timestamps and with camera-only observations at asynchronous timestamps, enabling higher-frequency motion and existence updates.

## Highlighted Results

| Split | Setting | Detectors | Input | AMOTA↑ | MOTA↑ | IDS↓ |
|-------|---------|-----------|-------|--------|-------|------|
| **test** | paper main | Mv2DFusion & Cascade R-CNN | C+L | **76.5** | 65.1 | 282 |
| **val** | high-freq (4 Hz cam) | CenterPoint & Cascade R-CNN | C+L | **77.1** | 67.3 | 416 |
| **val** | low-freq / sync. (2 Hz) | CenterPoint & Cascade R-CNN | C+L | **76.9** | 67.0 | 446 |

- Low-frequency (sync.): CenterPoint 3D + Cascade R-CNN 2D at the official nuScenes keyframe rate (**2 Hz**).
- High-frequency (async.): the same 2 Hz CenterPoint stream, plus Cascade R-CNN 2D densified to **~4 Hz** by inserting mid-interval camera frames between keyframes.

## Installation

```bash
conda create -n fusionpoly python=3.7 -y
conda activate fusionpoly
pip install -r requirements.txt
```

Or install from the provided Conda env file:

```bash
conda env create -f environment.yaml
conda activate fusionpoly
```

> Tracking / evaluation on CPU is sufficient. GPU is only needed if you regenerate Cascade R-CNN 2D detections with MMDetection.

## Data Preparation

### 1. nuScenes

Download [nuScenes](https://www.nuscenes.org/nuscenes) `v1.0-trainval` and set `--nusc_path` to the dataset root (the folder that contains `samples/`, `sweeps/`, `v1.0-trainval/`, ...).

### 2. Detection inputs (CenterPoint + Cascade R-CNN)

Download the prepared detection files from the project appendix
[[Google Drive](https://drive.google.com/drive/folders/1bUCXuWorg-vvcLo4BeOOcg27ZUiOnnWo?usp=sharing)]
(`trainval/` for val, `test/` for test). You do **not** need to regenerate CenterPoint or Cascade R-CNN outputs yourself.

Place (or symlink) them under `data/`:

```text
data/
├── detector/
│   └── val/
│       └── nuscenes_val_centerpoint_3d.json          # CenterPoint 3D on val
└── utils/
    ├── first_token_table/trainval/
    │   └── nuscenes_val_first_sample_tokens.json         # first sample token of each val scene
    └── cascade_rcnn_4hz/
        └── nuscenes_val_cascade_2d_4hz.json             # Cascade R-CNN 2D (key + mid frames)
```

Or pass absolute paths to `test.py`.

### 3. (Optional) How the 4 Hz camera stream is built

Official nuScenes annotations / keyframe tracking run at **2 Hz**. To form the paper’s ~**4 Hz** camera-only async stream, we insert one non-keyframe image per camera between consecutive keyframes, detect on those images, then interleave with keyframe 2D results.

**Step A — Non-keyframe token selection** (`data/script/detect2d.py`, mode `select_tokens` or `all`):

- Walk each scene from `nuscenes_val_first_sample_tokens.json`.
- Between two consecutive keyframe `sample`s, for every camera channel, pick the raw `sample_data` whose timestamp is closest to the interval midpoint (`--fractions 0.5`).
- Save the selected mid-frame camera tokens (used as the detection input list).

**Step B — Detection on mid-frame images** (same script, mode `detect` or `all`):

- Run the official MMDetection3D nuImages **Cascade Mask R-CNN (X-101_32x4d, 1x)** on the selected images:
  - Config: [`cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim.py`](https://github.com/open-mmlab/mmdetection3d/blob/1.0/configs/nuimages/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim.py)
  - Weights: [`cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim_20201024_135753-e0e49778.pth`](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim/cascade_mask_rcnn_x101_32x4d_fpn_1x_nuim_20201024_135753-e0e49778.pth) (listed in the [nuImages model zoo](https://github.com/open-mmlab/mmdetection3d/tree/1.0/configs/nuimages))
- These are also the defaults of `data/script/detect2d.py` (`--config_file` / `--checkpoint_file`).
- Write raw mid-frame 2D boxes to `<RAW_MID_DET_JSON>`.

**Step C — Concatenate with keyframe 2D** (`data/script/concat_2d_detection.py`):

- Convert / sort the mid-frame detections into the tracker format.
- Interleave them with keyframe Cascade results (`data/detector/nuscenes_val_cascade_2d_keyframe.json`) using the first-token table for sequence order.
- Mid-frame entries are keyed as `{keyframe_sample_token}_mid`.
- Output: `nuscenes_val_cascade_2d_4hz.json` (~2 Hz key + ~2 Hz mid ⇒ ~4 Hz camera stream).

Requires [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) (≥1.0 nuImages configs) and the X-101 Cascade Mask R-CNN checkpoint above:

```bash
# Steps A+B: select mid-frame tokens and run Cascade R-CNN
python data/script/detect2d.py \
  --mode all \
  --fractions 0.5 \
  --nusc_path <NUSCENES_ROOT> \
  --sample_token_path data/utils/first_token_table/trainval/nuscenes_val_first_sample_tokens.json \
  --token_output_path <MID_TOKEN_JSON> \
  --detection_output_path <RAW_MID_DET_JSON>

# Step C: interleave keyframe 2D and mid-frame 2D
python data/script/concat_2d_detection.py \
  --nusc_path <NUSCENES_ROOT> \
  --high_rate_det_path <RAW_MID_DET_JSON> \
  --key_frame_det_path data/detector/nuscenes_val_cascade_2d_keyframe.json \
  --first_token_path data/utils/first_token_table/trainval/nuscenes_val_first_sample_tokens.json \
  --output_path data/utils/cascade_rcnn_4hz/nuscenes_val_cascade_2d_4hz.json
```

If you downloaded `nuscenes_val_cascade_2d_4hz.json` from the [Google Drive appendix](https://drive.google.com/drive/folders/1bUCXuWorg-vvcLo4BeOOcg27ZUiOnnWo?usp=sharing), you do **not** need to re-run this section.

## Run on nuScenes val

Working directory: repository root. Use `config/nusc_config.yaml` for low frequency and `config/nusc_config_high.yaml` for high frequency.

### Low frequency (sync. 2 Hz)

Uses only keyframe tokens. Mid-frame entries (`*_mid`) in the 2D JSON are skipped when `basic.freq: low`.

```bash
python test.py \
  --process 1 \
  --nusc_path <NUSCENES_ROOT> \
  --config_path config/nusc_config.yaml \
  --detection_3d_path data/detector/val/nuscenes_val_centerpoint_3d.json \
  --detection_2d_path data/utils/cascade_rcnn_4hz/nuscenes_val_cascade_2d_4hz.json \
  --first_token_path data/utils/first_token_table/trainval/nuscenes_val_first_sample_tokens.json \
  --result_path Fusion_Poly_EXP/result/nusc_config_low/ \
  --eval_path Fusion_Poly_EXP/eval_results/nusc_config_low/
```

### High frequency (async. 4 Hz camera)

Same CenterPoint 3D detections; the tracker also consumes mid-frame Cascade R-CNN 2D for association / lifecycle updates (`basic.freq: high`, `LiDAR_interval: 0.25`).

On async mid-frames, **existence scores** are updated with attenuated 2D confidence (paper Eq. 6), while **motion states** only run prediction plus a 2D measurement update under huge observation noise \(R=\gamma^{n}C\) (Eq. 3, \(n=1\)). With the default large \(\gamma\), the Kalman gain is ~0, so async observations do not pull the 3D trajectory state (same outcome as skipping the motion update).

```bash
python test.py \
  --process 1 \
  --nusc_path <NUSCENES_ROOT> \
  --config_path config/nusc_config_high.yaml \
  --detection_3d_path data/detector/val/nuscenes_val_centerpoint_3d.json \
  --detection_2d_path data/utils/cascade_rcnn_4hz/nuscenes_val_cascade_2d_4hz.json \
  --first_token_path data/utils/first_token_table/trainval/nuscenes_val_first_sample_tokens.json \
  --result_path Fusion_Poly_EXP/result/nusc_config_high/ \
  --eval_path Fusion_Poly_EXP/eval_results/nusc_config_high/
```

### Where to read metrics

After `test.py` finishes, open:

```text
<eval_path>/tracking/metrics_summary.json   # tracking metrics
<eval_path>/metrics_summary.json            # aggregated tracking + detection summary
```

## Project Layout

```text
config/                 # only nusc_config.yaml (low) + nusc_config_high.yaml (high)
dataloader/             # nuScenes loader & fusion wiring
geometry/               # GAAM location coordinator, distances, boxes
motion_module/          # motion models / filters
pre_processing/         # NMS, 2D–3D data fusion
tracking/               # FACM / FATE / lifecycle / score management
data/script/            # Cascade R-CNN 2D generation & concat helpers
test.py                 # run tracking + official NuScenes evaluation
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wu2026fusionpoly,
  title     = {Fusion-Poly: A Polyhedral Framework Based on Spatial-Temporal Fusion for 3D Multi-Object Tracking},
  author    = {Wu, Xian and Wu, Yitao and Li, Xiaoyu and Li, Zijia and Zhao, Lijun and Sun, Lining},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## Acknowledgement

We thank the authors of the following open-source projects:

- [Fast-Poly](https://github.com/lixiaoyu2000/FastPoly)
- [EagerMOT](https://github.com/aleksandrkim61/EagerMOT)
- [CBMOT](https://github.com/cogsys-tuebingen/CBMOT)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [MV2DFusion](https://github.com/wangzt-halo/MV2DFusion)
