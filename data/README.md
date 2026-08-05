# Data layout

Detection inputs and token tables are **not** shipped in git. Place (or symlink) them under this directory:

```text
data/
├── detector/
│   └── val/
│       └── nuscenes_val_centerpoint_3d.json
└── utils/
    ├── cascade_rcnn_4hz/
    │   └── nuscenes_val_cascade_2d_4hz.json
    └── first_token_table/trainval/
        └── nuscenes_val_first_sample_tokens.json
```

See the repository root `README.md` for download / packing notes and reproduction commands.
