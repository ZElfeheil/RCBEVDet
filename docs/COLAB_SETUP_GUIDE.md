# RCBEVDet on MAN TruckScenes — Google Colab Setup Guide

> **Purpose**: Step-by-step instructions for running RCBEVDet inference on the
> MAN TruckScenes dataset using Google Colab with a Conda environment.

---

## Prerequisites

| Item | Details |
|------|---------|
| **Runtime** | Google Colab with **GPU** (T4 or better) |
| **Google Drive** | Dataset (`MAN-Truckscenes/`), pre-trained weights (`rcbevdet.pth`), and this repo (`rcbevdet-master/`) uploaded |
| **Dataset version** | `v1.1-mini` (also supports `v1.0-mini`, `v1.0-trainval`) |

### Google Drive Layout (expected)

```
MyDrive/
├── MAN-Truckscenes/          # Dataset root
│   └── v1.1-mini/            # Version subfolder
├── RCBEVDet_weights/
│   └── rcbevdet.pth          # Pre-trained checkpoint
└── rcbevdet-master/           # This repo
```

---

## 1 · Install Miniconda

```python
# Cell 1 — Download & install Miniconda
!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -p /content/miniconda
```

---

## 2 · Create Conda Environment

```bash
%%bash
# Cell 2 — Accept TOS and create the 'rcbev' env with Python 3.10
/content/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/content/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
/content/miniconda/bin/conda create -y -n rcbev python=3.10
```

---

## 3 · Install PyTorch (CUDA 11.7)

```bash
%%bash
# Cell 3 — PyTorch 1.13.1 + CUDA 11.7
set -e
CONDA=/content/miniconda/bin/conda
ENV=rcbev

$CONDA run -n $ENV pip install \
  torch==1.13.1+cu117 \
  torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

---

## 4 · Install OpenMMLab Stack

```bash
%%bash
# Cell 4 — mmcv-full, mmdet, mmsegmentation
set -e
CONDA=/content/miniconda/bin/conda
ENV=rcbev

# Old OpenMMLab stack needs old numpy / opencv
$CONDA run -n $ENV pip install "numpy<1.24" openmim
$CONDA run -n $ENV pip install opencv-python==4.6.0.66

# mmcv from the matching cu117 / torch1.13 wheel index
$CONDA run -n $ENV pip install \
  mmcv-full==1.7.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# OpenMMLab packages
$CONDA run -n $ENV pip install mmdet==2.28.2
$CONDA run -n $ENV pip install mmsegmentation==0.30.0
```

> **Note**: `mmdet3d` is *not* pip-installed — this repo *is* mmdet3d
> (added to `sys.path` at runtime).

---

## 5 · Install Extra Dependencies

```bash
%%bash
# Cell 5 — CUDA toolkit + Python extras
CONDA=/content/miniconda/bin/conda
ENV=rcbev

# CUDA runtime needed for MMCV ops
$CONDA install -y -n $ENV -c conda-forge cudatoolkit=11.7

# Python packages
$CONDA run -n $ENV pip install \
  nuscenes-devkit \
  lyft-dataset-sdk \
  prettytable \
  trimesh \
  tensorboard \
  timm \
  einops \
  numba==0.57.1 \
  llvmlite==0.40.1 \
  ipython \
  truckscenes-devkit \
  shapely \
  pyquaternion
```

---

## 6 · Mount Google Drive & Verify Paths

```python
# Cell 6 — Mount Drive and validate dataset / weights
from google.colab import drive
drive.mount('/content/drive')

import os

# ──── EDIT THESE PATHS TO MATCH YOUR GOOGLE DRIVE LAYOUT ────
TRUCKSCENES_ROOT = '/content/drive/MyDrive/MAN-Truckscenes'
WEIGHTS_PATH     = '/content/drive/MyDrive/RCBEVDet_weights/rcbevdet.pth'
REPO_PATH        = '/content/drive/MyDrive/rcbevdet-master'
# ─────────────────────────────────────────────────────────────

assert os.path.isdir(TRUCKSCENES_ROOT), f"Dataset not found at {TRUCKSCENES_ROOT}"
assert os.path.isfile(WEIGHTS_PATH),    f"Weights not found at {WEIGHTS_PATH}"
assert os.path.isdir(REPO_PATH),        f"Repo not found at {REPO_PATH}"
print(f"✅ Dataset : {TRUCKSCENES_ROOT}")
print(f"✅ Weights : {WEIGHTS_PATH}")
print(f"✅ Repo    : {REPO_PATH}")

# Auto-detect dataset version
DATASET_VERSION = None
for v in ['v1.1-mini', 'v1.0-mini', 'v1.0-trainval']:
    if os.path.isdir(os.path.join(TRUCKSCENES_ROOT, v)):
        DATASET_VERSION = v
        break
assert DATASET_VERSION, "Could not find version folder in dataset root"
print(f"✅ Version : {DATASET_VERSION}")
```

---

## 7 · Run Data Conversion

Convert TruckScenes metadata (JSON tables) into the `.pkl` info format
that RCBEVDet expects. This uses `tools/data_converter/truckscenes_converter.py`.

```bash
%%bash
# Cell 7 — Generate .pkl info files
CONDA=/content/miniconda/bin/conda
ENV=rcbev

$CONDA run -n $ENV python \
  /content/drive/MyDrive/rcbevdet-master/tools/data_converter/truckscenes_converter.py \
  --root "/content/drive/MyDrive/MAN-Truckscenes" \
  --version "v1.1-mini" \
  --out /content/ts_data
```

Verify the output:

```python
# Cell 7b — Check generated pkl files
import os, glob

files = sorted(glob.glob('/content/ts_data/*.pkl'))
for f in files:
    print(f, f"{os.path.getsize(f) / 1e6:.2f} MB")
```

Expected output:
```
/content/ts_data/truckscenes_infos_train.pkl  X.XX MB
/content/ts_data/truckscenes_infos_val.pkl    X.XX MB
```

---

## 8 · Smoke-Test: Load Model

Quick check that the model builds and the checkpoint loads.

```bash
%%bash
# Cell 8 — Load model from original config
CONDA=/content/miniconda/bin/conda
ENV=rcbev

MPLBACKEND=Agg $CONDA run --no-capture-output -n $ENV bash -c '
export LD_LIBRARY_PATH=/content/miniconda/envs/rcbev/lib:\
/content/miniconda/envs/rcbev/lib/python3.10/site-packages/torch/lib:\
$LD_LIBRARY_PATH

python - <<'"'"'PY'"'"'
import sys, torch
from mmcv import Config

sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master")
sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master/mmdet3d/models/model_utils/ops")

from mmdet3d.models import build_model

CONFIG     = "/content/drive/MyDrive/rcbevdet-master/configs/rcbevdet/rcbevdet-256x704-r50-BEV128-9kf-depth-cbgs12e-circlelarger.py"
CHECKPOINT = "/content/drive/MyDrive/RCBEVDet_weights/rcbevdet.pth"

cfg   = Config.fromfile(CONFIG)
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=False)
model.cuda().eval()

print("✅ RCBEVDet model loaded successfully")
PY
'
```

---

## 9 · Build Dataset with TruckScenes Config

The TruckScenes-specific config lives at
`configs/rcbevdet/truckscenes_rcbevdet_config.py`.

```bash
%%bash
# Cell 9 — Build the dataset object
CONDA=/content/miniconda/bin/conda
ENV=rcbev

MPLBACKEND=Agg $CONDA run --no-capture-output -n $ENV bash -c '
export LD_LIBRARY_PATH=/content/miniconda/envs/rcbev/lib:\
/content/miniconda/envs/rcbev/lib/python3.10/site-packages/torch/lib:\
$LD_LIBRARY_PATH

python - <<'"'"'PY'"'"'
import sys
from mmcv import Config

sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master")
sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master/mmdet3d/models/model_utils/ops")

from mmdet3d.datasets import build_dataset

cfg = Config.fromfile(
    "/content/drive/MyDrive/rcbevdet-master/configs/rcbevdet/truckscenes_rcbevdet_config.py"
)

dataset = build_dataset(cfg.data.train)
print("✅ Dataset built successfully")
print("Length:", len(dataset))
print("Dataset type:", type(dataset).__name__)
PY
'
```

### Inspect one sample

```bash
%%bash
# Cell 9b — Peek at raw data info
CONDA=/content/miniconda/bin/conda
ENV=rcbev

MPLBACKEND=Agg $CONDA run --no-capture-output -n $ENV bash -c '
export LD_LIBRARY_PATH=/content/miniconda/envs/rcbev/lib:\
/content/miniconda/envs/rcbev/lib/python3.10/site-packages/torch/lib:\
$LD_LIBRARY_PATH

python - <<'"'"'PY'"'"'
import sys
from mmcv import Config

sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master")
sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master/mmdet3d/models/model_utils/ops")

from mmdet3d.datasets import build_dataset

cfg = Config.fromfile(
    "/content/drive/MyDrive/rcbevdet-master/configs/rcbevdet/truckscenes_rcbevdet_config.py"
)
dataset = build_dataset(cfg.data.train)
base_dataset = dataset.dataset
info = base_dataset.data_infos[0]

print("token:     ", info["token"])
print("num cams:  ", len(info.get("cams", {})))
print("num radars:", len(info.get("radars", {})))
print("has gt_boxes:", "gt_boxes" in info)
print("cam names: ", list(info.get("cams", {}).keys()))
print("radar names:", list(info.get("radars", {}).keys()))
PY
'
```

---

## 10 · Run Inference (Single Sample)

```bash
%%bash
# Cell 10 — Single-sample inference
CONDA=/content/miniconda/bin/conda
ENV=rcbev

MPLBACKEND=Agg $CONDA run --no-capture-output -n $ENV bash -c '
export LD_LIBRARY_PATH=/content/miniconda/envs/rcbev/lib:\
/content/miniconda/envs/rcbev/lib/python3.10/site-packages/torch/lib:\
$LD_LIBRARY_PATH

python - <<'"'"'PY'"'"'
import sys, torch
from mmcv import Config
from mmcv.parallel import collate, scatter

sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master")
sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master/mmdet3d/models/model_utils/ops")

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset

CONFIG     = "/content/drive/MyDrive/rcbevdet-master/configs/rcbevdet/truckscenes_rcbevdet_config.py"
CHECKPOINT = "/content/drive/MyDrive/RCBEVDet_weights/rcbevdet.pth"

cfg = Config.fromfile(CONFIG)

# Use train dataset (provides full img_inputs incl. bda augmentation)
dataset = build_dataset(cfg.data.train.dataset)
print("dataset length:", len(dataset))

model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=False)
model.cuda().eval()
print("✅ Model ready")

sample = dataset[0]
data   = collate([sample], samples_per_gpu=1)
data   = scatter(data, [torch.cuda.current_device()])[0]

img_inputs = [data["img_inputs"]]
img_metas  = [data["img_metas"]]
radar      = [data["radar"]]

with torch.no_grad():
    result = model(
        return_loss=False, rescale=True,
        img_inputs=img_inputs, img_metas=img_metas,
        radar=radar, points=[None]
    )

print("✅ Inference done")

pred = result[0]["pts_bbox"]
print("num boxes:", len(pred["boxes_3d"]))
print("scores shape:", pred["scores_3d"].shape)
print("first 10 scores:", pred["scores_3d"][:10])
print("first 10 labels:", pred["labels_3d"][:10])

# Save result
out = {
    "boxes_3d":  pred["boxes_3d"].tensor.cpu(),
    "scores_3d": pred["scores_3d"].cpu(),
    "labels_3d": pred["labels_3d"].cpu(),
}
torch.save(out, "/content/sample_result_plain.pt")
print("Saved to /content/sample_result_plain.pt")
PY
'
```

---

## 11 · Run Inference (Batch — First 20 Samples)

```bash
%%bash
# Cell 11 — Loop over first 20 samples, save results.pkl
CONDA=/content/miniconda/bin/conda
ENV=rcbev

MPLBACKEND=Agg $CONDA run --no-capture-output -n $ENV bash -c '
export LD_LIBRARY_PATH=/content/miniconda/envs/rcbev/lib:\
/content/miniconda/envs/rcbev/lib/python3.10/site-packages/torch/lib:\
$LD_LIBRARY_PATH

python - <<'"'"'PY'"'"'
import sys, pickle, torch
from mmcv import Config
from mmcv.parallel import collate, scatter

sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master")
sys.path.insert(0, "/content/drive/MyDrive/rcbevdet-master/mmdet3d/models/model_utils/ops")

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset

CONFIG     = "/content/drive/MyDrive/rcbevdet-master/configs/rcbevdet/truckscenes_rcbevdet_config.py"
CHECKPOINT = "/content/drive/MyDrive/RCBEVDet_weights/rcbevdet.pth"
OUT        = "/content/results.pkl"

cfg     = Config.fromfile(CONFIG)
dataset = build_dataset(cfg.data.train.dataset)

model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=False)
model.cuda().eval()

results     = []
num_samples = min(20, len(dataset))

for idx in range(num_samples):
    sample = dataset[idx]
    data   = collate([sample], samples_per_gpu=1)
    data   = scatter(data, [torch.cuda.current_device()])[0]

    with torch.no_grad():
        result = model(
            return_loss=False, rescale=True,
            img_inputs=[data["img_inputs"]],
            img_metas=[data["img_metas"]],
            radar=[data["radar"]],
            points=[None]
        )
    results.append(result[0])

with open(OUT, "wb") as f:
    pickle.dump(results, f)

print(f"Saved {len(results)} predictions to {OUT}")
PY
'
```

---

## 12 · Full Evaluation (Optional — Long Running)

To run over **all** validation samples using the built-in test script:

```python
import os
os.chdir('/content/drive/MyDrive/rcbevdet-master')

WEIGHTS_PATH = '/content/drive/MyDrive/RCBEVDet_weights/rcbevdet.pth'

!python tools/test.py \
    configs/rcbevdet/truckscenes_rcbevdet_config.py \
    "{WEIGHTS_PATH}" \
    --eval bbox \
    --out /content/results.pkl \
    2>&1
```

> ⚠️ This will take a **significant amount of time** depending on dataset
> size and GPU. The 20-sample loop in Section 11 is recommended for quick
> validation.

---

## Key Files Modified for TruckScenes

| File | Purpose |
|------|---------|
| `mmdet3d/datasets/truckscenes_dataset_rc.py` | Dataset class that maps TruckScenes categories → nuScenes 10-class taxonomy and aliases camera names |
| `mmdet3d/datasets/__init__.py` | Registers `TruckScenesDatasetRC` |
| `tools/data_converter/truckscenes_converter.py` | Converts TruckScenes metadata to `.pkl` info files with camera, radar, lidar, and annotation data |
| `configs/rcbevdet/truckscenes_rcbevdet_config.py` | TruckScenes-specific config (2 cameras, 6 radars, data paths, pipelines) |

---

## TruckScenes → nuScenes Camera Mapping

TruckScenes uses different camera names than nuScenes. The dataset class
handles this transparently:

| TruckScenes Camera | → nuScenes Alias |
|--------------------|------------------|
| `CAMERA_LEFT_FRONT` | `CAM_FRONT`, `CAM_FRONT_LEFT` |
| `CAMERA_RIGHT_FRONT` | `CAM_FRONT_RIGHT` |
| `CAMERA_LEFT_BACK` | `CAM_BACK_LEFT`, `CAM_BACK` (fallback) |
| `CAMERA_RIGHT_BACK` | `CAM_BACK_RIGHT` |

---

## TruckScenes Category Mapping

All TruckScenes categories are mapped to the 10-class nuScenes taxonomy
so the pre-trained weights work directly:

| TruckScenes Category | → nuScenes Class |
|----------------------|------------------|
| `vehicle.car`, `vehicle.emergency.*`, `vehicle.other` | `car` |
| `vehicle.truck`, `vehicle.train` | `truck` |
| `vehicle.trailer`, `vehicle.ego_trailer` | `trailer` |
| `vehicle.bus.*` | `bus` |
| `vehicle.construction` | `construction_vehicle` |
| `vehicle.motorcycle` | `motorcycle` |
| `vehicle.bicycle` | `bicycle` |
| `human.pedestrian.*`, `animal` | `pedestrian` |
| `movable_object.trafficcone`, `static_object.traffic_sign` | `traffic_cone` |
| `movable_object.barrier`, `movable_object.*`, `static_object.bicycle_rack` | `barrier` |

---

## Dependency Version Summary

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | Via Conda |
| PyTorch | 1.13.1+cu117 | CUDA 11.7 |
| mmcv-full | 1.7.0 | cu117/torch1.13 wheel |
| mmdet | 2.28.2 | |
| mmsegmentation | 0.30.0 | |
| mmdet3d | *this repo* | Not pip-installed |
| numpy | <1.24 | Required by old OpenMMLab |
| opencv-python | 4.6.0.66 | |
| numba | 0.57.1 | |
| llvmlite | 0.40.1 | |
| truckscenes-devkit | latest | MAN TruckScenes SDK |
| nuscenes-devkit | latest | nuScenes SDK |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: mmdet3d` | Make sure `sys.path.insert(0, REPO_PATH)` is called before imports |
| `ImportError` for CUDA ops | Ensure `LD_LIBRARY_PATH` includes the conda env lib dirs |
| `GlobalRotScaleTrans_radar` error in test pipeline | Remove it from `test_pipeline` — only needed in train |
| Missing camera file errors | Verify all camera images exist in the dataset folder |
| `strict=False` warnings | Expected — TruckScenes uses 2 cameras vs nuScenes 6, so some weights are unused |
