# TruckScenes Code Modifications — Detailed Reference

> This document details every file that was added or modified to adapt
> RCBEVDet for the MAN TruckScenes dataset.

---

## 1. New Files

### 1.1 `mmdet3d/datasets/truckscenes_dataset_rc.py`

**Purpose**: Dataset class that bridges TruckScenes data into the nuScenes-based
RCBEVDet pipeline.

**How it works**: Inherits from `NuScenesDatasetRC` and overrides two things:

#### Category Mapping (27 → 10 classes)

```python
TRUCKSCENES_NAME_MAPPING = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.trailer': 'trailer',
    'vehicle.emergency.police': 'car',
    'vehicle.emergency.ambulance': 'car',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'human.pedestrian.wheelchair': 'pedestrian',
    'human.pedestrian.personal_mobility': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'animal': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.barrier': 'barrier',
    'movable_object.pushable_pullable': 'barrier',
    'movable_object.debris': 'barrier',
    'static_object.bicycle_rack': 'barrier',
    'static_object.traffic_sign': 'traffic_cone',
    'vehicle.train': 'truck',
    'vehicle.other': 'car',
    'vehicle.ego_trailer': 'trailer',
}
```

#### Camera Name Aliasing (`get_data_info`)

The method temporarily injects nuScenes-style camera names into the data info
so the parent class can process them:

| TruckScenes | → nuScenes alias | Fallback |
|-------------|------------------|----------|
| `CAMERA_LEFT_FRONT` | `CAM_FRONT`, `CAM_FRONT_LEFT` | Primary front camera |
| `CAMERA_RIGHT_FRONT` | `CAM_FRONT_RIGHT`, `CAM_FRONT` (if left not available) | — |
| `CAMERA_LEFT_BACK` | `CAM_BACK_LEFT`, `CAM_BACK` | Fallback rear camera |
| `CAMERA_RIGHT_BACK` | `CAM_BACK_RIGHT` | — |

Uses a try/finally pattern to restore the original `data_infos` after aliasing.

#### Annotation Handling

Converts `LiDARInstance3DBoxes` to plain numpy arrays for the `ann_infos` tuple
format expected by the pipeline:

```python
if hasattr(ann['gt_bboxes_3d'], 'tensor'):
    gt_boxes = ann['gt_bboxes_3d'].tensor.numpy()
else:
    gt_boxes = np.asarray(ann['gt_bboxes_3d'])
```

---

### 1.2 `tools/data_converter/truckscenes_converter.py`

**Purpose**: Converts TruckScenes metadata JSON tables into `.pkl` info files
(the format RCBEVDet reads at runtime).

**Key functions**:

| Function | Description |
|----------|-------------|
| `obtain_sensor2top()` | Computes the sensor-to-reference-lidar transformation matrix |
| `_fill_infos()` | Iterates over all samples, extracts camera/radar/lidar info, annotations, and splits into train/val |
| `create_infos()` | Entry point: loads TruckScenes via `TruckScenes` devkit, creates splits, calls `_fill_infos`, saves `.pkl` |

**Sensor configuration**:

```python
CAMERA_NAMES = ['CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT']

RADAR_NAMES = [
    'RADAR_LEFT_FRONT', 'RADAR_RIGHT_FRONT',
    'RADAR_LEFT_BACK', 'RADAR_LEFT_SIDE',
    'RADAR_RIGHT_BACK', 'RADAR_RIGHT_SIDE',
]

LIDAR_NAMES = [
    'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT',
    'LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_REAR',
]
```

**Radar sweep handling**: Each radar entry is stored as a list of sweep dicts
(key sample + up to `max_sweeps` previous frames), matching the nuScenes format
expected by `LoadRadarPointsMultiSweeps`.

**Annotation processing**: Converts global-frame ground truth boxes to lidar
frame, swaps `[w, l, h]` → `[l, w, h]` to match nuScenes convention.

---

### 1.3 `configs/rcbevdet/truckscenes_rcbevdet_config.py`

**Purpose**: TruckScenes-specific config inheriting from the base RCBEVDet
config.

**Key settings**:

| Parameter | Value | Note |
|-----------|-------|------|
| `dataset_type` | `TruckScenesDatasetRC` | New dataset class |
| `Ncams` | 2 | vs 6 in nuScenes |
| `cams` | `CAMERA_LEFT_FRONT`, `CAMERA_RIGHT_FRONT` | TruckScenes names |
| `input_size` | `(256, 704)` | Same as nuScenes config |
| `src_size` | `(943, 1980)` | TruckScenes native resolution |
| `sweeps_num` | 8 | Radar sweeps |
| `max_num` | 1200 | Max radar points |
| `samples_per_gpu` | 1 | Colab GPU memory |

**Note**: Contains Colab-specific absolute paths (`/content/drive/MyDrive/...`)
that users need to adjust for their environment.

---

## 2. Edited Files

### 2.1 `mmdet3d/datasets/pipelines/loading.py`

**3 classes modified** (compared against `loading_bk.py` backup):

#### Change A: PCD File Reader (3 instances)

**Classes affected**:
- `LoadRadarPointsMultiSweeps._load_points` (line ~666)
- `LoadRadarPointsMultiSweep2image._load_points` (line ~847)
- `LoadRadarPointsMultiSweeps_HoP._load_points` (line ~4012)

**Original** (nuScenes):
```python
def _load_points(self, pts_filename):
    radar_obj = RadarPointCloud.from_file(pts_filename)
    points = radar_obj.points       # [18, N]
    return points.transpose().astype(np.float32)
```

**Modified** (TruckScenes):
```python
def _load_points(self, pts_filename):
    with open(pts_filename, "rb") as f:
        # Parse PCD header
        num_points = None
        while True:
            line = f.readline().decode("utf-8").strip()
            if line.startswith("POINTS"):
                num_points = int(line.split()[1])
            if line.startswith("DATA"):
                break
        if num_points is None:
            raise ValueError(
                f"Could not parse POINTS from PCD header: {pts_filename}"
            )
        # Read binary payload after header
        data = np.fromfile(f, dtype=np.float32)
    points = data.reshape(num_points, -1)
    return points
```

**Why**: TruckScenes radar PCD files have a different binary layout than
nuScenes. The `RadarPointCloud.from_file()` from nuscenes-devkit expects the
nuScenes 18-field format and fails on TruckScenes 7-field radar data. The
manual reader parses the PCD header to determine the number of points, then
reads the raw float32 binary payload.

---

#### Change B: Velocity Field Remapping

**Class**: `LoadRadarPointsMultiSweeps.__call__` (line ~754)

**Original** (nuScenes — 18 radar fields):
```python
# compensated velocity at columns 8:10 (2D)
velo_comp = points_sweep[:, 8:10]
velo_comp = np.concatenate((velo_comp, np.zeros((N, 1))), 1)  # pad to 3D
velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
velo_comp = velo_comp[:, :2]  # back to 2D

# raw velocity at columns 6:8 (2D)
velo = points_sweep[:, 6:8]
velo = np.concatenate((velo, np.zeros((N, 1))), 1)
velo = velo @ sweep['sensor2lidar_rotation'].T
velo = velo[:, :2]

# concatenate: [pos(6), velo(2), velo_comp(2), other(6+), time(1)]
points_sweep_ = np.concatenate(
    [points_sweep[:, :6], velo, velo_comp, points_sweep[:, 10:], time_diff],
    axis=1)
```

**Modified** (TruckScenes — 7 radar fields `[x, y, z, vrel_x, vrel_y, vrel_z, rcs]`):
```python
# full 3D relative velocity at columns 3:6
velo_comp = points_sweep[:, 3:6]   # (N, 3)

# rotate from radar sensor frame to lidar frame
velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T

# write rotated velocity back
points_sweep[:, 3:6] = velo_comp

# transform position from sensor frame to lidar frame
points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
points_sweep[:, :3] += sweep['sensor2lidar_translation']

# simple concat: [all_fields(7), time(1)]
points_sweep_ = np.concatenate([points_sweep, time_diff], axis=1)
```

**Why**: nuScenes radar provides 18 fields per point with separate 2D raw and
compensated velocities at different column indices. TruckScenes radar provides
7 fields with full 3D relative velocity at indices 3:6. The concatenation is
simplified since there are no separate velocity channels to merge.

---

### 2.2 `mmdet3d/datasets/nuscenes_dataset_rc.py`

**Role**: Base dataset class for nuScenes + radar camera fusion.

**Notable characteristics** (not TruckScenes-specific changes, but important
for understanding the inheritance):

| Feature | Detail |
|---------|--------|
| Inherits from | `Custom3DDatasetradar` |
| `get_data_info()` | Reads `radars` from info dict, handles BEVDet4D adjacent frames |
| `get_ann_info()` | Filters by `valid_flag` or `num_lidar_pts > 0`, adds velocity |
| `_format_bbox()` | Converts predictions to nuScenes submission format |
| `_evaluate_single()` | Uses `NuScenesEval` for evaluation |
| `ego_cam` | Defaults to `CAM_FRONT` — this is why the TruckScenes dataset alias must create a `CAM_FRONT` entry |

**Line 268** is critical — the parent class hardcodes `CAM_FRONT`:
```python
cam_ego2global[:3, :3] = Quaternion(
    info['cams']['CAM_FRONT']['ego2global_rotation']
).rotation_matrix
```

This is why `TruckScenesDatasetRC.get_data_info()` must alias
`CAMERA_LEFT_FRONT` → `CAM_FRONT` before calling `super().get_data_info()`.

---

### 2.3 `mmdet3d/datasets/custom_3d_radar.py`

**Role**: Base dataset class that adds radar support to the standard
`Custom3DDataset`.

**Key difference from `Custom3DDataset`**: This class serves as the foundation
for `NuScenesDatasetRC`. It provides:

- Standard dataset loading from `.pkl` annotation files
- Train/test data preparation with pipeline execution
- Ground truth annotation extraction
- Class name → ID mapping

**No TruckScenes-specific modifications found.** This file provides the
infrastructure that both nuScenes and TruckScenes datasets inherit.

---

### 2.4 `mmdet3d/datasets/pipelines/transforms_3d.py`

**Role**: Data augmentation transforms for 3D detection.

**Key class — `GlobalRotScaleTrans_radar`** (line 868):

This is a radar-specific version of `GlobalRotScaleTrans` that applies
augmentations to the `radar` key instead of `points`:

```python
@PIPELINES.register_module()
class GlobalRotScaleTrans_radar(object):
    def __call__(self, input_dict):
        self._rot_bbox_points(input_dict)    # rotates input_dict['radar']
        self._scale_bbox_points(input_dict)  # scales input_dict['radar']
        self._flip_bbox_points(input_dict)   # flips input_dict['radar']
        return input_dict
```

| Method | What it does |
|--------|-------------|
| `_rot_bbox_points` | Rotates radar points by `input_dict['rotate_bda']` (BDA augmentation angle) |
| `_scale_bbox_points` | Scales radar points by `input_dict['scale_bda']` |
| `_flip_bbox_points` | Flips radar points based on `flip_dx`/`flip_dy` flags |

**Note**: This transform is used in `train_pipeline` but was **removed from
`test_pipeline`** during the Colab setup because `rotate_bda` and `scale_bda`
are not set during inference (only set by `LoadAnnotationsBEVDepth` in
training mode).

**No TruckScenes-specific modifications found** in this class — it operates on
the `radar` key generically.

---

### 2.5 `mmdet3d/datasets/__init__.py`

**Added** at the end of the file:

```python
from .truckscenes_dataset_rc import TruckScenesDatasetRC
```

This registers the new dataset class with the mmdet3d module system.

---

## 3. Radar Data Format Comparison

| Field | nuScenes Index | nuScenes Name | TruckScenes Index | TruckScenes Name |
|-------|---------------|---------------|-------------------|------------------|
| x | 0 | x | 0 | x |
| y | 1 | y | 1 | y |
| z | 2 | z | 2 | z |
| dyn_prop | 3 | — | — | — |
| id | 4 | — | — | — |
| rcs | 5 | rcs | 6 | rcs |
| vx (raw) | 6 | vx | 3 | vrel_x |
| vy (raw) | 7 | vy | 4 | vrel_y |
| vx_comp | 8 | vx_comp | 5 | vrel_z |
| vy_comp | 9 | vy_comp | — | — |
| ... | 10-17 | (8 more fields) | — | — |
| **Total** | **18 fields** | | **7 fields** | |

---

## 4. File Dependency Graph

```
Custom3DDatasetradar  (custom_3d_radar.py)
    └── NuScenesDatasetRC  (nuscenes_dataset_rc.py)
            └── TruckScenesDatasetRC  (truckscenes_dataset_rc.py)  [NEW]

Loading pipeline:
    LoadRadarPointsMultiSweeps  (loading.py)  [MODIFIED]
        ├── _load_points()  — PCD binary reader
        └── __call__()      — velocity field remapping

Transforms pipeline:
    GlobalRotScaleTrans_radar  (transforms_3d.py)
        ├── _rot_bbox_points()   — rotates radar points
        ├── _scale_bbox_points() — scales radar points
        └── _flip_bbox_points()  — flips radar points

Data conversion:
    truckscenes_converter.py  [NEW]
        ├── obtain_sensor2top()  — sensor to lidar transform
        ├── _fill_infos()        — extracts all sample info
        └── create_infos()       — entry point, saves .pkl
```
