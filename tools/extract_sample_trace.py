"""
Extract one TruckScenes sample trace through the full RCBEVDet pipeline.

Run on Google Colab after setting up the environment (see COLAB_SETUP_GUIDE.md).

Usage:
    python tools/extract_sample_trace.py \
        --config configs/rcbevdet/truckscenes_rcbevdet_config.py \
        --checkpoint /path/to/rcbevdet.pth \
        --pkl /path/to/truckscenes_infos_val.pkl \
        --data-root /path/to/MAN-Truckscenes/ \
        --out sample_trace.json \
        --sample-idx 0
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import mmcv

from collections import OrderedDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tensor_info(t, name=""):
    """Summarize a tensor's shape, dtype, range, mean."""
    if t is None:
        return {"name": name, "value": "None"}
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    if not isinstance(t, torch.Tensor):
        return {"name": name, "type": str(type(t).__name__), "value": str(t)[:200]}
    info = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "min": round(float(t.float().min()), 4),
        "max": round(float(t.float().max()), 4),
        "mean": round(float(t.float().mean()), 4),
        "std": round(float(t.float().std()), 4) if t.numel() > 1 else 0.0,
        "memory_bytes": int(t.numel() * t.element_size()),
    }
    return info


def numpy_info(arr, name=""):
    """Summarize a numpy array."""
    if arr is None:
        return {"name": name, "value": "None"}
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    if not isinstance(arr, np.ndarray):
        return {"name": name, "type": str(type(arr).__name__), "value": str(arr)[:200]}
    info = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
        "min": round(float(np.nanmin(arr).astype(float)), 4) if arr.size > 0 else None,
        "max": round(float(np.nanmax(arr).astype(float)), 4) if arr.size > 0 else None,
        "mean": round(float(np.nanmean(arr).astype(float)), 4) if arr.size > 0 else None,
    }
    # First few values as a sample
    flat = arr.flatten()
    info["sample_values"] = [round(float(v), 4) for v in flat[:8]]
    return info


# ---------------------------------------------------------------------------
# Stage 1: Raw Data Loading (from .pkl)
# ---------------------------------------------------------------------------
def trace_raw_data(data_infos, data_root, sample_idx):
    """Trace Stage 1: What the raw sample looks like from the pkl."""
    info = data_infos[sample_idx]
    stage = {
        "stage_name": "1_raw_data",
        "stage_title": "Raw TruckScenes Sample",
        "description": "Sample metadata loaded from .pkl info file",
        "sample_token": info.get("token", "N/A"),
        "timestamp": info.get("timestamp", 0),
        "scene_token": info.get("scene_token", "N/A"),
        "lidar_path": info.get("lidar_path", "N/A"),
        "location": info.get("location", "N/A"),
    }

    # Camera info
    cams = info.get("cams", {})
    cam_details = {}
    for cam_name, cam_info in cams.items():
        cam_detail = {
            "data_path": cam_info.get("data_path", ""),
            "cam_intrinsic": numpy_info(
                np.array(cam_info.get("cam_intrinsic", [])), f"{cam_name}_intrinsic"
            ),
            "sensor2lidar_rotation": numpy_info(
                np.array(cam_info.get("sensor2lidar_rotation", [])),
                f"{cam_name}_s2l_rot",
            ),
            "sensor2lidar_translation": numpy_info(
                np.array(cam_info.get("sensor2lidar_translation", [])),
                f"{cam_name}_s2l_trans",
            ),
        }
        cam_details[cam_name] = cam_detail
    stage["cameras"] = cam_details
    stage["num_cameras"] = len(cams)

    # Radar info
    radars = info.get("radars", {})
    radar_details = {}
    for radar_name, sweeps in radars.items():
        sweep_info = []
        for i, sweep in enumerate(sweeps[:3]):  # first 3 sweeps only
            si = {
                "sweep_idx": i,
                "data_path": sweep.get("data_path", ""),
                "timestamp": sweep.get("timestamp", 0),
                "sensor2lidar_rotation": numpy_info(
                    np.array(sweep.get("sensor2lidar_rotation", [])),
                    f"{radar_name}_sweep{i}_s2l_rot",
                ),
                "sensor2lidar_translation": numpy_info(
                    np.array(sweep.get("sensor2lidar_translation", [])),
                    f"{radar_name}_sweep{i}_s2l_trans",
                ),
            }
            sweep_info.append(si)
        radar_details[radar_name] = {
            "num_sweeps": len(sweeps),
            "sweeps_sample": sweep_info,
        }
    stage["radars"] = radar_details
    stage["num_radars"] = len(radars)

    # Ground truth annotations
    gt_boxes = info.get("gt_boxes", np.array([]))
    gt_names = info.get("gt_names", np.array([]))
    gt_velocity = info.get("gt_velocity", np.array([]))
    stage["annotations"] = {
        "num_objects": len(gt_names),
        "gt_names": list(gt_names[:20]) if len(gt_names) > 0 else [],
        "gt_boxes": numpy_info(gt_boxes, "gt_boxes"),
        "gt_velocity": numpy_info(gt_velocity, "gt_velocity"),
        "class_distribution": {},
    }
    if len(gt_names) > 0:
        unique, counts = np.unique(gt_names, return_counts=True)
        stage["annotations"]["class_distribution"] = {
            str(k): int(v) for k, v in zip(unique, counts)
        }

    # Ego and global transforms
    stage["transforms"] = {
        "lidar2ego_translation": list(info.get("lidar2ego_translation", [])),
        "lidar2ego_rotation": list(info.get("lidar2ego_rotation", [])),
        "ego2global_translation": list(info.get("ego2global_translation", [])),
        "ego2global_rotation": list(info.get("ego2global_rotation", [])),
    }

    return stage


# ---------------------------------------------------------------------------
# Stage 2: Load one radar PCD file and show field parsing
# ---------------------------------------------------------------------------
def trace_radar_loading(data_infos, data_root, sample_idx):
    """Trace Stage 2: Radar PCD loading and point parsing."""
    info = data_infos[sample_idx]
    radars = info.get("radars", {})

    stage = {
        "stage_name": "2_radar_loading",
        "stage_title": "LoadRadarPointsMultiSweeps",
        "description": "Load radar PCD files, parse binary header, transform to lidar frame",
        "config": {
            "load_dim": 7,
            "sweeps_num": 8,
            "use_dim": [0, 1, 2, 3, 4, 5, 6],
            "max_num": 1200,
        },
        "per_radar": {},
    }

    total_points = 0
    all_points_list = []

    for radar_name, sweeps in radars.items():
        radar_stage = {"sweeps_available": len(sweeps), "points_per_sweep": []}
        for i, sweep in enumerate(sweeps[:8]):
            pcd_path = sweep.get("data_path", "")
            if not os.path.isabs(pcd_path):
                pcd_path = os.path.join(data_root, pcd_path)

            if not os.path.exists(pcd_path):
                radar_stage["points_per_sweep"].append(
                    {"sweep": i, "error": f"File not found: {pcd_path}"}
                )
                continue

            # Parse PCD manually (same as modified loading.py)
            try:
                with open(pcd_path, "rb") as f:
                    num_points = None
                    header_lines = []
                    while True:
                        line = f.readline().decode("utf-8").strip()
                        header_lines.append(line)
                        if line.startswith("POINTS"):
                            num_points = int(line.split()[1])
                        if line.startswith("DATA"):
                            break
                    if num_points is None:
                        raise ValueError("Could not parse POINTS from PCD header")
                    data = np.fromfile(f, dtype=np.float32)

                points = data.reshape(num_points, -1)
                total_points += num_points

                sweep_detail = {
                    "sweep": i,
                    "pcd_path": os.path.basename(pcd_path),
                    "pcd_header": header_lines[:10],
                    "num_points": num_points,
                    "fields_per_point": points.shape[1] if len(points.shape) > 1 else 0,
                    "raw_points": numpy_info(points, f"{radar_name}_sweep{i}_raw"),
                }

                # Show field breakdown
                if points.shape[1] >= 7:
                    sweep_detail["field_breakdown"] = {
                        "x": numpy_info(points[:, 0], "x"),
                        "y": numpy_info(points[:, 1], "y"),
                        "z": numpy_info(points[:, 2], "z"),
                        "vrel_x": numpy_info(points[:, 3], "vrel_x"),
                        "vrel_y": numpy_info(points[:, 4], "vrel_y"),
                        "vrel_z": numpy_info(points[:, 5], "vrel_z"),
                        "rcs": numpy_info(points[:, 6], "rcs"),
                    }

                # Show coordinate transform
                s2l_rot = np.array(sweep.get("sensor2lidar_rotation", np.eye(3)))
                s2l_trans = np.array(
                    sweep.get("sensor2lidar_translation", np.zeros(3))
                )
                if len(points) > 0:
                    sample_pt = points[0, :3].copy()
                    vel = points[0, 3:6].copy()
                    # Transform position
                    transformed_pos = sample_pt @ s2l_rot.T + s2l_trans
                    # Transform velocity
                    transformed_vel = vel @ s2l_rot.T
                    sweep_detail["transform_example"] = {
                        "description": "First point: sensor → lidar frame",
                        "original_position": [round(float(v), 4) for v in sample_pt],
                        "rotation_matrix": s2l_rot.tolist(),
                        "translation": [round(float(v), 4) for v in s2l_trans],
                        "transformed_position": [
                            round(float(v), 4) for v in transformed_pos
                        ],
                        "original_velocity": [round(float(v), 4) for v in vel],
                        "transformed_velocity": [
                            round(float(v), 4) for v in transformed_vel
                        ],
                    }

                all_points_list.append(points)
                radar_stage["points_per_sweep"].append(sweep_detail)
            except Exception as e:
                radar_stage["points_per_sweep"].append(
                    {"sweep": i, "error": str(e)}
                )

        stage["per_radar"][radar_name] = radar_stage

    stage["total_raw_points"] = total_points
    stage["radar_field_format"] = [
        "x", "y", "z", "vrel_x", "vrel_y", "vrel_z", "rcs"
    ]

    # After concatenation + use_dim filtering
    if all_points_list:
        all_pts = np.concatenate(all_points_list, axis=0)
        use_dim = [0, 1, 2, 3, 4, 5, 6]
        filtered = all_pts[:, use_dim]
        stage["after_concat_and_filter"] = numpy_info(
            filtered, "radar_points_filtered"
        )
    return stage


# ---------------------------------------------------------------------------
# Stage 3: Camera Image Loading
# ---------------------------------------------------------------------------
def trace_camera_loading(data_infos, data_root, sample_idx):
    """Trace Stage 3: Camera image loading and preprocessing."""
    info = data_infos[sample_idx]
    cams = info.get("cams", {})

    stage = {
        "stage_name": "3_camera_loading",
        "stage_title": "PrepareImageInputs",
        "description": "Load camera images, apply augmentation, prepare for backbone",
        "config": {
            "input_size": [256, 704],
            "src_size": [943, 1980],
            "Ncams": 2,
            "cams": ["CAMERA_LEFT_FRONT", "CAMERA_RIGHT_FRONT"],
            "sequential": True,
            "num_adj_frames": 8,
        },
        "per_camera": {},
    }

    for cam_name in ["CAMERA_LEFT_FRONT", "CAMERA_RIGHT_FRONT"]:
        if cam_name not in cams:
            stage["per_camera"][cam_name] = {"error": "Camera not found in info"}
            continue

        cam_info = cams[cam_name]
        img_path = cam_info.get("data_path", "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(data_root, img_path)

        cam_detail = {
            "image_path": os.path.basename(img_path),
            "exists": os.path.exists(img_path),
            "cam_intrinsic": numpy_info(
                np.array(cam_info.get("cam_intrinsic", [])), "intrinsic"
            ),
        }

        if os.path.exists(img_path):
            try:
                import cv2
                img = cv2.imread(img_path)
                cam_detail["original_size"] = list(img.shape[:2])  # [H, W]
                cam_detail["original_channels"] = img.shape[2] if len(img.shape) > 2 else 1
                cam_detail["original_dtype"] = str(img.dtype)
                cam_detail["pixel_range"] = [int(img.min()), int(img.max())]

                # After resize to input_size
                resized = cv2.resize(img, (704, 256))
                cam_detail["after_resize"] = {
                    "size": [256, 704],
                    "shape": list(resized.shape),
                }
            except Exception as e:
                cam_detail["load_error"] = str(e)

        stage["per_camera"][cam_name] = cam_detail

    # Tensor output shape
    stage["output_tensor"] = {
        "description": "After PrepareImageInputs, images become a tensor",
        "shape": "(B, N*num_frames, C, H, W)",
        "example": f"(1, {2 * 9}, 3, 256, 704) = (1, 18, 3, 256, 704)",
        "note": "2 cameras × (1 key + 8 adjacent) frames = 18 views",
    }

    return stage


# ---------------------------------------------------------------------------
# Stage 4-7: Model forward pass (requires GPU + weights)
# ---------------------------------------------------------------------------
def trace_model_forward(config_path, checkpoint_path, data_infos, data_root, sample_idx):
    """Trace Stage 4-7: Full model forward pass with hook instrumentation."""
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset, build_dataloader

    cfg = Config.fromfile(config_path)

    # Override data root + ann file if needed
    cfg.data.test.data_root = data_root
    cfg.data.test.test_mode = True

    stages = {}

    # --- Stage 4: Model Architecture Summary ---
    model_stage = {
        "stage_name": "4_model_architecture",
        "stage_title": "RCBEVDet Model Architecture",
        "model_type": cfg.model.type,
        "components": {
            "img_backbone": {
                "type": cfg.model.img_backbone.type,
                "depth": cfg.model.img_backbone.depth,
                "out_indices": list(cfg.model.img_backbone.out_indices),
                "output_channels": [1024, 2048],
            },
            "img_neck": {
                "type": cfg.model.img_neck.type,
                "in_channels": cfg.model.img_neck.in_channels,
                "out_channels": cfg.model.img_neck.out_channels,
            },
            "img_view_transformer": {
                "type": cfg.model.img_view_transformer.type,
                "in_channels": cfg.model.img_view_transformer.in_channels,
                "out_channels": cfg.model.img_view_transformer.out_channels,
                "downsample": cfg.model.img_view_transformer.downsample,
                "grid_config": dict(cfg.model.img_view_transformer.grid_config),
            },
            "img_bev_encoder": {
                "backbone_type": cfg.model.img_bev_encoder_backbone.type,
                "neck_type": cfg.model.img_bev_encoder_neck.type,
                "neck_out_channels": cfg.model.img_bev_encoder_neck.out_channels,
            },
            "radar_voxel_encoder": {
                "type": cfg.model.radar_voxel_encoder.type,
                "in_channels": cfg.model.radar_voxel_encoder.in_channels,
                "feat_channels": list(cfg.model.radar_voxel_encoder.feat_channels),
                "voxel_size": list(cfg.model.radar_voxel_layer.voxel_size),
            },
            "radar_middle_encoder": {
                "type": cfg.model.radar_middle_encoder.type,
                "in_channels": cfg.model.radar_middle_encoder.in_channels,
                "output_shape": list(cfg.model.radar_middle_encoder.output_shape),
            },
            "radar_bev_backbone": {
                "type": cfg.model.radar_bev_backbone.type,
                "in_channels": cfg.model.radar_bev_backbone.in_channels,
                "out_channels": list(cfg.model.radar_bev_backbone.out_channels),
            },
            "radar_bev_neck": {
                "type": cfg.model.radar_bev_neck.type,
                "out_channels": list(cfg.model.radar_bev_neck.out_channels),
            },
            "fusion": {
                "type": "CrossAttention + MSDeformAttn",
                "deform_attn_1": {"d_model": 256, "n_levels": 1, "n_heads": 8, "n_points": 8},
                "deform_attn_2": {"d_model": 256, "n_levels": 1, "n_heads": 8, "n_points": 8},
                "conv_fuser": {"in_channels": [256, 256], "out_channels": 256, "deconv_blocks": 3},
            },
            "detection_head": {
                "type": cfg.model.pts_bbox_head.type,
                "in_channels": cfg.model.pts_bbox_head.in_channels,
                "num_classes": 10,
                "common_heads": dict(cfg.model.pts_bbox_head.common_heads),
            },
        },
    }
    stages["4_model_architecture"] = model_stage

    # Try to build model and run inference
    try:
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_stage["parameters"] = {
            "total": total_params,
            "total_millions": round(total_params / 1e6, 2),
            "trainable": trainable_params,
            "trainable_millions": round(trainable_params / 1e6, 2),
        }

        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
            model_stage["checkpoint"] = {
                "path": os.path.basename(checkpoint_path),
                "size_mb": round(os.path.getsize(checkpoint_path) / 1e6, 1),
                "keys_count": len(checkpoint.get("state_dict", checkpoint).keys()),
            }

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Build test dataset to get one properly formatted sample
        test_dataset = build_dataset(cfg.data.test)
        sample = test_dataset[sample_idx]

        # --- Stage 5: Image Branch Trace ---
        img_inputs = sample["img_inputs"]
        if isinstance(img_inputs, (list, tuple)):
            img_stage = {
                "stage_name": "5_image_branch",
                "stage_title": "Camera → BEV Feature Extraction",
                "steps": [],
            }
            for j, inp in enumerate(img_inputs):
                if isinstance(inp, torch.Tensor):
                    img_stage["steps"].append(
                        tensor_info(inp, f"img_input_{j}")
                    )
            stages["5_image_branch"] = img_stage

        # --- Stage 6: Radar Branch Trace ---
        radar_data = sample.get("radar", None)
        if radar_data is not None:
            radar_tensor = radar_data if isinstance(radar_data, torch.Tensor) else radar_data.data
            radar_stage = {
                "stage_name": "6_radar_branch",
                "stage_title": "Radar → BEV Feature Extraction",
                "input_radar": tensor_info(radar_tensor, "radar_points"),
                "pipeline_steps": [
                    {
                        "step": "Voxelization",
                        "voxel_size": [0.2, 0.2, 8],
                        "max_points_per_voxel": 10,
                        "output": "voxels (N_vox, 10, 7), coors (N_vox, 3), num_points (N_vox,)",
                    },
                    {
                        "step": "RadarBEVNet (Voxel Encoder)",
                        "description": "Dual-stream: Point-based (RFN) + Transformer-based with Injector/Extractor",
                        "input": "(N_vox, 10, 7+2) → 9 features per point",
                        "output": "(N_vox, 64)",
                    },
                    {
                        "step": "PointPillarsScatterRCS",
                        "description": "Scatter voxel features to BEV grid + RCS channel",
                        "output": "(1, 64, 512, 512)",
                    },
                    {
                        "step": "SECOND backbone",
                        "in_channels": 64,
                        "layers": [3, 5, 5],
                        "strides": [2, 2, 2],
                        "output": "[(1, 64, 256, 256), (1, 128, 128, 128), (1, 256, 64, 64)]",
                    },
                    {
                        "step": "SECONDFPN neck",
                        "upsample_strides": [0.5, 1, 2],
                        "output": "(1, 384, 128, 128) → conv → (1, 256, 128, 128)",
                    },
                ],
            }
            stages["6_radar_branch"] = radar_stage

        # --- Stage 7: Fusion + Detection ---
        fusion_stage = {
            "stage_name": "7_fusion_detection",
            "stage_title": "Cross-Attention Fusion → Detection",
            "steps": [
                {
                    "step": "radar_reduc_conv",
                    "description": "Reduce radar channel: 384 → 256",
                    "input": "(1, 384, 128, 128)",
                    "output": "(1, 256, 128, 128)",
                },
                {
                    "step": "Flatten for attention",
                    "description": "Rearrange BEV features for deformable attention",
                    "camera_bev": "(1, 128*128, 256) = (1, 16384, 256)",
                    "radar_bev": "(1, 128*128, 256) = (1, 16384, 256)",
                },
                {
                    "step": "Positional Encoding",
                    "description": "LearnedPositionalEncoding3D for both modalities",
                    "shape": "(1, 256, 128, 128)",
                },
                {
                    "step": "DeformAttn1: Radar queries Camera",
                    "description": "MSDeformAttn(query=radar+PE, ref_pts, input=camera+PE)",
                    "n_heads": 8,
                    "n_points": 8,
                    "output": "(1, 16384, 256)",
                },
                {
                    "step": "DeformAttn2: Camera queries Radar",
                    "description": "MSDeformAttn(query=camera+PE, ref_pts, input=radar+PE)",
                    "n_heads": 8,
                    "n_points": 8,
                    "output": "(1, 16384, 256)",
                },
                {
                    "step": "Reshape to BEV",
                    "description": "Both fusion features back to spatial",
                    "fusion_f1": "(1, 256, 128, 128)",
                    "fusion_f2": "(1, 256, 128, 128)",
                },
                {
                    "step": "RadarConvFuser",
                    "description": "Channel + spatial fusion with 3 deconv blocks",
                    "input": "cat(f1, f2) → (1, 512, 128, 128)",
                    "output": "(1, 256, 128, 128)",
                },
                {
                    "step": "CenterHead Detection",
                    "description": "Predict heatmap + regression targets",
                    "input": "(1, 256, 128, 128)",
                    "outputs": {
                        "heatmap": "(1, 10, 128, 128) — 10 class scores",
                        "reg": "(1, 2, 128, 128) — xy offset",
                        "height": "(1, 1, 128, 128) — z",
                        "dim": "(1, 3, 128, 128) — lwh",
                        "rot": "(1, 2, 128, 128) — sin/cos yaw",
                        "vel": "(1, 2, 128, 128) — vx/vy",
                    },
                },
                {
                    "step": "NMS + Post-processing",
                    "description": "Score threshold → NMS → final boxes",
                    "score_threshold": 0.1,
                    "max_detections": 500,
                    "nms_type": "rotate",
                    "nms_threshold": 0.2,
                    "output": "boxes_3d (N, 9), scores_3d (N,), labels_3d (N,)",
                },
            ],
        }
        stages["7_fusion_detection"] = fusion_stage

        # Run actual forward pass
        try:
            with torch.no_grad():
                # Prepare batch
                from mmcv.parallel import collate, scatter
                data_batch = collate([sample], samples_per_gpu=1)
                data_batch = scatter(data_batch, [device])[0]

                t0 = time.time()
                results = model(return_loss=False, rescale=True, **data_batch)
                t1 = time.time()

                # Extract detection results
                if results and len(results) > 0:
                    result = results[0]
                    if "pts_bbox" in result:
                        pts_bbox = result["pts_bbox"]
                        det_stage = {
                            "stage_name": "8_detection_results",
                            "stage_title": "Final Detection Output",
                            "inference_time_ms": round((t1 - t0) * 1000, 1),
                            "num_detections": len(pts_bbox["scores_3d"]),
                            "boxes_3d": tensor_info(
                                pts_bbox["boxes_3d"].tensor, "boxes_3d"
                            ),
                            "scores_3d": tensor_info(
                                pts_bbox["scores_3d"], "scores_3d"
                            ),
                            "labels_3d": tensor_info(
                                pts_bbox["labels_3d"], "labels_3d"
                            ),
                        }
                        # Top-5 detections
                        scores = pts_bbox["scores_3d"].cpu().numpy()
                        labels = pts_bbox["labels_3d"].cpu().numpy()
                        boxes = pts_bbox["boxes_3d"].tensor.cpu().numpy()
                        class_names = [
                            "car", "truck", "trailer", "bus",
                            "construction_vehicle", "bicycle", "motorcycle",
                            "pedestrian", "traffic_cone", "barrier",
                        ]
                        top_k = min(5, len(scores))
                        top_idx = np.argsort(scores)[::-1][:top_k]
                        top_dets = []
                        for idx in top_idx:
                            det = {
                                "rank": int(np.where(top_idx == idx)[0][0]) + 1,
                                "class": class_names[labels[idx]] if labels[idx] < len(class_names) else f"class_{labels[idx]}",
                                "score": round(float(scores[idx]), 4),
                                "position_xyz": [round(float(v), 2) for v in boxes[idx, :3]],
                                "size_lwh": [round(float(v), 2) for v in boxes[idx, 3:6]],
                                "yaw_rad": round(float(boxes[idx, 6]), 4),
                                "velocity_xy": [round(float(v), 2) for v in boxes[idx, 7:9]] if boxes.shape[1] > 8 else [0, 0],
                            }
                            top_dets.append(det)
                        det_stage["top_detections"] = top_dets

                        # Class distribution
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        det_stage["detection_class_distribution"] = {
                            class_names[int(l)] if int(l) < len(class_names) else f"class_{l}": int(c)
                            for l, c in zip(unique_labels, counts)
                        }
                        stages["8_detection_results"] = det_stage

        except Exception as e:
            stages["forward_pass_error"] = str(e)

    except Exception as e:
        stages["model_build_error"] = str(e)
        import traceback
        stages["model_build_traceback"] = traceback.format_exc()

    return stages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract sample trace for visualization")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file path")
    parser.add_argument("--pkl", required=True, help="Info pkl file path")
    parser.add_argument("--data-root", required=True, help="TruckScenes data root")
    parser.add_argument("--out", default="sample_trace.json", help="Output JSON path")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index")
    parser.add_argument("--skip-model", action="store_true", help="Skip model forward pass")
    args = parser.parse_args()

    print(f"Loading info from {args.pkl}...")
    pkl_data = mmcv.load(args.pkl, file_format="pkl")
    data_infos = list(sorted(pkl_data["infos"], key=lambda e: e["timestamp"]))
    print(f"  Found {len(data_infos)} samples. Using index {args.sample_idx}.")

    trace = OrderedDict()
    trace["metadata"] = {
        "dataset": "MAN TruckScenes",
        "model": "RCBEVDet (BEVDepth4D_RC)",
        "sample_idx": args.sample_idx,
        "total_samples": len(data_infos),
        "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Stage 1: Raw data
    print("Stage 1: Extracting raw data info...")
    trace["1_raw_data"] = trace_raw_data(data_infos, args.data_root, args.sample_idx)

    # Stage 2: Radar loading
    print("Stage 2: Tracing radar PCD loading...")
    trace["2_radar_loading"] = trace_radar_loading(data_infos, args.data_root, args.sample_idx)

    # Stage 3: Camera loading
    print("Stage 3: Tracing camera image loading...")
    trace["3_camera_loading"] = trace_camera_loading(data_infos, args.data_root, args.sample_idx)

    # Stages 4-8: Model forward
    if not args.skip_model:
        print("Stages 4-8: Tracing model forward pass...")
        model_stages = trace_model_forward(
            args.config, args.checkpoint, data_infos, args.data_root, args.sample_idx
        )
        trace.update(model_stages)
    else:
        print("Skipping model forward pass (--skip-model)")
        # Still add architecture info from config
        from mmcv import Config
        cfg = Config.fromfile(args.config)
        trace["4_model_architecture"] = {
            "stage_name": "4_model_architecture",
            "stage_title": "RCBEVDet Model Architecture (from config only)",
            "model_type": cfg.model.type,
            "note": "Model forward pass skipped. Architecture from config only.",
        }

    # Save
    print(f"Saving trace to {args.out}...")
    with open(args.out, "w") as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"Done! Trace saved ({os.path.getsize(args.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
