_base_ = ['./rcbevdet-256x704-r50-BEV128-9kf-depth-cbgs12e-circlelarger.py']

# -------------------------
# Basic dataset settings
# -------------------------
dataset_type = 'TruckScenesDatasetRC'
data_root = '/content/drive/MyDrive/MAN-Truckscenes/'
ann_root = '/content/ts_data/'
file_client_args = dict(backend='disk')

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False,
)

# -------------------------
# TruckScenes cameras
# -------------------------
data_config = dict(
    cams=[
        'CAMERA_LEFT_FRONT',
        'CAMERA_RIGHT_FRONT',
    ],
    Ncams=2,
    input_size=(256, 704),
    src_size=(943, 1980),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0,
)

# -------------------------
# Radar / geometry
# -------------------------
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
radar_voxel_size = [0.2, 0.2, 8]
radar_use_dims = [0, 1, 2, 3, 4, 5, 6]

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

# -------------------------
# Annotation files
# -------------------------
train_ann_file = ann_root + 'truckscenes_infos_train.pkl'
val_ann_file = ann_root + 'truckscenes_infos_val.pkl'
test_ann_file = ann_root + 'truckscenes_infos_val.pkl'

# -------------------------
# Pipelines
# -------------------------
train_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=7,
        sweeps_num=8,
        use_dim=radar_use_dims,
        max_num=1200),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='GlobalRotScaleTrans_radar'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'radar'])
]


test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=7,
        sweeps_num=8,
        use_dim=radar_use_dims,
        max_num=1200),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img_inputs', 'radar'])
        ])
]

# -------------------------
# Data
# -------------------------
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,

    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=train_ann_file,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            img_info_prototype='bevdet4d',
            multi_adj_frame_id_cfg=(1, 9, 1),
        )
    ),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 9, 1),
    ),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=(1, 9, 1),
    ),
)

evaluation = dict(interval=1)
load_from = '/content/drive/MyDrive/RCBEVDet_weights/rcbevdet.pth'
work_dir = '/content/work_dirs/truckscenes_rcbevdet'
