"""
Convert MAN TruckScenes dataset to RCBEVDet-compatible .pkl info files.

Usage:
    python truckscenes_converter.py --root /path/to/man-truckscenes --version v1.1-mini --out /content/ts_data
"""
import os
import os.path as osp
import argparse
import numpy as np
import mmcv
from pyquaternion import Quaternion

# 2-camera setup: only cameras that actually exist for all samples in this dataset copy
CAMERA_NAMES = [
    'CAMERA_LEFT_FRONT',
    'CAMERA_RIGHT_FRONT',
]

RADAR_NAMES = [
    'RADAR_LEFT_FRONT', 'RADAR_RIGHT_FRONT',
    'RADAR_LEFT_BACK', 'RADAR_LEFT_SIDE',
    'RADAR_RIGHT_BACK', 'RADAR_RIGHT_SIDE',
]

LIDAR_NAMES = [
    'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT',
    'LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_REAR',
]

# Map TruckScenes category -> nuScenes 10 classes
TS_NAME_MAP = {
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


def obtain_sensor2top(nusc, sd_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat):
    """Get sensor-to-reference-lidar transform."""
    sd_rec = nusc.get('sample_data', sd_token)
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_token))

    # sensor -> ego
    s2e_r = Quaternion(cs_rec['rotation']).rotation_matrix
    s2e_t = np.array(cs_rec['translation'])

    # ego -> global
    e2g_r_s = Quaternion(pose_rec['rotation']).rotation_matrix
    e2g_t_s = np.array(pose_rec['translation'])

    # sensor -> lidar
    R = s2e_r.T @ e2g_r_s.T @ e2g_r_mat @ l2e_r_mat.T
    T = (
        s2e_t @ e2g_r_s.T @ e2g_r_mat @ l2e_r_mat.T
        + (e2g_t_s - e2g_t) @ e2g_r_mat @ l2e_r_mat.T
        + (-l2e_t)
    )

    sensor2lidar_rotation = R.T
    sensor2lidar_translation = T
    return data_path, sensor2lidar_rotation, sensor2lidar_translation


def _fill_infos(nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    train_infos, val_infos = [], []
    skipped_missing_lidar = 0
    skipped_missing_files = 0

    for sample in mmcv.track_iter_progress(nusc.sample):
        # pick first available lidar as reference
        lidar_token = None
        for ln in LIDAR_NAMES:
            if ln in sample['data']:
                lidar_token = sample['data'][ln]
                break

        if lidar_token is None:
            skipped_missing_lidar += 1
            continue

        sd_rec = nusc.get('sample_data', lidar_token)
        cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path = str(nusc.get_sample_data_path(lidar_token))

        l2e_r = cs_rec['rotation']
        l2e_t = cs_rec['translation']
        e2g_r = pose_rec['rotation']
        e2g_t = pose_rec['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': {},
            'radars': {},
            'lidar2ego_translation': l2e_t,
            'lidar2ego_rotation': l2e_r,
            'ego2global_translation': e2g_t,
            'ego2global_rotation': e2g_r,
            'timestamp': sample['timestamp'],
            'location': 'unknown',
            'scene_token': sample['scene_token'],
        }

        # cameras
        for cam_name in CAMERA_NAMES:
            if cam_name not in sample['data']:
                continue

            cam_token = sample['data'][cam_name]
            cam_path, s2l_r, s2l_t = obtain_sensor2top(
                nusc, cam_token, np.array(l2e_t), l2e_r_mat, np.array(e2g_t), e2g_r_mat
            )

            cam_sd = nusc.get('sample_data', cam_token)
            cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
            cam_pose = nusc.get('ego_pose', cam_sd['ego_pose_token'])

            info['cams'][cam_name] = dict(
                data_path=cam_path,
                type=cam_name,
                sample_data_token=cam_token,
                timestamp=cam_sd['timestamp'],
                sensor2ego_translation=cam_cs['translation'],
                sensor2ego_rotation=cam_cs['rotation'],
                ego2global_translation=cam_pose['translation'],
                ego2global_rotation=cam_pose['rotation'],
                sensor2lidar_rotation=s2l_r,
                sensor2lidar_translation=s2l_t,
                cam_intrinsic=np.array(cam_cs['camera_intrinsic']),
            )

        # require both front cameras and existing files
        required_cams_present = True
        for cam_name in CAMERA_NAMES:
            if cam_name not in info['cams']:
                required_cams_present = False
                break
            if not osp.exists(info['cams'][cam_name]['data_path']):
                required_cams_present = False
                break

        if not required_cams_present:
            skipped_missing_files += 1
            continue

        # radars: IMPORTANT -> each radar entry must be a LIST of sweep dicts
        for radar_name in RADAR_NAMES:
            if radar_name not in sample['data']:
                continue

            radar_token = sample['data'][radar_name]
            radar_path, s2l_r, s2l_t = obtain_sensor2top(
                nusc, radar_token, np.array(l2e_t), l2e_r_mat, np.array(e2g_t), e2g_r_mat
            )

            radar_sd = nusc.get('sample_data', radar_token)
            radar_cs = nusc.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])
            radar_pose = nusc.get('ego_pose', radar_sd['ego_pose_token'])

            current_radar = {
                'data_path': radar_path,
                'type': radar_name,
                'sample_data_token': radar_token,
                'timestamp': radar_sd['timestamp'],
                'sensor2ego_translation': radar_cs['translation'],
                'sensor2ego_rotation': radar_cs['rotation'],
                'ego2global_translation': radar_pose['translation'],
                'ego2global_rotation': radar_pose['rotation'],
                'sensor2lidar_rotation': s2l_r,
                'sensor2lidar_translation': s2l_t,
            }

            radar_sweeps = []
            sweep_sd = radar_sd
            for _ in range(max_sweeps):
                if sweep_sd['prev'] == '':
                    break

                sweep_sd = nusc.get('sample_data', sweep_sd['prev'])
                sweep_cs = nusc.get('calibrated_sensor', sweep_sd['calibrated_sensor_token'])
                sweep_pose = nusc.get('ego_pose', sweep_sd['ego_pose_token'])
                sweep_path = str(nusc.get_sample_data_path(sweep_sd['token']))

                if not osp.exists(sweep_path):
                    continue

                sweep_path_r, sweep_s2l_r, sweep_s2l_t = obtain_sensor2top(
                    nusc, sweep_sd['token'], np.array(l2e_t), l2e_r_mat, np.array(e2g_t), e2g_r_mat
                )

                radar_sweeps.append({
                    'data_path': sweep_path_r,
                    'type': radar_name,
                    'sample_data_token': sweep_sd['token'],
                    'timestamp': sweep_sd['timestamp'],
                    'sensor2ego_translation': sweep_cs['translation'],
                    'sensor2ego_rotation': sweep_cs['rotation'],
                    'ego2global_translation': sweep_pose['translation'],
                    'ego2global_rotation': sweep_pose['rotation'],
                    'sensor2lidar_rotation': sweep_s2l_r,
                    'sensor2lidar_translation': sweep_s2l_t,
                })

            info['radars'][radar_name] = [current_radar] + radar_sweeps

        # lidar sweeps
        sweep_sd = sd_rec
        for _ in range(max_sweeps):
            if sweep_sd['prev'] == '':
                break

            sweep_sd = nusc.get('sample_data', sweep_sd['prev'])
            sweep_cs = nusc.get('calibrated_sensor', sweep_sd['calibrated_sensor_token'])
            sweep_pose = nusc.get('ego_pose', sweep_sd['ego_pose_token'])

            info['sweeps'].append({
                'data_path': str(nusc.get_sample_data_path(sweep_sd['token'])),
                'type': sd_rec['channel'],
                'sample_data_token': sweep_sd['token'],
                'sensor2ego_translation': sweep_cs['translation'],
                'sensor2ego_rotation': sweep_cs['rotation'],
                'ego2global_translation': sweep_pose['translation'],
                'ego2global_rotation': sweep_pose['rotation'],
                'timestamp': sweep_sd['timestamp'],
                'sensor2lidar_rotation': l2e_r_mat.T,
                'sensor2lidar_translation': np.zeros(3),
            })

        # annotations
        if not test:
            annotations = [nusc.get('sample_annotation', tok) for tok in sample['anns']]
            locs = np.array([a['translation'] for a in annotations]).reshape(-1, 3)
            dims = np.array([a['size'] for a in annotations]).reshape(-1, 3)
            rots = np.array([
                Quaternion(a['rotation']).yaw_pitch_roll[0] for a in annotations
            ]).reshape(-1, 1)
            velocity = np.array([
                nusc.box_velocity(a['token'])[:2] for a in annotations
            ]).reshape(-1, 2)
            names = np.array([TS_NAME_MAP.get(a['category_name'], 'car') for a in annotations])
            num_lidar_pts = np.array([a['num_lidar_pts'] for a in annotations])
            num_radar_pts = np.array([a['num_radar_pts'] for a in annotations])
            valid_flag = np.array(
                [(num_lidar_pts[i] + num_radar_pts[i]) > 0 for i in range(len(annotations))],
                dtype=bool
            )

            gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            for i, anno in enumerate(annotations):
                center = np.array(anno['translation'])

                # global -> ego
                center -= np.array(e2g_t)
                center = center @ Quaternion(e2g_r).inverse.rotation_matrix.T

                # ego -> lidar
                center -= np.array(l2e_t)
                center = center @ Quaternion(l2e_r).inverse.rotation_matrix.T
                gt_boxes[i, :3] = center

                q_global = Quaternion(anno['rotation'])
                q_l2e = Quaternion(l2e_r)
                q_e2g = Quaternion(e2g_r)
                q_lidar = q_l2e.inverse * q_e2g.inverse * q_global
                gt_boxes[i, 6] = q_lidar.yaw_pitch_roll[0]

            # [w, l, h] -> [l, w, h]
            gt_boxes[:, 3:6] = gt_boxes[:, [4, 3, 5]]

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity
            info['num_lidar_pts'] = num_lidar_pts
            info['num_radar_pts'] = num_radar_pts
            info['valid_flag'] = valid_flag

        if sample['scene_token'] in train_scenes:
            train_infos.append(info)
        else:
            val_infos.append(info)

    print(f"Skipped samples with no lidar reference: {skipped_missing_lidar}")
    print(f"Skipped samples with missing camera files: {skipped_missing_files}")
    return train_infos, val_infos


def create_infos(root_path, out_dir, version='v1.1-mini', max_sweeps=10):
    from truckscenes import TruckScenes
    from truckscenes.utils import splits

    print(f'Loading TruckScenes {version} from {root_path} ...')
    nusc = TruckScenes(version=version, dataroot=root_path, verbose=True)

    if 'mini' in version:
        train_scene_names = splits.mini_train
        val_scene_names = splits.mini_val
    else:
        train_scene_names = splits.train
        val_scene_names = splits.val

    scene_name2token = {s['name']: s['token'] for s in nusc.scene}
    train_scenes = {scene_name2token[n] for n in train_scene_names if n in scene_name2token}
    val_scenes = {scene_name2token[n] for n in val_scene_names if n in scene_name2token}

    print(f'Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}')

    train_infos, val_infos = _fill_infos(
        nusc, train_scenes, val_scenes, test=False, max_sweeps=max_sweeps
    )

    metadata = dict(version=version)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Train samples: {len(train_infos)}, Val samples: {len(val_infos)}')
    mmcv.dump(dict(infos=train_infos, metadata=metadata),
              osp.join(out_dir, 'truckscenes_infos_train.pkl'))
    mmcv.dump(dict(infos=val_infos, metadata=metadata),
              osp.join(out_dir, 'truckscenes_infos_val.pkl'))
    print('Done!')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--version', default='v1.1-mini')
    p.add_argument('--out', default='/content/ts_data')
    p.add_argument('--max-sweeps', type=int, default=10)
    args = p.parse_args()
    create_infos(args.root, args.out, args.version, args.max_sweeps)
