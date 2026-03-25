from copy import deepcopy
import numpy as np
from .nuscenes_dataset_rc import NuScenesDatasetRC
from .builder import DATASETS

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


@DATASETS.register_module()
class TruckScenesDatasetRC(NuScenesDatasetRC):

    NameMapping = TRUCKSCENES_NAME_MAPPING

    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }

    CLASSES = (
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    )

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('eval_version', 'detection_cvpr_2019')
        kwargs.pop('dataset', None)
        super().__init__(*args, **kwargs)

    def get_data_info(self, index):
        original_info = self.data_infos[index]
        info = deepcopy(original_info)

        cams = info.get('cams', {})

        # nuScenes-style aliases expected by RCBEVDet
        if 'CAM_FRONT' not in cams:
            if 'CAMERA_LEFT_FRONT' in cams:
                cams['CAM_FRONT'] = deepcopy(cams['CAMERA_LEFT_FRONT'])
            elif 'CAMERA_RIGHT_FRONT' in cams:
                cams['CAM_FRONT'] = deepcopy(cams['CAMERA_RIGHT_FRONT'])

        if 'CAM_FRONT_LEFT' not in cams and 'CAMERA_LEFT_FRONT' in cams:
            cams['CAM_FRONT_LEFT'] = deepcopy(cams['CAMERA_LEFT_FRONT'])

        if 'CAM_FRONT_RIGHT' not in cams and 'CAMERA_RIGHT_FRONT' in cams:
            cams['CAM_FRONT_RIGHT'] = deepcopy(cams['CAMERA_RIGHT_FRONT'])

        if 'CAM_BACK_LEFT' not in cams and 'CAMERA_LEFT_BACK' in cams:
            cams['CAM_BACK_LEFT'] = deepcopy(cams['CAMERA_LEFT_BACK'])

        if 'CAM_BACK_RIGHT' not in cams and 'CAMERA_RIGHT_BACK' in cams:
            cams['CAM_BACK_RIGHT'] = deepcopy(cams['CAMERA_RIGHT_BACK'])

        if 'CAM_BACK' not in cams:
            if 'CAMERA_LEFT_BACK' in cams:
                cams['CAM_BACK'] = deepcopy(cams['CAMERA_LEFT_BACK'])
            elif 'CAMERA_RIGHT_BACK' in cams:
                cams['CAM_BACK'] = deepcopy(cams['CAMERA_RIGHT_BACK'])

        info['cams'] = cams
        self.data_infos[index] = info

        try:
            input_dict = super().get_data_info(index)
        finally:
            self.data_infos[index] = original_info

        if not self.test_mode and 'ann_infos' not in input_dict:
            ann = self.get_ann_info(index)

            # Convert LiDARInstance3DBoxes -> plain ndarray
            if hasattr(ann['gt_bboxes_3d'], 'tensor'):
                gt_boxes = ann['gt_bboxes_3d'].tensor.numpy()
            else:
                gt_boxes = np.asarray(ann['gt_bboxes_3d'])

            gt_labels = np.asarray(ann['gt_labels_3d'], dtype=np.int64)

            input_dict['ann_infos'] = (gt_boxes, gt_labels)

        return input_dict
