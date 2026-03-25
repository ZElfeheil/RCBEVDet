# RCBEVDet — MAN TruckScenes Adaptation

This repository adapts [**RCBEVDet**](https://arxiv.org/abs/2403.16440) (CVPR 2024) to run on the [**MAN TruckScenes**](https://brandportal.man/d/QSf8mPdU5Hgj/downloads#/-/dataset) dataset. The original paper is: *RCBEVDet: Radar-camera Fusion in Bird's Eye View for 3D Object Detection* and its extended version RCBEVDet++.

## Introduction

We present RCBEVDet, a radar-camera fusion 3D object detection method in the bird's eye view (BEV). Specifically, we first design RadarBEVNet for radar BEV feature extraction. RadarBEVNet consists of a dual-stream radar backbone and a Radar Cross-Section (RCS) aware BEV encoder. In the dual-stream radar backbone, a point-based encoder and a transformer-based encoder are proposed to extract radar features, with an injection and extraction module to facilitate communication between the two encoders. The RCS-aware BEV encoder takes RCS as the object size prior to scattering the point feature in BEV. Besides, we present the Cross-Attention Multi-layer Fusion module to automatically align the multi-modal BEV feature from radar and camera with the deformable attention mechanism, and then fuse the feature with channel and spatial fusion layers.

![RCBEVDet](RCBEVDet.JPG)

## Weight & Code

* Model weights for RCBEVDet are released: [google drive](https://drive.google.com/drive/folders/1VhOBcJ7wT71R8Dqyr5MlQUKv7lVcjfrz?usp=sharing)

## Results

##### 3D Object Detection (nuScenes Validation)

| Method     | Input | Backbone  | NDS  | mAP  |
| :--------- | ----- | --------- | ---- | :--- |
| BEVDepth4D | C     | ResNet-50 | 51.9 | 40.5 |
| RCBEVDet   | C+R   | ResNet-50 | 56.8 | 45.3 |
| SparseBEV  | C     | ResNet-50 | 54.5 | 43.2 |
| RCBEVDet++ | C+R   | ResNet-50 | 60.4 | 51.9 |

##### 3D Object Detection (nuScenes Test)

| Method     | Input | Backbone | Future frame | NDS  | mAP  |
| :--------- | ----- | -------- | ------------ | ---- | :--- |
| BEVDepth4D | C     | V2-99    | No           | 60.5 | 51.5 |
| RCBEVDet   | C+R   | V2-99    | No           | 63.9 | 55.0 |
| SparseBEV  | C     | V2-99    | No           | 63.6 | 55.6 |
| RCBEVDet++ | C+R   | V2-99    | No           | 68.7 | 62.6 |
| SparseBEV  | C     | ViT-L    | Yes          | 70.2 | ——   |
| RCBEVDet++ | C+R   | ViT-L    | Yes          | 72.7 | 67.3 |

---

## 🚛 TruckScenes Adaptation

This fork adapts RCBEVDet for the [MAN TruckScenes](https://brandportal.man/d/QSf8mPdU5Hgj/downloads#/-/dataset) dataset.

**📖 [Google Colab Setup Guide](docs/COLAB_SETUP_GUIDE.md)** — Full step-by-step instructions for running inference on TruckScenes using Google Colab.

Key changes:
- **`mmdet3d/datasets/truckscenes_dataset_rc.py`** — Dataset class mapping TruckScenes categories & cameras to nuScenes format
- **`tools/data_converter/truckscenes_converter.py`** — Converts TruckScenes metadata to `.pkl` info files
- **`configs/rcbevdet/truckscenes_rcbevdet_config.py`** — TruckScenes-specific config (2 cameras, 6 radars)

---

## Getting Started

### environment

The code is tested in the following two environment:

```
python                       3.8.13
cuda                         12.1
pytorch                      2.0.1+cu118
torchvision                  0.15.2+cu118
numpy                        1.23.4
mmcv-full                    1.6.0
mmcls                        0.25.0
mmdet                        2.28.2
nuscenes-devkit              1.1.11
av2                          0.2.1
detectron2                   0.6
(for A800 or A40 + cuda 12.1)
```

```
python                       3.8.13
cuda                         11.6
pytorch                      1.12.1+cu116
torchvision                  0.13.0+cu116
numpy                        1.19.5
mmcv-full                    1.6.2
mmcls                        1.0.0rc1
mmdet                        2.24.0
nuscenes-devkit              1.1.9
detectron2                   0.6
(for other GPUs + cuda 11.6)
```

If you encounter slow download speed or timeout when downloading dependency packages, 
you need to consider installing the dependency packages from the mirror website first, 
and then execute the installation:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install {Find the dependencies in setup.py:setup(install_requires=[...]) and write them down here} -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py develop
```

The most recommended installation steps are:

1. Create a Python environment. Install [PyTorch](https://pytorch.org/get-started/previous-versions/)
corresponding to your machine's CUDA version;

2. Install [mmcv](https://github.com/open-mmlab/mmcv) corresponding to your PyTorch and CUDA version;

3. Install other dependencies of mmdet and install [mmdet](https://github.com/open-mmlab/mmdetection);

4. Install other dependencies of this project (Please change the spconv version
in the requirements.txt to the CUDA version you are using) and setup this project;

5. Compile some operators manually.
```bash
cd mmdet3d/ops/csrc
python setup.py build_ext --inplace
cd ../deformattn
python setup.py build install
```

6. Install other dependencies of detectron2 and install [detectron2](https://github.com/facebookresearch/detectron2);


### data preparation

The MAN TruckScenes dataset can be downloaded from the [MAN Brand Portal](https://brandportal.man/d/QSf8mPdU5Hgj/downloads#/-/dataset).

TruckScenes follows the same folder structure as nuScenes. Your data directory should look like this:

```
├── mmdet3d
├── tools
├── configs
├── data
│   ├── MAN-Truckscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-mini          # or v1.0-trainval / v1.1-mini
```

For **nuScenes** (original), prepare data by running:
```bash
python tools/create_data_nuscenes_RC.py
```

For **TruckScenes**, convert the dataset metadata to `.pkl` info files:
```bash
python tools/data_converter/truckscenes_converter.py \
  --root /path/to/MAN-Truckscenes \
  --version v1.1-mini \
  --out /path/to/output
```


### training

```bash
./tools/dist_train.sh $config_path $gpus
```

### testing

testing on validation set

```bash
./tools/dist_test.sh $config_path $checkpoint_path $gpus --eval bbox
```

testing on test set

```bash
./tools/dist_test.sh $config_path $checkpoint_path $gpus --format-only --eval-options 'jsonfile_prefix=work_dirs'
mv work_dirs/pts_bbox/results_nusc.json work_dirs/pts_bbox/{$name}.json
```

benchmarking test latency

```bash
python tools/analysis_tools/benchmark_sequential.py $config 1 --fuse-conv-bn
```

TTA or Ensemble

```bash
python tools/merge_result_json.py --paths a.json,b.json,c.json
mv work_dirs/ens/results.json work_dirs/ens/{$name}.json
```

If you have any other questions, please refer to 
<a href='https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/'>mmdet3d docs</a>.

## Acknowledgements

The overall code is based on [mmdetection3D](https://github.com/open-mmlab/mmdetection3d), [BEVDet](https://github.com/HuangJunJie2017/BEVDet) and [SparseBEV](https://github.com/MCG-NJU/SparseBEV/tree/main). We sincerely thank the authors for their great work.

## License

The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.
