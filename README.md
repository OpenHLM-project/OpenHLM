<p align="center">
  <img width="953" height="517" alt="image" src="https://raw.githubusercontent.com/Tendourisu/images/master/20260601202158247.png" />
</p>

# OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-placehoder-b31b1b)](https://arxiv.org/abs/placehoder)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://openhlm-corl.github.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>



Whole-body humanoid loco-manipulation requires coordinating the robot's entire kinematic chain. However, most existing systems typically decouple the upper and lower bodies into separate controllers, limiting such coordination and yielding behaviors similar to those of a wheeled dual-arm platform. 

In this project, we ask what it takes to build a whole-body native vision-language-action (VLA) model that maps language and pixels directly to all of the humanoid's degrees of freedom. We conduct a systematic empirical study organized as a roadmap of one-variable-at-a-time experiments across three phases: whole-body teleoperation, VLA model design, and heterogeneous co-training. 

Following this roadmap yields **OpenHLM**, an open-source recipe for whole-body humanoid loco-manipulation. In a challenging long-horizon task that spans a wide vertical range of the humanoid, \method outperforms two state-of-the-art humanoid VLA baselines (GR00T N1.6 and $\Psi_0$) using less than half the total demonstration time. 

This repository contains the full stack for **OpenHLM**, including hardware setup, data collection and processing, policy training, and deployment. Each component is organized into a separate folder.

---

## Table of Contents

- [📖 Overview](#overview)
- [🛠️ Hardware Setup](#hardware-setup)
  - [ Robot Data Collection Hardware](#robot-data-collection-hardware)
  - [ HuMI Data Collection Hardware](#humi-data-collection-hardware)
- [🤖 Robot Data Collection Guide](#robot-data-collection-guide)
  - [Environment Setup(PC side)](#environment-setup-PC-side)
  - [Environment Setup(robot side)](#environment-setup-robot-side)
  - [Robot Data Collection](#robot-data-collection)
- [🧑‍🤝‍🧑 HuMI Data Collection Guide](#humi-data-collection-guide)

- [🚀 Model Training](#model-training)
  - [Evironment Setup](#environment-setup)
  - [Converting Data to Lerobot Format](#converting-data-to-lerobot-format)
  - [Compute Normalization Statistics](#compute-normalization-statistics)
  - [Run Training](#run-training)
- [📦Deployment](#deployment)
  - [Policy Server](#policy-server)
  - [Robot Inference](#robot-inference)
- [Citation](#citation)
---

<a id="overview"></a>

## 📖 Overview



OpenHLM consists of **three main components**:

```
.
├── GR00T-WholeBodyControl-4-OpenHLM
├── openpi4OpenHLM
└── HuMI4OpenHLM

```


---

<a id="hardware-setup"></a>

## 🛠️ Hardware Setup

<a id="robot-data-collection-hardware"></a>

### Robot Data Collection Hardware
**required:**
- Unitree G1 humanoid robot
- Ubuntu 24.04/25.04 (25.04 is tested our version )
- NVIDIA GPU (RTX 5080 is tested for data collection, but any GPU with >8 GB VRAM should work)
- two ChangingTek CTAG2F90-D grippers equipped on its wrists 
- two Intel RealSense D405 cameras mounted on the wrists
- one Unitree SV1-25 fisheye stereo camera mounted on the robot’s head
- one PICO4U VR kit consisting of a head-mounted display (HMD), two handheld controllers, and two leg-mounted motion tracker
- 3D-printed mounts for the head-cameras and wrist-cameras (designs available in `this website`)

---

<a id="humi-data-collection-hardware"></a>

### HuMI Data Collection Hardware
**required:**
**TODO**


<a id="robot-data-collection-guide"></a>

## 🤖 Robot Data Collection Guide

<a id="environment-setup-PC-side"></a>

### Environment Setup(PC side)
You can refer to the [GR00T-WholeBodyControl](https://nvlabs.github.io/GR00T-WholeBodyControl/index.html) documentation for detailed instructions on setting up the GR00T control framework, which is used for both simulation and real robot data collection.

**Completed the [Installation Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)** — TensorRT is installed, the repo is cloned, and the C++ deployment is built.\
**Downloaded the model checkpoints** — run `python download_from_hf.py` from the repo root. See [Downloading Model Checkpoints](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html) for details.\
**Complete the [VR Setup Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)** — Set up the PICO VR system and ensure it is properly calibrated and connected to the data collection computer. Refer to the PICO SDK documentation for detailed instructions.

TODO: thirdparty LFS

<a id="environment-setup-robot-side"></a>

### Environment Setup(robot side)
refer to [hardware4OpenHLM](https://github.com/Tendourisu/hardware-4-cotraining)

<a id="robot-data-collection"></a>
### Robot Data Collection

Run the data collection script on the robot's onboard computer to capture synchronized full-body motion and camera views.

```bash
cd src/GR00T-WholeBodyControl/scripts
bash main.sh
```
<a id="humi-data-collection-guide"></a>

## 🧑‍🤝‍🧑 HuMI Data Collection Guide

**TODO**

---

<a id="model-training"></a>

## 🚀 Model Training

<a id="environment-setup"></a>

### Environment Setup

```bash
cd src/openpi4OpenHLM
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

### Download Checkpoint
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --config_name pi05_aloha \
    --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

<a id="converting-data-to-lerobot-format"></a>

### Converting Data to Lerobot Format

The training code expects data in the Lerobot format. Use the provided conversion script to convert your collected data:

```bash
### Convert a single dataset
uv run examples/unitree_g1/convert_g1_data_to_lerobot.py \
    --data_dir ~/codebase/OpenHLM/src/openpi-humanoid/sonic_demonstration/20260521_1102_test \
    --repo_name test_video
```
```bash
### Convert multiple datasets
uv run examples/unitree_g1/convert_g1_data_to_lerobot_multi.py \
    --parent_dir ~/codebase/OpenHLM/src/openpi-humanoid/sonic_demonstration \
    --dataset_folders 20260521_1102_test 20260521_1102_test_2 \
    --repo_name test_video2
```

<a id="compute-normalization-statistics"></a>

### Compute Normalization Statistics

Before training, compute normalization statistics for your dataset:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
    --config-name openhlm_example \
    --max-frames 500000
```

<a id="run-training"></a>

### Run Training

Train the model using the computed normalization statistics:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py \
    openhlm_example \
    --save_interval 5000 \
    --batch_size 32 \
    --no-enable-gradient-checkpointing \
    --enable-training-compile \
    --exp_name g1_test_video \
    --pytorch-weight-path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

---

<a id="deployment"></a>

## 📦 Deployment

### Policy Server

Start a policy server for remote inference:

```bash
# Use a trained checkpoint
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
--env SONICG1 \
--num-steps 10 \
policy:checkpoint \
--policy.config=openhlm_example \
--policy.dir=~/codebase/OpenHLM/src/openpi-humanoid/checkpoints/openhlm_example/g1_test_video_freeze_paligemma/30000
```

<a id="robot-inference"></a>

### Robot Inference

The deployment client connects to the OpenPI policy server via websocket for action inference and controls the G1 robot via the GR00T WBC framework.

**On the robot/client side:**

```bash
### terminal 1: start the low-level control server
cd src/GR00T-WholeBodyControl4OpenHLM/scripts
bash deploy_stream.sh
```

```bash
### terminal 2: start the OpenPI deployment client
python openpi-eval/main.py \
--control_hz 30 \
--max_steps 10000 \
--save_video \
--instruction "test" \
--exp_name g1_test_video
```

**🎛️ Keyboard Controls:**
TODO

| Key | Action |
|:---:|:-------|
| `]` | ▶️ Activate low level policy |


**📋 Workflow:**

![image.png](https://raw.githubusercontent.com/Tendourisu/images/master/20260302193639710.png)


---



<a id="citation"></a>

## 📝 Citation

If you find OpenHLM useful in your research, please consider citing:

```bibtex
TODO
```

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

</div>

---

## 📜 License

This project is licensed under the **Apache 2.0 License**.

The OpenPI models and code are provided by [Physical Intelligence](https://www.physicalintelligence.company/) under the Apache 2.0 License.

---

## 🙏 Acknowledgments

We sincerely thank the following projects and teams:

<table align="center">
<tr>
<td align="center" width="25%">
<a href="https://www.physicalintelligence.company/">
<img src="https://img.shields.io/badge/OpenPI-Physical_Intelligence-orange" alt="OpenPI"/>
</a><br>
Vision-language-action models
</td>
<td align="center" width="25%">
<a href="https://github.com/NVlabs/GR00T-WholeBodyControl">
<img src="https://img.shields.io/badge/GR00T-NVIDIA-green" alt="GR00T"/>
</a><br>
Humanoid control framework
</td>
<td align="center" width="25%">
<a href="https://github.com/XR-Robotics">
<img src="https://img.shields.io/badge/XR_Robotics-PICO_VR-blue" alt="XR Robotics"/>
</a><br>
PICO VR integration
</td>
</tr>
</table>
