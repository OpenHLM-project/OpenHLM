<p align="center">
  <img  alt="image" src="assets/teaser.png" />
</p>

# OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2606.22174-b31b1b)](https://arxiv.org/abs/2606.22174)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://openhlm-corl.github.io/)
[![Dataset](https://img.shields.io/badge/%20Dataset-OpenHLM--data-yellow)](https://huggingface.co/datasets/OpenHLM/OpenHLM-data)
[![Checkpoints](https://img.shields.io/badge/%20Checkpoints-OpenHLM--ckpts-yellow)](https://huggingface.co/OpenHLM/OpenHLM-ckpts)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>



Whole-body humanoid loco-manipulation requires coordinating the robot's entire kinematic chain. However, most existing systems typically decouple the upper and lower bodies into separate controllers, limiting such coordination and yielding behaviors similar to those of a wheeled dual-arm platform. 

In this project, we ask what it takes to build a whole-body native vision-language-action (VLA) model that maps language and pixels directly to all of the humanoid's degrees of freedom. We conduct a systematic empirical study organized as a roadmap of one-variable-at-a-time experiments across three phases: whole-body teleoperation, VLA model design, and heterogeneous co-training. 

Following this roadmap yields **OpenHLM**, an open-source recipe for whole-body humanoid loco-manipulation. In a challenging long-horizon task that spans a wide vertical range of the humanoid, **OpenHLM** outperforms two state-of-the-art humanoid VLA baselines (GR00T N1.6 and $\Psi_0$) using less than half the total demonstration time. 

This repository contains the full stack for **OpenHLM**, including hardware setup, data collection and processing, policy training, and deployment. Each component is organized into a separate folder.

---

## Table of Contents

- [📖 Overview](#overview)
- [🛠️ Hardware Setup](#hardware-setup)
  - [Robot Data Collection Hardware](#robot-data-collection-hardware)
  - [HuMI Data Collection Hardware](#humi-data-collection-hardware)
- [🤖 Robot Data Collection Guide](#robot-data-collection-guide)
  - [Environment Setup( PC side )](#environment-setup-PC-side)
  - [Environment Setup( robot side )](#environment-setup-robot-side)
  - [Robot Data Collection](#robot-data-collection)
- [🧑‍🤝‍🧑 HuMI Data Collection Guide](#humi-data-collection-guide)
- [🤗 Checkpoints and Datasets](#checkpoints-and-datasets)
  - [Checkpoints](#checkpoints)
  - [Datasets](#datasets)
  - [Example Raw Demonstration](#example-raw-demonstration)
- [🚀 Model Training](#model-training)
  - [Environment Setup](#environment-setup)
  - [Configure Training Config](#configure-training-config)
  - [Converting Data to Lerobot Format](#converting-data-to-lerobot-format)
  - [Compute Normalization Statistics](#compute-normalization-statistics)
  - [Run Training](#run-training)
- [📦Deployment](#deployment)
  - [Policy Server](#policy-server)
  - [Robot Inference](#robot-inference)
---

<a id="overview"></a>

## 📖 Overview

OpenHLM consists of **three main components**:

```
.
├── GR00T-WholeBodyControl-4-OpenHLM
├── HuMI4OpenHLM
└── openpi4OpenHLM

```


**GR00T-WholeBodyControl-4-OpenHLM** provides the robot-side whole-body control stack for Unitree G1, including VR teleoperation, synchronized robot data collection, simulation utilities, and deployment interfaces for executing full-body loco-manipulation policies.

**HuMI4OpenHLM** provides the HuMI data collection pipeline used to build heterogeneous co-training data, complementing robot demonstrations with human motion and interaction data for broader whole-body behavior coverage.

**openpi4OpenHLM** contains the VLA training and inference code, including data conversion to LeRobot format, normalization statistics, policy training, policy serving, and robot-side inference clients for OpenHLM deployment.

---

<a id="hardware-setup"></a>

## 🛠️ Hardware Setup

<a id="robot-data-collection-hardware"></a>

### Robot Data Collection Hardware
**Required:**
- Unitree G1 humanoid robot. Robot IP: `192.168.123.164`
- Ubuntu (25.04 is our tested version). PC IP: `192.168.123.222`
- NVIDIA GPU (RTX 5080 is tested for data collection, but any GPU with >8 GB VRAM should work)
- Two ChangingTek CTAG2F90-D grippers equipped on its wrists 
- Two Intel RealSense D405 cameras mounted on the wrists
- One Unitree SV1-25 fisheye stereo camera mounted on the robot’s head
- One PICO4U VR kit consisting of a head-mounted display (HMD), two handheld controllers, and two leg-mounted motion tracker
- 3D-printed mounts for the head-cameras and wrist-cameras (available in `assets/`)

---

<a id="humi-data-collection-hardware"></a>

### HuMI Data Collection Hardware
**Required:**
- Five HTC Vive trackers for two hand-held grippers, pelvis, left foot, and right foot tracking
- Windows machine for tracker pose streaming through SteamVR
- Linux or macOS machine for online/offline IK and data processing
- Two GoPro cameras for wrist-view video recording
- 3D-printed clamps or grippers for mounting the GoPro cameras on the wrists

See the HuMI [hardware guide](https://github.com/Richard-coder-Nai/HuMI/blob/main/hardware_guide.md)
for detailed hardware preparation and mounting instructions.


<a id="robot-data-collection-guide"></a>

## 🤖 Robot Data Collection Guide

<a id="environment-setup-PC-side"></a>

### Environment Setup( PC side )
You can refer to the [GR00T-WholeBodyControl](https://nvlabs.github.io/GR00T-WholeBodyControl/index.html) documentation for detailed instructions on setting up the GR00T control framework.

Cd to the `src/GR00T-WholeBodyControl4OpenHLM` directory and follow these steps:\
**Completed the [Installation Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)** — TensorRT is installed, the repo is cloned, and the C++ deployment is built.\
**Downloaded the model checkpoints** — Run `python download_from_hf.py` from the repo root. See [Downloading Model Checkpoints](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html) for details.\
**Completed the [Quick Start](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html)** — You can run the sim2sim loop.\
**Complete the [VR Setup Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)** — Set up the PICO VR system and ensure it is properly calibrated and connected to the data collection computer. Refer to the PICO SDK documentation for detailed instructions.

Then, install the additional dependencies for the OpenHLM data collection scripts:

```bash
## Assuming you are now back at the repository root

cd src/GR00T-WholeBodyControl4OpenHLM/
uv pip install --python .venv_teleop/bin/python -e ./gear_sonic/thirdparty/GMR
uv pip install --python .venv_teleop/bin/python -e ./openpi-client

cd ./gear_sonic/thirdparty/XRoboToolkit-Orin-Video-Sender
sudo apt update
sudo apt install \
  build-essential \
  pkg-config \
  libopencv-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libglib2.0-dev \
  libzmq3-dev \
  libssl-dev
make

```
This will install the GMR for retargeting openpi-client for inference and build the video sender for streaming the views to the VR headset.

<a id="environment-setup-robot-side"></a>

### Environment Setup( robot side )
git clone the hardware setup repository on the robot's onboard computer and follow the instructions there to set up the hardwares
```
git clone https://github.com/Tendourisu/hardware4OpenHLM.git
```
refer to [hardware4OpenHLM](https://github.com/Tendourisu/hardware4OpenHLM)

<a id="robot-data-collection"></a>

### Robot Data Collection

**Attention:** Teleoperation can be dangerous. Always ensure the robot is in a safe environment and start with low gains to prevent any potential damage or injury. Before running the data collection script, you are suggested to first teleop in simulation and then teleop in the real world following the instructions in [vr_wholebody_teleop](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html) to get familiar with the controls.

```bash
### g1 side, ssh into the g1's onboard computer and run the hardware setup script to initialize the cameras, grippers, and VR system
cd path/to/hardware4OpenHLM
uv run hardware_setup
```

```bash
### PC side
cd src/GR00T-WholeBodyControl4OpenHLM/scripts
bash launch_data_collection.sh
```
See [src/GR00T-WholeBodyControl4OpenHLM/robot_data_collection.md](src/GR00T-WholeBodyControl4OpenHLM/robot_data_collection.md) for detailed instructions.
<a id="humi-data-collection-guide"></a>

## 🧑‍🤝‍🧑 HuMI Data Collection Guide

HuMI collects human demonstrations with Vive body trackers and wrist-mounted
GoPros, then converts them into OpenHLM-style episode folders for heterogeneous
co-training with robot teleoperation data.

The high-level workflow is:

1. Run online IK while streaming tracker poses from the Windows machine.
   The web UI is used to start and stop each human demonstration episode.
```bash
uv run online-ik pick --rpc-address tcp://<windows_machine_ip>:4242
```

2. Align the GoPro videos with the tracker/IK trajectories and build a
   `dataset_plan.pkl` for the session.
```bash
uv run run-pipeline data/<humi_session> --gopro_timezone +08:00
```

3. Export the synchronized session into OpenHLM-style episode folders.
```bash
uv run generate-final data/<humi_session>
```

The generated HuMI episodes are written to `data/<humi_session>/final_data`.
For detailed Windows tracker setup, IK recomputation, GoPro synchronization,
visualization, and OpenPI co-training conversion, see
[src/HuMI4OpenHLM/README.md](src/HuMI4OpenHLM/README.md).

---

<a id="checkpoints-and-datasets"></a>

## 🤗 Checkpoints and Datasets

We provide fine-tuned checkpoints and corresponding datasets on Hugging Face for quick start:

<a id="checkpoints"></a>

### Checkpoints

Five checkpoints are available at [OpenHLM/OpenHLM-ckpts](https://huggingface.co/OpenHLM/OpenHLM-ckpts):

- `12tasks_12full-teleop` — 12 tasks with 12 full teleoperation demonstrations
- `12tasks_8full-teleop_4humi` — 12 tasks with 8 full teleoperation + 4 HuMI demonstrations
- `12tasks_8full-teleop_4stationary_teleop` — 12 tasks with 8 full teleoperation + 4 stationary teleoperation demonstrations
- `20fruit-arrangement_20full-teleop` — 20 fruit arrangement task with 20 full teleoperation demonstrations
- `20fruit-arrangement_6full-teleop_14humi` — 20 fruit arrangement task with 6 full teleoperation + 14 HuMI demonstrations

<a id="datasets"></a>

### Datasets

Corresponding datasets in LeRobot format are available at [OpenHLM/OpenHLM-data](https://huggingface.co/datasets/OpenHLM/OpenHLM-data). **Each checkpoint above is trained from its corresponding dataset below**:

- `g1_HLM-12_full_teleop` → trains `12tasks_12full-teleop`
- `g1_HLM-12_humi` → trains `12tasks_8full-teleop_4humi`
- `g1_HLM-12_stationary_teleop` → trains `12tasks_8full-teleop_4stationary_teleop`
- `long_g1_5_fruits_full_teleop` → trains `20fruit-arrangement_20full-teleop`
- `long_g1_5_fruits_humi` → trains `20fruit-arrangement_6full-teleop_14humi`

<a id="example-raw-demonstration"></a>

### Example Raw Demonstration

A separate raw example demonstration is provided for testing the data conversion pipeline:

- `example_demonstration` (2 examples, 2 episodes) — Raw demonstration data for running the data conversion scripts

You can use this example data to test the conversion pipeline before collecting your own data (see [Converting Data to Lerobot Format](#converting-data-to-lerobot-format)).

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

<a id="configure-training-config"></a>

### Configure Training Config

Before training, you need to create a TrainConfig that defines your training setup. The config specifies the model architecture, dataset location, hyperparameters, and more.

You can add your custom config to [src/openpi4OpenHLM/src/openpi/training/config.py](src/openpi4OpenHLM/src/openpi/training/config.py). Here's an example based on `openhlm_example`:

```python
TrainConfig(
    name="openhlm_example",  # Your config name (must be unique)
    model=pi0_config.Pi0Config(
        pi05=True, 
        action_dim=34,           
        action_horizon=50,       
        discrete_state_input=True
    ),
    data=LeRobotG1DataConfig(
        repo_id="OpenHLM/example",  # IMPORTANT: Your dataset location
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=False,
    ),
    batch_size=256,
    num_workers=16,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-4,
        decay_steps=30_000,
        decay_lr=1e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    pytorch_weight_path="~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch",
    num_train_steps=30_000,
    save_interval=10_000,
)
```

**Key parameters to customize:**

- **`name`**: Unique identifier for your config (used to reference it during training)
- **`repo_id`**: Your dataset identifier in LeRobot format. **Important**: The actual data is stored in `$LEROBOT_HOME/<repo_id>/` (defaults to `~/.cache/huggingface/lerobot/<repo_id>/`). For example:
  - `"OpenHLM/example"` → data should be in `$LEROBOT_HOME/OpenHLM/example/`
  - `"my_org/my_dataset"` → data should be in `$LEROBOT_HOME/my_org/my_dataset/`
  - You can set `$LEROBOT_HOME` environment variable to customize the base directory
- **`pytorch_weight_path`**: Path to the pretrained PyTorch checkpoint (downloaded in Environment Setup)


After adding your config to [config.py](src/openpi4OpenHLM/src/openpi/training/config.py), you can reference it by name in the training commands below.

<a id="converting-data-to-lerobot-format"></a>

### Converting Data to Lerobot Format

The training code expects data in the Lerobot format. Use the provided conversion script to convert your collected data:

```bash
### Convert a single dataset
uv run examples/unitree_g1/convert_g1_data_to_lerobot.py \
    --data_dir ./example_demonstration/20260520_1602_example_1 \
    --repo_name OpenHLM/example
```
```bash
### Convert multiple datasets
uv run examples/unitree_g1/convert_g1_data_to_lerobot_multi.py \
    --parent_dir ./example_demonstration \
    --dataset_folders 20260520_1602_example_1 20260520_1602_example_2 \
    --repo_name OpenHLM/example
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
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py \
  openhlm_example \
  --save-interval 10000 \
  --batch_size 128 \
  --no-enable-gradient-checkpointing \
  --enable-training-compile \
  --pytorch-weight-path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \
  --exp-name openhlm_example
```

For more details on the training repository, please refer to [src/openpi4OpenHLM/README.md](src/openpi4OpenHLM/README.md).

---

<a id="deployment"></a>

## 📦 Deployment

### Policy Server

Start a policy server for remote inference:

```bash
# Use a trained checkpoint
cd src/openpi4OpenHLM
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
--env SONICG1 \
--num-steps 10 \
policy:checkpoint \
--policy.config=openhlm_example \
--policy.dir= path/to/trained/checkpoint 
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
cd src/GR00T-WholeBodyControl4OpenHLM/scripts
python openpi-eval/main.py \
--control_hz 30 \
--max_steps 10000 \
--save_video \
--instruction "example" \
--exp_name openhlm_example
```

See [src/openpi4OpenHLM/inference.md](src/openpi4OpenHLM/inference.md) for detailed instructions.




---

## 📜 License

This project is licensed under the **Apache 2.0 License**.
See [NOTICE](NOTICE) for third-party attributions and model/asset license notes.
