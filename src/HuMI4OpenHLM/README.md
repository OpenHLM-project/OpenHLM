# HuMI4OpenHLM Data Collection

This repository converts human demonstrations recorded with HTC Vive trackers
and wrist-mounted GoPros into OpenHLM-style episode data for training.

Unless otherwise noted, run every command in this README from the
`src/HuMI4OpenHLM` directory.

## Installation

This setup utilizes a Windows machine to interface with trackers and a Linux or macOS machine for real-time Inverse Kinematics (IK).

### Windows Setup (Tracker Interface)

The HTC Vive tracker interface has been extracted into a separate package (`htc-interface`) under the `packages/` directory.

To install it on a Windows machine, navigate to `packages/htc_interface` and run:

```powershell
pip install .
```

See [packages/htc_interface/README.md](packages/htc_interface/README.md) for more detailed Windows setup instructions, including SteamVR configuration.

### Linux/macOS Setup (IK Solver)

We use `uv` for dependency and virtual environment management.

1.  **Install System Dependencies**:
    `exiftool` is required for processing GoPro video metadata.
    - **macOS**: `brew install exiftool`
    - **Linux (Ubuntu/Debian)**: `sudo apt-get install libimage-exiftool-perl`

2.  **Install Python Dependencies**:
    ```bash
    uv sync
    ```

## Configuration

The `tracker_config.json` file maps specific tracker IDs to their respective body roles.

Required fields:
- `right_hand`: ID for the right hand tracker.
- `left_hand`: ID for the left hand tracker.
- `root`: ID for the root (waist/pelvis) tracker.
- `left_foot`: ID for the left foot tracker.
- `right_foot`: ID for the right foot tracker.

Example `tracker_config.json`:
```json
{
  "right_hand": "3A-A33H02641",
  "left_hand": "3A-A33H02658",
  "root": "39-A33L00039",
  "left_foot": "39-A33L03913",
  "right_foot": "39-A33L03953"
}
```

## Usage

### 1. Start Tracker Pose Recording (Windows)

Connect the five trackers to the Vive hub and calibrate the tracking space according to the on-screen prompts. Once the setup is complete, start recording tracker poses using the installed `htc-interface` package:

```powershell
record-pose --rpc.serve --output-dir data/my-tracker-recordings
```

### 2. Start Online IK Solver (Linux/macOS)

Run the `online-ik` command to launch the IK solver and its web interface. Use
the `pick` task configuration for the current data collection workflow.

```bash
uv run online-ik pick --rpc-address tcp://<windows_machine_ip>:4242
```

This launches a control interface in your browser. Use the web UI to start/stop recordings, delete the last episode, and monitor real-time tracking data.

You can attach language instructions to each episode with either a single
fallback prompt:

```bash
uv run online-ik pick \
    --rpc-address tcp://<windows_machine_ip>:4242 \
    --prompt "Pick up the bottle and put it on the mouse pad."
```

or a prompt list:

```bash
uv run online-ik pick \
    --rpc-address tcp://<windows_machine_ip>:4242 \
    --prompt-file configs/prompts/fruits_shelf.txt
```

For a `.txt` prompt file, each non-empty line is one prompt. Episode `N` uses
line `N`; if the recording count exceeds the number of prompts, the list wraps
around.

The online IK UI shows the prompt for the current episode. The same prompt is
saved into the final data as `text.goal`.

> **Note on IK Parameters:**  
> The IK optimization settings and task weights (like task gains, posture costs, and tracking offsets) are defined in code. To adjust these IK parameters for existing tasks or to create new ones, please modify `src/ikumi/config.py`.

### 3. Offline IK Recomputation (Linux/macOS)

After transferring the raw JSON tracker data from the Windows machine, you need to recompute the full-body Inverse Kinematics (IK) to generate the robot joint trajectories.

1.  **Run Offline IK**:
    ```bash
    uv run offline-ik <task_name> -i data/my-raw-recordings
    ```
    This will process all JSON files in the input directory and save the results (containing joint angles) into a new directory, typically named `data/my-raw-recordings_ik_recomputed`.

2.  **Verify Results**:
    You can visualize the recomputed trajectories using the `replay` command:
    ```bash
    uv run replay -i data/my-raw-recordings_ik_recomputed	
    ```

### 4. Data Processing Pipeline

After recomputing the IK, organize your data directory as follows. Note that the `session_name_hand_root_foot_ik` (or similar) directory used here should be the **output** of the `offline-ik` step.

Organize your data directory as follows:
```text
data/<humi_session>/
├── card0/                      # GoPro videos from camera 0
│   ├── GX019551.MP4
│   └── ...
├── card1/                      # GoPro videos from camera 1
│   ├── GX019551.MP4
│   └── ...
└── session_name_hand_root_foot_ik/ # Recomputed IK JSONs from offline-ik
    ├── recording_2026.01.01_12.12.03.794766.json
    └── ...
```

> **Note:** The data processing pipeline will automatically filter and ignore raw, unprocessed tracker JSONs. It is completely safe to place both the raw recordings from the Windows machine and the recomputed IK outputs in the same directory (or to run `offline-ik` in-place). The pipeline will only group and process the final IK-solved trajectories.

> **Note on GoPro Timezones:** GoPro MP4 files store their creation time using local time without specifying the time zone. When the pipeline pairs videos with tracker trajectories (which use absolute system time), it assumes the GoPro was in the same time zone as the machine running the pipeline. If your GoPro recorded in a different time zone, use the `--gopro_timezone` (or `-tz`) parameter to specify the GoPro's time zone (e.g., `--gopro_timezone Asia/Shanghai` or `--gopro_timezone +08:00`).

Execute the processing pipeline:
```bash
uv run run-pipeline data/<humi_session> --gopro_timezone +08:00
```
Upon completion, a `dataset_plan.pkl` file will be generated in the session directory.

### 5. Generate OpenHLM Episodes

Generate the final OpenHLM-style episode directory:

```bash
uv run generate-final data/<humi_session>
```

This creates:

```text
data/<humi_session>/final_data/
├── episode_0000/
│   ├── data.json
│   └── videos/
│       ├── wrist_rgb_left.mp4
│       └── wrist_rgb_right.mp4
└── episode_0001/
    └── ...
```

Each frame in `data.json` stores wrist camera references using the same
video-backed format as the real-world teleop recorder:

```json
{
  "idx": 0,
  "wrist_rgb_left": {
    "video_path": "videos/wrist_rgb_left.mp4",
    "frame_index": 0
  },
  "wrist_rgb_right": {
    "video_path": "videos/wrist_rgb_right.mp4",
    "frame_index": 0
  },
  "state_body": [],
  "action_body": [],
  "state_hand_left": 0.0,
  "state_hand_right": 0.0,
  "action_hand_left": 0.0,
  "action_hand_right": 0.0
}
```

HuMI final data contains wrist-view videos only
(`wrist_rgb_left` and `wrist_rgb_right`). Since HuMI does not include a head
view, downstream OpenPI conversion fills the corresponding head image with a
black placeholder.

### 6. Visualize Final Episodes (Optional)

```bash
uv run visualize-final -i data/<humi_session>
```

This starts a `viser` server for inspecting the G1 trajectory, wrist videos,
prompt text, and gripper actions.

### 7. Convert for VLA Co-training

For co-training, include the HuMI `final_data` directory alongside robot
teleoperation datasets when running the LeRobot conversion script. For example,
from the `src/openpi4OpenHLM` directory:

```bash
uv run examples/unitree_g1/convert_g1_data_to_lerobot_multi.py \
    --parent_dir .. \
    --dataset_folders openpi4OpenHLM/example_demonstration/<teleop_session_1> \
    --humi_dataset_folders HuMI4OpenHLM/data/<humi_session>/final_data \
    --repo_name <your_org>/<your_dataset_name>
```

See [../openpi4OpenHLM/README.md](../openpi4OpenHLM/README.md) for the full
OpenPI training workflow, including LeRobot conversion, normalization
statistics, and training.
