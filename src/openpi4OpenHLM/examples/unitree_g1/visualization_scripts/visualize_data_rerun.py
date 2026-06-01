"""Visualize recorded episode data using OpenPIRerunVisualizer.

Reads data from a recorded episode directory (data.json + image folders)
and replays it through the Rerun visualizer at 30 FPS.

Usage:
    python visualize_data_rerun.py --episode_dir /path/to/episode_0001
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root so we can import from sibling directories.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from rerun_visualizer import OpenPIRerunVisualizer


def mimic_to_openpi(body_32: list[float], hand_left: float, hand_right: float) -> np.ndarray:
    """Convert 32-dim mimic body format + gripper values to 34-dim OpenPI format.

    Mimic layout (32 dims):
        0-2:   root (3): roll, pitch, yaw angular velocity
        3-8:   left leg (6)
        9-14:  right leg (6)
        15-17: waist (3)
        18-24: left arm (7)
        25-31: right arm (7)

    OpenPI layout (34 dims):
        0-6:   left arm (7)
        7:     left gripper (1)
        8-14:  right arm (7)
        15:    right gripper (1)
        16-21: left leg (6)
        22-27: right leg (6)
        28-30: waist (3)
        31-33: root (3)
    """
    m = np.array(body_32, dtype=np.float32)
    openpi = np.zeros(34, dtype=np.float32)
    openpi[0:7] = m[18:25]    # left arm
    openpi[7] = hand_left      # left gripper
    openpi[8:15] = m[25:32]   # right arm
    openpi[15] = hand_right    # right gripper
    openpi[16:22] = m[3:9]    # left leg
    openpi[22:28] = m[9:15]   # right leg
    openpi[28:31] = m[15:18]  # waist
    openpi[31:34] = m[0:3]    # root
    return openpi


def load_image_or_black(episode_dir: Path, relative_path: str | None, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load an image from disk, or return a black image if the path is missing."""
    if relative_path is not None:
        img_path = episode_dir / relative_path
        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                return img_rgb
    # Return a black placeholder image.
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize recorded episode data with Rerun.")
    parser.add_argument(
        "--episode_dir",
        type=str,
        default="/Users/huyingdong/Downloads/openpi-humanoid/data/final_data/episode_0000",
        help="Path to the episode directory containing data.json and image folders.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        print("\033[91mDebug server started on 127.0.0.1:5678\033[0m")
        print("\033[91mWaiting for debugger to attach...\033[0m")
        debugpy.wait_for_client()
        print("Debugger attached, continuing execution...")

    episode_dir = Path(args.episode_dir)
    data_path = episode_dir / "data.json"
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    with open(data_path, "r") as f:
        episode = json.load(f)

    frames = episode["data"]
    num_frames = len(frames)
    fps = episode.get("info", {}).get("wrist_image", {}).get("fps", 30)
    goal = episode.get("text", {}).get("goal", "")
    print(f"Episode: {episode_dir.name}")
    print(f"Frames: {num_frames}, FPS: {fps}")
    print(f"Goal: {goal}")

    # Initialize the visualizer.
    mujoco_xml_path = os.path.join(os.path.dirname(__file__), "../robot_assets/g1_mocap_29dof.xml")
    viz = OpenPIRerunVisualizer(
        app_id="openpi_data_viewer",
        recording_name=episode_dir.name,
        mujoco_xml_path=mujoco_xml_path,
        control_hz=float(fps),
    )

    dt = 1.0 / fps
    print(f"Playing back at {fps} FPS (dt={dt:.4f}s)...")

    for frame in frames:
        t_start = time.time()
        step_idx = frame["idx"]

        # Build the state vector (34 dims).
        state = mimic_to_openpi(
            frame["state_body"],
            frame.get("state_hand_left", 0.0),
            frame.get("state_hand_right", 0.0),
        )

        # Build the action vector (34 dims).
        action = mimic_to_openpi(
            frame["action_body"],
            frame.get("action_hand_left", 0.0),
            frame.get("action_hand_right", 0.0),
        )

        # Load images (use black placeholders for missing ones).
        left_wrist = load_image_or_black(episode_dir, frame.get("wrist_rgb_left"))
        right_wrist = load_image_or_black(episode_dir, frame.get("wrist_rgb_right"))
        head_image = load_image_or_black(episode_dir, frame.get("head_image_left"))

        # Build observation dict matching the format expected by log_observation.
        obs = {
            "head_image_left": head_image,
            "left_wrist_image": left_wrist,
            "right_wrist_image": right_wrist,
            "state": state,
        }

        viz.log_observation(step_idx, obs)
        viz.log_action(step_idx, action)

        # Maintain playback rate.
        elapsed = time.time() - t_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print("Playback complete.")


if __name__ == "__main__":
    main()
