import json
from dataclasses import asdict
from pathlib import Path

import openvr  # type: ignore
import tyro
import win_precise_time as wpt  # type: ignore
import zmq
from htc.pose_common import (
    PoseData,
    PoseFrame,
    make_table,
    offset_ground_height,
    transform_poses,
)
from rich import print
from rich.live import Live


def precise_wait_until(
    target_time: float,
    slack_time: float = 0.002,
):
    while True:
        current_time = wpt.time()
        time_to_wait = target_time - current_time
        if time_to_wait <= 0:
            break
        elif time_to_wait > slack_time:
            wpt.sleep(time_to_wait - slack_time)
        else:
            while wpt.time() < target_time:
                pass


def main(
    port: int = 1234,
    frequency: float = 120.0,
    roles_to_send: list[str] | None = None,
    config_path: Path = Path(__file__).parents[1]
    / Path("tracker_config_ground.json"),
    offset_ground: bool = True,
):
    """Start the ZMQ server to send pose data.

    Args:
        port (int, optional): The port to bind the server to. Defaults to 1234.
        frequency (float, optional): Frequency to send pose data. Defaults to 120.0.
        roles_to_send (list[str], optional): List of roles to send. Defaults to ['root'].
        config_path (Path, optional): Path to the tracker configuration file. Defaults to Path(__file__).parents[1] / Path("tracker_config.json").
        offset_ground (bool, optional): Whether to offset poses by ground height. Defaults to True.
    """
    if roles_to_send is None:
        roles_to_send = ["root"]
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    print(f"ZMQ server started on port {port}.")

    # Initialize VR system
    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()
    print("VR system initialized.")

    with open(config_path, "r") as f:
        tracker_config = json.load(f)

    serial_to_role: dict[str, str] = {v: k for k, v in tracker_config.items()}

    print("Waiting for VR system to detect all trackers...\n")
    wpt.sleep(2)

    time_func = wpt.time
    dt = 1.0 / frequency

    poses = PoseFrame.read_from_vr_system(vr_system, time_func=time_func)

    assert len(poses) > 0, (
        "No trackers detected. Please ensure your trackers are powered on and connected."
    )

    if not all(pose.serial_number in serial_to_role for pose in poses):
        unknown_serials = [
            pose.serial_number
            for pose in poses
            if pose.serial_number not in serial_to_role
        ]
        print(
            f"Error: The following trackers are not in the config file: {unknown_serials}"
        )
        openvr.shutdown()
        return

    try:
        with Live(refresh_per_second=30) as live:
            while True:
                poses = PoseFrame.read_from_vr_system(
                    vr_system, time_func=time_func
                )
                poses = transform_poses(poses, serial_to_role)
                if offset_ground:
                    try:
                        poses = offset_ground_height(poses, serial_to_role)
                    except ValueError as e:
                        print(f"Warning: {e}. Skipping ground offset.")
                        continue

                loop_start_time = time_func()
                loop_end_time = loop_start_time + dt

                # send poses
                try:
                    pose_data = PoseData.from_pose_frames(
                        pose_frames=poses,
                        roles_to_send=roles_to_send,
                        serial_to_role=serial_to_role,
                    )
                    socket.send_json(asdict(pose_data), flags=zmq.NOBLOCK)
                except ValueError as e:
                    print(f"Warning: {e}. Skipping this frame.")

                table = make_table(poses, serial_to_role)
                live.update(table)
                precise_wait_until(target_time=loop_end_time)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        openvr.shutdown()
        socket.close()


if __name__ == "__main__":
    tyro.cli(main)


def cli():
    """Entry point for the `send-pose` console command.

    Wraps the Tyro-driven `main` so the script can be installed and invoked as
    a standard command without relying on `python -m` execution.
    """
    tyro.cli(main)
