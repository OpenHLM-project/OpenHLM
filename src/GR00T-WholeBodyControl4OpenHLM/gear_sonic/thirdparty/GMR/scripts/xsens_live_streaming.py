#!/usr/bin/env python3
"""
Xsens MVN Live Streaming to Unitree G1 Retargeting.
Real-time motion capture retargeting following bvh_xsens_to_robot.py format.
"""

import argparse
import os
import signal
import sys
import time

import numpy as np
from rich import print

from general_motion_retargeting.utils.xsens_vendor.xsens_to_gmr_adapter import XsensToGMR
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

# Global flag for graceful shutdown
g_running = True


def signal_handler(signum, frame):
    global g_running
    print(f"\nReceived signal {signum}, shutting down...")
    g_running = False


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Xsens MVN Live Streaming to Robot Retargeting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9763,
        help="UDP port for Xsens MVN streaming",
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1"],
        default="unitree_g1",
    )

    parser.add_argument(
        "--human_height",
        type=float,
        default=None,
        help="Actual height of human in meters (optional, for better scaling)",
    )

    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/xsens_live.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--save_dir",
        default=None,
        help="Directory to save the robot motion CSV. If not set, no file is saved.",
    )

    args = parser.parse_args()

    target_fps = 60  # Xsens MVN typical streaming rate

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        qpos_list = []

    # ---- Initialize Xsens adapter ----
    print("[1/3] Initializing Xsens adapter...")
    xsens = XsensToGMR(port=args.port, verbose=True)
    if not xsens.initialize():
        print("Failed to initialize Xsens adapter")
        sys.exit(1)

    # ---- Initialize retargeter ----
    print("[2/3] Initializing retargeter...")
    retargeter = GMR(
        src_human="xsens_mvn",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        solver="daqp",
        damping=1.0,
        use_velocity_limit=True,
    )

    # ---- Initialize viewer ----
    print("[3/3] Initializing viewer...")
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=target_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    # ---- Start streaming ----
    xsens.start()
    time.sleep(1.0)  # Wait for stream to stabilize

    print(f"mocap_frame_rate: {target_fps}")
    print("Starting live retargeting... Press Ctrl+C to stop\n")

    # FPS measurement
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    total_frames = 0
    dropped_frames = 0
    last_valid_qpos = None
    last_valid_human_frame = None

    try:
        while g_running:
            human_frame = xsens.get_human_frame()

            if human_frame is None:
                dropped_frames += 1
                # Show last valid pose if available
                if last_valid_qpos is not None:
                    robot_motion_viewer.step(
                        root_pos=last_valid_qpos[:3],
                        root_rot=last_valid_qpos[3:7],
                        dof_pos=last_valid_qpos[7:],
                        human_motion_data=last_valid_human_frame,
                        rate_limit=args.rate_limit,
                    )
                time.sleep(0.001)
                continue

            total_frames += 1

            # Retarget
            try:
                qpos = retargeter.retarget(human_frame, offset_to_ground=False)
            except Exception as e:
                print(f"Retargeting failed: {e}")
                dropped_frames += 1
                continue

            last_valid_qpos = qpos.copy()
            last_valid_human_frame = retargeter.scaled_human_data

            # Visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            if args.save_dir is not None:
                qpos_list.append(qpos)

            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(
                    f"FPS: {actual_fps:.1f} | "
                    f"Retargeted: {total_frames} | "
                    f"Dropped: {dropped_frames}"
                )
                fps_counter = 0
                fps_start_time = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Stop streaming
        xsens.stop()

        # Save trajectory
        if args.save_dir is not None and qpos_list:
            import pickle
            root_pos = np.array([qpos[:3] for qpos in qpos_list])
            # save from wxyz to xyzw
            root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
            dof_pos = np.array([qpos[7:] for qpos in qpos_list])
            local_body_pos = None
            body_names = None
            
            motion_data = {
                "fps": target_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos,
                "link_body_list": body_names,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"Saved to {args.save_path}")

        # Print final stats
        print(f"\nTotal retargeted: {total_frames}")
        print(f"Total dropped: {dropped_frames}")

        robot_motion_viewer.close()