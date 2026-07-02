#!/usr/bin/env python3

"""
Data collection script.
Collect data via ZMQ from pico manager, robot deploy, and hand controller.
The data includes:
- vision data
- body and hand state (from robot ZMQ)
- body and hand action (from pico ZMQ)


"""

from dataclasses import dataclass, field
from datetime import datetime
import os
import json
import time
import zmq
import msgpack
import cv2
import numpy as np
import threading
from gear_sonic.data_utils.episode_writer import EpisodeWriter
from gear_sonic.data_utils.vision_client import HeadZMQClient, WristZMQClient
from gear_sonic.data_utils.recording_visualizer import RecordingVisualizer
from gear_sonic.data_utils.quat_processor import QuatProcessor
from rich import print
from gear_sonic.data_utils.speaker import Speaker
from typing import Any, Dict, Literal, Optional
from scipy.spatial.transform import Rotation
import tyro


POSE_HEADER_SIZE = 1280
POSE_TOPIC = b"pose"


@dataclass
class ServerDataRecordConfig:
    """CLI config for recording demonstration data via ZMQ."""

    # Dataset
    data_folder: str = "openhlm_demonstration"
    """Data output folder."""

    task_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M"))
    """Task name. Defaults to the current timestamp."""

    desc: str = "Walk forward to the table and then put the bottle on the mouse pad."
    """Text description of the task."""

    frequency: int = 30
    """Recording frequency (Hz)."""

    # Robot
    robot: Literal["unitree_g1"] = "unitree_g1"
    """Robot name."""

    robot_ip: str = "192.168.123.164"
    """Robot IP (orin/onboard)."""

    pc_ip: str = "192.168.123.222"
    """Host PC IP (pico manager & deploy)."""

    # ZMQ: Pico action stream
    pico_port: int = 5556
    """ZMQ port for pico action stream."""

    # ZMQ: Robot state streams
    body_state_port: int = 5557
    """ZMQ port for body state from C++ deploy."""

    hand_state_port: int = 5558
    """ZMQ port for hand state from orin."""

    # Head cameras
    image_height: int = 400
    """Image height for head cameras."""

    image_width: int = 464
    """Image width per head camera."""

    # Wrist cameras
    wrist_height: int = 480
    """Image height for wrist cameras."""

    wrist_width: int = 640
    """Image width per wrist camera."""

    # Visualization
    rerun_visualize: bool = False
    """Enable real-time Rerun visualization."""

    # Optional exports
    record_human_motion_data: bool = False
    """Record human motion data from pico to pico_data.json."""


def _unpack_pose_message(raw: bytes) -> Optional[Dict[str, np.ndarray]]:
    """Parse [topic][1280-byte JSON header][binary payload] into a dict of numpy arrays.

    Wire format from pico_manager_thread_server.py / pack_pose_message():
      [topic_prefix bytes][1280-byte zero-padded JSON header][concatenated binary fields]
    """
    offset = len(POSE_TOPIC)
    if len(raw) < offset + POSE_HEADER_SIZE:
        return None
    header = json.loads(raw[offset: offset + POSE_HEADER_SIZE].rstrip(b"\x00"))
    payload = raw[offset + POSE_HEADER_SIZE:]

    dtype_map = {
        "f32": np.float32, "f64": np.float64,
        "i32": np.int32, "i64": np.int64,
        "bool": bool,
    }
    result: Dict[str, np.ndarray] = {}
    pos = 0
    for field in header["fields"]:
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arr = np.frombuffer(payload[pos: pos + nbytes], dtype=dtype).reshape(shape)
        result[field["name"]] = arr
        pos += nbytes
    return result


def main(config: ServerDataRecordConfig):

    # ---- ZMQ setup ----
    zmq_context = zmq.Context()

    # 1) Pico action data (PUB/SUB): joint_pos, action_hand_left/right, toggle controls
    #    Protocol: [topic "pose"][1280-byte JSON header][binary payload]  (pack_pose_message format)
    pico_sub = zmq_context.socket(zmq.SUB)
    pico_sub.setsockopt(zmq.SUBSCRIBE, POSE_TOPIC)
    pico_sub.setsockopt(zmq.RCVHWM, 2)          # buffer messages so we don't miss toggles
    pico_sub.connect(f"tcp://{config.pc_ip}:{config.pico_port}")
    print(f"Connected to Pico action stream at {config.pc_ip}:{config.pico_port}")

    # 2) Body state (PUB/SUB, msgpack): body_q_measured from C++ deploy
    #    Protocol: [topic "g1_debug"][msgpack map<string, vector<double>>]
    body_state_sub = zmq_context.socket(zmq.SUB)
    body_state_sub.setsockopt_string(zmq.SUBSCRIBE, "g1_debug")
    body_state_sub.setsockopt(zmq.CONFLATE, 1)     # only keep latest
    body_state_sub.connect(f"tcp://{config.pc_ip}:{config.body_state_port}")
    print(f"Connected to body state stream at {config.pc_ip}:{config.body_state_port}")

    # 3) Hand state (PUB/SUB, msgpack): state_hand_left/right from orin_hand_controller
    #    Protocol: msgpack dict (no topic prefix)
    hand_state_sub = zmq_context.socket(zmq.SUB)
    hand_state_sub.setsockopt_string(zmq.SUBSCRIBE, "")  # no topic prefix
    hand_state_sub.setsockopt(zmq.CONFLATE, 1)     # only keep latest
    hand_state_sub.connect(f"tcp://{config.robot_ip}:{config.hand_state_port}")
    print(f"Connected to hand state stream at {config.robot_ip}:{config.hand_state_port}")

    num_cameras = 2
    image_shape = (config.image_height, config.image_width * num_cameras, 3)  # Height, Width, Channels for OpenCV format

    wrist_shape = (config.wrist_height, config.wrist_width * num_cameras, 3)
    wrist_client = WristZMQClient(
        server_address=config.robot_ip,
        port=5554,
        image_shape=wrist_shape,
        image_show=False,
    )
    wrist_thread = threading.Thread(target=wrist_client.receive_process, daemon=True)
    wrist_thread.start()


    # Display settings for single camera
    # If rerun_visualize is enabled, disable OpenCV image_show
    image_show = not config.rerun_visualize

    vision_client = HeadZMQClient(
        server_address=config.robot_ip,  # robot IP
        port=5555,
        image_shape=image_shape,
        image_show=False,
    )
    vision_thread = threading.Thread(target=vision_client.receive_process, daemon=True)
    vision_thread.start()
    
    # create recorder
    recording = False
    # recording = True
    save_data_keys = ['rgb_left', 'rgb_right', 'wrist_rgb_left', 'wrist_rgb_right']
    task_dir = os.path.join(config.data_folder, config.task_name)
    recorder = EpisodeWriter(task_dir = task_dir, frequency = config.frequency,
                             image_shape=image_shape,
                             wrist_shape=wrist_shape,
                             data_keys=save_data_keys)
    # recorder.text_desc(goal="walk ahead and pick a box.",
    #                    desc="a humanoid robot walk head and pick a box from the table.",
    #                    steps="step1: walk ahead 1 meter. step2: pick a box from the table.")
    # recorder.text_desc(goal="Pick up the purple soft finger on the table and place it on the mouse pad.", desc="", steps="")
    recorder.text_desc(goal=config.desc, desc=config.desc, steps="")

    # Initialize Rerun visualizer
    rerun_viz = None
    if config.rerun_visualize:
        rerun_viz = RecordingVisualizer(
            app_id="data_recording",
            recording_name=f"recording_{config.task_name}",
            jpeg_quality=85,
            enabled=True,
            enable_wrist=True,
        )
        print("[green]Rerun visualizer initialized[/green]")
    
    # Print text description and require user confirmation
    print("\n" + "="*60)
    print("[bold yellow]Task Description:[/bold yellow]")
    print(f"  [cyan]Goal:[/cyan]  {recorder.text['goal']}")
    print("="*60)
    
    user_confirm = input("\n[Confirm] Is the task description correct? (y/n): ").strip().lower()
    if user_confirm != 'y':
        print("[red]Task description not confirmed. Exiting...[/red]")
        vision_client.stop()
        vision_thread.join(timeout=1.0)
        wrist_client.stop()
        wrist_thread.join(timeout=1.0)
        recorder.close()
        pico_sub.close()
        body_state_sub.close()
        hand_state_sub.close()
        zmq_context.term()
        return
    
    print("[green]Task description confirmed. Starting recording...[/green]\n")
    print("="*60)
    print("[bold]Controls (Pico VR):[/bold]")
    print("  [cyan]Left grip + A[/cyan]  : Toggle recording start/stop (save episode)")
    print("  [cyan]Left grip + B[/cyan]  : Discard current episode (while recording)")
    print("  [cyan]Ctrl+C[/cyan]         : Quit program")
    print("="*60 + "\n")
    
    control_dt = 1 / config.frequency
    step_count = 0
    running = True
    
    print("Recorded control frequency: ", config.frequency)
    
   
    speaker = Speaker(network_interface="enp131s0")
    
    # Cache for latest ZMQ data (persists across loop iterations)
    latest_pico_data: Optional[Dict[str, np.ndarray]] = None
    latest_body_state: Optional[dict] = None
    latest_hand_state: Optional[dict] = None

    # Cache for previous yaw values (for delta_yaw calculation)
    prev_state_yaw: Optional[float] = None
    prev_action_yaw: Optional[float] = None

    state_quat_processor = QuatProcessor()
    action_quat_processor = QuatProcessor()

    # Pico human motion data recording
    pico_data_list = []
    
    try:
        while running:

            start_time = time.time()
            
            # ---- Drain all pending pico messages to catch toggle events ----
            toggle_collection = False
            toggle_abort = False
            pico_data = None
            while True:
                try:
                    raw = pico_sub.recv(zmq.NOBLOCK)
                    pico_data = _unpack_pose_message(raw)

                    if pico_data is not None:
                        latest_pico_data = pico_data
                        if pico_data.get("toggle_data_collection", np.array([False]))[0]:
                            toggle_collection = True
                        if pico_data.get("toggle_data_abort", np.array([False]))[0]:
                            toggle_abort = True
                except zmq.Again:
                    break

            # ---- Poll body state (latest only) ----
            try:
                raw = body_state_sub.recv(zmq.NOBLOCK)
                topic_prefix = b"g1_debug"
                unpacked = msgpack.unpackb(raw[len(topic_prefix):], raw=False)
                latest_body_state = unpacked
            except zmq.Again:
                pass

            # ---- Poll hand state (latest only) ----
            try:
                raw = hand_state_sub.recv(zmq.NOBLOCK)
                latest_hand_state = msgpack.unpackb(raw, raw=False)
            except zmq.Again:
                pass

            print(f"==> toggle_collection: {toggle_collection}, toggle_abort: {toggle_abort}", end="\r")

            # Detect discard toggle - only when recording
            if toggle_abort and recording:
                print("\n[!] Discard button pressed")
                recording = False
                recorder.discard_episode()
                speaker.speak("episode discarded.")
                print("Episode discarded.")
            
            # Detect recording toggle (start/stop)
            if toggle_collection:
                print("\ntoggle recording pressed")
                recording = not recording
                if recording:
                    speaker.speak("episode recording started.")
                    if not recorder.create_episode():
                        recording = False
                    step_count = 0
                    pico_data_list.clear()
                    print("episode recording started...")
                    # Start new episode in Rerun visualizer
                    if rerun_viz is not None:
                        rerun_viz.new_episode(f"episode_{recorder.episode_id:04d}")
                else:
                    recorder.save_episode()
                    if config.record_human_motion_data and pico_data_list:
                        pico_file = os.path.join(recorder.episode_dir, "pico_data.json")
                        with open(pico_file, 'w') as f:
                            f.write('[\n')
                            for i, item in enumerate(pico_data_list):
                                suffix = ',\n' if i < len(pico_data_list) - 1 else '\n'
                                f.write(json.dumps(item) + suffix)
                            f.write(']\n')
                        print(f"Saved {len(pico_data_list)} pico data frames to {pico_file}")
                    speaker.speak("episode saved.")
           
            if recording:
                data_dict: dict[str, Any] = {'idx': step_count}
                # receive vision data (split stereo into left/right)
                stereo_img = vision_client.get_latest_image()
                if stereo_img is None:
                    print("Warning: No head camera data available")
                    time.sleep(control_dt)
                    continue
                if stereo_img.shape[1] % 2 != 0:
                    print("Warning: stereo image width not even, skipping split")
                    continue
                half_w = stereo_img.shape[1] // 2
                data_dict["rgb_left"] = stereo_img[:, :half_w, :]
                data_dict["rgb_right"] = stereo_img[:, half_w:, :]
                data_dict["t_img"] = int(time.time() * 1000) # current timestamp in ms

                # Split wrist stereo. Wrist cameras are part of the required
                # video-backed recording format.
                wrist_img = wrist_client.get_latest_image()
                if wrist_img is None:
                    print("Warning: No wrist camera data available")
                    time.sleep(control_dt)
                    continue
                if wrist_img.shape[1] % 2 != 0:
                    print("Warning: wrist image width not even, skipping wrist split")
                    continue
                wrist_half_w = wrist_img.shape[1] // 2
                data_dict["wrist_rgb_left"] = wrist_img[:, :wrist_half_w, :]
                data_dict["wrist_rgb_right"] = wrist_img[:, wrist_half_w:, :]
                data_dict["t_wrist"] = int(time.time() * 1000)

                # ---- Populate state data from ZMQ ----
                # state_body: [roll, pitch, delta_yaw] + body_q_measured (3+29=32 DOF)
                if latest_body_state is not None and "base_quat_measured" in latest_body_state and "body_q_measured" in latest_body_state:
                    quat_wxyz = latest_body_state["base_quat_measured"]
                    roll, pitch, yaw_vel = state_quat_processor.process_output_yaw_vel(quat_wxyz, timestamp_ns=int(time.time() * 1e9))
                    data_dict["state_body"] = [roll, pitch, yaw_vel] + latest_body_state["body_q_measured"]
                else:
                    print("Warning: No body state data available")
                    data_dict["state_body"] = None

                # state_hand_left / state_hand_right from orin_hand_controller (msgpack)
                if latest_hand_state is not None:
                    data_dict["state_hand_left"] = latest_hand_state.get("state_hand_left")
                    data_dict["state_hand_right"] = latest_hand_state.get("state_hand_right")
                else:
                    print("Warning: No hand state data available")
                    data_dict["state_hand_left"] = None
                    data_dict["state_hand_right"] = None

                data_dict["t_state"] = int(time.time() * 1000)

                # ---- Populate action data from pico ZMQ ----
                if latest_pico_data is not None:
                    # joint_pos: stacked (N, 29), take the latest frame
                    joint_pos = latest_pico_data.get("joint_pos_real")
                    body_quat_w = latest_pico_data.get("body_quat_w")

                    if joint_pos is not None and body_quat_w is not None:
                        if joint_pos.ndim == 2:
                            joint_data = joint_pos[-1].tolist()
                        else:
                            joint_data = joint_pos.tolist()

                        # body_quat_w is (N, 4) in xyzw format, take the latest frame
                        if body_quat_w.ndim == 2:
                            quat_xyzw = body_quat_w[-1]
                        else:
                            quat_xyzw = body_quat_w

                        # Convert to roll, pitch, yaw
                        roll, pitch, yaw_vel = action_quat_processor.process_output_yaw_vel(quat_xyzw, timestamp_ns=int(time.time() * 1e9))
                        data_dict["action_body"] = [roll, pitch, yaw_vel] + joint_data
                    else:
                        data_dict["action_body"] = None

                    action_hl = latest_pico_data.get("action_hand_left")
                    data_dict["action_hand_left"] = float(action_hl[0]) if action_hl is not None else None

                    action_hr = latest_pico_data.get("action_hand_right")
                    data_dict["action_hand_right"] = float(action_hr[0]) if action_hr is not None else None
                else:
                    print("Warning: No pico action data available")
                    data_dict["action_body"] = None
                    data_dict["action_hand_left"] = None
                    data_dict["action_hand_right"] = None

                data_dict["t_action"] = int(time.time() * 1000)

                # Record human motion data if enabled
                if config.record_human_motion_data and latest_pico_data is not None:
                    human_motion_data = latest_pico_data.get("human_motion_data")
                    if human_motion_data is not None:
                        pico_data_list.append({
                            "step": step_count,
                            "timestamp": int(time.time() * 1000),
                            "human_motion_data": human_motion_data.tolist()
                        })

                print(f"[Recording] step {step_count} data keys: {list(data_dict.keys())}")
                print(f"  state_hand_left: {data_dict['state_hand_left']}")
                print(f"  state_hand_right: {data_dict['state_hand_right']}")
                print(f"  action_hand_left: {data_dict['action_hand_left']}")
                print(f"  action_hand_right: {data_dict['action_hand_right']}")
                
                # Log to Rerun visualizer first (before recorder modifies data_dict)
                if rerun_viz is not None:
                    rerun_viz.log_frame(step_count, data_dict)
                
                # write data to recorder (this may modify data_dict in background thread)
                recorder.add_item(data_dict)
                
                if image_show:
                    image_display = vision_client.get_latest_image(copy=False)
                    if image_display is not None and image_display.size > 0:
                        # Create window with size matching image
                        window_name = "Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_display.shape[1], image_display.shape[0])
                        cv2.moveWindow(window_name, 50, 50)  # Position window on left side
                        cv2.imshow(window_name, image_display)
                        cv2.waitKey(1)
                
                step_count += 1
                elapsed = time.time() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                if image_show:
                    image_display = vision_client.get_latest_image(copy=False)
                    if image_display is not None and image_display.size > 0:
                        # Create window with size matching image
                        window_name = "Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_display.shape[1], image_display.shape[0])
                        cv2.moveWindow(window_name, 50, 50)  # Position window on left side
                        cv2.imshow(window_name, image_display)
                        cv2.waitKey(1)
                else:
                    # For keyboard mode, just sleep to avoid busy waiting
                    time.sleep(0.1)
                    
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, exiting...")
        running = False
    finally:
        print(f"\nDone! Recorded {recorder.episode_id + 1} episodes to {task_dir}")

        vision_client.stop()
        vision_thread.join(timeout=1.0)
        wrist_client.stop()
        wrist_thread.join(timeout=1.0)
        recorder.close()
        
        # Close ZMQ sockets
        pico_sub.close()
        body_state_sub.close()
        hand_state_sub.close()
        zmq_context.term()
        
        # Close Rerun visualizer
        if rerun_viz is not None:
            rerun_viz.close()
        
        cv2.destroyAllWindows()  # Close OpenCV window
        
        print("Exiting the recording...")

if __name__ == "__main__":
    config = tyro.cli(ServerDataRecordConfig)
    main(config)
