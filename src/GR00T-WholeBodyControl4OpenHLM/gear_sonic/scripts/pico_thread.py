import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from enum import Enum

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R

from pico_manager_thread_server import PicoReader, get_abxy_buttons, get_controller_inputs, xrt
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)
from gear_sonic.data_utils.quat_processor import QuatProcessor

from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

class StreamMode(Enum):   
    POSE = 1
    POSE_PAUSE = 2


PICO_XROBOT_INDEX = {
    "Pelvis": 0,
    "Left_Hip": 1,
    "Right_Hip": 2,
    "Left_Knee": 4,
    "Right_Knee": 5,
    "Spine3": 9,
    "Left_Foot": 10,
    "Right_Foot": 11,
    "Left_Shoulder": 16,
    "Right_Shoulder": 17,
    "Left_Elbow": 18,
    "Right_Elbow": 19,
    "Left_Wrist": 20,
    "Right_Wrist": 21,
}


G1_REAL_TO_ISAACLAB_IDX = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q.astype(np.float32)


def _unity_to_xrobot_pose(pose_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_xyzw = np.asarray(pose_xyzw, dtype=np.float64)
    pos_unity = pose_xyzw[:3]
    quat_wxyz = np.array([pose_xyzw[6], pose_xyzw[3], pose_xyzw[4], pose_xyzw[5]], dtype=np.float64)

    rotation_matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    pos_robot = pos_unity @ rotation_matrix.T

    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
    rot_robot = R.from_quat(rotation_quat, scalar_first=True) * R.from_quat(
        quat_wxyz, scalar_first=True
    )
    return pos_robot, rot_robot.as_quat(scalar_first=True)


def pico_body_poses_to_xrobot_frame(body_poses_np: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    body_poses_np = np.asarray(body_poses_np, dtype=np.float64)
    if body_poses_np.ndim != 2 or body_poses_np.shape[1] != 7 or body_poses_np.shape[0] <= 21:
        raise ValueError(f"Expected body_poses_np shape (N, 7) with N >= 22, got {body_poses_np.shape}")

    frame = {}
    for name, idx in PICO_XROBOT_INDEX.items():
        frame[name] = _unity_to_xrobot_pose(body_poses_np[idx])
    return frame


def _gmr_worker(
    input_queue,
    output_queue,
    stop_event,
    robot: str,
    verbose: bool,
    return_human_motion_data: bool,
) -> None:
    """Retargeting worker running in a dedicated subprocess.

    Completely isolated from the main process GIL, so the IK solver can use a
    full CPU core without being interrupted by the main loop.

    Data flow:
        main process (feeder thread) --[input_queue]--> this process
        this process                 --[output_queue]--> main process
    """
    import time as _time  # local import keeps namespace clean in forked child

    from general_motion_retargeting import GeneralMotionRetargeting

    retarget = GeneralMotionRetargeting(
        src_human="xrobot",
        tgt_robot=robot,
        actual_human_height=1.6,
        verbose=verbose,
    )

    last_source_stamp_ns: int | None = None
    prev_joint_pos: np.ndarray | None = None
    last_report = _time.time()

    while not stop_event.is_set():
        # Drain queue: keep only the newest sample to avoid pile-up
        sample = None
        while True:
            try:
                sample = input_queue.get_nowait()
            except Exception:
                break

        if sample is None:
            # Nothing yet – block briefly so we don't busy-spin
            try:
                sample = input_queue.get(timeout=0.005)
            except Exception:
                continue

        source_stamp_ns = int(sample.get("timestamp_ns", 0))
        if source_stamp_ns <= 0 or source_stamp_ns == last_source_stamp_ns:
            continue

        start_mono = _time.monotonic()
        try:
            frame = pico_body_poses_to_xrobot_frame(sample["body_poses_np"])
            qpos = retarget.retarget(frame, offset_to_ground=True)
            joint_pos = np.asarray(qpos[7:], dtype=np.float32)
            body_quat_w = np.asarray(qpos[3:7], dtype=np.float32)

            if prev_joint_pos is None or last_source_stamp_ns is None:
                joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
            else:
                dt = (source_stamp_ns - last_source_stamp_ns) * 1e-9
                joint_vel = (
                    ((joint_pos - prev_joint_pos) / dt).astype(np.float32)
                    if dt > 1e-6
                    else np.zeros_like(joint_pos, dtype=np.float32)
                )

            result = {
                "qpos": qpos,
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "body_quat_w": body_quat_w,
                "timestamp_ns": source_stamp_ns,
                "timestamp_realtime": float(sample.get("timestamp_realtime", 0.0)),
                "timestamp_monotonic": float(sample.get("timestamp_monotonic", 0.0)),
                "retarget_latency_ms": (_time.monotonic() - start_mono) * 1000.0,
                "source_fps": float(sample.get("fps", 0.0)),
            }
            if return_human_motion_data:
                result["human_motion_data"] = retarget.scaled_human_data

            # Keep output queue lean: drop stale results
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except Exception:
                    break
            try:
                output_queue.put_nowait(result)
            except Exception:
                pass

            last_source_stamp_ns = source_stamp_ns
            prev_joint_pos = joint_pos

            now = _time.time()
            if now - last_report >= 5.0:
                print(
                    f"[GMRReader] latency: {result['retarget_latency_ms']:.2f} ms, "
                    f"source_fps: {result['source_fps']:.2f}"
                )
                last_report = now

        except Exception as exc:
            print(f"[GMRProcess] retarget error: {exc}")
            _time.sleep(0.005)


class GMRReader:
    """Multiprocess retarget worker – runs IK on a dedicated CPU core, bypassing the GIL.

    Architecture
    ~~~~~~~~~~~~
    ::

        PicoReader (HW thread)
            │  get_latest()
            ▼
        _feed()  [lightweight daemon thread in main process]
            │  put_nowait()  – keeps only latest sample
            ▼
        input_queue  (mp.Queue)
            │
            ▼
        _gmr_worker()  [subprocess – own Python interpreter, no GIL contention]
            │  put_nowait()  – keeps only latest result
            ▼
        output_queue  (mp.Queue)
            │
            ▼
        get_latest()  [called by GMRPoseStreamer in main process]
    """

    def __init__(
        self,
        reader: PicoReader,
        robot: str = "unitree_g1",
        verbose: bool = False,
        return_human_motion_data: bool = False,
    ):
        self.reader = reader
        # Use fork (Linux default) so child inherits already-loaded .so modules,
        # which avoids re-loading heavy robot URDF/mesh assets from disk.
        self._ctx = mp.get_context("fork")
        self._input_queue = self._ctx.Queue(maxsize=4)
        self._output_queue = self._ctx.Queue(maxsize=4)
        self._stop_event = self._ctx.Event()
        self._latest: dict | None = None

        # Lightweight feeder thread: pushes PicoReader data into the subprocess
        self._feeder_thread = threading.Thread(target=self._feed, daemon=True)

        # Dedicated retarget subprocess
        self._process = self._ctx.Process(
            target=_gmr_worker,
            args=(
                self._input_queue,
                self._output_queue,
                self._stop_event,
                robot,
                verbose,
                return_human_motion_data,
            ),
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()
        self._feeder_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._feeder_thread.join(timeout=1.0)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()

    def get_latest(self) -> dict | None:
        """Drain the output queue and return the most recent retargeted result."""
        while not self._output_queue.empty():
            try:
                self._latest = self._output_queue.get_nowait()
            except Exception:
                break
        return self._latest

    def _feed(self) -> None:
        """Daemon thread: reads PicoReader and forwards new samples to the subprocess."""
        last_stamp: int | None = None
        while not self._stop_event.is_set():
            sample = self.reader.get_latest()
            if sample is None:
                time.sleep(0.001)
                continue

            stamp = int(sample.get("timestamp_ns", 0))
            if stamp <= 0 or stamp == last_stamp:
                time.sleep(0.001)
                continue

            # Drop stale input so subprocess always sees the freshest pose
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except Exception:
                    break
            try:
                self._input_queue.put_nowait(sample)
                last_stamp = stamp
            except Exception:
                pass
            time.sleep(0.001)

class GMRPoseStreamer:
    """Minimal pose-v1 sender that resamples GMR outputs onto a steady time grid."""

    def __init__(
        self,
        socket,
        gmr_reader: GMRReader,
        num_frames_to_send: int,
        target_fps: int,
        record_dir: str = "",
        viewer=None,
        visualize_human_motion: bool = False,
        yaw_vel_ema_alpha: float = 0.3,
    ):
        self.socket = socket
        self.gmr_reader = gmr_reader
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self.viewer = viewer
        self.visualize_human_motion = visualize_human_motion

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

        self.step = 0
        self.frame_time = 0.95 / max(1, target_fps)
        self.frame_start = time.time()
        self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))
        self.toggle_data_collection_last = False
        self.toggle_data_abort_last = False
        self._pending_toggle_data_collection = False
        self._pending_toggle_data_abort = False
        self.action_hand_left = 0.0
        self.action_hand_right = 0.0

        self.prev_stamp_ns = None
        self.prev_joint_pos = None
        self.prev_joint_vel = None
        self.prev_body_quat = None
        self.next_target_ns = None
        self.last_fps_report = time.time()
        self.fps_counter = 0
        self._last_view_stamp_ns = None

        self.real2isaac_idx = G1_REAL_TO_ISAACLAB_IDX

        # QuatProcessor: converts absolute body_quat_w into an accumulated abstract quaternion
        # (roll/pitch directly replaced, yaw accumulated as delta from first frame)
        self.quat_processor = QuatProcessor(yaw_vel_ema_alpha)

        # State machine
        self.current_mode = StreamMode.POSE
        self.prev_left_menu_button = False
        self.pending_start_command = False

    def _check_emergency_stop(self, a_pressed, b_pressed, x_pressed, y_pressed):
        if a_pressed and b_pressed and x_pressed and y_pressed:
            print("[EmergencyStop] A+B+X+Y pressed; sending stop command")
            for _ in range(3):
                self.socket.send(build_command_message(start=False, stop=True, planner=False))
                time.sleep(0.02)
            raise SystemExit

    def run_once(self):
        # State machine control
        left_menu_button, left_trigger, right_trigger, left_grip, _ = get_controller_inputs()
        a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons()
        self._check_emergency_stop(a_pressed, b_pressed, x_pressed, y_pressed)

        new_mode = self.current_mode
        if self.current_mode == StreamMode.POSE:
            if left_menu_button:
                new_mode = StreamMode.POSE_PAUSE
        elif self.current_mode == StreamMode.POSE_PAUSE:
            if not left_menu_button:
                new_mode = StreamMode.POSE

        if new_mode != self.current_mode:
            print(f"[State] {self.current_mode.name} -> {new_mode.name}")

            # Clear state when exiting POSE mode
            if self.current_mode == StreamMode.POSE:
                self.frame_buffer.clear()
                self.prev_stamp_ns = None
                self.prev_joint_pos = None
                self.prev_joint_vel = None
                self.prev_body_quat = None
                self.next_target_ns = None
                self.step = 0
                print("[State] Cleared buffers on POSE exit")

            # Reset yaw when entering POSE mode
            if new_mode == StreamMode.POSE:
                self.quat_processor.reset()
                self.pending_start_command = True
                print("[State] Reset yaw, will send start command after first frame")

            self.current_mode = new_mode
        self.prev_left_menu_button = left_menu_button

        # Pause: skip processing
        if self.current_mode == StreamMode.POSE_PAUSE:
            time.sleep(0.01)
            return

        sample = self.gmr_reader.get_latest()
        if sample is None:
            time.sleep(0.005)
            return

        if self.viewer is not None:
            view_stamp_ns = int(sample["timestamp_ns"])
            if view_stamp_ns != self._last_view_stamp_ns:
                qpos = sample["qpos"]
                human_motion_data = sample.get("human_motion_data") if self.visualize_human_motion else None
                self.viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=human_motion_data,
                    rate_limit=False,
                )
                self._last_view_stamp_ns = view_stamp_ns

        toggle_data_collection_tmp = a_pressed and left_grip > 0.5
        toggle_data_abort_tmp = b_pressed and left_grip > 0.5
        if toggle_data_collection_tmp and not self.toggle_data_collection_last:
            self._pending_toggle_data_collection = True
        if toggle_data_abort_tmp and not self.toggle_data_abort_last:
            self._pending_toggle_data_abort = True
        self.toggle_data_collection_last = toggle_data_collection_tmp
        self.toggle_data_abort_last = toggle_data_abort_tmp

        ## use left_trigger to control action_hand_left 
        # left_trigger > 0.8 → open (0.0), left_trigger <= 0.2 → close (1.0), else hold last state
        if left_trigger > 0.8 and self.action_hand_left == 1.0:
            self.action_hand_left = 0.0
        elif left_trigger <= 0.2 and self.action_hand_left == 0.0:
            self.action_hand_left = 1.0

        if right_trigger > 0.8 and self.action_hand_right == 1.0:
            self.action_hand_right = 0.0
        elif right_trigger <= 0.2 and self.action_hand_right == 0.0:
            self.action_hand_right = 1.0

        curr_stamp_ns = int(sample["timestamp_ns"])
        curr_joint_pos = np.asarray(sample["joint_pos"], dtype=np.float32)
        curr_joint_vel = np.asarray(sample["joint_vel"], dtype=np.float32)
        curr_body_quat = np.asarray(sample["body_quat_w"], dtype=np.float32)
        step_ns = int(1e9 / max(1, self.target_fps))

        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_joint_pos = curr_joint_pos
            self.prev_joint_vel = curr_joint_vel
            self.prev_body_quat = curr_body_quat
            self.next_target_ns = curr_stamp_ns
            return

        if curr_stamp_ns <= self.prev_stamp_ns:
            return

        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        if self.next_target_ns < self.prev_stamp_ns:
            self.next_target_ns = self.prev_stamp_ns
        if self.next_target_ns > curr_stamp_ns:
            return
        if self.prev_joint_pos is None or self.prev_joint_vel is None or self.prev_body_quat is None:
            return

        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0.0 else 1.0
        alpha = float(np.clip(alpha, 0.0, 1.0))

        use_joint_pos = ((1.0 - alpha) * self.prev_joint_pos + alpha * curr_joint_pos).astype(np.float32)
        use_joint_vel = ((1.0 - alpha) * self.prev_joint_vel + alpha * curr_joint_vel).astype(np.float32)
        use_body_quat_raw = _quat_lerp_normalized(self.prev_body_quat, curr_body_quat, alpha)
        # Convert to abstract accumulated quaternion (yaw starts at 0, accumulates delta)
        use_body_quat = self.quat_processor.process(use_body_quat_raw, curr_stamp_ns)

        self.frame_buffer["joint_pos_real"].append(use_joint_pos)
        self.frame_buffer["joint_vel_real"].append(use_joint_vel)
        use_joint_pos_isaac = use_joint_pos[self.real2isaac_idx]
        use_joint_vel_isaac = use_joint_vel[self.real2isaac_idx]
        self.frame_buffer["joint_pos"].append(use_joint_pos_isaac)
        self.frame_buffer["joint_vel"].append(use_joint_vel_isaac)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))

        human_motion_data = sample.get("human_motion_data") if self.visualize_human_motion else None

        buffer_is_full = len(self.frame_buffer["frame_index"]) >= self.num_frames_to_send
        if buffer_is_full:
            n = len(self.frame_buffer["frame_index"])
            numpy_data = {
                "joint_pos_real": np.stack(self.frame_buffer["joint_pos_real"], axis=0),
                "joint_vel_real": np.stack(self.frame_buffer["joint_vel_real"], axis=0),
                "joint_pos": np.stack(self.frame_buffer["joint_pos"], axis=0),
                "joint_vel": np.stack(self.frame_buffer["joint_vel"], axis=0),
                "body_quat_w": np.stack(self.frame_buffer["body_quat_w"], axis=0),
                "frame_index": np.array(self.frame_buffer["frame_index"], dtype=np.int64),
                "action_hand_left": np.array([self.action_hand_left], dtype=np.float32),
                "action_hand_right": np.array([self.action_hand_right], dtype=np.float32),
                "toggle_data_collection": np.array(
                    [self._pending_toggle_data_collection], dtype=bool
                ),
                "toggle_data_abort": np.array([self._pending_toggle_data_abort], dtype=bool),
            }
            if human_motion_data is not None and isinstance(human_motion_data, dict):
                body_order = sorted(PICO_XROBOT_INDEX.keys(), key=lambda k: PICO_XROBOT_INDEX[k])
                flat = np.concatenate(
                    [np.concatenate([np.asarray(human_motion_data[k][0], dtype=np.float32),
                                     np.asarray(human_motion_data[k][1], dtype=np.float32)])
                     for k in body_order if k in human_motion_data]
                )
                numpy_data["human_motion_data"] = flat
            self.socket.send(pack_pose_message(numpy_data, topic="pose", version=1))
            self._pending_toggle_data_collection = False
            self._pending_toggle_data_abort = False

            # Send start command after first frame when resuming from pause
            if self.pending_start_command:
                self.socket.send(build_command_message(start=True, stop=False, planner=False))
                print("[State] Sent start command after first frame")
                self.pending_start_command = False

            if self.record_dir:
                out_path = os.path.join(self.record_dir, f"pose_v1_{self.record_idx:06d}.npz")
                np.savez_compressed(out_path, **numpy_data)
                self.record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_joint_pos = curr_joint_pos
        self.prev_joint_vel = curr_joint_vel
        self.prev_body_quat = curr_body_quat
        self.fps_counter += 1

        now = time.time()
        if now - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (now - self.last_fps_report)
            print(f"[GMRPoseStreamer] FPS: {fps:.2f}, step: {self.step}")
            print(f"  [DEBUG] target_fps={self.target_fps}, frame_time={self.frame_time*1000:.2f}ms, expected_fps={1.0/self.frame_time:.2f}")
            self.fps_counter = 0
            self.last_fps_report = now

        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()


def run_pico_gmr(
    port: int = 5556,
    buffer_size: int = 15,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    record_dir: str = "",
    record_format: str = "npz",
    robot: str = "unitree_g1",
    viewer_fps: int = 30,
    verbose: bool = False,
    visualize: bool = False,
    visualize_human_motion: bool = False,
    yaw_vel_ema_alpha: float = 0.3,
):
    if xrt is None:
        raise ImportError("XRoboToolkit SDK not available. Install xrobotoolkit_sdk first.")

    if record_format != "npz":
        raise ValueError("Only npz recording is supported in this minimal GMR sender.")

    effective_visualize_human_motion = visualize and visualize_human_motion

    if visualize_human_motion and not visualize:
        print("[GMRPoseStreamer] --visualize_human_motion requires --visualize; human overlay is disabled.")

    subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for Pico body tracking data...")
    while not xrt.is_body_data_available():
        time.sleep(1)
        print("waiting for body data...")
        

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"ZMQ PUB bound on tcp://*:{port}")

    reader = PicoReader(max_queue_size=buffer_size)
    reader.start()

    gmr_reader = GMRReader(
        reader=reader,
        robot=robot,
        verbose=verbose,
        return_human_motion_data=effective_visualize_human_motion,
    )
    gmr_reader.start()

    viewer = RobotMotionViewer(robot_type=robot, motion_fps=viewer_fps) if visualize else None
    streamer = GMRPoseStreamer(
        socket=socket,
        gmr_reader=gmr_reader,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        record_dir=record_dir,
        viewer=viewer,
        visualize_human_motion=effective_visualize_human_motion,
        yaw_vel_ema_alpha=yaw_vel_ema_alpha,
    )

    socket.send(build_command_message(start=True, stop=False, planner=False))
    print("[GMRPoseStreamer] Sent initial command(start=True, planner=False)")

    try:
        while True:
            streamer.run_once()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        gmr_reader.stop()
        reader.stop()
        if viewer is not None:
            viewer.close()
        socket.close()
        context.term()
        print("[GMRPoseStreamer] Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15, help="PicoReader buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--num_frames_to_send",
        type=int,
        default=2,
        help="Number of frames per pose packet (default: 5)",
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target send FPS (default: 50)")
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: only 'npz' is supported here (default: npz)",
    )
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--viewer_fps", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable RobotMotionViewer while streaming",
    )
    parser.add_argument(
        "--visualize_human_motion",
        action="store_true",
        help="Overlay GMR scaled human motion in the viewer (requires --visualize)",
    )
    parser.add_argument("--yaw_vel_ema_alpha", default=0.1, type=float, help="EMA alpha for yaw_vel filtering (0-1, higher=less smoothing)")
    
    args = parser.parse_args()

    run_pico_gmr(
        port=args.port,
        buffer_size=args.buffer_size,
        num_frames_to_send=args.num_frames_to_send,
        target_fps=args.target_fps,
        record_dir=args.record_dir,
        record_format=args.record_format,
        robot=args.robot,
        viewer_fps=args.viewer_fps,
        verbose=args.verbose,
        visualize=args.visualize,
        visualize_human_motion=args.visualize_human_motion,
        yaw_vel_ema_alpha=args.yaw_vel_ema_alpha,
    )
