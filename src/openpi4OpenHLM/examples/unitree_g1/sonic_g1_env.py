# flake8: noqa: E501
import json
import threading
import time
from collections import defaultdict, deque

import msgpack
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R
# from gear_sonic.data_utils.vision_client import HeadZMQClient, WristZMQClient

########################################################
# copy and paste the following code to SONIC codebase
########################################################

_REAL_TO_ISAAC_IDX = np.array([
    0, 6, 12,   # left_hip_pitch, right_hip_pitch, waist_yaw
    1, 7, 13,   # left_hip_roll, right_hip_roll, waist_roll
    2, 8, 14,   # left_hip_yaw, right_hip_yaw, waist_pitch
    3, 9,       # left_knee, right_knee
    15, 22,     # left_shoulder_pitch, right_shoulder_pitch
    4, 10,      # left_ankle_pitch, right_ankle_pitch
    16, 23,     # left_shoulder_roll, right_shoulder_roll
    5, 11,      # left_ankle_roll, right_ankle_roll
    17, 24,     # left_shoulder_yaw, right_shoulder_yaw
    18, 25,     # left_elbow, right_elbow
    19, 26,     # left_wrist_roll, right_wrist_roll
    20, 27,     # left_wrist_pitch, right_wrist_pitch
    21, 28,     # left_wrist_yaw, right_wrist_yaw
], dtype=np.int64)

# Head camera image shape (height, width, channels)
HEAD_IMAGE_HEIGHT = 400
HEAD_IMAGE_WIDTH = 464
HEAD_IMAGE_CHANNELS = 3

# Wrist camera image shape (height, width, channels)
WRIST_IMAGE_HEIGHT = 480
WRIST_IMAGE_WIDTH = 640
WRIST_IMAGE_CHANNELS = 3

# Default robot observation: both arms hanging down vertically (rest position)
# 32 dims total: 3 (root: roll, pitch, yaw angular velocity) + 29 (joints)
DEFAULT_MIMIC_OBS_G1 = np.concatenate([
    np.array([0, 0, 0]),  # roll, pitch, yaw angular velocity
    np.array([
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,      # left leg (6)
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,      # right leg (6)
        0.0, 0.0, 0.0,                       # waist (3)
        0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,   # left arm (7)
        0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,  # right arm (7)
    ])
])

# Default robot observation: both arms raised up (active/ready position)
# 32 dims total: 3 (root: roll, pitch, yaw angular velocity) + 29 (joints)
DEFAULT_MIMIC_OBS_G1_RAISED = np.concatenate([
    np.array([0, 0, 0]),  # roll, pitch, yaw angular velocity
    np.array([
        0.05221500247716904,
        0.035042937844991684,
        0.14586453139781952,
        0.14763712882995605,
        -0.09440571814775467,
        -0.050025228410959244,  # left leg (6)
        0.04669266939163208,
        -0.10003259032964706,
        -0.21058756113052368,
        0.16901911795139313,
        -0.1263013780117035,
        0.08077848702669144,    # right leg (6)
        -0.015427463687956333,
        -0.030686339363455772,
        0.0030055493116378784,  # waist (3)
        0.02636529505252838,
        0.4650239050388336,
        -0.07923969626426697,
        0.06137121841311455,
        -0.3445344865322113,
        -0.0475073866546154,
        -0.11823924630880356,  # left arm (7)
        0.05132843554019928,
        -0.4928153157234192,
        0.15939019620418549,
        -0.051064785569906235,
        0.4574618339538574,
        -0.009871167130768299,
        0.26929032802581787    # right arm (7)
    ])
])


# ---------------------------------------------------------------------------
# ZMQ message packing utilities
# ---------------------------------------------------------------------------

_HEADER_SIZE = 1280


def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    """Build a fixed-size JSON header for ZMQ wire-format messages."""
    header = {"v": version, "endian": "le", "count": count, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > _HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {_HEADER_SIZE}")
    return header_json.ljust(_HEADER_SIZE, b"\x00")


def pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 1) -> bytes:
    """Pack a dict of numpy arrays into a ZMQ wire-format message.

    Message layout: [topic_bytes][fixed JSON header][packed binary payload].

    Args:
        pose_data: Dict mapping field names to numpy arrays.
        topic: Topic prefix string.
        version: Protocol version written into the header.

    Returns:
        Packed message bytes ready to send via ``socket.send()``.
    """
    fields = []
    binary_parts = []

    _DTYPE_MAP = {
        np.float32: "f32",
        np.float64: "f64",
        np.int32: "i32",
        np.int64: "i64",
        np.bool_: "bool",
    }

    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        dtype_str = _DTYPE_MAP.get(value.dtype.type, None)
        if dtype_str is None:
            value = value.astype(np.float32)
            dtype_str = "f32"
        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        # Ensure little-endian byte order
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        binary_parts.append(value.tobytes())

    header_bytes = _build_header(fields, version=version, count=1)
    return topic.encode("utf-8") + header_bytes + b"".join(binary_parts)


# ---------------------------------------------------------------------------
# Helpers: quaternion <-> euler conversions
# ---------------------------------------------------------------------------

def _euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles (radians) to a wxyz quaternion (float32)."""
    rot = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    x, y, z, w = rot.as_quat()  # scipy returns xyzw
    return np.array([w, x, y, z], dtype=np.float32)


def _quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """Convert a wxyz quaternion to intrinsic XYZ Euler angles (radians).

    Args:
        quat_wxyz: shape (4,), format [w, x, y, z].

    Returns:
        (roll, pitch, yaw) in radians.
    """
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    rot = R.from_quat([x, y, z, w])  # scipy uses xyzw
    roll, pitch, yaw = rot.as_euler("xyz", degrees=False)
    return roll, pitch, yaw


class SonicG1Env:
    def __init__(
        self,
        control_hz: int = 30,
        mock: bool = False,
        vision_server_address: str = '192.168.123.164',  # g1
        vision_server_port: int = 5555,
        wrist_server_port: int = 5554,
        action_zmq_address: str = '*',
        action_zmq_port: int = 5556,
        body_state_address: str = '192.168.123.222',  # PC
        body_state_port: int = 5557,
        hand_state_address: str = '192.168.123.164',  # g1
        hand_state_port: int = 5558,
        num_frames_to_send: int = 2,
    ):
        """
        Initialize the SonicG1Env.

        Args:
            control_hz: Control frequency in Hz.
            mock: Whether to run in mock mode (no real robot connection).
            vision_server_address: IP address of the vision server (head camera).
            vision_server_port: Port number of the vision server.
            wrist_server_port: Port number of the wrist camera server (default: 5554).
            action_zmq_address: Address to bind the ZMQ action publisher socket.
                Use '*' to bind on all interfaces (default), or a specific IP.
            action_zmq_port: Port for the ZMQ action publisher socket (default: 5556).
            body_state_address: IP address of the PC running the C++ deploy node
                (publishes body state on the "g1_debug" topic via msgpack).
            body_state_port: ZMQ port for the body state stream (default: 5557).
            hand_state_address: IP address of the robot orin publishing hand state
                (msgpack, no topic prefix).
            hand_state_port: ZMQ port for the hand state stream (default: 5558).
            num_frames_to_send: Number of frames to buffer before sending a single ZMQ message (default: 2).
        """
        self.mock = mock
        self.control_hz = control_hz
        self._dt = 1.0 / control_hz
        self.action_dim = 34  # OpenPI format: 7+1+7+1+6+6+3+3
        self.state_dim = 34

        # State tracking for action transformation
        self._prev_joint_pos: np.ndarray | None = None
        self._accumulated_yaw: float = 0.0
        self._frame_index: int = 0

        # Multi-frame action buffer: accumulates frames and sends when full
        self._num_frames_to_send = num_frames_to_send
        self._action_frame_buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=num_frames_to_send))

        # State tracking for observation (yaw integration from measured quaternion)
        self._prev_state_yaw: float | None = None
        self._latest_body_state: dict | None = None
        self._latest_hand_state: dict | None = None

        # ZMQ context (shared by all sockets)
        self._zmq_context = zmq.Context()

        # ZMQ PUB socket for sending actions to the robot controller
        self._zmq_socket = self._zmq_context.socket(zmq.PUB)
        if not self.mock:
            self._zmq_socket.bind(f"tcp://{action_zmq_address}:{action_zmq_port}")
            print(f"ZMQ PUB bound on tcp://{action_zmq_address}:{action_zmq_port}")

        # ZMQ SUB socket for body state ("g1_debug" topic, msgpack)
        # Published by the C++ deploy node; contains base_quat_measured and body_q_measured.
        self._body_state_sub = self._zmq_context.socket(zmq.SUB)
        self._body_state_sub.setsockopt_string(zmq.SUBSCRIBE, "g1_debug")
        self._body_state_sub.setsockopt(zmq.CONFLATE, 1)  # keep only the latest message
        if not self.mock:
            self._body_state_sub.connect(f"tcp://{body_state_address}:{body_state_port}")
            print(f"ZMQ SUB connected to body state at {body_state_address}:{body_state_port}")

        # ZMQ SUB socket for hand state (no topic, msgpack)
        # Published by the orin hand controller; contains state_hand_left/right.
        self._hand_state_sub = self._zmq_context.socket(zmq.SUB)
        self._hand_state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._hand_state_sub.setsockopt(zmq.CONFLATE, 1)
        if not self.mock:
            self._hand_state_sub.connect(f"tcp://{hand_state_address}:{hand_state_port}")
            print(f"ZMQ SUB connected to hand state at {hand_state_address}:{hand_state_port}")

        num_cameras = 2
        self.image_shape = (HEAD_IMAGE_HEIGHT, HEAD_IMAGE_WIDTH * num_cameras, HEAD_IMAGE_CHANNELS)
        self.wrist_image_shape = (WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH * num_cameras, WRIST_IMAGE_CHANNELS)

        # Initialize vision clients. The clients keep their own latest-image cache.
        if not self.mock:
            self.vision_client = HeadZMQClient(
                server_address=vision_server_address,
                port=vision_server_port,
                image_shape=self.image_shape,
                image_show=False,
            )

            self.vision_thread = threading.Thread(target=self.vision_client.receive_process, daemon=True)
            self.vision_thread.start()
        else:
            self.vision_client = None
            self.vision_thread = None

        # Initialize wrist camera client.
        if not self.mock:
            self.wrist_client = WristZMQClient(
                server_address=vision_server_address,
                port=wrist_server_port,
                image_shape=self.wrist_image_shape,
                image_show=False
            )

            self.wrist_thread = threading.Thread(target=self.wrist_client.receive_process, daemon=True)
            self.wrist_thread.start()
        else:
            self.wrist_client = None
            self.wrist_thread = None

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'vision_client') and self.vision_client is not None:
            self.vision_client.stop()
        if hasattr(self, 'vision_thread') and self.vision_thread is not None:
            self.vision_thread.join(timeout=1.0)

        if hasattr(self, 'wrist_client') and self.wrist_client is not None:
            self.wrist_client.stop()
        if hasattr(self, 'wrist_thread') and self.wrist_thread is not None:
            self.wrist_thread.join(timeout=1.0)

        for attr in ('_zmq_socket', '_body_state_sub', '_hand_state_sub'):
            sock = getattr(self, attr, None)
            if sock is not None:
                sock.close()
        if hasattr(self, '_zmq_context') and self._zmq_context is not None:
            self._zmq_context.term()

    def _send_action_zmq(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        body_quat_w: np.ndarray,
        action_hand_left: float,
        action_hand_right: float,
    ) -> None:
        """Pack and publish a multi-frame action via ZMQ.

        Frames are buffered in a deque of size ``num_frames_to_send``. A ZMQ
        message is only sent once the buffer is full, so each message contains
        exactly ``num_frames_to_send`` consecutive frames stacked along axis 0.

        All joint arrays follow robot format order:
          left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7) = 29 dims.
        body_quat_w is in wxyz format.

        Args:
            joint_pos: Joint positions, shape (29,).
            joint_vel: Joint velocities, shape (29,).
            body_quat_w: Body orientation quaternion wxyz, shape (4,).
            action_hand_left: Left hand gripper command (scalar).
            action_hand_right: Right hand gripper command (scalar).
        """
        # Reorder joints from real robot order to IsaacLab order before buffering
        joint_pos_isaac = joint_pos[_REAL_TO_ISAAC_IDX]
        joint_vel_isaac = joint_vel[_REAL_TO_ISAAC_IDX]

        buf = self._action_frame_buffer
        buf["joint_pos"].append(joint_pos_isaac.astype(np.float32))
        buf["joint_vel"].append(joint_vel_isaac.astype(np.float32))
        buf["body_quat_w"].append(body_quat_w.astype(np.float32))
        buf["frame_index"].append(self._frame_index)
        buf["action_hand_left"].append(float(action_hand_left))
        buf["action_hand_right"].append(float(action_hand_right))

        self._frame_index += 1

        if len(buf["frame_index"]) < self._num_frames_to_send:
            return

        numpy_data = {
            "joint_pos": np.stack(buf["joint_pos"], axis=0),        # (N, 29)
            "joint_vel": np.stack(buf["joint_vel"], axis=0),        # (N, 29)
            "body_quat_w": np.stack(buf["body_quat_w"], axis=0),    # (N, 4)
            "frame_index": np.array(buf["frame_index"], dtype=np.int64),  # (N,)
            # Hand actions are scalar open/close commands, not per-frame trajectories,
            # so only the latest value is sent with shape (1,) rather than (N,).
            "action_hand_left": np.array([buf["action_hand_left"][-1]], dtype=np.float32),
            "action_hand_right": np.array([buf["action_hand_right"][-1]], dtype=np.float32),
        }
        try:
            self._zmq_socket.send(pack_pose_message(numpy_data, topic="pose", version=1))
        except Exception as e:
            print(f"Error sending action via ZMQ: {e}")

    def _poll_state_zmq(self) -> None:
        """Drain the latest body and hand state messages from ZMQ (non-blocking).

        Updates ``self._latest_body_state`` and ``self._latest_hand_state`` caches.
        Both sockets use CONFLATE=1, so at most one message is buffered per socket.
        """
        # Body state: topic prefix "g1_debug", remainder is msgpack payload
        _BODY_TOPIC = b"g1_debug"
        try:
            raw = self._body_state_sub.recv(zmq.NOBLOCK)
            self._latest_body_state = msgpack.unpackb(raw[len(_BODY_TOPIC):], raw=False)
        except zmq.Again:
            pass
        except Exception as e:
            print(f"Error reading body state from ZMQ: {e}")

        # Hand state: no topic prefix, full payload is msgpack
        try:
            raw = self._hand_state_sub.recv(zmq.NOBLOCK)
            self._latest_hand_state = msgpack.unpackb(raw, raw=False)
        except zmq.Again:
            pass
        except Exception as e:
            print(f"Error reading hand state from ZMQ: {e}")

    def reset(self, default_pose=None):
        """
        Reset the environment to the specified default pose.

        Args:
            default_pose: Target pose array (32 dims: 3 root + 29 joints) to reset to.
                         If None, uses DEFAULT_MIMIC_OBS_G1 (arms down position).

        Returns:
            dict: Observation after reset.
        """
        if default_pose is None:
            default_pose = DEFAULT_MIMIC_OBS_G1

        if self.mock:
            return self.get_observation()

        # Get current joint positions from the latest ZMQ body state
        self._poll_state_zmq()
        if self._latest_body_state is None or "body_q_measured" not in self._latest_body_state:
            print("Warning: No body state available for reset. Skipping reset motion.")
            return self.get_observation()

        # body_q_measured: 29 joint positions in robot format order
        # left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)
        current_joints = np.array(self._latest_body_state["body_q_measured"], dtype=np.float32)

        # default_pose is 32 dims: 0-2 root (3), 3-31 joints (29)
        target_joints = default_pose[3:32]
        root_action = default_pose[0:3]  # [roll, pitch, yaw_vel]

        # During reset the root command is [0, 0, 0], so accumulated_yaw stays unchanged.
        roll = float(root_action[0])
        pitch = float(root_action[1])
        yaw_vel = float(root_action[2])
        body_quat_w = _euler_xyz_to_quat_wxyz(roll, pitch, self._accumulated_yaw)

        # Interpolate over 2 seconds
        duration = 2.0
        steps = int(duration * self.control_hz)
        prev_joints = current_joints.copy()

        for i in range(steps):
            alpha = (i + 1) / steps
            interp_joints = (current_joints * (1 - alpha) + target_joints * alpha).astype(np.float32)

            joint_vel = (interp_joints - prev_joints) / self._dt
            prev_joints = interp_joints.copy()

            # Accumulate yaw (zero during default reset, but honour whatever is passed)
            self._accumulated_yaw += yaw_vel * self._dt
            body_quat_w = _euler_xyz_to_quat_wxyz(roll, pitch, self._accumulated_yaw)

            self._send_action_zmq(
                joint_pos=interp_joints,
                joint_vel=joint_vel,
                body_quat_w=body_quat_w,
                action_hand_left=1.0,
                action_hand_right=1.0,
            )
            self._prev_joint_pos = interp_joints.copy()

            time.sleep(self._dt)

        # Hold the final frame for 0.3 s to let the robot settle at the target pose.
        hold_steps = max(1, round(0.3 / self._dt))
        final_body_quat_w = _euler_xyz_to_quat_wxyz(roll, pitch, self._accumulated_yaw)
        for _ in range(hold_steps):
            self._send_action_zmq(
                joint_pos=interp_joints,
                joint_vel=np.zeros_like(interp_joints),
                body_quat_w=final_body_quat_w,
                action_hand_left=1.0,
                action_hand_right=1.0,
            )
            time.sleep(self._dt)

        return self.get_observation()

    def step(self, action):
        """
        Execute a single action in the environment.

        Args:
            action: numpy array with shape (34,) in OpenPI format:
                0-6:   left arm (7)
                7:     left gripper (1)
                8-14:  right arm (7)
                15:    right gripper (1)
                16-21: left leg (6)
                22-27: right leg (6)
                28-30: waist (3)
                31-33: root (3) -- roll, pitch, yaw angular velocity

        Action transformation (OpenPI -> robot format):
          - Joint positions (29 dims, robot order): left_leg, right_leg, waist, left_arm, right_arm
          - Joint velocities: finite difference from previous joint positions / dt
          - Body orientation: roll and pitch taken directly from action[31:33];
            yaw is accumulated by integrating yaw_angular_velocity * dt from 0 at startup.
            The three Euler angles are then converted to a wxyz quaternion.
        """
        action = np.asarray(action)

        if action.shape != (self.action_dim,):
            raise ValueError(f"Action shape must be ({self.action_dim},), got {action.shape}")

        # Decompose OpenPI action format
        left_arm = action[0:7]        # 7 dims
        left_gripper = float(action[7])
        right_arm = action[8:15]      # 7 dims
        right_gripper = float(action[15])
        left_leg = action[16:22]      # 6 dims
        right_leg = action[22:28]     # 6 dims
        waist = action[28:31]         # 3 dims
        root = action[31:34]          # 3 dims: roll, pitch, yaw_angular_velocity

        # 29 joint positions in robot format order
        joint_pos = np.concatenate([
            left_leg,   # 6
            right_leg,  # 6
            waist,      # 3
            left_arm,   # 7
            right_arm   # 7
        ]).astype(np.float32)

        # Joint velocities via finite difference
        if self._prev_joint_pos is None:
            joint_vel = np.zeros_like(joint_pos)
        else:
            joint_vel = (joint_pos - self._prev_joint_pos) / self._dt
        self._prev_joint_pos = joint_pos.copy()

        # Body orientation: roll/pitch direct, yaw accumulated
        roll = float(root[0])
        pitch = float(root[1])
        yaw_vel = float(root[2])
        self._accumulated_yaw += yaw_vel * self._dt
        body_quat_w = _euler_xyz_to_quat_wxyz(roll, pitch, self._accumulated_yaw)

        if not self.mock:
            self._send_action_zmq(
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                body_quat_w=body_quat_w,
                action_hand_left=left_gripper,
                action_hand_right=right_gripper,
            )

    def get_observation(self):
        """
        Get the current observation and transform it.

        Returns:
            dict: Observation dictionary with:
                - "head_image_left": head camera left image
                - "left_wrist_image": left wrist camera image
                - "right_wrist_image": right wrist camera image
                - "state": robot state in OpenPI format (34 dims)
        """
        # Get head camera image (left camera from stereo pair)
        if self.mock:
            head_image_left = np.random.randint(0, 255, (HEAD_IMAGE_HEIGHT, HEAD_IMAGE_WIDTH, HEAD_IMAGE_CHANNELS), dtype=np.uint8)
        else:
            stereo_img = self.vision_client.get_latest_image() if self.vision_client is not None else None
            if stereo_img is not None and stereo_img.size > 0:
                if stereo_img.shape[1] % 2 == 0:
                    half_w = stereo_img.shape[1] // 2
                    # Vision clients return OpenCV BGR frames; OpenPI expects RGB.
                    head_image_left = stereo_img[:, :half_w, ::-1].copy()
                else:
                    print("Warning: stereo image width not even")
                    head_image_left = np.zeros((HEAD_IMAGE_HEIGHT, HEAD_IMAGE_WIDTH, HEAD_IMAGE_CHANNELS), dtype=np.uint8)
            else:
                head_image_left = np.zeros((HEAD_IMAGE_HEIGHT, HEAD_IMAGE_WIDTH, HEAD_IMAGE_CHANNELS), dtype=np.uint8)

        # Get wrist camera images
        if self.mock:
            left_wrist_image = np.random.randint(0, 255, (WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH, WRIST_IMAGE_CHANNELS), dtype=np.uint8)
            right_wrist_image = np.random.randint(0, 255, (WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH, WRIST_IMAGE_CHANNELS), dtype=np.uint8)
        else:
            wrist_stereo_img = self.wrist_client.get_latest_image() if self.wrist_client is not None else None
            if wrist_stereo_img is not None and wrist_stereo_img.size > 0:
                if wrist_stereo_img.shape[1] % 2 == 0:
                    wrist_half_w = wrist_stereo_img.shape[1] // 2
                    # Vision clients return OpenCV BGR frames; OpenPI expects RGB.
                    left_wrist_image = wrist_stereo_img[:, :wrist_half_w, ::-1].copy()
                    right_wrist_image = wrist_stereo_img[:, wrist_half_w:, ::-1].copy()
                else:
                    print("Warning: wrist stereo image width not even")
                    left_wrist_image = np.zeros((WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH, WRIST_IMAGE_CHANNELS), dtype=np.uint8)
                    right_wrist_image = np.zeros((WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH, WRIST_IMAGE_CHANNELS), dtype=np.uint8)
            else:
                left_wrist_image = np.zeros((WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH, WRIST_IMAGE_CHANNELS), dtype=np.uint8)
                right_wrist_image = np.zeros((WRIST_IMAGE_HEIGHT, WRIST_IMAGE_WIDTH, WRIST_IMAGE_CHANNELS), dtype=np.uint8)

        if self.mock:
            state = np.random.uniform(-1.0, 1.0, size=(self.state_dim,)).astype(np.float32)
        else:
            state = np.zeros(self.state_dim, dtype=np.float32)
            self._poll_state_zmq()
            try:
                if self._latest_body_state is not None:
                    # Convert measured base quaternion (wxyz) to Euler angles
                    # then compute yaw angular velocity via delta_yaw / dt
                    #
                    # body state from "g1_debug" (msgpack):
                    #   base_quat_measured: wxyz quaternion of the robot body
                    #   body_q_measured:    29 joint positions in robot format order
                    #     left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)
                    quat_wxyz = np.array(self._latest_body_state["base_quat_measured"], dtype=np.float64)
                    roll, pitch, yaw = _quat_wxyz_to_euler_xyz(quat_wxyz)

                    if self._prev_state_yaw is not None:
                        delta_yaw = float(yaw - self._prev_state_yaw)
                        # Wrap delta to [-pi, pi] to handle discontinuities near ±pi
                        delta_yaw = (delta_yaw + np.pi) % (2.0 * np.pi) - np.pi
                        yaw_vel = delta_yaw / self._dt
                    else:
                        yaw_vel = 0.0
                    self._prev_state_yaw = yaw

                    state_root = np.array([roll, pitch, yaw_vel], dtype=np.float32)  # 3 dims

                    body_q = np.array(self._latest_body_state["body_q_measured"], dtype=np.float32)
                    state_leg_left  = body_q[0:6]    # 6 dims
                    state_leg_right = body_q[6:12]   # 6 dims
                    state_waist     = body_q[12:15]  # 3 dims
                    state_arm_left  = body_q[15:22]  # 7 dims
                    state_arm_right = body_q[22:29]  # 7 dims

                    # Hand state from hand controller (msgpack, no topic prefix)
                    state_hand_left = np.zeros(1, dtype=np.float32)
                    state_hand_right = np.zeros(1, dtype=np.float32)
                    if self._latest_hand_state is not None:
                        hl = self._latest_hand_state.get("state_hand_left")
                        if hl is not None:
                            v = hl[0] if isinstance(hl, (list, tuple)) else float(hl)
                            state_hand_left = np.array([float(v)], dtype=np.float32)
                        hr = self._latest_hand_state.get("state_hand_right")
                        if hr is not None:
                            v = hr[0] if isinstance(hr, (list, tuple)) else float(hr)
                            state_hand_right = np.array([float(v)], dtype=np.float32)

                    # Construct OpenPI State (34 dims)
                    # 0-6: left arm (7), 7: left gripper (1),
                    # 8-14: right arm (7), 15: right gripper (1),
                    # 16-21: left leg (6), 22-27: right leg (6),
                    # 28-30: waist (3), 31-33: root (3)
                    state = np.concatenate([
                        state_arm_left,     # 7
                        state_hand_left,    # 1
                        state_arm_right,    # 7
                        state_hand_right,   # 1
                        state_leg_left,     # 6
                        state_leg_right,    # 6
                        state_waist,        # 3
                        state_root          # 3
                    ])  # Total: 34

            except Exception as e:
                print(f"Error reading state from ZMQ: {e}")

        obs = {
            "head_image_left": head_image_left,
            "left_wrist_image": left_wrist_image,
            "right_wrist_image": right_wrist_image,
            "state": state
        }

        return obs
