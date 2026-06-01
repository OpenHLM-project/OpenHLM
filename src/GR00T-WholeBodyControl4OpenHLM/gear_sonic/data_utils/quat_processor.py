import numpy as np
from scipy.spatial.transform import Rotation as R

class QuatProcessor:
    """Helper class for processing quaternions.

    Convention: all quaternions are wxyz.

    Maintains an abstract accumulated Euler state starting at [roll=0, pitch=0, yaw=0].
    Each call to ``process`` does:
      1. Convert the incoming wxyz quaternion to [roll, pitch, raw_yaw].
      2. Compute delta_yaw = raw_yaw - prev_raw_yaw  (wrapped to [-pi, pi]).
      3. Update state: roll/pitch replaced directly; yaw += delta_yaw.
      4. Convert the accumulated [roll, pitch, yaw] back to a wxyz quaternion and return it.
    """

    def __init__(self, yaw_vel_ema_alpha: float = 0.1):
        # Accumulated abstract Euler state (radians)
        self._roll: float = 0.0
        self._pitch: float = 0.0
        self._yaw: float = 0.0
        # Previous raw yaw from the real quaternion stream (for delta computation)
        self._prev_raw_yaw: float | None = None
        self._prev_timestamp_ns: int | None = None
        # EMA filter for yaw_vel
        self._yaw_vel_ema_alpha: float = yaw_vel_ema_alpha
        self._yaw_vel_ema: float | None = None

    def reset(self):
        """Reset accumulated state to zero."""
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
        self._prev_raw_yaw = None
        self._prev_timestamp_ns = None
        self._yaw_vel_ema = None

    def reset_delta_tracking(self):
        """Reset delta tracking without clearing accumulated yaw. Use when resuming from pause."""
        self._prev_raw_yaw = None
        self._prev_timestamp_ns = None
        self._yaw_vel_ema = None

    def process(self, quat_wxyz: np.ndarray, timestamp_ns: int, return_yaw_vel: bool = False) -> np.ndarray:
        """Process one quaternion and return the abstract accumulated quaternion (wxyz).

        Args:
            quat_wxyz: shape (4,), format wxyz.
            timestamp_ns: timestamp in nanoseconds.
            return_yaw_vel: if True, return [roll, pitch, yaw_vel] instead of wxyz quaternion.

        Returns:
            np.ndarray: shape (4,) wxyz quaternion if return_yaw_vel=False, else shape (3,) [roll, pitch, yaw_vel].
        """
        roll, pitch, raw_yaw = self._quat_to_euler(quat_wxyz)

        if self._prev_raw_yaw is None:
            self._prev_timestamp_ns = timestamp_ns
            self._prev_raw_yaw = raw_yaw
            yaw_vel = 0.0
            self._yaw_vel_ema = 0.0
        else:
            if self._prev_timestamp_ns is None:
                delta_time = 0.0
            else:
                delta_time = (timestamp_ns - self._prev_timestamp_ns) * 1e-9
            yaw_vel_raw = self._wrap_angle(raw_yaw - self._prev_raw_yaw) / delta_time if delta_time > 1e-6 else 0.0
            # Apply EMA filter
            if self._yaw_vel_ema is None:
                self._yaw_vel_ema = yaw_vel_raw
            else:
                self._yaw_vel_ema = self._yaw_vel_ema_alpha * yaw_vel_raw + (1 - self._yaw_vel_ema_alpha) * self._yaw_vel_ema
            yaw_vel = self._yaw_vel_ema
            # Add clip for _yaw_vel to prevent extreme jumps (e.g., due to tracking loss)
            yaw_vel = float(np.clip(yaw_vel, -2.0, 2.0))
            self._yaw += yaw_vel * delta_time
            self._prev_timestamp_ns = timestamp_ns
            self._prev_raw_yaw = raw_yaw

        self._roll = roll
        self._pitch = pitch
        return np.array([roll, pitch, yaw_vel], dtype=np.float32) if return_yaw_vel else self._euler_to_quat(roll, pitch, self._yaw)

    def process_output_yaw_vel(self, quat_wxyz: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Process quaternion and return euler with yaw velocity instead of accumulated yaw."""
        return self.process(quat_wxyz, timestamp_ns, return_yaw_vel=True)

    def process_input_yaw_vel(self, roll: float, pitch: float, yaw_vel: float, timestamp_ns: int) -> np.ndarray:
        """Directly input roll, pitch, and yaw velocity to update state.

        Args:
            roll: roll angle in radians.
            pitch: pitch angle in radians.
            yaw_vel: yaw velocity in radians/second.
            timestamp_ns: timestamp in nanoseconds.

        Returns:
            np.ndarray shape (4,) float32, format wxyz.
        """
        self._roll = roll
        self._pitch = pitch
        if self._prev_timestamp_ns is None:
            self._prev_timestamp_ns = timestamp_ns
            delta_time = 0.0
        else:
            delta_time = (timestamp_ns - self._prev_timestamp_ns) * 1e-9
            self._prev_timestamp_ns = timestamp_ns
        self._yaw += yaw_vel * delta_time
        return self._euler_to_quat(self._roll, self._pitch, self._yaw)

    @staticmethod
    def _quat_to_euler(quat_wxyz: np.ndarray) -> np.ndarray:
        """Convert wxyz quaternion to (roll, pitch, yaw) via 'xyz' intrinsic Euler angles."""
        w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
        # scipy uses xyzw convention
        rot = R.from_quat([x, y, z, w])
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert (roll, pitch, yaw) 'xyz' intrinsic Euler angles to wxyz quaternion."""
        rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        x, y, z, w = rot.as_quat()  # scipy returns xyzw
        return np.array([w, x, y, z], dtype=np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
