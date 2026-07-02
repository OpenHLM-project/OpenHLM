import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation


def pos_rot_to_pose(pos: np.ndarray, rot: st.Rotation) -> np.ndarray:
    """Convert position and rotation to a 6D pose representation.
    Args:
        pos: (..., 3) array of positions
        rot: scipy Rotation object representing rotations
    Returns:
        pose: (..., 6) array of poses (first 3 are position, last 3 are rotation axis-angle)
    """
    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def mat_to_pos_rot(
    mat: np.ndarray,
) -> tuple[np.ndarray, st.Rotation]:
    """Convert a 4x4 transformation matrix to position and rotation.
    Args:
        mat: (..., 4, 4) array of transformation matrices
    Returns:
        tuple:
            - pos: (..., 3) array of positions
            - rot: scipy Rotation object representing rotations
    """
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot = st.Rotation.from_matrix(mat[..., :3, :3])
    return pos, rot


def mat_to_pose(
    mat: np.ndarray,
) -> np.ndarray:
    """Convert a 4x4 transformation matrix to a 6D pose representation.
    Args:
        mat: (..., 4, 4) array of transformation matrices
    Returns:
        pose: (..., 6) array of poses (first 3 are position, last 3 are rotation axis-angle)
    """
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def get_interp1d(t: np.ndarray, x: np.ndarray):
    gripper_interp = si.interp1d(
        t,
        x,
        axis=0,
        bounds_error=False,
        fill_value=(x[0], x[-1]),  # type: ignore
    )
    return gripper_interp


class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:, :3]
        rot = st.Rotation.from_rotvec(x[:, 3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)

    @property
    def x(self):
        return self.pos_interp.x

    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose


def get_gripper_calibration_interpolator(
    aruco_measured_width, aruco_actual_width
):
    """
    Assumes the minimum width in aruco_actual_width
    is measured when the gripper is fully closed
    and maximum width is when the gripper is fully opened
    """
    aruco_measured_width = np.array(aruco_measured_width)
    aruco_actual_width = np.array(aruco_actual_width)
    assert len(aruco_measured_width) == len(aruco_actual_width)
    assert len(aruco_actual_width) >= 2
    aruco_min_width = np.min(aruco_actual_width)
    gripper_actual_width = aruco_actual_width - aruco_min_width
    interp = get_interp1d(aruco_measured_width, gripper_actual_width)
    return interp


def get_tcp_pose_interpolator(df):
    timestamp_sec = df["timestamp"].to_numpy()

    pos = df[["x", "y", "z"]].to_numpy()

    quat_xyzw = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy()
    rot = Rotation.from_quat(quat_xyzw)

    pose_matrix = np.zeros((pos.shape[0], 4, 4), dtype=np.float32)
    pose_matrix[:, 3, 3] = 1
    pose_matrix[:, :3, 3] = pos
    pose_matrix[:, :3, :3] = rot.as_matrix()

    # Create interpolator
    pose_interp = PoseInterpolator(t=timestamp_sec, x=mat_to_pose(pose_matrix))

    return pose_interp
