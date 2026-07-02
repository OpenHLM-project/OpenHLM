from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

import numpy as np
import numpy.typing as npt
import openvr  # type: ignore
from rich.table import Table
from scipy.spatial.transform import Rotation as R

# Default role ordering used for ZMQ payloads
DEFAULT_ROLES: list[str] = [
    "root",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
]


@dataclass(frozen=True)
class PoseFrame:
    serial_number: str
    timestamp: float
    position: npt.NDArray[np.float64]
    quaternion_wxyz: npt.NDArray[np.float64]

    @classmethod
    def read_from_vr_system(
        cls, vr_system: openvr.IVRSystem, time_func: Callable[[], float]
    ) -> list["PoseFrame"]:
        poses = vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        current_time = time_func()
        frames = []
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bPoseIsValid:
                device_class = vr_system.getTrackedDeviceClass(i)
                if device_class != openvr.TrackedDeviceClass_GenericTracker:
                    continue
                serial_number = vr_system.getStringTrackedDeviceProperty(
                    i, openvr.Prop_SerialNumber_String
                )
                m34 = poses[i].mDeviceToAbsoluteTracking
                pose_matrix = np.array(
                    [
                        [m34[0][0], m34[0][1], m34[0][2], m34[0][3]],
                        [m34[1][0], m34[1][1], m34[1][2], m34[1][3]],
                        [m34[2][0], m34[2][1], m34[2][2], m34[2][3]],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                position = pose_matrix[:3, 3]
                quaternion_wxyz = R.from_matrix(pose_matrix[:3, :3]).as_quat(
                    scalar_first=True  # type: ignore
                )
                frames.append(
                    cls(
                        serial_number=serial_number,
                        timestamp=current_time,
                        position=position,
                        quaternion_wxyz=quaternion_wxyz,
                    )
                )
        return frames


def make_table(
    poses: list[PoseFrame], serial_to_role: dict[str, str]
) -> Table:
    table = Table(title="Connected Trackers")
    table.add_column("Role", justify="center", style="cyan")
    table.add_column("Timestamp", justify="center", style="magenta")
    table.add_column("Position (m)", justify="center", style="green")
    table.add_column("Orientation (wxyz)", justify="center", style="yellow")

    for pose in poses:
        role = serial_to_role.get(
            pose.serial_number, f"Unknown ({pose.serial_number})"
        )
        if role == "ground":
            role += " (raw)"
        table.add_row(
            f"{role.replace('_', ' ').title()}",
            f"{datetime.fromtimestamp(pose.timestamp).strftime('%H:%M:%S.%f')[:-3]}",
            f"({pose.position[0]:+.3f}, {pose.position[1]:+.3f}, {pose.position[2]:+.3f})",
            f"({pose.quaternion_wxyz[0]:+.3f}, {pose.quaternion_wxyz[1]:+.3f}, {pose.quaternion_wxyz[2]:+.3f}, {pose.quaternion_wxyz[3]:+.3f})",
        )
    return table


# Transform matrices consolidated in a mapping for clarity
_TX_OROBOT_OTRACKER = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

_ROLE_TO_TX_TRACKER = {
    "root": np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, -1.0, -0.082],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "left_hand": np.array(
        [
            [-1.0, 0.0, 0.0, -0.155],
            [0.0, 0.0, 1.0, -0.009],
            [0.0, 1.0, 0.0, -0.042],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "right_hand": np.array(
        [
            [-1.0, 0.0, 0.0, -0.155],
            [0.0, 0.0, 1.0, -0.009],
            [0.0, 1.0, 0.0, -0.042],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "left_foot": np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "right_foot": np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "ground": np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}


def transform_poses(
    poses: list[PoseFrame],
    serial_to_role: dict[str, str],
) -> list[PoseFrame]:
    poses_transformed = []
    for pose in poses:
        role = serial_to_role[pose.serial_number]
        rot_otracker_tracker = R.from_quat(
            pose.quaternion_wxyz,
            scalar_first=True,  # type: ignore
        ).as_matrix()
        tx_otracker_tracker = np.eye(4)
        tx_otracker_tracker[:3, :3] = rot_otracker_tracker
        tx_otracker_tracker[:3, 3] = pose.position

        if role not in _ROLE_TO_TX_TRACKER:
            raise ValueError(f"Unknown role: {role}")
        tx_tracker_role = _ROLE_TO_TX_TRACKER[role]

        tx_orobot_role = (
            _TX_OROBOT_OTRACKER @ tx_otracker_tracker @ tx_tracker_role
        )
        rot_orobot_role = tx_orobot_role[:3, :3]
        quat_orobot_role = R.from_matrix(rot_orobot_role).as_quat(
            scalar_first=True  # type: ignore
        )
        pos_orobot_role = tx_orobot_role[:3, 3]

        poses_transformed.append(
            replace(
                pose,
                position=pos_orobot_role,
                quaternion_wxyz=quat_orobot_role,
            )
        )

    return poses_transformed


def offset_ground_height(
    poses: list[PoseFrame],
    serial_to_role: dict[str, str],
) -> list[PoseFrame]:
    """Offset the height of all poses by the ground height."""
    role_to_pose = {serial_to_role[pose.serial_number]: pose for pose in poses}
    ground_pose = role_to_pose.get("ground", None)
    if ground_pose is None:
        raise ValueError("Ground pose not found in poses.")
    ground_height = ground_pose.position[2]

    for role, pose in role_to_pose.items():
        if role == "ground":
            continue
        new_position = pose.position.copy()
        new_position[2] -= ground_height
        role_to_pose[role] = replace(pose, position=new_position)

    return list(role_to_pose.values())


@dataclass(frozen=True)
class PoseData:
    """
    Minimal structure used for ZMQ messages. Keep field names stable.
    pos: [num_roles, 3]
    quat_wxyz: [num_roles, 4]
    """

    pos: list[list[float]]
    quat_wxyz: list[list[float]]

    @classmethod
    def from_pose_frames(
        cls,
        pose_frames: list[PoseFrame],
        roles_to_send: list[str],
        serial_to_role: dict[str, str],
    ) -> "PoseData":
        """Create PoseData from a list of PoseFrames.
        Args:
            pose_frames (list[PoseFrame]): List of PoseFrame objects.
            roles_to_send (list[str]): List of roles to include in the PoseData.
            serial_to_role (dict[str, str]): Mapping from tracker serial numbers to roles.
        Returns:
            PoseData: The resulting PoseData object.
        """
        pos_dict: dict[str, list[float]] = {}
        quat_dict: dict[str, list[float]] = {}
        for pose in pose_frames:
            role = serial_to_role[pose.serial_number]
            if role in roles_to_send:
                pos_dict[role] = pose.position.tolist()
                quat_dict[role] = pose.quaternion_wxyz.tolist()

        for role in roles_to_send:
            if role not in pos_dict:
                raise ValueError(f"Role {role} not found in pose frames.")

        return cls(
            pos=[pos_dict[role] for role in roles_to_send],
            quat_wxyz=[quat_dict[role] for role in roles_to_send],
        )
