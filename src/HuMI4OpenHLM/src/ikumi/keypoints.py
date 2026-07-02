from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mink import SE3, SO3


@dataclass(frozen=True)
class HandRootFootKeyPoints:
    """Tracker poses for both wrists, ankles, and the pelvis tracker."""

    left_hand_pose: SE3
    right_hand_pose: SE3
    left_foot_pose: SE3
    right_foot_pose: SE3
    root_pose: SE3
    timestamp: float = 0.0

    def transformed(
        self,
        transform: SE3,
        left_tracker_to_foot_transform: SE3,
        right_tracker_to_foot_transform: SE3,
    ) -> HandRootFootKeyPoints:
        return HandRootFootKeyPoints(
            left_hand_pose=transform.multiply(self.left_hand_pose),
            right_hand_pose=transform.multiply(self.right_hand_pose),
            left_foot_pose=transform.multiply(self.left_foot_pose).multiply(
                left_tracker_to_foot_transform
            ),
            right_foot_pose=transform.multiply(self.right_foot_pose).multiply(
                right_tracker_to_foot_transform
            ),
            root_pose=transform.multiply(self.root_pose),
            timestamp=self.timestamp,
        )

    @property
    def feet_midpoint_pose(self) -> SE3:
        mid_translation = 0.5 * (
            self.left_foot_pose.translation()
            + self.right_foot_pose.translation()
        )
        mid_rotation = self.left_foot_pose.rotation().interpolate(
            self.right_foot_pose.rotation(), 0.5
        )
        return SE3.from_rotation_and_translation(
            rotation=mid_rotation,
            translation=mid_translation,
        )

    def to_json_serializable(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "left_hand_pose": _se3_to_pose_dict(self.left_hand_pose),
            "right_hand_pose": _se3_to_pose_dict(self.right_hand_pose),
            "left_foot_pose": _se3_to_pose_dict(self.left_foot_pose),
            "right_foot_pose": _se3_to_pose_dict(self.right_foot_pose),
            "root_pose": _se3_to_pose_dict(self.root_pose),
        }

    @classmethod
    def from_json_frame(cls, frame: dict) -> HandRootFootKeyPoints:
        return cls(
            left_hand_pose=_pose_dict_to_se3(frame["left_hand_pose"]),
            right_hand_pose=_pose_dict_to_se3(frame["right_hand_pose"]),
            left_foot_pose=_pose_dict_to_se3(frame["left_foot_pose"]),
            right_foot_pose=_pose_dict_to_se3(frame["right_foot_pose"]),
            root_pose=_pose_dict_to_se3(frame["root_pose"]),
            timestamp=float(frame.get("timestamp", 0.0)),
        )


def load_target_from_frame(frame: dict) -> HandRootFootKeyPoints:
    """Decode a single frame into the target structure."""
    required_keys = (
        "left_hand_pose",
        "right_hand_pose",
        "left_foot_pose",
        "right_foot_pose",
        "root_pose",
    )
    missing = [key for key in required_keys if key not in frame]
    if missing:
        raise ValueError(f"Frame missing required keys {missing}.")
    return HandRootFootKeyPoints.from_json_frame(frame)


def _pose_dict_to_se3(pose: dict) -> SE3:
    """
    Convert a pose dict into SE3, accepting either quaternion key/order:
    - "quaternion_wxyz": [w, x, y, z]
    - "quaternion_xyzw": [x, y, z, w]
    """
    translation = np.array(pose["position"], dtype=np.float64)
    if "quaternion_wxyz" in pose:
        quat_wxyz = np.array(pose["quaternion_wxyz"], dtype=np.float64)
        if quat_wxyz.shape != (4,):
            raise ValueError("quaternion_wxyz must have 4 elements.")
        wxyz = quat_wxyz
    elif "quaternion_xyzw" in pose:
        quat_xyzw = np.array(pose["quaternion_xyzw"], dtype=np.float64)
        if quat_xyzw.shape != (4,):
            raise ValueError("quaternion_xyzw must have 4 elements.")
        wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
    else:
        raise KeyError(
            "Pose dict missing 'quaternion_wxyz' or 'quaternion_xyzw'"
        )
    rotation = SO3(wxyz=wxyz)
    return SE3.from_rotation_and_translation(
        rotation=rotation,
        translation=translation,
    )


def _se3_to_pose_dict(se3: SE3) -> dict:
    """Convert an SE3 into a pose dict (xyzw for compatibility)."""
    translation = se3.translation().tolist()
    wxyz = se3.rotation().wxyz
    quaternion_xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    return {
        "position": translation,
        "quaternion_xyzw": quaternion_xyzw,
    }


__all__ = [
    "HandRootFootKeyPoints",
    "load_target_from_frame",
]
