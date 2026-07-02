import enum
import json
import pathlib
from dataclasses import dataclass
from typing import NotRequired, TypedDict

import pandas as pd


class PoseDict(TypedDict):
    position: list[float]
    quaternion_xyzw: list[float]


class TargetDict(TypedDict):
    timestamp: float
    left_hand_pose: PoseDict
    right_hand_pose: PoseDict
    root_pose: PoseDict
    left_foot_pose: PoseDict
    right_foot_pose: PoseDict


class EpFrameDict(TypedDict):
    timestamp: float
    realized_target: TargetDict
    q: list[float]


class TrajPayload(TypedDict):
    episode: list[EpFrameDict]
    mjcf_path: str
    prompt: NotRequired[str]


def load_prompt_from_json(json_path: pathlib.Path) -> str | None:
    with open(json_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    prompt = payload.get("prompt")
    if prompt is None:
        return None
    return str(prompt)


@dataclass(frozen=True)
class Pose:
    position: list[float]
    quaternion_xyzw: list[float]

    @classmethod
    def from_dict(cls, data: PoseDict) -> "Pose":
        return cls(
            position=data["position"],
            quaternion_xyzw=data["quaternion_xyzw"],
        )


@dataclass(frozen=True)
class KeypointFrame:
    timestamp: float
    left_hand_pose: Pose
    right_hand_pose: Pose
    root_pose: Pose
    left_foot_pose: Pose
    right_foot_pose: Pose

    @classmethod
    def from_dict(cls, data: TargetDict) -> "KeypointFrame":
        return cls(
            timestamp=data["timestamp"],
            left_hand_pose=Pose.from_dict(data["left_hand_pose"]),
            right_hand_pose=Pose.from_dict(data["right_hand_pose"]),
            root_pose=Pose.from_dict(data["root_pose"]),
            left_foot_pose=Pose.from_dict(data["left_foot_pose"]),
            right_foot_pose=Pose.from_dict(data["right_foot_pose"]),
        )


def load_trajectory_from_json(json_path: pathlib.Path) -> list[KeypointFrame]:
    with open(json_path) as f:
        traj_payload: TrajPayload = json.load(f)

    keypoint_frames: list[KeypointFrame] = []
    for ep in traj_payload["episode"]:
        target: TargetDict = ep["realized_target"]
        keypoint_frame = KeypointFrame.from_dict(target)
        keypoint_frames.append(keypoint_frame)

    return keypoint_frames


class RoleEnum(enum.Enum):
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    ROOT = "root"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"


def traj_to_pd_dataframe(
    traj: list[KeypointFrame],
    role: RoleEnum,
) -> pd.DataFrame:
    """Convert a list of KeypointFrame to a pandas DataFrame for a specific role.
    DataFrame heads:
        frame_idx, timestamp, x, y, z, q_x, q_y, q_z, q_w

    Args:
        traj: List of KeypointFrame objects.
        role: Role to extract ('left_hand', 'right_hand', 'root', 'left_foot', 'right_foot').

    Returns:
        pd.DataFrame: DataFrame containing the trajectory data for the specified role.
    """
    data = []
    for frame_idx, keypoint_frame in enumerate(traj):
        pose: Pose
        if role == RoleEnum.LEFT_HAND:
            pose = keypoint_frame.left_hand_pose
        elif role == RoleEnum.RIGHT_HAND:
            pose = keypoint_frame.right_hand_pose
        elif role == RoleEnum.ROOT:
            pose = keypoint_frame.root_pose
        elif role == RoleEnum.LEFT_FOOT:
            pose = keypoint_frame.left_foot_pose
        elif role == RoleEnum.RIGHT_FOOT:
            pose = keypoint_frame.right_foot_pose
        else:
            raise ValueError(f"Unknown role: {role}")

        row = {
            "frame_idx": frame_idx,
            "timestamp": keypoint_frame.timestamp,
            "x": pose.position[0],
            "y": pose.position[1],
            "z": pose.position[2],
            "q_x": pose.quaternion_xyzw[0],
            "q_y": pose.quaternion_xyzw[1],
            "q_z": pose.quaternion_xyzw[2],
            "q_w": pose.quaternion_xyzw[3],
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


__all__ = [
    "PoseDict",
    "TargetDict",
    "EpFrameDict",
    "TrajPayload",
    "load_prompt_from_json",
    "Pose",
    "KeypointFrame",
    "load_trajectory_from_json",
    "RoleEnum",
    "traj_to_pd_dataframe",
]
