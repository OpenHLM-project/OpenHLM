from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from mink import Configuration
from mink.tasks import ComTask, FrameTask

from .keypoints import (
    HandRootFootKeyPoints,
    load_target_from_frame,
)


@dataclass(frozen=True)
class FrameErrorNorm:
    """Norm of a frame task error (meters for position, radians for orientation)."""

    position: float
    orientation: float

    def to_json(self) -> dict:
        return {
            "position": self.position,
            "orientation": self.orientation,
        }

    @classmethod
    def from_configuration_and_task(
        cls,
        configuration: Configuration,
        task: FrameTask,
    ) -> FrameErrorNorm:
        """Compute frame-task error norms for the provided configuration/task pair."""
        task_error = task.compute_error(configuration=configuration)
        position_error = np.linalg.norm(task_error[:3])
        orientation_error = np.linalg.norm(task_error[3:])
        return FrameErrorNorm(
            position=float(position_error),
            orientation=float(orientation_error),
        )

    @classmethod
    def from_json(cls, data: dict) -> FrameErrorNorm:
        return cls(
            position=float(data.get("position", 0.0)),
            orientation=float(data.get("orientation", 0.0)),
        )


@dataclass(frozen=True)
class IKSolution:
    """Solved joint configuration plus achieved end-effector errors."""

    q: np.ndarray
    target: HandRootFootKeyPoints
    transformed_target: HandRootFootKeyPoints
    realized_target: HandRootFootKeyPoints
    errors: dict[str, FrameErrorNorm]
    com_target: np.ndarray
    realized_com: np.ndarray

    @classmethod
    def from_configuration_and_tasks(
        cls,
        configuration: Configuration,
        left_hand_task: FrameTask,
        right_hand_task: FrameTask,
        left_foot_task: FrameTask,
        right_foot_task: FrameTask,
        pelvis_task: FrameTask | None,
        com_task: ComTask,
        target: HandRootFootKeyPoints,
        transformed_target: HandRootFootKeyPoints,
    ) -> Self:
        """Create an `IKSolution` snapshot from the current Mink configuration."""
        q_sol = configuration.q.copy()

        assert pelvis_task is not None, "Pelvis task must be provided"

        errors: dict[str, FrameErrorNorm] = {
            "left_hand": FrameErrorNorm.from_configuration_and_task(
                configuration=configuration, task=left_hand_task
            ),
            "right_hand": FrameErrorNorm.from_configuration_and_task(
                configuration=configuration, task=right_hand_task
            ),
            "left_foot": FrameErrorNorm.from_configuration_and_task(
                configuration=configuration,
                task=left_foot_task,
            ),
            "right_foot": FrameErrorNorm.from_configuration_and_task(
                configuration=configuration,
                task=right_foot_task,
            ),
            "root": FrameErrorNorm.from_configuration_and_task(
                configuration=configuration, task=pelvis_task
            ),
        }

        assert com_task.target_com is not None, "CoM target must be set"
        com_target = com_task.target_com.copy()

        realized_left_hand = configuration.get_transform_frame_to_world(
            frame_name="left_hand_site",
            frame_type="site",
        )
        realized_right_hand = configuration.get_transform_frame_to_world(
            frame_name="right_hand_site",
            frame_type="site",
        )
        realized_left_foot = configuration.get_transform_frame_to_world(
            frame_name="left_ankle_pitch_link",
            frame_type="body",
        )
        realized_right_foot = configuration.get_transform_frame_to_world(
            frame_name="right_ankle_pitch_link",
            frame_type="body",
        )
        realized_pelvis = configuration.get_transform_frame_to_world(
            frame_name="pelvis",
            frame_type="body",
        )

        realized_target = HandRootFootKeyPoints(
            left_hand_pose=realized_left_hand,
            right_hand_pose=realized_right_hand,
            left_foot_pose=realized_left_foot,
            right_foot_pose=realized_right_foot,
            root_pose=realized_pelvis,
            timestamp=getattr(target, "timestamp", 0.0),
        )

        realized_com = configuration.data.subtree_com[1].copy()

        return cls(
            q=q_sol,
            target=target,
            transformed_target=transformed_target,
            realized_target=realized_target,
            errors=errors,
            com_target=com_target,
            realized_com=realized_com,
        )

    def to_frame_dict(self) -> dict:
        """Serialize the IK solution by extending the original target frame."""
        frame = self.target.to_json_serializable()
        frame["q"] = self.q.tolist()
        if self.errors:
            frame["error"] = {
                key: err.to_json() for key, err in self.errors.items()
            }
        frame["com_target"] = self.com_target.tolist()
        realized_target = self.realized_target
        transformed_target = self.transformed_target
        transformed_frame = transformed_target.to_json_serializable()
        frame["transformed_target"] = transformed_frame
        realized_frame = realized_target.to_json_serializable()
        frame["realized_target"] = realized_frame
        frame["realized_com"] = self.realized_com.tolist()
        return frame

    @classmethod
    def from_frame_dict(cls, frame: dict) -> IKSolution:
        """Decode an IK solution that extends a target frame."""
        realized_target = load_target_from_frame(frame["realized_target"])
        q = np.array(frame["q"], dtype=np.float64)
        error_block = frame.get("error", {})
        errors = {
            key: FrameErrorNorm.from_json(err_dict)
            for key, err_dict in error_block.items()
        }
        com_target = np.array(
            frame.get("com_target", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        realized_com = np.array(
            frame.get("realized_com", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        target = load_target_from_frame(frame)
        transformed_target = load_target_from_frame(
            frame["transformed_target"]
        )
        return IKSolution(
            target=target,
            q=q,
            realized_target=realized_target,
            transformed_target=transformed_target,
            errors=errors,
            com_target=com_target,
            realized_com=realized_com,
        )

    def _get_error(self, key: str) -> FrameErrorNorm:
        return self.errors.get(
            key, FrameErrorNorm(position=0.0, orientation=0.0)
        )

    @property
    def left_hand_error(self) -> FrameErrorNorm:
        return self._get_error("left_hand")

    @property
    def right_hand_error(self) -> FrameErrorNorm:
        return self._get_error("right_hand")

    @property
    def left_foot_error(self) -> FrameErrorNorm:
        return self._get_error("left_foot")

    @property
    def right_foot_error(self) -> FrameErrorNorm:
        return self._get_error("right_foot")

    @property
    def timestamp(self) -> float:
        return getattr(self.target, "timestamp", 0.0)


__all__ = [
    "FrameErrorNorm",
    "IKSolution",
]
