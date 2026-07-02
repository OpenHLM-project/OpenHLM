from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser
from mink import SE3
from viser.extras import ViserUrdf

from ikumi.keypoints import (
    HandRootFootKeyPoints,
)
from ikumi.solution import IKSolution


@dataclass(frozen=True)
class ViserConfig:
    urdf_path: Path = Path(
        "g1_description/g1_29dof_rev_1_0.urdf"
    )  # Path to URDF model file for Viser.
    port: int = 8732  # Port for Viser server.

    robot_default_height: float = (
        0.70  # Default height of robot base from ground (in meters).
    )

    def create(self) -> "ViserVisualizer":
        """Create a ViserVisualizer from this config."""
        return ViserVisualizer(config=self)


class ViserVisualizer:
    """Visualizer to show current targets and configurations using Viser."""

    def __init__(self, config: ViserConfig):
        self._config = config

        self.server = viser.ViserServer(port=self._config.port)
        # set up scene
        server = self.server
        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = True
        server.scene.add_grid(
            "/grid",
            plane="xy",
            shadow_opacity=0.0,
            infinite_grid=True,
            cell_size=0.25,
        )

        # set up robot urdf model
        self._robot_base_frame = server.scene.add_frame(
            name="/robot/g1",
            show_axes=True,
            origin_color=(0.8, 0.8, 0.8),
            axes_length=0.1,
            axes_radius=0.004,
        )
        self._robot = ViserUrdf(
            target=server,
            urdf_or_path=self._config.urdf_path,
            root_node_name="/robot/g1",
        )

        # set up frames to show targets
        self._left_hand_target_frame = server.scene.add_frame(
            name="/targets/left_hand",
            show_axes=True,
            origin_color=(1.0, 0.2, 0.2),
            axes_length=0.08,
            axes_radius=0.003,
        )
        self._right_hand_target_frame = server.scene.add_frame(
            name="/targets/right_hand",
            show_axes=True,
            origin_color=(0.2, 0.2, 1.0),
            axes_length=0.08,
            axes_radius=0.003,
        )
        self._left_foot_target_frame = server.scene.add_frame(
            name="/targets/left_foot",
            show_axes=True,
            origin_color=(0.2, 1.0, 0.2),
            axes_length=0.08,
            axes_radius=0.003,
        )
        self._right_foot_target_frame = server.scene.add_frame(
            name="/targets/right_foot",
            show_axes=True,
            origin_color=(1.0, 1.0, 0.2),
            axes_length=0.08,
            axes_radius=0.003,
        )
        self._com_target_frame = server.scene.add_frame(
            name="/targets/com",
            show_axes=True,
            origin_color=(1.0, 0.2, 1.0),
            axes_length=0.08,
            axes_radius=0.003,
        )
        self._root_target_frame = server.scene.add_frame(
            name="/targets/root",
            show_axes=True,
            origin_color=(0.9, 0.4, 0.1),
            axes_length=0.08,
            axes_radius=0.003,
        )
        self._target_frames = [
            self._left_hand_target_frame,
            self._right_hand_target_frame,
            self._left_foot_target_frame,
            self._right_foot_target_frame,
            self._com_target_frame,
            self._root_target_frame,
        ]
        self._realized_com = server.scene.add_icosphere(
            name="/targets/realized_com", color=(0.5, 0.5, 0.5), radius=0.02
        )

        # gui
        self._info_format = """
        # {title} \n
        {description}
        """
        self._info_text = server.gui.add_markdown(
            content=self._info_format.format(
                title="IK Visualizer",
                description="Visualizing IK targets and robot configuration.",
            )
        )

        self.reset()

        # GUI: error sliders for hands (read-only)
        with server.gui.add_folder("IK Errors", order=1):
            # Position errors (meters)
            self._left_hand_pos_err = server.gui.add_slider(
                "Left Hand Position (m)",
                min=0.0,
                max=0.03,
                step=0.0001,
                initial_value=0.0,
                disabled=True,
            )
            self._right_hand_pos_err = server.gui.add_slider(
                "Right Hand Position (m)",
                min=0.0,
                max=0.03,
                step=0.0001,
                initial_value=0.0,
                disabled=True,
            )
            # Orientation errors (radians)
            self._left_hand_rot_err = server.gui.add_slider(
                "Left Hand Rotation (deg)",
                min=0.0,
                max=5.0,
                step=0.1,
                initial_value=0.0,
                disabled=True,
            )
            self._right_hand_rot_err = server.gui.add_slider(
                "Right Hand Rotation (deg)",
                min=0.0,
                max=5.0,
                step=0.1,
                initial_value=0.0,
                disabled=True,
            )

    def reset(self):
        """Reset robot to default configuration and hide target frames."""
        with self.server.atomic():
            self._robot.update_cfg(
                np.zeros(len(self._robot.get_actuated_joint_names()))
            )
            self._robot_base_frame.position = np.array(
                [0.0, 0.0, self._config.robot_default_height]
            )
            self._robot_base_frame.wxyz = np.array(
                [1.0, 0.0, 0.0, 0.0]
            )  # no rotation
            self._robot_base_frame.visible = True

            for frame in self._target_frames:
                frame.visible = False

    def _set_robot_visible(self, visible: bool) -> None:
        """Toggle visibility for the whole robot tree."""
        self._robot_base_frame.visible = visible

    def _set_frame_pose(
        self, frame: viser.FrameHandle, pose: SE3 | None
    ) -> None:
        if pose is None:
            frame.visible = False
            return
        frame.visible = True
        frame.position = pose.translation()
        frame.wxyz = pose.rotation().wxyz

    def _update_target_frames(
        self,
        target: HandRootFootKeyPoints,
        com_target: np.ndarray | None,
        realized_com: np.ndarray | None = None,
    ) -> None:
        self._set_frame_pose(
            self._left_hand_target_frame,
            getattr(target, "left_hand_pose", None),
        )
        self._set_frame_pose(
            self._right_hand_target_frame,
            getattr(target, "right_hand_pose", None),
        )
        self._set_frame_pose(
            self._left_foot_target_frame,
            getattr(target, "left_foot_pose", None),
        )
        self._set_frame_pose(
            self._right_foot_target_frame,
            getattr(target, "right_foot_pose", None),
        )
        root_pose = getattr(target, "root_pose", None)
        self._set_frame_pose(self._root_target_frame, root_pose)

        if com_target is not None:
            self._com_target_frame.visible = True
            self._com_target_frame.position = com_target
            self._com_target_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self._com_target_frame.visible = False

        if realized_com is not None:
            self._realized_com.visible = True
            realized_com_xy = realized_com.copy()
            realized_com_xy[2] = 0.0  # project to ground for better visibility
            self._realized_com.position = realized_com_xy
        else:
            self._realized_com.visible = False

    def visualize_target(self, target: HandRootFootKeyPoints):
        """Only display tracker targets without changing the robot configuration."""
        with self.server.atomic():
            self._set_robot_visible(False)
            self._update_target_frames(target, com_target=None)

    def visualize_ik_solution(self, ik_solution: IKSolution):
        """Visualize targets frames and robot configuration from IK solution."""
        with self.server.atomic():
            self._set_robot_visible(True)
            # update robot configuration
            self._robot.update_cfg(
                ik_solution.q[7:]
            )  # skip first 7 dofs (floating base)
            self._robot_base_frame.position = ik_solution.q[0:3]
            self._robot_base_frame.wxyz = ik_solution.q[3:7]

            # update targets and COM
            self._update_target_frames(
                ik_solution.transformed_target,
                ik_solution.com_target,
                ik_solution.realized_com,
            )

            # update error sliders (left/right hands)
            try:
                # Position errors: meters, clipped to [0, 0.03]
                left_pos = float(ik_solution.left_hand_error.position)
                right_pos = float(ik_solution.right_hand_error.position)
                self._left_hand_pos_err.value = max(0.0, min(0.03, left_pos))
                self._right_hand_pos_err.value = max(0.0, min(0.03, right_pos))

                # Rotation errors: radians -> degrees, clipped to [0, 5]
                left_rot_deg = float(
                    np.degrees(ik_solution.left_hand_error.orientation)
                )
                right_rot_deg = float(
                    np.degrees(ik_solution.right_hand_error.orientation)
                )
                self._left_hand_rot_err.value = max(
                    0.0, min(5.0, left_rot_deg)
                )
                self._right_hand_rot_err.value = max(
                    0.0, min(5.0, right_rot_deg)
                )
            except Exception:
                # In case GUI elements are not available, skip updating
                pass

    def update_info_text(self, content: str, title: str = "IK Visualizer"):
        """Update the info text in the GUI."""
        self._info_text.content = self._info_format.format(
            title=title,
            description=content,
        )
