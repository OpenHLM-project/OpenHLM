from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from mink import SE3, SO3, Configuration, solve_ik
from mink.limits import (
    CollisionAvoidanceLimit,
    ConfigurationLimit,
    VelocityLimit,
)
from mink.tasks import BaseTask, ComTask, FrameTask, PostureTask

from .config import IKConfig
from .episodes import load_ik_episode, load_target_episode
from .keypoints import (
    HandRootFootKeyPoints,
)
from .solution import IKSolution
from .utils import geom_group_from_bodies


class IKSolver:
    """Inverse kinematics solver for G1 using HandRootFoot mode."""

    def __init__(self, config: IKConfig):
        self._config = config

        # Load Mujoco model and initialize the nominal configuration.
        self.model = mujoco.MjModel.from_xml_path(str(config.mjcf_path))
        self._ground_plane_geom_id = self._ensure_ground_plane_at_origin()
        self.configuration = self.home_configuration
        self._home_q = self.configuration.q.copy()
        self._home_com_z = float(self.configuration.data.subtree_com[1][2])
        self._home_feet_to_pelvis_offset = (
            self._compute_home_feet_to_pelvis_offset()
        )

        # Build frame tasks for the tracked end effectors.
        self._left_hand_task = self._build_frame_task(
            frame_name="left_hand_site", frame_type="site"
        )
        self._right_hand_task = self._build_frame_task(
            frame_name="right_hand_site", frame_type="site"
        )
        self._left_foot_task = self._build_frame_task(
            frame_name="left_ankle_pitch_link",
            frame_type="body",
        )
        self._right_foot_task = self._build_frame_task(
            frame_name="right_ankle_pitch_link",
            frame_type="body",
        )
        # Build posture task.
        self._posture_task = self._build_posture_task()
        # Build COM task and initialize target.
        self._com_task = self._build_com_task()

        # Build pelvis task
        self._pelvis_task = self._build_pelvis_task()
        self._torso_upright_task = self._build_torso_upright_task()

        # Aggregate tasks for the solver loop.
        self._tasks: list[BaseTask] = [
            self._left_hand_task,
            self._right_hand_task,
            self._left_foot_task,
            self._right_foot_task,
            self._posture_task,
            self._com_task,
            self._pelvis_task,
        ]
        if self._torso_upright_task is not None:
            self._tasks.append(self._torso_upright_task)

        # Configure joint-limit constraints.
        self._configuration_limit = ConfigurationLimit(
            self.model,
            gain=self._config.joint_limit_gain,
            min_distance_from_limits=self._config.joint_limit_margin,
        )
        # Configure collision-avoidance constraints.
        self._collision_limit = self._build_collision_limit()
        self._velocity_limit = self._build_velocity_limit()
        self._limits = [
            self._configuration_limit,
            self._velocity_limit,
        ]
        if not self._config.disable_collision:
            self._limits.append(self._collision_limit)

        # Track per-episode transforms.
        self._reference_transform: SE3 = SE3.identity()
        self._left_tracker_to_foot_transform: SE3 = SE3.identity()
        self._right_tracker_to_foot_transform: SE3 = SE3.identity()
        self._root_z_scale_factor: float = 1.0
        self._tracker_to_root_transform: SE3 = SE3.identity()

    @property
    def config(self) -> IKConfig:
        """Return the IK configuration used by this solver."""
        return self._config

    @property
    def home_configuration(self) -> Configuration:
        """Return a fresh copy of the stand keyframe configuration."""
        configuration = Configuration(self.model)
        if self._config.home_keyframe is not None:
            configuration.update_from_keyframe(self._config.home_keyframe)
        return configuration

    def _get_postrue_configuration(self) -> Configuration:
        """Return a fresh copy of the posture keyframe configuration."""
        configuration = Configuration(self.model)
        if self._config.posture_keyframe is not None:
            configuration.update_from_keyframe(self._config.posture_keyframe)
        return configuration

    def _build_frame_task(
        self,
        frame_name: str,
        frame_type: Literal["site", "body"],
    ) -> FrameTask:
        """Build a frame task for the requested frame with config-defined weights."""
        match frame_name:
            case "left_hand_site" | "right_hand_site":
                position_cost = self._config.hand_position_cost
                orientation_cost = self._config.hand_orientation_cost
                gain = self._config.hand_gain
                lm_damping = self._config.hand_damping
            case "left_ankle_pitch_link" | "right_ankle_pitch_link":
                position_cost = self._config.foot_position_cost
                orientation_cost = self._config.foot_orientation_cost
                gain = self._config.foot_gain
                lm_damping = self._config.foot_damping
            case _:
                raise ValueError(f"Unsupported frame name: {frame_name}")
        task = FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        target_pose = self.configuration.get_transform_frame_to_world(
            frame_name=frame_name,
            frame_type=frame_type,
        )

        task.set_target(target_pose)
        return task

    def _build_posture_task(self) -> PostureTask:
        """Build the posture task that keeps all joints near the nominal pose."""
        costs = np.full(
            self.model.nv, self._config.posture_cost, dtype=np.float64
        )
        if self._config.waist_posture_cost is not None:
            for joint_name in ("waist_roll_joint", "waist_pitch_joint"):
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                if joint_id == -1:
                    raise ValueError(f"Joint not found in model: {joint_name}")
                dof_adr = int(self.model.jnt_dofadr[joint_id])
                joint_type = self.model.jnt_type[joint_id]
                dof_width = 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
                costs[dof_adr : dof_adr + dof_width] = float(
                    self._config.waist_posture_cost
                )

        task = PostureTask(
            model=self.model,
            cost=costs,
            gain=self._config.posture_gain,
            lm_damping=self._config.posture_damping,
        )

        posture_q = self._get_postrue_configuration().q.copy()

        task.set_target(posture_q)

        return task

    def _build_com_task(self) -> ComTask:
        """Build the COM regulation task that projects the COM between both feet."""
        cost = np.array(
            [
                self._config.com_xy_cost,
                self._config.com_xy_cost,
                self._config.com_z_cost,
            ],
            dtype=np.float64,
        )
        task = ComTask(
            cost=cost,
            gain=self._config.com_gain,
            lm_damping=self._config.com_damping,
        )
        left_foot_pose = self.configuration.get_transform_frame_to_world(
            frame_name="left_ankle_pitch_link",
            frame_type="body",
        )
        right_foot_pose = self.configuration.get_transform_frame_to_world(
            frame_name="right_ankle_pitch_link",
            frame_type="body",
        )
        feet_midpoint_pos = 0.5 * (
            left_foot_pose.translation() + right_foot_pose.translation()
        )
        current_com = self.configuration.data.subtree_com[1].copy()
        com_target = np.array(
            [
                feet_midpoint_pos[0],
                feet_midpoint_pos[1],
                current_com[2],
            ]
        )
        task.set_target(com_target)
        # Initialize COM task target only.
        return task

    def _build_pelvis_task(self) -> FrameTask:
        """Build the pelvis tracking task."""
        pos_cost = np.array(
            [
                self._config.pelvis_xy_cost,
                self._config.pelvis_xy_cost,
                self._config.pelvis_z_cost,
            ],
            dtype=np.float64,
        )
        task = FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=pos_cost,
            orientation_cost=self._config.pelvis_orientation_cost,
            gain=self._config.pelvis_gain,
            lm_damping=self._config.pelvis_damping,
        )
        pelvis_pose = self.configuration.get_transform_frame_to_world(
            frame_name="pelvis",
            frame_type="body",
        )
        task.set_target(pelvis_pose)
        return task

    def _build_torso_upright_task(self) -> FrameTask | None:
        """Build an optional torso orientation task that prefers upright posture."""
        if self._config.torso_upright_orientation_cost <= 0.0:
            return None
        task = FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=np.zeros(3, dtype=np.float64),
            orientation_cost=self._config.torso_upright_orientation_cost,
            gain=self._config.torso_upright_gain,
            lm_damping=self._config.torso_upright_damping,
        )
        torso_pose = self.configuration.get_transform_frame_to_world(
            frame_name="torso_link",
            frame_type="body",
        )
        task.set_target(self._make_upright_pose(torso_pose))
        return task

    def _ensure_ground_plane_at_origin(self) -> int:
        """
        Ensure the model has a ground plane geom at z=0 and return its geom id.

        The MJCF files already include a plane named "floor", but we enforce its
        presence and placement so collision checks stay consistent if the model
        changes.
        """
        plane_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        if plane_geom_id == -1:
            raise ValueError(
                "Ground plane geom named 'floor' is required in the MJCF."
            )
        # Force the plane to lie on z=0.
        self.model.geom_pos[plane_geom_id, 2] = 0.0
        return plane_geom_id

    def _robot_geom_ids_excluding_world(self) -> list[int]:
        """Return geom ids belonging to non-world bodies and not the ground plane."""
        geom_ids: list[int] = []
        for geom_id in range(self.model.ngeom):
            if geom_id == self._ground_plane_geom_id:
                continue
            body_id = self.model.geom_bodyid[geom_id]
            if body_id <= 0:
                continue
            geom_ids.append(geom_id)
        return geom_ids

    def _build_collision_limit(self) -> CollisionAvoidanceLimit:
        """Collision avoidance between arms, torso, and the ground plane."""
        left_arm = geom_group_from_bodies(
            self.model,
            [
                "left_elbow_link",
                "left_shoulder_yaw_link",
                "left_wrist_roll_link",
                "left_wrist_pitch_link",
                "left_wrist_yaw_link",
            ],
        )
        right_arm = geom_group_from_bodies(
            self.model,
            [
                "right_elbow_link",
                "right_shoulder_yaw_link",
                "right_wrist_roll_link",
                "right_wrist_pitch_link",
                "right_wrist_yaw_link",
            ],
        )
        left_leg = geom_group_from_bodies(
            self.model, ["left_hip_yaw_link", "left_knee_link"]
        )
        right_leg = geom_group_from_bodies(
            self.model, ["right_hip_yaw_link", "right_knee_link"]
        )
        torso = geom_group_from_bodies(self.model, ["torso_link"])
        robot_geoms = self._robot_geom_ids_excluding_world()
        ground_plane = [self._ground_plane_geom_id]
        collision_pairs = [
            (left_arm, torso),
            (right_arm, torso),
            (left_arm, left_leg),
            (right_arm, right_leg),
            (robot_geoms, ground_plane),
        ]
        return CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=collision_pairs,
            gain=self._config.collision_gain,
            minimum_distance_from_collisions=self._config.minimum_distance_from_collisions,
            collision_detection_distance=self._config.collision_detection_distance,
        )

    def _build_velocity_limit(self) -> VelocityLimit:
        """Velocity limit only for lower-limb joints (hips, knees, ankles)."""
        velocities: dict[str, np.ndarray] = {}
        lower_limb_prefixes = (
            "left_hip_",
            "left_knee_",
            "left_ankle_",
            "right_hip_",
            "right_knee_",
            "right_ankle_",
        )
        for jid in range(self.model.njnt):
            jname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jid
            )
            if jname is None:
                continue
            if not any(jname.startswith(pref) for pref in lower_limb_prefixes):
                continue
            jtype = self.model.jnt_type[jid]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if jtype == mujoco.mjtJoint.mjJNT_BALL:
                vdim = 3
            else:
                vdim = 1
            velocities[jname] = np.full(
                vdim, self._config.max_joint_velocity, dtype=np.float64
            )
        return VelocityLimit(
            model=self.model,
            velocities=velocities,
        )

    def _compute_home_feet_to_pelvis_offset(self) -> np.ndarray:
        """Return pelvis translation relative to the midpoint of both feet."""
        pelvis_pose = self.configuration.get_transform_frame_to_world(
            frame_name="pelvis",
            frame_type="body",
        )
        left_foot_pose = self.configuration.get_transform_frame_to_world(
            frame_name="left_ankle_pitch_link",
            frame_type="body",
        )
        right_foot_pose = self.configuration.get_transform_frame_to_world(
            frame_name="right_ankle_pitch_link",
            frame_type="body",
        )
        feet_midpoint = 0.5 * (
            left_foot_pose.translation() + right_foot_pose.translation()
        )
        return pelvis_pose.translation() - feet_midpoint

    def _update_com_target(self, desired_target: np.ndarray) -> None:
        """Set the COM target directly."""
        self._com_task.set_target(desired_target)

    def _get_transformed_root_pose(self, root_pose: SE3) -> SE3:
        """Return root pose with z scaled, and tracker-to-root transform applied."""
        transformed_root_pose = root_pose.multiply(
            self._tracker_to_root_transform
        )
        scaled_translation = transformed_root_pose.translation().copy()
        scaled_translation[2] *= self._root_z_scale_factor
        return SE3.from_rotation_and_translation(
            rotation=transformed_root_pose.rotation(),
            translation=scaled_translation,
        )

    def _get_feet_based_root_pose(
        self, transformed: HandRootFootKeyPoints
    ) -> SE3:
        """Synthesize a pelvis pose from the feet when no root tracker is used."""
        feet_midpoint_pose = transformed.feet_midpoint_pose
        feet_yaw = feet_midpoint_pose.rotation().as_rpy_radians().yaw
        pelvis_translation = feet_midpoint_pose.translation().copy()
        pelvis_translation[0] += (
            np.cos(feet_yaw) * self._home_feet_to_pelvis_offset[0]
            - np.sin(feet_yaw) * self._home_feet_to_pelvis_offset[1]
        )
        pelvis_translation[1] += (
            np.sin(feet_yaw) * self._home_feet_to_pelvis_offset[0]
            + np.cos(feet_yaw) * self._home_feet_to_pelvis_offset[1]
        )
        pelvis_translation[2] = self._config.standing_pelvis_height
        pelvis_rotation = SO3.from_rpy_radians(
            roll=0.0,
            pitch=self._config.standing_pelvis_pitch,
            yaw=feet_yaw,
        )
        return SE3.from_rotation_and_translation(
            rotation=pelvis_rotation,
            translation=pelvis_translation,
        )

    def _setup_reference_from_feet(self, target_update: HandRootFootKeyPoints):
        # Compute tracker-to-foot transforms used to offset ankle trackers.
        fm_rot = target_update.feet_midpoint_pose.rotation()
        fm_rpy = fm_rot.as_rpy_radians()
        left_tracker_pose = target_update.left_foot_pose
        left_tracker_rot = left_tracker_pose.rotation()
        left_foot_rot = SO3.from_rpy_radians(
            roll=0.0,
            pitch=0.0,
            yaw=fm_rpy.yaw,
        )
        left_foot_to_tracker_rot = left_foot_rot.inverse().multiply(
            left_tracker_rot
        )
        left_foot_to_tracker_translation = np.array(
            [
                self._config.foot_tracker_x_offset,
                0.0,
                0.0,
            ]
        )
        left_foot_to_tracker_transform = SE3.from_rotation_and_translation(
            rotation=left_foot_to_tracker_rot,
            translation=left_foot_to_tracker_translation,
        )
        self._left_tracker_to_foot_transform = (
            left_foot_to_tracker_transform.inverse()
        )
        right_tracker_pose = target_update.right_foot_pose
        right_tracker_rot = right_tracker_pose.rotation()
        right_foot_rot = SO3.from_rpy_radians(
            roll=0.0,
            pitch=0.0,
            yaw=fm_rpy.yaw,
        )
        right_foot_to_tracker_rot = right_foot_rot.inverse().multiply(
            right_tracker_rot
        )
        right_foot_to_tracker_translation = np.array(
            [
                self._config.foot_tracker_x_offset,
                0.0,
                0.0,
            ]
        )
        right_foot_to_tracker_transform = SE3.from_rotation_and_translation(
            rotation=right_foot_to_tracker_rot,
            translation=right_foot_to_tracker_translation,
        )
        self._right_tracker_to_foot_transform = (
            right_foot_to_tracker_transform.inverse()
        )
        target_fm_pos_goal = np.array(
            [
                self._home_q[0],
                self._home_q[1],
                self.home_configuration.get_transform_frame_to_world(
                    frame_name="left_ankle_pitch_link",
                    frame_type="body",
                ).translation()[2],
            ]
        )
        target_fm_rpy_orig = (
            target_update.feet_midpoint_pose.rotation().as_rpy_radians()
        )
        target_fm_rot_goal = SO3.from_rpy_radians(
            roll=target_fm_rpy_orig.roll,
            pitch=target_fm_rpy_orig.pitch,
            yaw=0.0,
        )
        target_transform_goal = SE3.from_rotation_and_translation(
            rotation=target_fm_rot_goal,
            translation=target_fm_pos_goal,
        )
        self._reference_transform = target_transform_goal.multiply(
            target_update.feet_midpoint_pose.inverse()
        )

    def _update_root_transform(
        self, target_update: HandRootFootKeyPoints
    ) -> None:
        transformed = target_update.transformed(
            transform=self._reference_transform,
            left_tracker_to_foot_transform=self._left_tracker_to_foot_transform,
            right_tracker_to_foot_transform=self._right_tracker_to_foot_transform,
        )

        current_tracker_pose = transformed.root_pose
        # Root should have no pitch and roll
        current_tracker_rot = current_tracker_pose.rotation()
        current_tracker_rpy = current_tracker_rot.as_rpy_radians()
        current_root_rot = SO3.from_rpy_radians(
            roll=0.0,
            pitch=self._config.standing_pelvis_pitch,
            yaw=current_tracker_rpy.yaw,
        )
        rot_tracker_to_root = current_tracker_rot.inverse().multiply(
            current_root_rot
        )
        translation_tracker_to_root = np.array(
            [
                -self._config.root_tracker_x_offset,
                0.0,
                0.0,
            ]
        )
        self._tracker_to_root_transform = SE3.from_rotation_and_translation(
            rotation=rot_tracker_to_root,
            translation=translation_tracker_to_root,
        )

        current_root_pose = transformed.root_pose.multiply(
            self._tracker_to_root_transform
        )
        self._root_z_scale_factor = (
            self._config.standing_pelvis_height
            / current_root_pose.translation()[2]
        )

    def _suppress_pitch(self, transform: SE3) -> SE3:
        """Return transform with pitch component removed."""
        rpy = transform.rotation().as_rpy_radians()
        no_pitch_rot = SO3.from_rpy_radians(
            roll=rpy.roll,
            pitch=0.0,
            yaw=rpy.yaw,
        )
        return SE3.from_rotation_and_translation(
            rotation=no_pitch_rot,
            translation=transform.translation(),
        )

    def _make_upright_pose(self, reference_pose: SE3) -> SE3:
        """Return pose with reference yaw, but zero roll/pitch, i.e. upright vs gravity."""
        rpy = reference_pose.rotation().as_rpy_radians()
        upright_rot = SO3.from_rpy_radians(
            roll=0.0,
            pitch=0.0,
            yaw=rpy.yaw,
        )
        return SE3.from_rotation_and_translation(
            rotation=upright_rot,
            translation=reference_pose.translation(),
        )

    def reset(self):
        """Reset the IK solver to the home configuration."""
        self.configuration.update(q=self._home_q.copy())

    def recompute_reference_transform(
        self, target_update: HandRootFootKeyPoints
    ) -> None:
        """Normalize tracker poses so the initial frame aligns with the home pose."""
        self._setup_reference_from_feet(target_update)
        if self._config.use_root_tracker:
            self._update_root_transform(target_update)
        else:
            self._tracker_to_root_transform = SE3.identity()
            self._root_z_scale_factor = 1.0

    def _compute_com_target(
        self, transformed: HandRootFootKeyPoints
    ) -> np.ndarray:
        desired_xy = transformed.feet_midpoint_pose.translation()[:2]
        desired_z = float(self.configuration.data.subtree_com[1][2])
        return np.array(
            [desired_xy[0], desired_xy[1], desired_z], dtype=np.float64
        )

    def _update_task_targets(
        self, target_update: HandRootFootKeyPoints
    ) -> HandRootFootKeyPoints:
        """Update IK tasks based on normalized tracker inputs."""
        transformed = target_update.transformed(
            transform=self._reference_transform,
            left_tracker_to_foot_transform=self._left_tracker_to_foot_transform,
            right_tracker_to_foot_transform=self._right_tracker_to_foot_transform,
        )
        self._left_hand_task.set_target(transformed.left_hand_pose)
        self._right_hand_task.set_target(transformed.right_hand_pose)
        if self._config.suppress_foot_pitch:
            left_foot_pose = self._suppress_pitch(transformed.left_foot_pose)
            right_foot_pose = self._suppress_pitch(transformed.right_foot_pose)
            self._left_foot_task.set_target(left_foot_pose)
            self._right_foot_task.set_target(right_foot_pose)
        else:
            self._left_foot_task.set_target(transformed.left_foot_pose)
            self._right_foot_task.set_target(transformed.right_foot_pose)
        self._update_com_target(self._compute_com_target(transformed))
        if self._pelvis_task is None:
            raise RuntimeError("Pelvis task is required.")
        if self._config.use_root_tracker:
            scaled_root_pose = self._get_transformed_root_pose(
                transformed.root_pose
            )
        else:
            scaled_root_pose = self._get_feet_based_root_pose(transformed)
        self._pelvis_task.set_target(scaled_root_pose)
        if self._torso_upright_task is not None:
            self._torso_upright_task.set_target(
                self._make_upright_pose(scaled_root_pose)
            )
        return transformed

    def solve_one_step(
        self, target_update: HandRootFootKeyPoints, dt: float | None = None
    ) -> IKSolution:
        """Advance the IK solver towards the provided tracker targets."""
        outer_dt = float(self._config.dt if dt is None else dt)
        transformed_target = self._update_task_targets(target_update)

        inner_dt = outer_dt / max(self._config.inner_steps, 1)
        for _ in range(self._config.inner_steps):
            velocity = solve_ik(
                configuration=self.configuration,
                tasks=self._tasks,
                dt=inner_dt,
                solver=self._config.solver,
                limits=self._limits,
                damping=self._config.damping,
            )
            self.configuration.integrate_inplace(velocity, inner_dt)

        com_task = self._com_task
        pelvis_task = self._pelvis_task
        return IKSolution.from_configuration_and_tasks(
            configuration=self.configuration,
            left_hand_task=self._left_hand_task,
            right_hand_task=self._right_hand_task,
            left_foot_task=self._left_foot_task,
            right_foot_task=self._right_foot_task,
            com_task=com_task,
            pelvis_task=pelvis_task,
            target=target_update,
            transformed_target=transformed_target,
        )

    def load_ik_solution(self, json_path: Path) -> list[IKSolution]:
        """Load an IK episode from a JSON file."""
        return load_ik_episode(json_path=json_path)

    def load_targets(self, json_path: Path) -> list[HandRootFootKeyPoints]:
        """Load episode targets from a JSON file."""
        return load_target_episode(json_path=json_path)


__all__ = ["IKSolver"]
