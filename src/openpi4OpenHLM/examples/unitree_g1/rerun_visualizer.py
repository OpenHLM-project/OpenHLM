"""
Rerun visualization for OpenPI evaluation (self-contained).

`openpi-eval/main.py` uses this module only when `--rerun-visualize` is enabled.

What gets logged:
  - `/head_image_left`, `/left_wrist_image`, `/right_wrist_image`: head and wrist camera images
  - `/state_body/*`: groups sliced from OpenPI `obs["state"]` (34 dims)
  - `/action_body/*`: groups sliced from OpenPI `action` (34 dims)
  - `/state_hand` + `/action_hand`: 2D signals (left/right gripper)

State/action layout (34 dims):
  0-6:   left arm (7)
  7:     left gripper (1)
  8-14:  right arm (7)
  15:    right gripper (1)
  16-21: left leg (6)
  22-27: right leg (6)
  28-30: waist (3)
  31-33: root (3) -- roll, pitch, yaw angular velocity

Layout:
  - Uses a fixed blueprint with stable entity paths (e.g. `/state_body/root`,
    `/action_body/left_arm_joint_pos`).

Recording behavior:
  - Each `OpenPIRerunVisualizer` instance creates a fresh `RecordingStream`.
    The evaluation loop typically creates one instance per episode, so each
    episode becomes a separate recording in the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

import numpy as np

import mujoco
import rerun as rr
from rerun import blueprint as rrb


@dataclass(frozen=True)
class GroupSpec:
    tab_name: str
    entity_path: str
    start: int  # inclusive, 0-based
    end: int  # exclusive, 0-based
    axis_names: list[str]


def _palette_rgb() -> list[tuple[int, int, int]]:
    # A small, high-contrast palette (cycled as needed).
    return [
        (231, 76, 60),  # red
        (39, 174, 96),  # green
        (52, 120, 219),  # blue
        (241, 196, 15),  # yellow
        (155, 89, 182),  # purple
        (149, 165, 166),  # gray
        (230, 126, 34),  # orange
        (26, 188, 156),  # teal
    ]


def _series_style(axis_names: list[str]) -> rr.SeriesLines:
    colors = _palette_rgb()
    per_axis = [colors[i % len(colors)] for i in range(len(axis_names))]
    widths = [2.0] * len(axis_names)
    # `colors` expects a list per series; Rerun's python typing is a bit loose here.
    return rr.SeriesLines.from_fields(names=axis_names, colors=[per_axis], widths=widths)  # type: ignore[arg-type]


def _timeseries_view(*, origin: str, name: str, axis_names: list[str]) -> rrb.BlueprintLike:
    """Create a `TimeSeriesView` with consistent styling."""
    return rrb.TimeSeriesView(
        origin=origin,
        name=name,
        plot_legend=rrb.Corner2D.RightTop,
        overrides={
            origin: _series_style(axis_names),
        },
    )


def _state_group_specs_openpi() -> list[GroupSpec]:
    # Root is at indices 31-34 (3 dims): roll, pitch, yaw angular velocity
    return [
        GroupSpec(
            tab_name="State Root",
            entity_path="/state_body/root",
            start=31,
            end=34,
            axis_names=["roll", "pitch", "yaw_vel"],
        ),
        GroupSpec(
            tab_name="State Left leg",
            entity_path="/state_body/left_leg_joint_pos",
            start=16,
            end=22,
            axis_names=[f"l_leg_j{i}" for i in range(1, 7)],
        ),
        GroupSpec(
            tab_name="State Right leg",
            entity_path="/state_body/right_leg_joint_pos",
            start=22,
            end=28,
            axis_names=[f"r_leg_j{i}" for i in range(1, 7)],
        ),
        GroupSpec(
            tab_name="State Waist",
            entity_path="/state_body/waist_joint_pos",
            start=28,
            end=31,
            axis_names=[f"waist_j{i}" for i in range(1, 4)],
        ),
        GroupSpec(
            tab_name="State Left arm",
            entity_path="/state_body/left_arm_joint_pos",
            start=0,
            end=7,
            axis_names=[f"l_arm_j{i}" for i in range(1, 8)],
        ),
        GroupSpec(
            tab_name="State Right arm",
            entity_path="/state_body/right_arm_joint_pos",
            start=8,
            end=15,
            axis_names=[f"r_arm_j{i}" for i in range(1, 8)],
        ),
    ]


def _action_group_specs_openpi() -> list[GroupSpec]:
    # Root is at indices 31-34 (3 dims): roll, pitch, yaw angular velocity
    return [
        GroupSpec(
            tab_name="Action Root",
            entity_path="/action_body/root",
            start=31,
            end=34,
            axis_names=["roll", "pitch", "yaw_vel"],
        ),
        GroupSpec(
            tab_name="Action Left leg",
            entity_path="/action_body/left_leg_joint_pos",
            start=16,
            end=22,
            axis_names=[f"l_leg_j{i}" for i in range(1, 7)],
        ),
        GroupSpec(
            tab_name="Action Right leg",
            entity_path="/action_body/right_leg_joint_pos",
            start=22,
            end=28,
            axis_names=[f"r_leg_j{i}" for i in range(1, 7)],
        ),
        GroupSpec(
            tab_name="Action Waist",
            entity_path="/action_body/waist_joint_pos",
            start=28,
            end=31,
            axis_names=[f"waist_j{i}" for i in range(1, 4)],
        ),
        GroupSpec(
            tab_name="Action Left arm",
            entity_path="/action_body/left_arm_joint_pos",
            start=0,
            end=7,
            axis_names=[f"l_arm_j{i}" for i in range(1, 8)],
        ),
        GroupSpec(
            tab_name="Action Right arm",
            entity_path="/action_body/right_arm_joint_pos",
            start=8,
            end=15,
            axis_names=[f"r_arm_j{i}" for i in range(1, 8)],
        ),
    ]


def _make_state_tabs(*, state_groups: list[GroupSpec], active_tab: int = 0) -> rrb.BlueprintLike:
    tabs: list[rrb.BlueprintLike] = []

    for g in state_groups:
        tabs.append(_timeseries_view(origin=g.entity_path, name=g.tab_name, axis_names=g.axis_names))

    tabs.append(_timeseries_view(origin="/state_hand", name="State Hand", axis_names=["left hand", "right hand"]))

    return rrb.Tabs(*tabs, active_tab=active_tab, name="State body")


def _make_action_tabs(*, action_groups: list[GroupSpec], active_tab: int = 0) -> rrb.BlueprintLike:
    tabs: list[rrb.BlueprintLike] = []

    for g in action_groups:
        tabs.append(_timeseries_view(origin=g.entity_path, name=g.tab_name, axis_names=g.axis_names))

    tabs.append(_timeseries_view(origin="/action_hand", name="Action Hand", axis_names=["left hand", "right hand"]))

    return rrb.Tabs(*tabs, active_tab=active_tab, name="Action body")


def _make_blueprint(
    state_groups: list[GroupSpec],
    action_groups: list[GroupSpec],
) -> rrb.BlueprintLike:
    return rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(origin="/", name="left_wrist_image", contents=["/left_wrist_image"]),
            rrb.Spatial2DView(origin="/", name="right_wrist_image", contents=["/right_wrist_image"]),
            row_shares=[1, 1],
            name="Wrist Cameras",
        ),
        rrb.Vertical(
            rrb.Spatial2DView(origin="/", name="head_image_left", contents=["/head_image_left"]),
            rrb.Spatial3DView(origin="/robot", name="robot (reference motion)"),
            row_shares=[1, 1],
            name="Head + Robot",
        ),
        rrb.Vertical(
            rrb.Vertical(
                _make_state_tabs(state_groups=state_groups, active_tab=0),
                name="State",
            ),
            rrb.Vertical(
                _make_action_tabs(action_groups=action_groups, active_tab=0),
                name="Action",
            ),
            row_shares=[1, 1],
            name="State/Action",
        ),
        column_shares=[1, 1, 1],
        name="RGB + Signals",
    )



def _as_float_vector(x: Any, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = arr.reshape(-1)
    if arr.shape != expected_shape:
        raise ValueError(f"Expected vector shape {expected_shape}, got {arr.shape}")
    return arr


def _as_image_hwc_u8(x: Any) -> np.ndarray:
    """
    Convert an input image into HxWxC uint8 for `rr.Image`.

    Supported inputs:
      - numpy arrays in HWC or CHW
      - nested lists convertible to numpy
    """
    img = np.asarray(x)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
        # Likely CHW -> HWC
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        # Common cases: float in [0,1] or int types. Clip to [0,255] and cast.
        img = np.clip(img, 0, 255)
        if img.dtype.kind == "f" and img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255)
        img = img.astype(np.uint8, copy=False)

    return img


_G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _vector_to_joint_positions(vector: np.ndarray) -> np.ndarray:
    left_leg = vector[16:22]
    right_leg = vector[22:28]
    waist = vector[28:31]
    left_arm = vector[0:7]
    right_arm = vector[8:15]
    return np.concatenate([left_leg, right_leg, waist, left_arm, right_arm], axis=0)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz], dtype=np.float32)


def _resolve_xml_path(xml_path: str | Path) -> Path:
    path = Path(xml_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


_VIEWER_SPAWNED = False


def _spawn_viewer_once() -> bool:
    """
    Spawn the Rerun viewer only once per Python process.

    The evaluation loop may create a new visualizer per episode to start a fresh
    recording; we still want to reuse the same viewer window instead of opening
    a new one every episode.
    """
    global _VIEWER_SPAWNED
    spawn = not _VIEWER_SPAWNED
    _VIEWER_SPAWNED = True
    return spawn


class OpenPIRerunVisualizer:
    """
    Minimal wrapper around a per-episode Rerun `RecordingStream`.

    Typical usage (per-episode):
      viz = OpenPIRerunVisualizer(app_id="openpi_eval", recording_name="episode_0001")
      viz.log_observation(step_idx, obs)
      viz.log_action(step_idx, action)
    """

    def __init__(
        self,
        *,
        app_id: str = "openpi_eval",
        recording_name: str | None = None,
        jpeg_quality: int = 90,
        mujoco_xml_path: str | Path = "g1_mocap_29dof.xml",
        control_hz: float = 30.0,
    ) -> None:
        """
        Initialize the visualizer.

        Args:
            app_id: Rerun application ID.
            recording_name: Name for this recording (typically episode ID).
            jpeg_quality: JPEG compression quality for images (0-100).
            mujoco_xml_path: Path to the MuJoCo XML model.
            control_hz: Control loop frequency for integrating yaw velocity.
        """
        self.state_dim = 34
        self.action_dim = 34

        self._state_groups = _state_group_specs_openpi()
        self._action_groups = _action_group_specs_openpi()
        self._blueprint = _make_blueprint(
            self._state_groups,
            self._action_groups,
        )
        self._jpeg_quality = int(jpeg_quality)
        self._control_hz = float(control_hz)
        self._base_yaw = 0.0
        self._last_step_idx: int | None = None

        self._mujoco_model = mujoco.MjModel.from_xml_path(str(_resolve_xml_path(mujoco_xml_path)))
        self._mujoco_data = mujoco.MjData(self._mujoco_model)
        self._mujoco_base_qpos = self._mujoco_data.qpos.copy()
        self._mujoco_joint_qposadr = []
        for name in _G1_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self._mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model.")
            self._mujoco_joint_qposadr.append(self._mujoco_model.jnt_qposadr[joint_id])
        self._mujoco_geom_indices: list[int] = []
        self._mujoco_geom_paths: list[str] = []

        # Create a fresh recording per episode.
        self._rec = rr.RecordingStream(app_id, recording_id=str(uuid.uuid4()))

        if _spawn_viewer_once():
            # First time: spawn the viewer and connect this recording.
            self._rec.spawn(port=9876, connect=True, detach_process=True)
        else:
            # Subsequent episodes: connect to the already-running local viewer/server.
            # `url=None` uses Rerun's default endpoint (typically the local viewer proxy).
            self._rec.connect_grpc()

        if recording_name is not None:
            rr.send_recording_name(recording_name, recording=self._rec)

        rr.send_blueprint(self._blueprint, make_active=True, make_default=True, recording=self._rec)
        self._log_mujoco_meshes()

    def _log_mujoco_meshes(self) -> None:
        model = self._mujoco_model
        ground_extent = 12.0
        tiles = 12
        step = (ground_extent * 2.0) / tiles
        verts = []
        faces = []
        colors = []
        light = np.array([235, 235, 235], dtype=np.uint8)
        dark = np.array([200, 200, 200], dtype=np.uint8)
        for yi in range(tiles):
            for xi in range(tiles):
                x0 = -ground_extent + xi * step
                y0 = -ground_extent + yi * step
                x1 = x0 + step
                y1 = y0 + step
                base = len(verts)
                verts.extend([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
                faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
                tile_color = light if (xi + yi) % 2 == 0 else dark
                colors.extend([tile_color, tile_color, tile_color, tile_color])

        ground_verts = np.asarray(verts, dtype=np.float32)
        ground_faces = np.asarray(faces, dtype=np.int32)
        ground_colors = np.asarray(colors, dtype=np.uint8)
        rr.log(
            "/robot/ground",
            rr.Mesh3D(
                vertex_positions=ground_verts,
                triangle_indices=ground_faces,
                vertex_colors=ground_colors,
            ),
            recording=self._rec,
        )

        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = model.geom_dataid[geom_id]
            if mesh_id < 0:
                continue

            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            geom_name = name if name else f"geom_{geom_id}"
            entity_path = f"/robot/geoms/{geom_name}"

            vert_start = model.mesh_vertadr[mesh_id]
            vert_count = model.mesh_vertnum[mesh_id]
            verts = model.mesh_vert[vert_start : vert_start + vert_count].reshape(-1, 3)

            face_start = model.mesh_faceadr[mesh_id]
            face_count = model.mesh_facenum[mesh_id]
            faces = model.mesh_face[face_start : face_start + face_count].reshape(-1, 3)

            rgba = model.geom_rgba[geom_id]
            rgb_u8 = (np.clip(rgba[:3], 0.0, 1.0) * 255).astype(np.uint8)
            vertex_colors = np.tile(rgb_u8, (verts.shape[0], 1))

            rr.log(
                entity_path,
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=faces,
                    vertex_colors=vertex_colors,
                ),
                recording=self._rec,
            )
            self._mujoco_geom_indices.append(geom_id)
            self._mujoco_geom_paths.append(entity_path)

    def log_observation(self, step_idx: int, obs: dict[str, Any]) -> None:
        rr.set_time("step", sequence=int(step_idx), recording=self._rec)

        head_img = obs.get("head_image_left")
        if head_img is not None:
            img = _as_image_hwc_u8(head_img)
            rr_img = rr.Image(img, color_model="RGB").compress(jpeg_quality=self._jpeg_quality)
            rr.log("/head_image_left", rr_img, recording=self._rec)

        left_wrist = obs.get("left_wrist_image")
        if left_wrist is not None:
            img = _as_image_hwc_u8(left_wrist)
            rr_img = rr.Image(img, color_model="RGB").compress(jpeg_quality=self._jpeg_quality)
            rr.log("/left_wrist_image", rr_img, recording=self._rec)

        right_wrist = obs.get("right_wrist_image")
        if right_wrist is not None:
            img = _as_image_hwc_u8(right_wrist)
            rr_img = rr.Image(img, color_model="RGB").compress(jpeg_quality=self._jpeg_quality)
            rr.log("/right_wrist_image", rr_img, recording=self._rec)

        state = _as_float_vector(obs["state"], expected_shape=(self.state_dim,))

        # Body groups.
        for g in self._state_groups:
            rr.log(g.entity_path, rr.Scalars(state[g.start : g.end]), recording=self._rec)

        rr.log("/state_hand", rr.Scalars(np.array([state[7], state[15]], dtype=np.float32)), recording=self._rec)

    def log_action(self, step_idx: int, action: Any) -> None:
        rr.set_time("step", sequence=int(step_idx), recording=self._rec)

        act = _as_float_vector(action, expected_shape=(self.action_dim,))

        for g in self._action_groups:
            rr.log(g.entity_path, rr.Scalars(act[g.start : g.end]), recording=self._rec)

        rr.log("/action_hand", rr.Scalars(np.array([act[7], act[15]], dtype=np.float32)), recording=self._rec)

        # root (3 dims): roll, pitch, yaw angular velocity
        root = act[31:34]
        roll, pitch, yaw_vel = root

        if self._last_step_idx is None:
            dt = 0.0
        else:
            dt = (step_idx - self._last_step_idx) / self._control_hz
        self._last_step_idx = step_idx

        self._base_yaw += yaw_vel * dt

        base_pos = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        base_quat = _quat_from_rpy(float(roll), float(pitch), float(self._base_yaw))

        joint_positions = _vector_to_joint_positions(act)
        self._mujoco_data.qpos[:] = self._mujoco_base_qpos
        self._mujoco_data.qpos[0:3] = base_pos
        self._mujoco_data.qpos[3:7] = base_quat
        for adr, value in zip(self._mujoco_joint_qposadr, joint_positions):
            self._mujoco_data.qpos[adr] = value
        mujoco.mj_forward(self._mujoco_model, self._mujoco_data)
        for geom_id, geom_path in zip(self._mujoco_geom_indices, self._mujoco_geom_paths):
            pos = self._mujoco_data.geom_xpos[geom_id]
            xmat = self._mujoco_data.geom_xmat[geom_id].reshape(3, 3)
            rr.log(
                geom_path,
                rr.Transform3D(translation=pos, mat3x3=xmat),
                recording=self._rec,
            )
