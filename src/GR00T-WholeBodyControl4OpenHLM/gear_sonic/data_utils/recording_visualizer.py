#!/usr/bin/env python3
"""
Real-time Rerun visualization for data recording (server_data_record.py).

This module provides live visualization of state and action data during teleoperation
data collection. It visualizes:
  - `/rgb_left`: Left camera image (JPEG-compressed)
  - `/wrist_rgb_left`: Wrist left camera image (JPEG-compressed, optional)
  - `/wrist_rgb_right`: Wrist right camera image (JPEG-compressed, optional)
  - `/state_body/*`: State body groups (34 dims total)
  - `/action_body/*`: Action body groups (35 dims total)
  - `/state_hand`, `/action_hand`: Hand signals (left/right)
  - `/state_neck`, `/action_neck`: Neck signals (2 dims)

Data format (from server_data_record.py):
  state_body (32 dims):
    0-2   : root (roll, pitch, delta_yaw)
    3-8   : left leg joint positions (6)
    9-14  : right leg joint positions (6)
    15-17 : waist joint positions (3)
    18-24 : left arm joint positions (7)
    25-31 : right arm joint positions (7)

  action_body (32 dims):
    0-2   : root (roll, pitch, delta_yaw)
    3-8   : left leg joint positions (6)
    9-14  : right leg joint positions (6)
    15-17 : waist joint positions (3)
    18-24 : left arm joint positions (7)
    25-31 : right arm joint positions (7)

Usage:
    from data_utils.recording_visualizer import RecordingVisualizer
    
    viz = RecordingVisualizer()
    viz.log_frame(step_idx, data_dict)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import uuid

import cv2
import numpy as np
import rerun as rr
from rerun import blueprint as rrb



@dataclass(frozen=True)
class GroupSpec:
    """Specification for a group of signals to visualize."""
    tab_name: str
    entity_path: str
    start: int  # inclusive, 0-based
    end: int    # exclusive, 0-based
    axis_names: list[str]


def _palette_rgb() -> list[tuple[int, int, int]]:
    """High-contrast color palette for plotting."""
    return [
        (231, 76, 60),   # red
        (39, 174, 96),   # green
        (52, 120, 219),  # blue
        (241, 196, 15),  # yellow
        (155, 89, 182),  # purple
        (149, 165, 166), # gray
        (230, 126, 34),  # orange
        (26, 188, 156),  # teal
    ]


def _series_style(axis_names: list[str]) -> "rr.SeriesLines":
    """Create consistent line styling for time series."""
    colors = _palette_rgb()
    per_axis = [colors[i % len(colors)] for i in range(len(axis_names))]
    widths = [2.0] * len(axis_names)
    return rr.SeriesLines.from_fields(names=axis_names, colors=[per_axis], widths=widths)


def _timeseries_view(*, origin: str, name: str, axis_names: list[str]) -> "rrb.BlueprintLike":
    """Create a TimeSeriesView with consistent styling."""
    return rrb.TimeSeriesView(
        origin=origin,
        name=name,
        plot_legend=rrb.Corner2D.RightTop,
        overrides={
            origin: _series_style(axis_names),
        },
    )


def _state_group_specs() -> list[GroupSpec]:
    """State body grouping specs (32 dims total)."""
    return [
        GroupSpec(
            tab_name="State Root",
            entity_path="/state_body/root",
            start=0,
            end=3,
            axis_names=["roll", "pitch", "delta_yaw"],
        ),
        GroupSpec(
            tab_name="State Left Leg",
            entity_path="/state_body/left_leg",
            start=3,
            end=9,
            axis_names=[f"l_leg_j{i}" for i in range(1, 7)],
        ),
        GroupSpec(
            tab_name="State Right Leg",
            entity_path="/state_body/right_leg",
            start=9,
            end=15,
            axis_names=[f"r_leg_j{i}" for i in range(1, 7)],
        ),
        GroupSpec(
            tab_name="State Waist",
            entity_path="/state_body/waist",
            start=15,
            end=18,
            axis_names=[f"waist_j{i}" for i in range(1, 4)],
        ),
        GroupSpec(
            tab_name="State Left Arm",
            entity_path="/state_body/left_arm",
            start=18,
            end=25,
            axis_names=[f"l_arm_j{i}" for i in range(1, 8)],
        ),
        GroupSpec(
            tab_name="State Right Arm",
            entity_path="/state_body/right_arm",
            start=25,
            end=32,
            axis_names=[f"r_arm_j{i}" for i in range(1, 8)],
        ),
    ]


def _action_group_specs() -> list[GroupSpec]:
    """Action body grouping specs (32 dims total)."""
    return [
        GroupSpec(
            tab_name="Action Root",
            entity_path="/action_body/root",
            start=0,
            end=3,
            axis_names=["roll", "pitch", "delta_yaw"],
        ),
        GroupSpec(
            tab_name="Action Left Leg",
            entity_path="/action_body/left_leg",
            start=3,
            end=9,
            axis_names=[f"l_leg_j{i}" for i in range(1, 7)],
        ),
        GroupSpec(
            tab_name="Action Right Leg",
            entity_path="/action_body/right_leg",
            start=9,
            end=15,
            axis_names=[f"r_leg_j{i}" for i in range(1, 7)],
        ),
        GroupSpec(
            tab_name="Action Waist",
            entity_path="/action_body/waist",
            start=15,
            end=18,
            axis_names=[f"waist_j{i}" for i in range(1, 4)],
        ),
        GroupSpec(
            tab_name="Action Left Arm",
            entity_path="/action_body/left_arm",
            start=18,
            end=25,
            axis_names=[f"l_arm_j{i}" for i in range(1, 8)],
        ),
        GroupSpec(
            tab_name="Action Right Arm",
            entity_path="/action_body/right_arm",
            start=25,
            end=32,
            axis_names=[f"r_arm_j{i}" for i in range(1, 8)],
        ),
    ]


def _make_state_tabs(state_groups: list[GroupSpec], active_tab: int = 0) -> "rrb.Tabs":
    """Create state visualization tabs."""
    tabs = []
    for g in state_groups:
        tabs.append(_timeseries_view(origin=g.entity_path, name=g.tab_name, axis_names=g.axis_names))

    # Hand tab
    tabs.append(_timeseries_view(
        origin="/state_hand",
        name="State Hand",
        axis_names=["left_hand", "right_hand"]
    ))

    return rrb.Tabs(*tabs, active_tab=active_tab, name="State")


def _make_action_tabs(action_groups: list[GroupSpec], active_tab: int = 0) -> "rrb.Tabs":
    """Create action visualization tabs."""
    tabs = []
    for g in action_groups:
        tabs.append(_timeseries_view(origin=g.entity_path, name=g.tab_name, axis_names=g.axis_names))

    # Hand tab
    tabs.append(_timeseries_view(
        origin="/action_hand",
        name="Action Hand",
        axis_names=["left_hand", "right_hand"]
    ))

    return rrb.Tabs(*tabs, active_tab=active_tab, name="Action")


def _make_blueprint(
    state_groups: list[GroupSpec],
    action_groups: list[GroupSpec],
    enable_wrist: bool = False
) -> "rrb.BlueprintLike":
    """Create the full visualization blueprint."""
    # Build camera views as a single vertical column (no tab switching).
    # Note: rgb_right is intentionally not visualized.
    camera_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin="/", name="RGB Left", contents=["/rgb_left"]),
    ]
    if enable_wrist:
        camera_views.extend([
            rrb.Spatial2DView(origin="/", name="Wrist Left", contents=["/wrist_rgb_left"]),
            rrb.Spatial2DView(origin="/", name="Wrist Right", contents=["/wrist_rgb_right"]),
        ])
    cameras_column = rrb.Vertical(
        *camera_views,
        row_shares=[1] * len(camera_views),
        name="Cameras",
    )

    return rrb.Horizontal(
        cameras_column,
        rrb.Vertical(
            _make_state_tabs(state_groups, active_tab=0),
            _make_action_tabs(action_groups, active_tab=0),
            row_shares=[1, 1],
            name="State/Action",
        ),
        column_shares=[1, 2],
        name="Recording Visualization",
    )


# Global flag to track if viewer has been spawned
_VIEWER_SPAWNED = False


def _resolve_video_ref(episode_dir: Path, ref: object) -> tuple[Path, int] | None:
    if not isinstance(ref, dict):
        return None
    video_path = ref.get("video_path")
    frame_index = ref.get("frame_index")
    if video_path is None or frame_index is None:
        return None
    return episode_dir / str(video_path), int(frame_index)


class _VideoFrameCache:
    """Small cache for reading frame references from episode videos."""

    def __init__(self) -> None:
        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._positions: dict[Path, int] = {}

    def read_bgr(self, episode_dir: Path, ref: object) -> np.ndarray | None:
        resolved = _resolve_video_ref(episode_dir, ref)
        if resolved is None:
            return None
        video_path, frame_index = resolved
        if not video_path.exists():
            return None

        cap = self._captures.get(video_path)
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            self._captures[video_path] = cap
            self._positions[video_path] = 0

        if self._positions.get(video_path) != frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        self._positions[video_path] = frame_index + 1
        return frame

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        self._positions.clear()


class RecordingVisualizer:
    """
    Real-time Rerun visualizer for data recording.
      
    Uses a single recording stream. The viewer is spawned only once per Python
    process to avoid multiple windows.
    
    Usage:
        viz = RecordingVisualizer(app_id="data_recording")
        
        # For each frame during recording:
        viz.log_frame(step_idx, data_dict)
        
        # When starting a new episode:
        viz.new_episode("episode_0001")
    """
    
    def __init__(
        self,
        app_id: str = "data_recording",
        recording_name: Optional[str] = None,
        jpeg_quality: int = 85,
        enabled: bool = True,
        enable_wrist: bool = False,
    ) -> None:
        """
        Initialize the visualizer.

        Args:
            app_id: Rerun application ID
            recording_name: Name for the recording (shown in viewer)
            jpeg_quality: JPEG compression quality for images (1-100)
            enabled: Whether visualization is enabled
            enable_wrist: Whether to visualize wrist camera data
        """
        self._enabled = enabled
        self._enable_wrist = enable_wrist
        self._app_id = app_id
        self._jpeg_quality = jpeg_quality
        self._state_groups = _state_group_specs()
        self._action_groups = _action_group_specs()
        self._blueprint = _make_blueprint(
            self._state_groups, self._action_groups,
            enable_wrist=enable_wrist
        )
        self._rec: Optional[rr.RecordingStream] = None
        self._episode_counter = 0
        self._video_cache = _VideoFrameCache()

        self._activate_recording(recording_name)
    
    def _create_recording_stream(self) -> rr.RecordingStream:
        """Create a recording stream and attach it to the viewer."""
        global _VIEWER_SPAWNED

        rec = rr.RecordingStream(self._app_id, recording_id=str(uuid.uuid4()))
        if not _VIEWER_SPAWNED:
            rec.spawn(port=9876, connect=True, detach_process=True)
            _VIEWER_SPAWNED = True
        else:
            rec.connect_grpc()

        return rec

    def _activate_recording(self, recording_name: Optional[str] = None) -> None:
        """(Re)create the recording stream and apply the blueprint."""
        if not self._enabled:
            return

        rec = self._create_recording_stream()
        self._rec = rec
        rr.reset_time(recording=rec)

        if recording_name is not None:
            rr.send_recording_name(recording_name, recording=rec)

        rr.send_blueprint(self._blueprint, make_active=True, make_default=True, recording=rec)
    
    def new_episode(self, episode_name: str) -> None:
        """
        Start a new episode recording.
        
        Args:
            episode_name: Name for the new episode (e.g., "episode_0001")
        """
        self._activate_recording(episode_name)
        self._episode_counter += 1
    
    def log_frame(self, step_idx: int, data_dict: dict[str, Any], episode_dir: str | Path | None = None) -> None:
        """
        Log a single frame of data.
        
        Args:
            step_idx: Frame index/step number
            data_dict: Dictionary containing:
                - rgb_left: numpy array (H, W, 3) uint8 or
                  {"video_path": str, "frame_index": int}
                - state_body: list/array of 34 floats
                - action_body: list/array of 35 floats
                - state_hand_left: list/array of 1 float
                - state_hand_right: list/array of 1 float
                - action_hand_left: list/array of 1 float
                - action_hand_right: list/array of 1 float
                - state_neck: list/array of 2 floats
                - action_neck: list/array of 2 floats
        """
        if not self._enabled or self._rec is None:
            return
        
        rr.set_time("step", sequence=int(step_idx), recording=self._rec)
        
        # Helper to log image
        def _log_image(key: str, entity_path: str, is_bgr: bool = True):
            if key not in data_dict or data_dict[key] is None:
                return

            value = data_dict[key]
            if isinstance(value, dict):
                if episode_dir is None:
                    return
                img = self._video_cache.read_bgr(Path(episode_dir), value)
                if img is None:
                    return
            else:
                img = np.asarray(value, dtype=np.uint8)

            # Ensure HWC format
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            # Convert BGR to RGB for Rerun if needed
            if is_bgr and img.shape[-1] == 3:
                img = img[..., ::-1]  # BGR -> RGB
            rr_img = rr.Image(img, color_model="RGB").compress(jpeg_quality=self._jpeg_quality)
            rr.log(entity_path, rr_img, recording=self._rec)
        
        # Log head camera images (BGR from OpenCV)
        _log_image("rgb_left", "/rgb_left", is_bgr=True)
        
        # Log wrist camera images (BGR format, same as head camera)
        if self._enable_wrist:
            _log_image("wrist_rgb_left", "/wrist_rgb_left", is_bgr=True)
            _log_image("wrist_rgb_right", "/wrist_rgb_right", is_bgr=True)
        
        # Log state_body
        if "state_body" in data_dict and data_dict["state_body"] is not None:
            state_body = np.asarray(data_dict["state_body"], dtype=np.float32).reshape(-1)
            if state_body.shape[0] >= 32:
                for g in self._state_groups:
                    rr.log(g.entity_path, rr.Scalars(state_body[g.start:g.end]), recording=self._rec)

        # Log action_body
        if "action_body" in data_dict and data_dict["action_body"] is not None:
            action_body = np.asarray(data_dict["action_body"], dtype=np.float32).reshape(-1)
            if action_body.shape[0] >= 32:
                for g in self._action_groups:
                    rr.log(g.entity_path, rr.Scalars(action_body[g.start:g.end]), recording=self._rec)

        # Log state hands
        state_hand_left = data_dict.get("state_hand_left")
        state_hand_right = data_dict.get("state_hand_right")
        if state_hand_left is not None and state_hand_right is not None:
            left_val = float(np.asarray(state_hand_left).reshape(-1)[0])
            right_val = float(np.asarray(state_hand_right).reshape(-1)[0])
            rr.log("/state_hand", rr.Scalars([left_val, right_val]), recording=self._rec)

        # Log action hands
        action_hand_left = data_dict.get("action_hand_left")
        action_hand_right = data_dict.get("action_hand_right")
        if action_hand_left is not None and action_hand_right is not None:
            left_val = float(np.asarray(action_hand_left).reshape(-1)[0])
            right_val = float(np.asarray(action_hand_right).reshape(-1)[0])
            rr.log("/action_hand", rr.Scalars([left_val, right_val]), recording=self._rec)
    
    def close(self) -> None:
        """Close the recording stream."""
        self._video_cache.close()
        if self._rec is None:
            return
        try:
            self._rec.disconnect()
        except Exception:
            pass
        self._rec = None


# Convenience function for quick testing
if __name__ == "__main__":
    import time
    
    print("Testing RecordingVisualizer...")
    viz = RecordingVisualizer(app_id="test_recording", recording_name="test_episode")
    
    # Generate fake data
    for i in range(300):
        data = {
            "rgb_left": np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8),
            "state_body": np.random.randn(32).astype(np.float32) * 0.5,
            "action_body": np.random.randn(32).astype(np.float32) * 0.5,
            "state_hand_left": [np.random.rand()],
            "state_hand_right": [np.random.rand()],
            "action_hand_left": [np.random.rand()],
            "action_hand_right": [np.random.rand()],
        }
        viz.log_frame(i, data)
        time.sleep(1.0 / 30)  # 30 FPS

        if i % 30 == 0:
            print(f"Logged frame {i}")
    
    viz.close()
    print("Done!")
