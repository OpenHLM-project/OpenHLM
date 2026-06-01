#!/usr/bin/env python3
"""
Data check script for visualizing recorded episodes using Rerun.

Loads all episodes from a data folder and plays video-backed frames at a
specified speed.

Usage:
    python data_check.py <data_path> <playback_speed> [--start-number N]

Arguments:
    data_path       Path to the data folder (e.g., sonic_demonstration/20260401_1519_pp_sprite)
    playback_speed  Playback speed multiplier (e.g., 8 for 8x speed, 0.5 for half speed)

Example:
    python data_check.py sonic_demonstration/20260401_1519_pp_sprite 8 --start-number 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rerun as rr
from rerun import blueprint as rrb


# ---------------------------------------------------------------------------
# Blueprint helpers (mirrored from recording_visualizer.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupSpec:
    tab_name: str
    entity_path: str
    start: int
    end: int
    axis_names: list[str]


def _palette_rgb() -> list[tuple[int, int, int]]:
    return [
        (231, 76, 60),
        (39, 174, 96),
        (52, 120, 219),
        (241, 196, 15),
        (155, 89, 182),
        (149, 165, 166),
        (230, 126, 34),
        (26, 188, 156),
    ]


def _series_style(axis_names: list[str]) -> rr.SeriesLines:
    colors = _palette_rgb()
    per_axis = [colors[i % len(colors)] for i in range(len(axis_names))]
    return rr.SeriesLines.from_fields(names=axis_names, colors=[per_axis], widths=[2.0] * len(axis_names))


def _timeseries_view(*, origin: str, name: str, axis_names: list[str]) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
        origin=origin,
        name=name,
        plot_legend=rrb.Corner2D.RightTop,
        overrides={origin: _series_style(axis_names)},
    )


def _state_group_specs() -> list[GroupSpec]:
    return [
        GroupSpec("State Root",      "/state_body/root",      0,  3,  ["roll", "pitch", "delta_yaw"]),
        GroupSpec("State Left Leg",  "/state_body/left_leg",  3,  9,  [f"l_leg_j{i}" for i in range(1, 7)]),
        GroupSpec("State Right Leg", "/state_body/right_leg", 9,  15, [f"r_leg_j{i}" for i in range(1, 7)]),
        GroupSpec("State Waist",     "/state_body/waist",     15, 18, [f"waist_j{i}" for i in range(1, 4)]),
        GroupSpec("State Left Arm",  "/state_body/left_arm",  18, 25, [f"l_arm_j{i}" for i in range(1, 8)]),
        GroupSpec("State Right Arm", "/state_body/right_arm", 25, 32, [f"r_arm_j{i}" for i in range(1, 8)]),
    ]


def _action_group_specs() -> list[GroupSpec]:
    return [
        GroupSpec("Action Root",      "/action_body/root",      0,  3,  ["roll", "pitch", "delta_yaw"]),
        GroupSpec("Action Left Leg",  "/action_body/left_leg",  3,  9,  [f"l_leg_j{i}" for i in range(1, 7)]),
        GroupSpec("Action Right Leg", "/action_body/right_leg", 9,  15, [f"r_leg_j{i}" for i in range(1, 7)]),
        GroupSpec("Action Waist",     "/action_body/waist",     15, 18, [f"waist_j{i}" for i in range(1, 4)]),
        GroupSpec("Action Left Arm",  "/action_body/left_arm",  18, 25, [f"l_arm_j{i}" for i in range(1, 8)]),
        GroupSpec("Action Right Arm", "/action_body/right_arm", 25, 32, [f"r_arm_j{i}" for i in range(1, 8)]),
    ]


def _make_blueprint(enable_wrist: bool = False) -> rrb.BlueprintLike:
    state_groups = _state_group_specs()
    action_groups = _action_group_specs()

    # State tabs
    state_tabs_list = [_timeseries_view(origin=g.entity_path, name=g.tab_name, axis_names=g.axis_names) for g in state_groups]
    state_tabs_list.append(_timeseries_view(origin="/state_hand", name="State Hand", axis_names=["left_hand", "right_hand"]))
    state_tabs = rrb.Tabs(*state_tabs_list, active_tab=0, name="State")

    # Action tabs
    action_tabs_list = [_timeseries_view(origin=g.entity_path, name=g.tab_name, axis_names=g.axis_names) for g in action_groups]
    action_tabs_list.append(_timeseries_view(origin="/action_hand", name="Action Hand", axis_names=["left_hand", "right_hand"]))
    action_tabs = rrb.Tabs(*action_tabs_list, active_tab=0, name="Action")

    # Camera views
    camera_views: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin="/", name="RGB Left", contents=["/rgb_left"]),
    ]
    if enable_wrist:
        camera_views.extend([
            rrb.Spatial2DView(origin="/", name="Wrist Left",  contents=["/wrist_rgb_left"]),
            rrb.Spatial2DView(origin="/", name="Wrist Right", contents=["/wrist_rgb_right"]),
        ])

    cameras_col = rrb.Vertical(*camera_views, row_shares=[1] * len(camera_views), name="Cameras")

    return rrb.Horizontal(
        cameras_col,
        rrb.Vertical(state_tabs, action_tabs, row_shares=[1, 1], name="State/Action"),
        column_shares=[1, 2],
        name="Data Check",
    )


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------

def _find_episodes(data_path: Path, start_number: int = 0) -> list[Path]:
    """Return sorted list of episode directories inside *data_path*."""
    episodes = []
    if not data_path.exists():
        print(f"[ERROR] Data path does not exist: {data_path}")
        sys.exit(1)
    for entry in sorted(data_path.iterdir()):
        if entry.is_dir() and entry.name.startswith("episode_"):
            number_str = entry.name.split("episode_", 1)[1]
            if not number_str.isdigit():
                continue
            episode_number = int(number_str)
            if episode_number < start_number:
                continue
            json_file = entry / "data.json"
            if json_file.exists():
                episodes.append(entry)
    if not episodes:
        print(f"[WARNING] No episode directories found in: {data_path}")
    return episodes


# ---------------------------------------------------------------------------
# Main playback logic
# ---------------------------------------------------------------------------

def _resolve_video_ref(episode_dir: Path, ref: object) -> tuple[Path, int] | None:
    if not isinstance(ref, dict):
        return None
    video_path = ref.get("video_path")
    frame_index = ref.get("frame_index")
    if video_path is None or frame_index is None:
        return None
    return episode_dir / str(video_path), int(frame_index)


class EpisodeVideoReader:
    """Read RGB frames from one episode's per-camera videos."""

    def __init__(self, episode_dir: Path) -> None:
        self.episode_dir = episode_dir
        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._positions: dict[Path, int] = {}

    def read_rgb(self, frame: dict, key: str) -> np.ndarray | None:
        resolved = _resolve_video_ref(self.episode_dir, frame.get(key))
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

        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None

        self._positions[video_path] = frame_index + 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        self._positions.clear()

def _new_recording(app_id: str, episode_name: str, blueprint: rrb.BlueprintLike, port: int, first: bool) -> rr.RecordingStream:
    """Create a fresh RecordingStream for one episode."""
    rec = rr.RecordingStream(app_id, recording_id=str(uuid.uuid4()))
    if first:
        rec.spawn(port=port, connect=True, detach_process=True)
    else:
        rec.connect_grpc(f"rerun+http://0.0.0.0:{port}/proxy")
    rr.send_recording_name(episode_name, recording=rec)
    rr.send_blueprint(blueprint, make_active=True, make_default=True, recording=rec)
    return rec


def play_episode(
    episode_dir: Path,
    rec: rr.RecordingStream,
    state_groups: list[GroupSpec],
    action_groups: list[GroupSpec],
    fps: float,
    speed: float,
    enable_wrist: bool,
) -> None:
    """Load one episode from disk and stream it to the Rerun viewer."""
    json_path = episode_dir / "data.json"
    with open(json_path, "r", encoding="utf-8") as f:
        episode = json.load(f)

    frames: list[dict] = episode.get("data", [])
    if not frames:
        print(f"  [WARNING] Episode has no data frames: {episode_dir.name}")
        return

    # Prefer FPS from video metadata if available
    info = episode.get("info", {})
    info_fps = info.get("video", {}).get("fps", None) or info.get("image", {}).get("fps", None)
    if info_fps:
        fps = float(info_fps)

    n_frames = len(frames)
    duration_s = n_frames / fps
    print(f"  {n_frames} frames @ {fps:.1f} fps × {speed}x  "
          f"→ target wall-clock {duration_s / speed:.2f}s")

    rr.reset_time(recording=rec)

    wall_start = time.monotonic()
    logged = 0
    skipped = 0
    video_reader = EpisodeVideoReader(episode_dir)

    for step_idx, frame in enumerate(frames):
        orig_time = step_idx / fps          # position in original timeline (s)

        # How far we have consumed in original-timeline time, given elapsed wall clock.
        # Any frame whose orig_time is already behind this threshold is skipped.
        wall_elapsed = time.monotonic() - wall_start
        orig_consumed = wall_elapsed * speed
        if orig_time < orig_consumed - (1.0 / fps):
            skipped += 1
            continue

        # If we're ahead of schedule, sleep until this frame's wall-clock moment.
        target_wall = orig_time / speed
        sleep_ahead = target_wall - (time.monotonic() - wall_start)
        if sleep_ahead > 0:
            time.sleep(sleep_ahead)

        t0 = time.monotonic()

        rr.set_time("step", sequence=int(step_idx), recording=rec)
        rr.set_time("time_s", duration=orig_time, recording=rec)

        # --- Video-backed images ---
        def _log_image(key: str, entity_path: str) -> None:
            img = video_reader.read_rgb(frame, key)
            if img is None:
                return
            img_u8: np.ndarray = np.ascontiguousarray(img, dtype=np.uint8)
            rr.log(entity_path, rr.Image(img_u8, color_model="RGB").compress(jpeg_quality=85), recording=rec)

        _log_image("rgb_left", "/rgb_left")
        if enable_wrist:
            _log_image("wrist_rgb_left", "/wrist_rgb_left")
            _log_image("wrist_rgb_right", "/wrist_rgb_right")

        # --- State body ---
        state_body = frame.get("state_body")
        if state_body is not None:
            sb = np.asarray(state_body, dtype=np.float32).reshape(-1)
            if sb.shape[0] >= 32:
                for g in state_groups:
                    rr.log(g.entity_path, rr.Scalars(sb[g.start:g.end]), recording=rec)

        # --- Action body ---
        action_body = frame.get("action_body")
        if action_body is not None:
            ab = np.asarray(action_body, dtype=np.float32).reshape(-1)
            if ab.shape[0] >= 32:
                for g in action_groups:
                    rr.log(g.entity_path, rr.Scalars(ab[g.start:g.end]), recording=rec)

        # --- Hands ---
        shl = frame.get("state_hand_left")
        shr = frame.get("state_hand_right")
        if shl is not None and shr is not None:
            lv = float(np.asarray(shl).reshape(-1)[0])
            rv = float(np.asarray(shr).reshape(-1)[0])
            rr.log("/state_hand", rr.Scalars([lv, rv]), recording=rec)

        ahl = frame.get("action_hand_left")
        ahr = frame.get("action_hand_right")
        if ahl is not None and ahr is not None:
            lv = float(np.asarray(ahl).reshape(-1)[0])
            rv = float(np.asarray(ahr).reshape(-1)[0])
            rr.log("/action_hand", rr.Scalars([lv, rv]), recording=rec)

        logged += 1

    video_reader.close()

    wall_elapsed = time.monotonic() - wall_start
    print(f"  Done: {episode_dir.name}  "
          f"logged {logged}/{n_frames} frames  "
          f"(skipped {skipped})  wall {wall_elapsed:.2f}s")


def _detect_wrist(episodes: list[Path]) -> bool:
    """Check the first episode to see if wrist camera videos exist."""
    if not episodes:
        return False
    json_path = episodes[0] / "data.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            ep = json.load(f)
        cameras = ep.get("info", {}).get("video", {}).get("cameras", {})
        if "wrist_rgb_left" in cameras or "wrist_rgb_right" in cameras:
            return True
        frames = ep.get("data", [])
        if frames:
            return (
                isinstance(frames[0].get("wrist_rgb_left"), dict)
                or isinstance(frames[0].get("wrist_rgb_right"), dict)
            )
    except Exception:
        pass
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded episodes with Rerun at a specified playback speed."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the data folder (e.g., sonic_demonstration/20260401_1519_pp_sprite)",
    )
    parser.add_argument(
        "speed",
        type=float,
        help="Playback speed multiplier (e.g., 8 for 8× speed, 0.5 for half speed)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS if not stored in episode metadata (default: 30)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9876,
        help="Rerun viewer gRPC port (default: 9876)",
    )
    parser.add_argument(
        "--start-number",
        type=int,
        default=0,
        help="Start checking from this episode number, e.g. 8 means from episode_0008 (default: 0)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    speed: float = args.speed
    fps: float = args.fps
    start_number: int = args.start_number

    print(f"Data path   : {data_path}")
    print(f"Speed       : {speed}×")
    print(f"Fallback FPS: {fps}")
    print(f"Start number: {start_number}")

    episodes = _find_episodes(data_path, start_number=start_number)
    if not episodes:
        sys.exit(1)
    print(f"Found {len(episodes)} episode(s): {[e.name for e in episodes]}\n")

    enable_wrist = _detect_wrist(episodes)
    if enable_wrist:
        print("Wrist cameras detected.\n")

    state_groups = _state_group_specs()
    action_groups = _action_group_specs()
    blueprint = _make_blueprint(enable_wrist=enable_wrist)

    for i, ep_dir in enumerate(episodes):
        print(f"Episode: {ep_dir.name}")
        # Each episode gets its own recording slot; only the first one spawns the viewer.
        rec = _new_recording(
            app_id="data_check",
            episode_name=ep_dir.name,
            blueprint=blueprint,
            port=args.port,
            first=(i == 0),
        )
        play_episode(
            episode_dir=ep_dir,
            rec=rec,
            state_groups=state_groups,
            action_groups=action_groups,
            fps=fps,
            speed=speed,
            enable_wrist=enable_wrist,
        )
        try:
            rec.disconnect()
        except Exception:
            pass
        # Brief pause between episodes so the viewer can settle
        time.sleep(0.3)

    print("\nAll episodes played. Rerun viewer remains open.")


if __name__ == "__main__":
    main()
