import json
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction

import av
from exiftool import ExifToolHelper
from loguru import logger

from .trajectory import EpFrameDict, TrajPayload
from .video import mp4_get_start_datetime


@dataclass(frozen=True)
class VideoInfo:
    path: pathlib.Path
    start_timestamp: float
    end_timestamp: float
    camera_serial: str
    n_frames: int
    fps: Fraction

    @classmethod
    def from_file(
        cls, video_path: pathlib.Path, gopro_timezone: str | None = None
    ) -> "VideoInfo":
        with ExifToolHelper() as et:
            meta = list(et.get_metadata(str(video_path)))[0]
            camera_serial: str = meta["QuickTime:CameraSerialNumber"]

        start_datetime = mp4_get_start_datetime(str(video_path))

        if gopro_timezone is not None:
            import re
            import zoneinfo
            from datetime import timedelta, timezone

            match = re.match(r"^([+-])(\d{2}):?(\d{2})$", gopro_timezone)
            if match:
                sign = 1 if match.group(1) == "+" else -1
                hours = int(match.group(2))
                minutes = int(match.group(3))
                tz = timezone(timedelta(minutes=sign * (hours * 60 + minutes)))
            else:
                tz = zoneinfo.ZoneInfo(gopro_timezone)

            start_datetime = start_datetime.replace(tzinfo=tz)

        start_ts = start_datetime.timestamp()

        n_frames = 0
        fps: Fraction | None = None

        with av.open(str(video_path), "r") as container:
            stream = container.streams.video[0]
            n_frames = stream.frames
            fps = stream.average_rate
        assert fps is not None, f"Video {video_path} has no valid frame rate."
        end_ts = start_ts + float(n_frames / fps)

        return cls(
            path=video_path.absolute(),
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            camera_serial=camera_serial,
            n_frames=int(n_frames),
            fps=fps,
        )

    def save_as_gripper_calibration(self, session_dir: pathlib.Path):
        """Create symbol link in demos/ for gripper calibration video.
        Skip if the symlink already exists.
        """
        logger.info(f"Use video {self.path} as gripper calibration video.")
        demos_dir = session_dir / "demos"
        gripper_cal_dir = (
            demos_dir
            / f"gripper_calibration_{self.camera_serial}_{datetime.fromtimestamp(self.start_timestamp).strftime('%Y%m%d_%H%M%S.%f')}"
        )
        gripper_cal_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = gripper_cal_dir / "raw_video.mp4"
        if symlink_path.exists():
            logger.warning(
                f"Symlink path {symlink_path} already exists. Skipping."
            )
            return

        raw_videos_dir = session_dir / "raw_videos"
        target_dir = raw_videos_dir / "gripper_calibration"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = (
            self.path.relative_to(session_dir).as_posix().replace("/", "_")
        )
        target_path = target_dir / target_name
        self.path.rename(target_path)
        logger.debug(f"Moved original video {self.path} to {target_path}")
        rel_target_path = os.path.relpath(target_path, start=gripper_cal_dir)
        symlink_path.symlink_to(rel_target_path)

        # Clean up empty parent directory
        self.clean_up_empty_parent()

    def clean_up_empty_parent(self):
        """If the parent directory is empty after moving the video,
        remove the parent directory.
        """
        if not self.path.parent.exists():
            return
        if not any(self.path.parent.iterdir()):
            logger.debug(f"Removing empty parent directory {self.path.parent}")
            self.path.parent.rmdir()

    def get_normalized_target_path(
        self,
        session_dir: pathlib.Path,
        mark_unmatched: bool = False,
        mark_unpaired: bool = False,
    ) -> pathlib.Path:
        """
        Get target path under 'raw_videos' directory with normalized naming.
        - If mark_unmatched is True, append '_unmatched_with_traj' before the file extension.
        - If mark_unpaired is True, append '_unpaired' before the file extension.
        """
        if mark_unmatched and mark_unpaired:
            raise ValueError(
                f"Cannot mark {self.path} as both unmatched and unpaired."
            )
        raw_videos_dir = session_dir / "raw_videos"
        target_name = (
            self.path.relative_to(session_dir).as_posix().replace("/", "_")
        )
        # Append '_unmatched' if needed
        if mark_unmatched and not target_name.endswith(
            "_unmatched_with_traj.mp4"
        ):
            target_stem = target_name.rsplit(".", 1)[
                0
            ]  # Remove file extension
            target_name = target_stem + "_unmatched_with_traj.mp4"
        if not mark_unmatched and target_name.endswith(
            "_unmatched_with_traj.mp4"
        ):
            target_name = (
                target_name.rsplit("_unmatched_with_traj", 1)[0] + ".mp4"
            )
        if mark_unpaired and not target_name.endswith("_unpaired.mp4"):
            target_stem = target_name.rsplit(".", 1)[
                0
            ]  # Remove file extension
            target_name = target_stem + "_unpaired.mp4"
        if not mark_unpaired and target_name.endswith("_unpaired.mp4"):
            target_name = target_name.rsplit("_unpaired", 1)[0] + ".mp4"
        target_path = raw_videos_dir / target_name
        return target_path

    def save_as_unpaired(self, session_dir: pathlib.Path):
        """Move unmatched video into raw_videos/<name>_unpaired.mp4."""
        unmatched_dir = session_dir / "raw_videos"
        unmatched_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.get_normalized_target_path(
            session_dir, mark_unpaired=True
        )
        self.path.rename(target_path)
        logger.debug(f"Moved unpaired video {self.path} to {target_path}")
        # Clean up empty parent directory
        self.clean_up_empty_parent()

    def save_as_unmatched(self, session_dir: pathlib.Path):
        """Move unmatched video into raw_videos/<name>_unmatched_with_traj.mp4."""
        unmatched_dir = session_dir / "raw_videos"
        unmatched_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.get_normalized_target_path(
            session_dir, mark_unmatched=True
        )
        self.path.rename(target_path)
        logger.debug(f"Moved unmatched video {self.path} to {target_path}")
        # Clean up empty parent directory
        self.clean_up_empty_parent()


@dataclass(frozen=True)
class TrajectoryInfo:
    path: pathlib.Path
    start_timestamp: float
    end_timestamp: float
    is_ik_processed: bool

    @classmethod
    def from_file(cls, traj_path: pathlib.Path) -> "TrajectoryInfo":
        with traj_path.open("r") as f:
            data = json.load(f)

        is_ik_processed = "mjcf_path" in data

        if isinstance(data, dict) and "episode" in data:
            episode: list[EpFrameDict] = data["episode"]
        elif isinstance(data, list):
            episode = data
        else:
            raise ValueError(f"Invalid trajectory format in {traj_path}")

        start_ts = float(episode[0]["timestamp"])
        end_ts = float(episode[-1]["timestamp"])

        return cls(
            path=traj_path.absolute(),
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            is_ik_processed=is_ik_processed,
        )

    def clean_up_empty_parent(self):
        """If the parent directory is empty after moving the trajectory,
        remove the parent directory.
        """
        if not self.path.parent.exists():
            return
        if not any(self.path.parent.iterdir()):
            logger.debug(f"Removing empty parent directory {self.path.parent}")
            self.path.parent.rmdir()

    def get_normalized_target_path(
        self, session_dir: pathlib.Path, mark_unmatched: bool = False
    ) -> pathlib.Path:
        """
        Get target path under 'raw_trajectories' directory with normalized naming.
        - Ensure the target name starts with 'recording_'
        - If mark_unmatched is True, append '_unmatched' before the file extension.
        """
        raw_traj_dir = session_dir / "raw_trajectories"
        target_name = self.path.name
        # Ensure the target name starts with 'recording_'
        if not target_name.startswith("recording_"):
            target_name = "recording_" + target_name
        # Append '_unmatched' if needed
        if mark_unmatched and not target_name.endswith("_unmatched.json"):
            target_stem = target_name.rsplit(".", 1)[
                0
            ]  # Remove file extension
            target_name = target_stem + "_unmatched.json"
        if not mark_unmatched and target_name.endswith("_unmatched.json"):
            target_name = target_name.rsplit("_unmatched", 1)[0] + ".json"
        target_path = raw_traj_dir / target_name
        return target_path

    def save_as_unmatched(self, session_dir: pathlib.Path):
        """Move unmatched trajectory into raw_trajectories/unmatched/ directory."""
        target_path = self.get_normalized_target_path(
            session_dir, mark_unmatched=True
        )
        self.path.rename(target_path)
        logger.debug(
            f"Moved unmatched trajectory {self.path} to {target_path}"
        )
        # Clean up empty parent directory
        self.clean_up_empty_parent()


@dataclass(frozen=True)
class VideoPair:
    video0: VideoInfo
    video1: VideoInfo

    @property
    def start_timestamp(self) -> float:
        return max(self.video0.start_timestamp, self.video1.start_timestamp)

    @property
    def end_timestamp(self) -> float:
        return min(self.video0.end_timestamp, self.video1.end_timestamp)

    def as_tuple(self) -> tuple[VideoInfo, VideoInfo]:
        return (self.video0, self.video1)


@dataclass(frozen=True)
class DemoGroup:
    video_pair: VideoPair
    trajectory: TrajectoryInfo

    @property
    def start_timestamp(self) -> float:
        return self.video_pair.start_timestamp

    @property
    def end_timestamp(self) -> float:
        return self.video_pair.end_timestamp

    def save_and_symlink(self, session_dir: pathlib.Path):
        """Move videos and trajectory into raw_videos/ and raw_trajectories/,
        then create symlinks in demos/ directory.
        """
        start_datetime = datetime.fromtimestamp(self.start_timestamp)
        start_dt_str = start_datetime.strftime("%Y%m%d_%H%M%S.%f")
        demos_dir = session_dir / "demos"
        demo_dir = demos_dir / f"demo_{start_dt_str}"
        demo_dir.mkdir(parents=True, exist_ok=True)

        vis = (self.video_pair.video0, self.video_pair.video1)
        for vi in vis:
            cam_start_datetime = datetime.fromtimestamp(vi.start_timestamp)
            cam_start_dt_str = cam_start_datetime.strftime("%Y%m%d_%H%M%S.%f")
            cam_dir = demo_dir / f"{vi.camera_serial}_{cam_start_dt_str}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            video_symlink_path = cam_dir / "raw_video.mp4"
            if video_symlink_path.exists():
                logger.warning(
                    f"Symlink path {video_symlink_path} already exists. Skipping."
                )
                continue

            video_target_path = vi.get_normalized_target_path(session_dir)
            video_target_path.parent.mkdir(parents=True, exist_ok=True)
            vi.path.rename(video_target_path)
            logger.debug(
                f"Moved original video {vi.path} to {video_target_path}"
            )
            video_rel_target_path = os.path.relpath(
                video_target_path, start=cam_dir
            )
            video_symlink_path.symlink_to(video_rel_target_path)

        traj_start_datetime = datetime.fromtimestamp(
            self.trajectory.start_timestamp
        )
        traj_start_dt_str = traj_start_datetime.strftime("%Y%m%d_%H%M%S.%f")
        traj_symlink_path = demo_dir / f"recording_{traj_start_dt_str}.json"
        if traj_symlink_path.exists():
            self._trim_trajectory(session_dir, traj_symlink_path.resolve())
            logger.warning(
                f"Symlink path {traj_symlink_path} already exists. Skipping."
            )
            return
        else:
            self._trim_trajectory(session_dir, self.trajectory.path.resolve())
        traj_target_name = (
            self.trajectory.path.relative_to(session_dir)
            .as_posix()
            .replace("/", "_")
        )
        # Need to prefix with 'recording_' to match naming convention
        if not traj_target_name.startswith("recording_"):
            traj_target_name = "recording_" + traj_target_name
        traj_target_path = session_dir / "raw_trajectories" / traj_target_name
        traj_target_path.parent.mkdir(parents=True, exist_ok=True)
        # Trajectory can be matched multiple times
        if traj_target_path.exists():
            logger.warning(
                f"Trajectory target path {traj_target_path} already exists. Skipping moving trajectory."
            )
        else:
            self.trajectory.path.rename(traj_target_path)
            logger.debug(
                f"Moved original trajectory {self.trajectory.path} to {traj_target_path}"
            )
        traj_rel_target_path = os.path.relpath(
            traj_target_path, start=demo_dir
        )
        traj_symlink_path.symlink_to(traj_rel_target_path)

        # Clean up empty parent directories
        self.clean_up_empty_parent()

    def clean_up_empty_parent(self):
        """Clean up empty parent directories of videos and trajectory."""
        self.video_pair.video0.clean_up_empty_parent()
        self.video_pair.video1.clean_up_empty_parent()
        self.trajectory.clean_up_empty_parent()

    @classmethod
    def from_demo_dir(
        cls, demo_dir_path: pathlib.Path, gopro_timezone: str | None = None
    ) -> "DemoGroup":
        """Load DemoGroup from existing demo directory."""
        video_dirs = [x.parent for x in demo_dir_path.glob("**/raw_video.mp4")]
        if len(video_dirs) != 2:
            raise ValueError(
                f"Expected 2 video directories in {demo_dir_path}, found {len(video_dirs)}"
            )
        vis = [
            VideoInfo.from_file(
                video_dir / "raw_video.mp4", gopro_timezone=gopro_timezone
            )
            for video_dir in video_dirs
        ]
        vis.sort(key=lambda x: x.camera_serial)

        traj_paths = list(demo_dir_path.glob("recording_*.json"))
        if len(traj_paths) != 1:
            raise ValueError(
                f"Expected 1 trajectory file in {demo_dir_path}, found {len(traj_paths)}"
            )
        traj_info = TrajectoryInfo.from_file(traj_paths[0])

        return cls(
            video_pair=VideoPair(video0=vis[0], video1=vis[1]),
            trajectory=traj_info,
        )

    def _trim_trajectory(
        self,
        session_dir: pathlib.Path,
        traj_path: pathlib.Path,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        *,
        early_start_time: float = 5.0,
    ) -> None:
        """Trim the trajectory to match the demo group's start and end timestamps.
        Trimmed trajectory will be saved as `session_dir/trimmed_trajectories/trimmed_<original_name>.json`.
        """
        if start_timestamp is None:
            _start_timestamp = self.start_timestamp - early_start_time
        else:
            _start_timestamp = start_timestamp
        if end_timestamp is None:
            _end_timestamp = self.end_timestamp
        else:
            _end_timestamp = end_timestamp

        with traj_path.open("r") as f:
            traj_payload: TrajPayload = json.load(f)
        trimmed_episode: list[EpFrameDict] = []
        for ep in traj_payload["episode"]:
            timestamp = float(ep["timestamp"])
            if _start_timestamp <= timestamp <= _end_timestamp:
                trimmed_episode.append(ep)
        trimmed_traj_payload: TrajPayload = traj_payload.copy()
        trimmed_traj_payload["episode"] = trimmed_episode

        traj_start_datetime = datetime.fromtimestamp(
            self.trajectory.start_timestamp
        )
        traj_start_dt_str = traj_start_datetime.strftime("%Y%m%d_%H%M%S.%f")

        trimmed_traj_name = f"trimmed_recording_{traj_start_dt_str}.json"
        trimmed_traj_path = (
            session_dir / "trimmed_trajectories" / trimmed_traj_name
        )
        with trimmed_traj_path.open("w") as f:
            json.dump(trimmed_traj_payload, f, indent=4)
        logger.debug(
            f"Saved trimmed trajectory to {trimmed_traj_path} with {len(trimmed_episode)} frames."
        )


__all__ = [
    "VideoInfo",
    "TrajectoryInfo",
    "VideoPair",
    "DemoGroup",
]
