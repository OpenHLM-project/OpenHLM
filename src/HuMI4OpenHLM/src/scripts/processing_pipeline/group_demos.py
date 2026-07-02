import pathlib
import sys
from typing import Optional

import tyro
from loguru import logger

from processing import (
    DemoGroup,
    TrajectoryInfo,
    VideoInfo,
    VideoPair,
)


def process_one_session(
    session_dir: pathlib.Path, gopro_timezone: Optional[str] = None
):
    session_dir = session_dir.resolve().absolute()
    logger.info(f"Processing session directory: {session_dir}")

    raw_videos_dir = session_dir / "raw_videos"
    raw_trajectories_dir = session_dir / "raw_trajectories"
    trimmed_trajectories_dir = session_dir / "trimmed_trajectories"
    demos_dir = session_dir / "demos"

    if not raw_videos_dir.exists():
        raw_videos_dir.mkdir(parents=True)
        logger.info(f"Created raw_videos directory: {raw_videos_dir}")
    if not raw_trajectories_dir.exists():
        raw_trajectories_dir.mkdir(parents=True)
        logger.info(
            f"Created raw_trajectories directory: {raw_trajectories_dir}"
        )
    if not trimmed_trajectories_dir.exists():
        trimmed_trajectories_dir.mkdir(parents=True)
        logger.info(
            f"Created trimmed_trajectories directory: {trimmed_trajectories_dir}"
        )
    if not demos_dir.exists():
        demos_dir.mkdir(parents=True)
        logger.info(f"Created demos directory: {demos_dir}")

    # Get all video infos and trajectory infos
    video_infos = [
        VideoInfo.from_file(video_path, gopro_timezone=gopro_timezone)
        for video_path in list(session_dir.rglob("*.mp4"))
        + list(session_dir.rglob("*.MP4"))
        if not video_path.is_symlink()
    ]
    all_traj_infos = [
        TrajectoryInfo.from_file(traj_path)
        for traj_path in session_dir.rglob("recording_*.json")
        if not traj_path.is_symlink()
    ]
    traj_infos = [ti for ti in all_traj_infos if ti.is_ik_processed]
    ignored_raw_trajs = [ti for ti in all_traj_infos if not ti.is_ik_processed]
    if ignored_raw_trajs:
        logger.info(
            f"Ignored {len(ignored_raw_trajs)} raw trajectories (not IK processed)."
        )

    # Get all serial numbers
    all_serials = set(vi.camera_serial for vi in video_infos)

    assert len(all_serials) == 2, (
        f"Expected 2 unique camera serials, found {len(all_serials)}: {all_serials}"
    )
    serial0, serial1 = list(all_serials)

    # Mark the earliest video for each serial as gripper calibration
    serial_to_videos: dict[str, list[VideoInfo]] = {
        serial: [] for serial in all_serials
    }
    for vi in video_infos:
        serial_to_videos[vi.camera_serial].append(vi)

    for _serial, vis in serial_to_videos.items():
        vis.sort(key=lambda x: x.start_timestamp)
        gripper_cal_video = vis.pop(0)  # earliest video
        gripper_cal_video.save_as_gripper_calibration(session_dir)

    # Match remaining videos to trajectories
    # Videos can be matched only once
    # But trajectories can be matched multiple times
    def _video_overlaps_video(this_vi: VideoInfo, other_vi: VideoInfo) -> bool:
        return not (
            this_vi.end_timestamp <= other_vi.start_timestamp
            or this_vi.start_timestamp >= other_vi.end_timestamp
        )

    # Double pointer to find overlapping video pairs
    # Note that sorting is done based on start_timestamp
    video_pairs: list[VideoPair] = []
    unpaired_videos: set[VideoInfo] = set()
    i, j = 0, 0
    while i < len(serial_to_videos[serial0]) and j < len(
        serial_to_videos[serial1]
    ):
        vi0 = serial_to_videos[serial0][i]
        vi1 = serial_to_videos[serial1][j]
        if _video_overlaps_video(vi0, vi1):
            video_pairs.append(VideoPair(video0=vi0, video1=vi1))
            # Move both pointers forward
            i += 1
            j += 1
        else:
            # Move the pointer with the earlier ending video
            if vi0.end_timestamp < vi1.end_timestamp:
                i += 1
                unpaired_videos.add(vi0)
                logger.warning(
                    f"Video {vi0.path} did not find a matching pair video."
                )
            else:
                j += 1
                unpaired_videos.add(vi1)
                logger.warning(
                    f"Video {vi1.path} did not find a matching pair video."
                )
    # Handle remaining unpaired videos
    for k in range(i, len(serial_to_videos[serial0])):
        vi0 = serial_to_videos[serial0][k]
        unpaired_videos.add(vi0)
        logger.warning(f"Video {vi0.path} did not find a matching pair video.")
    for k in range(j, len(serial_to_videos[serial1])):
        vi1 = serial_to_videos[serial1][k]
        unpaired_videos.add(vi1)
        logger.warning(f"Video {vi1.path} did not find a matching pair video.")

    # Save unpaired videos
    for uv in unpaired_videos:
        uv.save_as_unpaired(session_dir)

    def _traj_covers_video_pair(traj: TrajectoryInfo, vp: VideoPair) -> bool:
        return (
            traj.start_timestamp <= vp.start_timestamp
            and traj.end_timestamp >= vp.end_timestamp
        )

    demo_groups: list[DemoGroup] = []
    video_pairs_wo_traj: set[VideoPair] = set()
    used_traj: set[TrajectoryInfo] = set()
    for vp in video_pairs:
        for traj in traj_infos:
            if _traj_covers_video_pair(traj, vp):
                demo_groups.append(DemoGroup(video_pair=vp, trajectory=traj))
                used_traj.add(traj)
                break
        else:
            video_pairs_wo_traj.add(vp)
            logger.warning(
                f"Video pair ({vp.video0.path}, {vp.video1.path}) did not find a matching trajectory."
            )
    unused_traj = set(traj_infos) - used_traj
    for traj in unused_traj:
        logger.warning(
            f"Trajectory {traj.path} was not used to match any video pair."
        )

    # Save demo groups
    for dg in demo_groups:
        dg.save_and_symlink(session_dir)

    # Save unmatched videos
    for vp in video_pairs_wo_traj:
        vp.video0.save_as_unmatched(session_dir)
        vp.video1.save_as_unmatched(session_dir)
    # Save unmatched trajectories
    for traj in unused_traj:
        traj.save_as_unmatched(session_dir)

    # Summarize unmatched videos and trajectories
    if (
        len(unpaired_videos) > 0
        or len(video_pairs_wo_traj) > 0
        or len(unused_traj) > 0
    ):
        logger.warning("Summary of unmatched items:")
        logger.warning(f"Total unmatched videos: {len(unpaired_videos)}")
        logger.warning(
            f"Total unmatched video pairs: {len(video_pairs_wo_traj)}"
        )
        logger.warning(f"Total unmatched trajectories: {len(unused_traj)}")
    else:
        logger.info(
            "All videos and trajectories have been successfully matched."
        )


def main(
    session_dirs: tuple[pathlib.Path, ...],
    /,
    htc_config: Optional[pathlib.Path] = None,
    gopro_timezone: Optional[str] = None,
    verbose: bool = False,
):
    """Group mp4s and jsons into demo dirs based on timestamps.

    We assume trajectories' timestamps always cover the video timestamps.

    The resulting structure is:
    ```text
    data/test-pipeline
    ├── demos
    │   ├── demo_20251123_173856
    │   │   ├── C3461326849443_20251123_173856
    │   │   │   └── raw_video.mp4 -> ../../../raw_videos/card0_00.mp4
    │   │   ├── C3461326882238_20251123_173856
    │   │   │   └── raw_video.mp4 -> ../../../raw_videos/card1_00.mp4
    │   │   └── recording_20251123_173849.json -> ../../raw_trajectories/recording_2025.11.23_17.38.49.283318.json
    │   ├── demo_20251123_173928
    │   │   ├── C3461326849443_20251123_173928
    │   │   │   └── raw_video.mp4 -> ../../../raw_videos/card0_01.mp4
    │   │   ├── C3461326882238_20251123_173928
    │   │   │   └── raw_video.mp4 -> ../../../raw_videos/card1_01.mp4
    │   │   └── recording_20251123_173920.json -> ../../raw_trajectories/recording_2025.11.23_17.39.20.585779.json
    │   ├── gripper_calibration_C3461326849443_20251123_173627
    │   │   └── raw_video.mp4 -> ../../raw_videos/gripper_calibration/card0_grip.mp4
    │   └── gripper_calibration_C3461326882238_20251123_173636
    │       └── raw_video.mp4 -> ../../raw_videos/gripper_calibration/card1_grip.mp4
    ├── raw_trajectories
    │   ├── recording_2025.11.23_17.38.49.283318.json
    │   └── recording_2025.11.23_17.39.20.585779.json
    └── raw_videos
        ├── card0_00.mp4
        ├── card0_01.mp4
        ├── card1_00.mp4
        ├── card1_01.mp4
        └── gripper_calibration
            ├── card0_grip.mp4
            └── card1_grip.mp4
    ```

    Args:
        session_dirs (tuple[pathlib.Path, ...]): Session directories containing raw videos and trajectories.
        htc_config (pathlib.Path | None, optional): Not used. Defaults to None.
        gopro_timezone (str | None, optional): Timezone of the GoPro cameras. Defaults to system timezone.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <level>{level: <8}</level> | {file.name}:{line} - {message}",
        colorize=True,
        level="DEBUG" if verbose else "INFO",
    )
    for session_dir in session_dirs:
        process_one_session(session_dir, gopro_timezone)


if __name__ == "__main__":
    tyro.cli(main)
