import json
import math
import pathlib
import pickle
import sys
from dataclasses import asdict, dataclass, replace
from typing import Annotated, Optional

import numpy as np
import numpy.typing as npt
import scipy.interpolate as si
import tyro
import yaml
from loguru import logger

from processing import (
    EXPECTED_ACTUATED_JOINTS,
    DemoGroup,
    RoleEnum,
    TrajectoryInfo,
    TrajPayload,
    VideoInfo,
    VideoPair,
    get_recording_joint_idx_to_poselib,
    load_prompt_from_json,
    load_trajectory_from_json,
    traj_to_pd_dataframe,
)
from processing.cv_util import (
    TagDetectionResultDict,
    detect_gripper_id,
    get_gripper_width,
)
from processing.interpolation_util import (
    get_gripper_calibration_interpolator,
    get_interp1d,
    get_tcp_pose_interpolator,
)


def gripper_id_from_video_info(video_info: VideoInfo, nominal_z: float) -> int:
    """Extract gripper id from VideoInfo object based on tag detection.
    Args:
        video_info: VideoInfo object.
        nominal_z: Nominal Z value for gripper finger tag detection.
    Returns:
        Gripper id as an integer.
    """
    tag_path = video_info.path.parent / "tag_detection.pkl"
    gripper_id = detect_gripper_id(tag_path, nominal_z=nominal_z)
    if gripper_id is None:
        raise ValueError(
            f"Could not detect gripper id from tag detection file: {tag_path}"
        )
    return gripper_id


def correct_video_pair_timestamp(video_pair: VideoPair) -> VideoPair:
    """Correct video pair timestamps based on their offset with trajectory timestamps
    (Read from `timestamp_calibration.json` file in the same directory as video files.).
    Note that we use tracker's trajectory timestamps as ground truth.
    """

    def _correct_video_info_timestamp(vi: VideoInfo) -> VideoInfo:
        timestamp_calibration_path = (
            vi.path.parent / "timestamp_calibration.json"
        )
        if not timestamp_calibration_path.exists():
            raise FileNotFoundError(
                f"Timestamp calibration file not found: {timestamp_calibration_path}"
            )
        with open(timestamp_calibration_path) as f:
            cal_data = json.load(f)
            time_offset: float = float(cal_data["optimal_time_offset_seconds"])
        corrected_start_timestamp = vi.start_timestamp + time_offset
        corrected_end_timestamp = vi.end_timestamp + time_offset

        return replace(
            vi,
            start_timestamp=corrected_start_timestamp,
            end_timestamp=corrected_end_timestamp,
        )

    corrected_video0 = _correct_video_info_timestamp(video_pair.video0)
    corrected_video1 = _correct_video_info_timestamp(video_pair.video1)
    return VideoPair(video0=corrected_video0, video1=corrected_video1)


def get_joint_pos_array(
    timestamps: npt.NDArray, traj_info: TrajectoryInfo
) -> npt.NDArray[np.float64]:
    """Get joint position array aligned with given timestamps.
    Args:
        timestamps: Timestamps to align joint positions to.
        traj_info: TrajectoryInfo object containing trajectory JSON path.
    Returns:
        Joint position array of shape (N, D) where N is number of timestamps and D is number of joints.
    """
    with open(traj_info.path) as f:
        traj_payload: TrajPayload = json.load(f)

    episode = traj_payload["episode"]
    ts = [frame["timestamp"] for frame in episode]
    act_qs = [episode_frame["q"][7:] for episode_frame in episode]
    # Assert timestamps are within trajectory time range
    assert timestamps[0] >= ts[0] and timestamps[-1] <= ts[-1], (
        f"Timestamps {timestamps[0]} to {timestamps[-1]} are out of trajectory time range {ts[0]} to {ts[-1]}"
    )
    q_interp = si.interp1d(
        x=np.array(ts),
        y=np.array(act_qs),
        axis=0,
    )
    q_resampled: npt.NDArray[np.float64] = q_interp(timestamps)
    recording_jid_to_poselib = get_recording_joint_idx_to_poselib(
        pathlib.Path(traj_payload["mjcf_path"])
    )
    joint_pos = np.zeros(
        (len(timestamps), len(EXPECTED_ACTUATED_JOINTS)), dtype=np.float64
    )
    for recording_idx, poselib_idx in recording_jid_to_poselib.items():
        joint_pos[:, poselib_idx] = q_resampled[:, recording_idx]
    return joint_pos


@dataclass(frozen=True)
class CameraPlan:
    gripper_id: int
    video_path: str
    video_start_end: tuple[int, int]

    @classmethod
    def from_video_pair(
        cls,
        video_pair: VideoPair,
        nominal_z: float,
    ) -> tuple[list["CameraPlan"], npt.NDArray[np.float64]]:
        """Carefully align timestamps between videos.
        Caveat: Video timestamps are discrete: they only exist at frame times.
        Decide which video as the start time reference
        ```text
        Video0: |-----|-----|-----|-----|
        Video1: ----|-----|-----|-----|
        t_min:  ^
        start_ts    ^
        t_max:                        ^
        ```

        Args:
            video_pair: VideoPair object containing two VideoInfo objects.
            latency_dict: Dictionary mapping video paths to their respective latencies.
                Note that we use tracker's trajectory timestamps as ground truth, so we need to
                adjust video timestamps by subtracting latency.
            nominal_z: Nominal Z value for gripper finger tag detection.
        Returns:
            `Tuple[List[CameraPlan], numpy.ndarray]`: List of CameraPlan objects and aligned timestamps.
        """
        # Correct video pair timestamps based on latency
        video_pair = correct_video_pair_timestamp(video_pair)
        if video_pair.video0.fps != video_pair.video1.fps:
            raise ValueError(
                f"Video fps mismatch: {video_pair.video0.path}: {video_pair.video0.fps}, {video_pair.video1.path}: {video_pair.video1.fps}"
            )
        dt = 1 / video_pair.video0.fps
        t_min = video_pair.start_timestamp
        t_max = video_pair.end_timestamp

        # Decide which video as the start time reference
        # Video0: |-----|-----|-----|-----|
        # Video1: ----|-----|-----|-----|
        #         ^   ^                 ^
        #       t_min start_time        t_max
        align_cost: dict[VideoInfo, float] = {}
        for vi in video_pair.as_tuple():
            align_cost[vi] = float(
                sum(
                    [
                        (vj.start_timestamp - vi.start_timestamp) % dt
                        for vj in video_pair.as_tuple()
                    ]
                )
            )
        referen_video = min(align_cost, key=lambda k: align_cost[k])
        # start_timestamp should be aligned to frame time of reference video
        eps = 1e-6
        n_frames_offset = math.ceil(
            (t_min - referen_video.start_timestamp) / dt - eps
        )
        start_timestamp_aligned = (
            referen_video.start_timestamp + n_frames_offset * dt
        )

        # Get start and end frame indices for both videos
        n_frames = math.floor((t_max - start_timestamp_aligned) / dt + eps)
        start_frame_dict: dict[VideoInfo, int] = {}
        for vi in video_pair.as_tuple():
            assert start_timestamp_aligned >= vi.start_timestamp - eps, (
                f"Aligned start timestamp {start_timestamp_aligned} is before video start timestamp {vi.start_timestamp}"
            )
            start_frame_idx = math.ceil(
                (start_timestamp_aligned - vi.start_timestamp) / dt - eps
            )
            assert start_frame_idx >= 0, (
                f"Computed start frame index {start_frame_idx} is negative for video {vi.path}"
            )
            end_frame_idx = math.floor((t_max - vi.start_timestamp) / dt + eps)
            assert end_frame_idx <= vi.n_frames, (
                f"Computed end frame index {end_frame_idx} exceeds number of frames {vi.n_frames} for video {vi.path}"
            )
            n_frames = min(n_frames, end_frame_idx - start_frame_idx)
            start_frame_dict[vi] = start_frame_idx
        aligned_timestamps = start_timestamp_aligned + np.arange(
            n_frames, dtype=np.float64
        ) * float(dt)

        gid_to_camera_plans: dict[int, CameraPlan] = {}

        for vi in video_pair.as_tuple():
            gid = gripper_id_from_video_info(vi, nominal_z=nominal_z)
            camera_plan = CameraPlan(
                video_path=vi.path.absolute().as_posix(),
                video_start_end=(
                    start_frame_dict[vi],
                    start_frame_dict[vi] + n_frames,
                ),
                gripper_id=gid,
            )
            gid_to_camera_plans[gid] = camera_plan

        sorted_gripper_ids = sorted(gid_to_camera_plans.keys())
        # Assert gripper ids are contiguous
        assert sorted_gripper_ids == list(range(len(sorted_gripper_ids))), (
            f"Gripper ids in video pair are not contiguous: {sorted_gripper_ids}"
        )
        camera_plans = [
            gid_to_camera_plans[gripper_id]
            for gripper_id in sorted_gripper_ids
        ]

        return camera_plans, aligned_timestamps

    @property
    def tag_detection_path(self) -> pathlib.Path:
        """Get tag detection file path corresponding to this camera plan."""
        return pathlib.Path(self.video_path).parent / "tag_detection.pkl"


@dataclass(frozen=True)
class GripperPlan:
    gripper_id: int
    tcp_pose: npt.NDArray[np.float64]
    gripper_width: Optional[npt.NDArray[np.float64]]
    """`None` for non-gripper end-effectors, e.g., root and feet."""
    demo_start_pose: npt.NDArray[np.float64]
    demo_end_pose: npt.NDArray[np.float64]

    @classmethod
    def from_traj_info_and_cam_plans(
        cls,
        traj_info: TrajectoryInfo,
        cam_plans: list[CameraPlan],
        ep_timestamps: npt.NDArray[np.float64],
        pose_mapping: dict[RoleEnum, int],
        gripper_id_to_gripper_cal: dict[int, si.interp1d],
        nominal_z: float,
    ) -> list["GripperPlan"]:
        traj = load_trajectory_from_json(json_path=traj_info.path)
        gid_to_pose_role = {v: k for k, v in pose_mapping.items()}

        gid_to_cam_plan = {cp.gripper_id: cp for cp in cam_plans}

        # Get interpolators for (calibrated) gripper width
        gid_to_gripper_width: dict[int, npt.NDArray[np.float64]] = {}
        for gripper_id, cam_plan in gid_to_cam_plan.items():
            tag_path = cam_plan.tag_detection_path
            if gripper_id == 0:
                left_finger_id = 0
                right_finger_id = 1
            elif gripper_id == 1:
                left_finger_id = 6
                right_finger_id = 7
            else:
                raise ValueError(
                    f"Unknown gripper id {gripper_id} for tag detection file {tag_path}"
                )
            with tag_path.open("rb") as f:
                _tag_data: list[TagDetectionResultDict] = pickle.load(f)
            start_idx, end_idx = cam_plan.video_start_end
            valid_tag_data = _tag_data[start_idx:end_idx]
            valid_gripper_timestamps = [
                frame["time"] for frame in valid_tag_data
            ]
            # NOTE: gripper's timestamps start from 0
            gripper_timestamps: list[float] = []
            gripper_widths: list[float] = []
            for tag_frame in valid_tag_data:
                width: Optional[float] = get_gripper_width(
                    tag_dict=tag_frame["tag_dict"],
                    left_id=left_finger_id,
                    right_id=right_finger_id,
                    nominal_z=nominal_z,
                )
                if width is not None:
                    gripper_cal_interp = gripper_id_to_gripper_cal[gripper_id]
                    calibrated_width = float(gripper_cal_interp(width))
                    gripper_timestamps.append(tag_frame["time"])
                    gripper_widths.append(calibrated_width)
            gripper_det_ratio = len(gripper_widths) / len(valid_tag_data)
            if gripper_det_ratio < 0.9:
                logger.warning(
                    f"Low gripper tag detection ratio {gripper_det_ratio:.2f} for gripper id {gripper_id} in {tag_path}, gripper width interpolation may be inaccurate."
                )
            gw_interp = get_interp1d(
                t=np.array(gripper_timestamps),
                x=np.array(gripper_widths),
            )

            gid_to_gripper_width[gripper_id] = gw_interp(
                np.array(valid_gripper_timestamps)
            )

        # Collect gripper plans
        sorted_gripper_ids = sorted(gid_to_pose_role.keys())
        # Ensure gripper_ids are contiguous
        assert sorted_gripper_ids == list(range(len(sorted_gripper_ids))), (
            f"Gripper ids in pose mapping are not contiguous: {sorted_gripper_ids}"
        )
        gripper_plans: list[GripperPlan] = []
        for gripper_id in sorted_gripper_ids:
            pose_role = gid_to_pose_role[gripper_id]
            # Get TCP pose
            traj_df = traj_to_pd_dataframe(traj, role=pose_role)
            tcp_pose_interp = get_tcp_pose_interpolator(traj_df)
            tcp_pose = tcp_pose_interp(ep_timestamps)
            # Get gripper width if applicable
            if gripper_id in gripper_id_to_gripper_cal:
                gripper_width = gid_to_gripper_width[gripper_id]
            else:
                gripper_width = None
            demo_start_pose = tcp_pose[0]
            demo_end_pose = tcp_pose[-1]
            gripper_plans.append(
                GripperPlan(
                    gripper_id=gripper_id,
                    tcp_pose=tcp_pose,
                    gripper_width=gripper_width,
                    demo_start_pose=demo_start_pose,
                    demo_end_pose=demo_end_pose,
                )
            )
        return gripper_plans


@dataclass(frozen=True)
class DemoPlan:
    episode_timestamps: npt.NDArray[np.float64]
    grippers: list[GripperPlan]
    cameras: list[CameraPlan]
    joint_pos: npt.NDArray[np.float64]
    prompt: str | None = None

    @classmethod
    def from_demo_dir(
        cls,
        demo_dir: pathlib.Path,
        gripper_id_to_gripper_cal: dict[int, si.interp1d],
        pose_mapping: dict[RoleEnum, int],
        nominal_z: float,
        gopro_timezone: str | None = None,
    ) -> "DemoPlan":
        demo_dir = demo_dir.resolve().absolute()
        _demo_group = DemoGroup.from_demo_dir(
            demo_dir, gopro_timezone=gopro_timezone
        )
        _video_pair = _demo_group.video_pair
        _traj_info = _demo_group.trajectory

        camera_plans, ep_timestamps = CameraPlan.from_video_pair(
            _video_pair, nominal_z=nominal_z
        )

        grippers = GripperPlan.from_traj_info_and_cam_plans(
            traj_info=_traj_info,
            cam_plans=camera_plans,
            ep_timestamps=ep_timestamps,
            pose_mapping=pose_mapping,
            gripper_id_to_gripper_cal=gripper_id_to_gripper_cal,
            nominal_z=nominal_z,
        )
        # Sort camera plans by gripper id
        camera_plans.sort(key=lambda x: x.gripper_id)
        grippers.sort(key=lambda x: x.gripper_id)

        # Get joint position array
        joint_pos = get_joint_pos_array(
            timestamps=ep_timestamps, traj_info=_traj_info
        )
        logger.debug(
            f"Generated DemoPlan for demo dir: {demo_dir} with {len(grippers)} grippers and {len(camera_plans)} cameras."
        )
        return DemoPlan(
            episode_timestamps=ep_timestamps,
            grippers=grippers,
            cameras=camera_plans,
            joint_pos=joint_pos,
            prompt=load_prompt_from_json(_traj_info.path),
        )


def main(
    input: Annotated[pathlib.Path, tyro.conf.arg(aliases=["-i"])],
    output: Annotated[
        Optional[pathlib.Path], tyro.conf.arg(aliases=["-o"])
    ] = None,
    nominal_z: Annotated[float, tyro.conf.arg(aliases=["-nz"])] = 0.045,
    htc_config: pathlib.Path = pathlib.Path("configs/htc/pose_mapping.yaml"),
    verbose: bool = False,
    gopro_timezone: Optional[str] = None,
):
    """Generate dataset plan for UMI G1 dataset.
    Plan format:
    ``` python
    all_plans = [
        {
            "episode_timestamps": np.ndarray,
            "grippers": [
                {"tcp_pose": np.ndarray, "gripper_width": np.ndarray}
            ],
            "cameras": [
                {"video_path": str, "video_start_end": Tuple[int, int]}
            ],
        }
    ]
    ```
    Args:
        input: Project directory containing demos and raw_trajectories subdirs.
        output: Path to output dataset plan pickle file. Defaults to <input>/dataset_plan.pkl
        nominal_z: Nominal Z value for gripper finger tag detection.
        htc_config: Path to HTC pose mapping config file.
        verbose: Whether to enable verbose logging.
        gopro_timezone: Timezone of the GoPro cameras. Defaults to system timezone.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <level>{level: <8}</level> | {file.name}:{line} - {message}",
        colorize=True,
        level="DEBUG" if verbose else "INFO",
    )
    input_path = input.resolve()
    demos_dir = input_path.joinpath("demos")
    if output is None:
        output = input_path.joinpath("dataset_plan.pkl")

    # Load HTC calibration config
    htc_config_path = pathlib.Path(htc_config)

    with open(htc_config_path) as f:
        htc_config_dict = yaml.safe_load(f)
        _pose_mapping: dict[str, int] = htc_config_dict["PoseMapping"]

    pose_mapping: dict[RoleEnum, int] = {
        RoleEnum(k): v for k, v in _pose_mapping.items()
    }

    # Get gripper width interpolators
    gripper_cal_dirs = list(demos_dir.glob("gripper*"))
    gripper_id_to_gripper_cal: dict[int, si.interp1d] = {}
    cam_serial_to_gripper_cal: dict[str, si.interp1d] = {}
    for gripper_cal_dir in gripper_cal_dirs:
        gr_path = gripper_cal_dir / "gripper_range.json"
        with gr_path.open("r") as f:
            gripper_range_data = json.load(f)
        gripper_id: int = int(gripper_range_data["gripper_id"])
        max_width: float = float(gripper_range_data["max_width"])
        min_width: float = float(gripper_range_data["min_width"])
        gripper_cal_interp = get_gripper_calibration_interpolator(
            aruco_measured_width=[min_width, max_width],
            aruco_actual_width=[min_width, max_width],
        )
        gripper_id_to_gripper_cal[gripper_id] = gripper_cal_interp
        gripper_cal_video_info = VideoInfo.from_file(
            gripper_cal_dir / "raw_video.mp4", gopro_timezone=gopro_timezone
        )
        cam_serial = gripper_cal_video_info.camera_serial
        cam_serial_to_gripper_cal[cam_serial] = gripper_cal_interp

    demo_plans: list[DemoPlan] = [
        DemoPlan.from_demo_dir(
            demo_dir=demo_dir,
            gripper_id_to_gripper_cal=gripper_id_to_gripper_cal,
            pose_mapping=pose_mapping,
            nominal_z=nominal_z,
            gopro_timezone=gopro_timezone,
        )
        for demo_dir in demos_dir.glob("demo_*/")
    ]

    # Sort by episode start time
    demo_plans.sort(key=lambda dp: dp.episode_timestamps[0])

    # Serialize dataset plan
    with output.open("wb") as f:
        demo_plans_dict = [asdict(dp) for dp in demo_plans]
        pickle.dump(demo_plans_dict, f)
    logger.info(f"Saved dataset plan with {len(demo_plans)} demos to {output}")


if __name__ == "__main__":
    tyro.cli(main)
