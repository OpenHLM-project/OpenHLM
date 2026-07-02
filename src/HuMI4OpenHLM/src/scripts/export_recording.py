import concurrent.futures
import json
import multiprocessing
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, TypedDict

import joblib
import mujoco
import numpy as np
import numpy.typing as npt
import scipy.interpolate as si
import tyro
from loguru import logger
from tqdm import tqdm


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


def get_recording_joint_idx_to_poselib(
    path: Path,
) -> dict[int, int]:
    """Extract joint ordering mapping from recording json files to PoseLib expected format.
    Args:
        path: Path to MJCF XML file.
    Returns:
        mapping: Dictionary mapping recording joint indices to PoseLib expected joint indices.

    """
    if not path.exists():
        # Fallback to local project directory if the absolute path is from another machine
        parts = path.parts
        project_root = Path(__file__).resolve().parents[2]

        # Try to match from "g1_description" onwards
        if "g1_description" in parts:
            idx = parts.index("g1_description")
            rel_path = Path(*parts[idx:])
            new_path = project_root / rel_path
            if new_path.exists():
                logger.info(
                    f"MJCF path {path} not found. Using fallback {new_path}"
                )
                path = new_path

        # If still not found, try just the filename in g1_description
        if not path.exists():
            fallback_path = project_root / "g1_description" / path.name
            if fallback_path.exists():
                logger.info(
                    f"MJCF path {path} not found. Using fallback {fallback_path}"
                )
                path = fallback_path

    model = mujoco.MjModel.from_xml_path(path.as_posix())
    joint_order: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        joint_order.append(joint_name)
    mapping: dict[int, int] = {}
    unexpected: list[str] = []
    for actual_idx, joint_name in enumerate(joint_order):
        expected_idx = EXPECTED_ACTUATED_INDEX.get(joint_name)
        if expected_idx is None:
            unexpected.append(joint_name)
            continue
        mapping[actual_idx] = expected_idx
    missing_expected = [
        joint_name
        for joint_name in EXPECTED_ACTUATED_JOINTS
        if joint_name not in joint_order
    ]
    if unexpected:
        logger.warning(
            f"Joints present in {path.as_posix()} but not in expected schema: {', '.join(unexpected)}"
        )
    if missing_expected:
        logger.warning(
            f"Expected joints missing from {path.as_posix()}: {', '.join(missing_expected)}, use default zero values."
        )
    return mapping


EXPECTED_ACTUATED_JOINTS = [
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
EXPECTED_ACTUATED_INDEX = {
    name: idx for idx, name in enumerate(EXPECTED_ACTUATED_JOINTS)
}


@dataclass(frozen=True)
class PoseLibEpisode:
    """A single episode in PoseLib format."""

    fps: int
    """Frames per second of the episode."""
    root_trans_offset: npt.NDArray[np.float64]
    """Root translation offset (3,)."""
    root_rot: npt.NDArray[np.float64]
    """Root rotation as quaternions (xyzw) (N, 4)."""
    dof: npt.NDArray[np.float64]
    """Joint angles (N, D)."""


def export_one_episode(
    json_path: Path,
    mjcf_path: Path | None = None,
    fps: int = 30,
) -> PoseLibEpisode:
    """Export a single trajectory JSON to PoseLib episode.
    Args:
        json_path: Path to trajectory JSON file.
        mjcf_path: Optional MJCF path to interpret recorded q vectors.
        fps: Target frames per second for exported data.
    Returns:
        pose_lib_episode: Exported PoseLibEpisode.
    """
    with json_path.open("r") as f:
        traj_payload: TrajPayload = json.load(f)

    if mjcf_path is None:
        mjcf_source = traj_payload["mjcf_path"]
    else:
        mjcf_source = mjcf_path.as_posix()

    recording_jid_to_poselib = get_recording_joint_idx_to_poselib(
        Path(mjcf_source)
    )

    episode = traj_payload["episode"]

    qs: list[npt.NDArray[np.float64]] = []
    timestamps: list[float] = []

    for frame in episode:
        qs.append(np.asarray(frame["q"], dtype=np.float64))
        timestamps.append(frame["timestamp"])

    q_interp = si.interp1d(
        timestamps,
        np.stack(qs, axis=0),
        axis=0,
    )

    # Downsample to target fps
    t0 = timestamps[0]
    t_end = timestamps[-1]
    target_dt = 1.0 / fps
    resample_times = np.arange(t0, t_end + 1e-9, target_dt, dtype=np.float64)
    q_resampled: npt.NDArray[np.float64] = q_interp(resample_times)

    root_trans_offset = q_resampled[:, :3]
    root_rot_wxyz = q_resampled[:, 3:7]
    root_rot = root_rot_wxyz[:, [1, 2, 3, 0]]  # to xyzw

    dof = np.zeros(
        (q_resampled.shape[0], len(EXPECTED_ACTUATED_JOINTS)),
        dtype=np.float64,
    )
    actuated = q_resampled[:, 7:]
    for recording_idx, poselib_idx in recording_jid_to_poselib.items():
        dof[:, poselib_idx] = actuated[:, recording_idx]

    return PoseLibEpisode(
        fps=fps,
        root_trans_offset=root_trans_offset,
        root_rot=root_rot,
        dof=dof,
    )


def main(
    input_dir: Annotated[Path, tyro.conf.arg(aliases=["-i"])],
    output_path: Annotated[Path, tyro.conf.arg(aliases=["-o"])],
    fps: int = 30,
    mjcf_path: Path | None = None,
    num_workers: int | None = None,
    verbose: bool = False,
):
    """Export Recorder JSON folder to PoseLib PKL.
    Args:
        input_dir: Directory containing recorded JSON files.
        output_path: Path to output PKL file.
        fps: Target frames per second for exported data.
        mjcf_path: Optional MJCF path to interpret recorded q vectors.
        num_workers: Number of parallel workers to use. Defaults to number of CPU cores.
        verbose: Whether to enable verbose logging.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <level>{level: <8}</level> | {file.name}:{line} - {message}",
        colorize=True,
        level="DEBUG" if verbose else "INFO",
    )

    ep_jsons = sorted(input_dir.glob("*recording*.json"))
    logger.info(f"Found {len(ep_jsons)} episodes in {input_dir.as_posix()}")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    episodes: list[PoseLibEpisode] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:
        futures: list[concurrent.futures.Future[PoseLibEpisode]] = []
        for ep_json in ep_jsons:
            futures.append(
                executor.submit(
                    export_one_episode,
                    json_path=ep_json,
                    mjcf_path=mjcf_path,
                    fps=fps,
                )
            )
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Exporting episodes",
        ):
            pass
        episodes = [f.result() for f in futures]

    # Save to PKL
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ep_data = {}
    for i, ep in enumerate(episodes):
        ep_data[f"episode_{i:03d}"] = asdict(ep)

    joblib.dump(ep_data, output_path.as_posix())
    logger.info(f"Saved exported episodes to {output_path.as_posix()}")


if __name__ == "__main__":
    tyro.cli(main)


def cli():
    tyro.cli(main)
