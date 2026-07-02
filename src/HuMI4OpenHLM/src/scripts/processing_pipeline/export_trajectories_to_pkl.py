import concurrent.futures
import json
import multiprocessing
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import numpy.typing as npt
import scipy.interpolate as si
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from processing import (
    EXPECTED_ACTUATED_JOINTS,
    TrajPayload,
    get_recording_joint_idx_to_poselib,
)


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

    qs_array = np.stack(qs, axis=0)
    root_trans = qs_array[:, :3]
    root_rot_wxyz = qs_array[:, 3:7]
    actuated = qs_array[:, 7:]

    root_trans_interp = si.interp1d(
        x=np.asarray(timestamps, dtype=np.float64),
        y=root_trans,
        axis=0,
    )
    root_rot = R.from_quat(root_rot_wxyz[:, [1, 2, 3, 0]])  # to xyzw
    root_rot_slerp = Slerp(
        times=np.asarray(timestamps, dtype=np.float64), rotations=root_rot
    )
    actuated_interp = si.interp1d(
        x=np.asarray(timestamps, dtype=np.float64),
        y=actuated,
        axis=0,
    )

    # Downsample to target fps
    t0 = timestamps[0]
    t_end = timestamps[-1]
    target_dt = 1.0 / fps
    resample_times = np.arange(t0, t_end + 1e-9, target_dt, dtype=np.float64)

    root_trans_offset: npt.NDArray[np.float64] = root_trans_interp(
        resample_times
    )
    root_rot_xyzw = root_rot_slerp(resample_times).as_quat()  # type: ignore

    dof = np.zeros(
        (len(resample_times), len(EXPECTED_ACTUATED_JOINTS)),
        dtype=np.float64,
    )
    actuated = actuated_interp(resample_times)
    for recording_idx, poselib_idx in recording_jid_to_poselib.items():
        dof[:, poselib_idx] = actuated[:, recording_idx]

    return PoseLibEpisode(
        fps=fps,
        root_trans_offset=root_trans_offset,
        root_rot=root_rot_xyzw,
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
