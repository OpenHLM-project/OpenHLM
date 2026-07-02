import pathlib
import shutil
import sys
from typing import Annotated, Literal

import tyro
import yaml
from loguru import logger

from processing import (
    RoleEnum,
    load_trajectory_from_json,
    traj_to_pd_dataframe,
)
from processing.cv_util import detect_gripper_id


def match_one_session(
    session_dir: pathlib.Path,
    gid_to_role: dict[Literal[0, 1], RoleEnum],
    nominal_z: float,
    run_again: bool,
):
    session_dir = session_dir.resolve().absolute()
    demo_dir = session_dir / "demos"
    demos = list(demo_dir.glob("demo_*/"))

    unk_gid_demos: set[pathlib.Path] = set()
    for demo in demos:
        tag_files = list(demo.glob("*/tag_detection.pkl"))
        if len(tag_files) != 2:
            raise ValueError(
                f"Expected 2 tag_detection.pkl files in {demo}, found {len(tag_files)}"
            )
        traj_files = list(demo.glob("recording_*.json"))
        if len(traj_files) != 1:
            raise ValueError(
                f"Expected 1 trajectory json file in {demo}, found {len(traj_files)}"
            )
        traj_file = traj_files[0]
        traj_data = load_trajectory_from_json(traj_file)
        for tag in tag_files:
            try:
                gripper_id = detect_gripper_id(tag, nominal_z=nominal_z)
            except ValueError as e:
                logger.warning(
                    f"Failed to detect gripper id from {tag}: {e}, skipping demo {demo}"
                )
                gripper_id = -1  # Mark unknown gripper id
            if gripper_id in (0, 1):
                role = gid_to_role[gripper_id]
                df = traj_to_pd_dataframe(traj_data, role=role)
                camera_csv = tag.parent / "camera_trajectory.csv"
                if camera_csv.is_file() and not run_again:
                    logger.info(
                        f"{role} camera_trajectory.csv already exists, skipping {demo}"
                    )
                else:
                    df.to_csv(camera_csv, index=False)
                    logger.debug(
                        f"Wrote {role} camera trajectory to {camera_csv}"
                    )
            else:
                logger.warning(
                    f"Unknown gripper id {gripper_id} in {tag}, skipping demo {demo}"
                )
                unk_gid_demos.add(demo)
                # Mark unknown gripper id demos to skip
                target_path = demo.with_name("unk_gid_" + demo.name)
                if target_path.exists():
                    shutil.rmtree(target_path)
                demo.rename(target_path)
                break

    # Summarize failed demos
    if unk_gid_demos:
        logger.warning(
            f"Total {len(unk_gid_demos)} demos with unknown gripper ids:"
        )
        for d in unk_gid_demos:
            logger.warning(f" - {d.as_posix()}")
    else:
        logger.info("All demos matched successfully.")


def main(
    session_dirs: tuple[pathlib.Path, ...],
    /,
    run_again: Annotated[bool, tyro.conf.arg(aliases=["-ra"])] = False,
    nominal_z: Annotated[float, tyro.conf.arg(aliases=["-nz"])] = 0.045,
    htc_config: pathlib.Path = pathlib.Path("configs/htc/pose_mapping.yaml"),
    verbose: bool = False,
):
    """Extract camera trajectories from trajectory json files. With role assignment based on gripper id.

    Args:
        session_dirs: List of session directories.
        run_again: If True, overwrite existing camera_trajectory.csv files.
        nominal_z: Nominal Z value for gripper finger tag detection.
        htc_config: Path to HTC configuration file.
                Requires to contain 'PoseMapping' field like:
                ```json
                PoseMapping:
                    right_hand: 0
                    left_hand: 1
                ```
        verbose: If True, print detailed logs.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <level>{level: <8}</level> | {file.name}:{line} - {message}",
        colorize=True,
        level="DEBUG" if verbose else "INFO",
    )
    with open(htc_config) as f:
        config = yaml.safe_load(f)
        pose_mapping: dict[str, int] = config["PoseMapping"]

    gid_to_role: dict[Literal[0, 1], RoleEnum] = {}
    if pose_mapping["left_hand"] == 0 and pose_mapping["right_hand"] == 1:
        gid_to_role[0] = RoleEnum.LEFT_HAND
        gid_to_role[1] = RoleEnum.RIGHT_HAND
    elif pose_mapping["left_hand"] == 1 and pose_mapping["right_hand"] == 0:
        gid_to_role[0] = RoleEnum.RIGHT_HAND
        gid_to_role[1] = RoleEnum.LEFT_HAND
    else:
        raise ValueError(
            f"HTC config PoseMapping must map 'left_hand' and 'right_hand' to gripper ids 0 and 1, but got: {pose_mapping}"
        )

    for session in session_dirs:
        match_one_session(
            session, gid_to_role, nominal_z=nominal_z, run_again=run_again
        )


if __name__ == "__main__":
    tyro.cli(main)
