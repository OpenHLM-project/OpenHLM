import pathlib
import subprocess
import sys
from typing import Annotated

import tyro
from loguru import logger


def main(
    session_dirs: tuple[pathlib.Path, ...],
    /,
    run_again: Annotated[bool, tyro.conf.arg(aliases=["-ra"])] = False,
    verbose: bool = False,
) -> None:
    """
    Calibrate gripper range for each gripper calibration demo folder.

    Args:
        session_dirs: Tuple of paths to session directories containing 'demos' subdirectories.
        run_again: If True, re-run calibration even if 'gripper_range.json' already exists.
        verbose: If True, sets log level to DEBUG.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <level>{level: <8}</level> | {file.name}:{line} - {message}",
        colorize=True,
        level="DEBUG" if verbose else "INFO",
    )

    script_path = (
        pathlib.Path(__file__).resolve().parent / "calibrate_gripper_range.py"
    )
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Calibration script not found at {script_path}"
        )

    for session in session_dirs:
        session = session.expanduser().resolve()
        demos_dir = session / "demos"
        if not demos_dir.is_dir():
            logger.info(f"{demos_dir} is not a directory, skipping.")
            continue

        gripper_dirs = sorted(demos_dir.glob("gripper_calibration*"))
        if not gripper_dirs:
            logger.warning(f"No gripper calibration dirs found in {session}")
            continue

        logger.info(
            f"Found {len(gripper_dirs)} gripper calibration dirs in {session}"
        )
        for gripper_dir in gripper_dirs:
            gripper_range_path = gripper_dir / "gripper_range.json"
            tag_path = gripper_dir / "tag_detection.pkl"
            if not tag_path.is_file():
                logger.error(
                    f"Missing tag_detection.pkl in {gripper_dir}, skipping."
                )
                continue

            if gripper_range_path.is_file() and not run_again:
                logger.info(
                    f"gripper_range.json already exists, skipping {gripper_dir.name}"
                )
                continue

            cmd = [
                "python",
                str(script_path),
                "--input",
                str(tag_path),
                "--output",
                str(gripper_range_path),
                "--nominal_z",
                "0.045",
            ]
            logger.debug(f"Running calibration: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(gripper_dir))
            if result.returncode != 0:
                logger.error(f"Calibration failed in {gripper_dir}")


if __name__ == "__main__":
    tyro.cli(main)
