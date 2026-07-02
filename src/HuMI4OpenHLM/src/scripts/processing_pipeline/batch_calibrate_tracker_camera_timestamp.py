import concurrent.futures
import multiprocessing
import pathlib
import subprocess
import sys
from typing import Annotated, Optional

import tyro
from loguru import logger
from tqdm import tqdm


def _run_subprocess(
    cmd: list[str],
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> subprocess.CompletedProcess:
    with (
        stdout_path.open("w") as stdout_file,
        stderr_path.open("w") as stderr_file,
    ):
        return subprocess.run(cmd, stdout=stdout_file, stderr=stderr_file)


def main(
    session_dirs: tuple[pathlib.Path, ...],
    /,
    max_offset: float = 1.0,
    visualize: bool = False,
    num_workers: Optional[int] = None,
    run_again: Annotated[bool, tyro.conf.arg(aliases=["-ra"])] = False,
    verbose: bool = False,
    gopro_timezone: Optional[str] = None,
) -> None:
    """
    Calibrate timestamp offset between tracker camera trajectory and GoPro gyro data for each demo folder.

    Args:
        session_dirs: Tuple of paths to session directories containing 'demos' subdirectories.
        max_offset: Maximum allowed timestamp offset in seconds for calibration.
        visualize: If True, generates visualization of the calibration results.
        num_workers: Number of concurrent processes to run. Defaults to CPU count.
        run_again: If True, re-runs calibration even if 'timestamp_calibration.json' already exists.
        verbose: If True, sets log level to DEBUG.
        gopro_timezone: Optional timezone for GoPro cameras (e.g., 'Asia/Shanghai' or '+08:00').
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <level>{level: <8}</level> | {file.name}:{line} - {message}",
        colorize=True,
        level="DEBUG" if verbose else "INFO",
    )

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    for session in session_dirs:
        session = session.expanduser().resolve()
        input_dir = session / "demos"

        demo_dirs = [
            x.parent for x in input_dir.glob("**/camera_trajectory.csv")
        ]

        futures: dict[
            concurrent.futures.Future[subprocess.CompletedProcess],
            tuple[pathlib.Path, pathlib.Path, pathlib.Path],
        ] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            for demo_dir in demo_dirs:
                demo_dir = demo_dir.resolve()

                raw_video = demo_dir / "raw_video.mp4"
                tracker_csv = demo_dir / "camera_trajectory.csv"
                gyro_csv = demo_dir / "gyro_data.csv"

                output_file = demo_dir / "timestamp_calibration.json"
                if output_file.is_file() and not run_again:
                    logger.info(
                        f"{output_file.name} already exists, skipping {demo_dir.name}"
                    )
                    continue

                script_path = (
                    pathlib.Path(__file__).parent
                    / "calibrate_tracker_camera_timestamp.py"
                )
                cmd = [
                    "python",
                    str(script_path),
                    "-v",
                    str(raw_video),
                    "-t",
                    str(tracker_csv),
                    "-g",
                    str(gyro_csv),
                    "--max_offset",
                    str(max_offset),
                    "--output",
                    str(output_file),
                ]
                if visualize:
                    cmd.append("--visualize")
                if gopro_timezone is not None:
                    cmd.extend(["--gopro_timezone", gopro_timezone])

                stdout_path = demo_dir / "calibrate_tracker_camera_stdout.txt"
                stderr_path = demo_dir / "calibrate_tracker_camera_stderr.txt"

                future = executor.submit(
                    _run_subprocess, cmd, stdout_path, stderr_path
                )
                futures[future] = (demo_dir, stdout_path, stderr_path)

            if not futures:
                logger.info(f"No calibration jobs scheduled for {session}")
                continue

            with tqdm(
                total=len(futures),
                desc=f"Calibrating timestamps ({session.name})",
                dynamic_ncols=True,
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result.returncode != 0:
                        demo_dir, stdout_path, stderr_path = futures[future]
                        logger.error(
                            f"Calibration failed for {demo_dir.name}. Logs: "
                            f"{stdout_path}, {stderr_path}"
                        )
                    pbar.update(1)


if __name__ == "__main__":
    tyro.cli(main)
