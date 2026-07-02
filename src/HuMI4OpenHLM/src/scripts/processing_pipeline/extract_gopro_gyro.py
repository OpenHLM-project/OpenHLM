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
    cwd: pathlib.Path,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> subprocess.CompletedProcess:
    with (
        stdout_path.open("w") as stdout_file,
        stderr_path.open("w") as stderr_file,
    ):
        return subprocess.run(
            cmd, cwd=str(cwd), stdout=stdout_file, stderr=stderr_file
        )


# %%
def main(
    session_dirs: tuple[pathlib.Path, ...],
    /,
    docker_image: str = "richardnai/gpmf-extract-gyro",
    num_workers: Optional[int] = None,
    skip_docker_pull: bool = False,
    run_again: Annotated[bool, tyro.conf.arg(aliases=["-ra"])] = False,
    verbose: bool = False,
) -> None:
    """
    Extract GoPro gyro data from videos in the specified session directories using Docker.

    Args:
        session_dirs: Tuple of paths to session directories containing 'demos' subdirectories.
        docker_image: Name of the Docker image used for gyro data extraction.
        num_workers: Number of concurrent processes to run. Defaults to CPU count.
        skip_docker_pull: If True, skips 'docker pull' for the image.
        run_again: If True, re-extracts gyro data even if 'gyro_data.csv' already exists.
        verbose: If True, sets log level to DEBUG.
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

    if not skip_docker_pull:
        logger.info(f"Pulling docker image {docker_image}")
        pull = subprocess.run(["docker", "pull", docker_image], check=False)
        if pull.returncode != 0:
            raise RuntimeError("Docker pull failed!")

    mount_target = pathlib.Path("/data")
    video_path = "/real_raw_video.mp4"
    csv_path = mount_target / "gyro_data.csv"

    for session in session_dirs:
        session = session.expanduser().resolve()
        input_dir = session / "demos"
        if not input_dir.is_dir():
            logger.info(f"{input_dir} is not a directory, skipping.")
            continue

        input_video_dirs = [
            x.parent for x in input_dir.glob("**/raw_video.mp4")
        ]
        logger.info(f"Found {len(input_video_dirs)} video dirs in {session}")

        futures: dict[
            concurrent.futures.Future[subprocess.CompletedProcess],
            tuple[pathlib.Path, pathlib.Path, pathlib.Path],
        ] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            for video_dir in input_video_dirs:
                video_dir = video_dir.resolve()
                if (
                    video_dir.joinpath("gyro_data.csv").is_file()
                    and not run_again
                ):
                    logger.info(
                        f"gyro_data.csv already exists, skipping {video_dir.name}"
                    )
                    continue

                real_raw_video_path = (video_dir / "raw_video.mp4").resolve()

                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "--volume",
                    f"{video_dir}:/data",
                    "--volume",
                    f"{real_raw_video_path}:{video_path}",
                    docker_image,
                    str(video_path),
                    str(csv_path),
                ]

                stdout_path = video_dir / "extract_gopro_gyro_stdout.txt"
                stderr_path = video_dir / "extract_gopro_gyro_stderr.txt"

                future = executor.submit(
                    _run_subprocess,
                    cmd,
                    video_dir,
                    stdout_path,
                    stderr_path,
                )
                futures[future] = (video_dir, stdout_path, stderr_path)

            with tqdm(
                total=len(futures),
                desc=f"Extracting gyro ({session.name})",
                dynamic_ncols=True,
            ) as pbar:
                completed_results: list[subprocess.CompletedProcess] = []
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    completed_results.append(result)
                    if result.returncode != 0:
                        video_dir, stdout_path, stderr_path = futures[future]
                        logger.error(
                            f"Error extracting gyro for {video_dir.name}. Logs: "
                            f"{stdout_path}, {stderr_path}"
                        )
                    pbar.update(1)


# %%
if __name__ == "__main__":
    tyro.cli(main)
