import concurrent.futures
import multiprocessing
import pathlib
import pickle
import re
import subprocess
import sys
from typing import Annotated

import numpy as np
import tyro
from loguru import logger
from tqdm import tqdm


# %%
def main(
    input_dir: Annotated[pathlib.Path, tyro.conf.arg(aliases=["-i"])],
    camera_intrinsics: pathlib.Path = pathlib.Path(
        "configs/calibration/gopro_intrinsics_2_7k.json"
    ),
    aruco_yaml: pathlib.Path = pathlib.Path(
        "configs/calibration/aruco_config.yaml"
    ),
    num_workers: int | None = None,
    demo_detect_fps: float = 10.0,
    crop_margin_px: int = 96,
    run_again: Annotated[bool, tyro.conf.arg(aliases=["-ra"])] = False,
    verbose: bool = False,
):
    """Detect aruco tags for every demo folder.
    Args:
        input_dir: Directory containing demo subdirectories with raw_video.mp4 files.
        camera_intrinsics: Path to camera intrinsics json file.
        aruco_yaml: Path to aruco configuration yaml file.
        num_workers: Number of parallel workers to use. Defaults to number of CPU cores.
        demo_detect_fps: Detection fps for non-gripper-calibration videos.
        crop_margin_px: Extra crop margin around calibration tag detections in pixels.
        run_again: If True, re-run detection even if tag_detection.pkl already exists.
        verbose: If True, print detailed logs.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <level>{level: <8}</level> | {file.name}:{line} - {message}",
        colorize=True,
        level="DEBUG" if verbose else "INFO",
    )
    input_dir = input_dir.expanduser().resolve()
    input_video_dirs = sorted(x.parent for x in input_dir.glob("**/raw_video.mp4"))
    logger.info(f"Found {len(input_video_dirs)} video dirs")

    camera_intrinsics = camera_intrinsics.expanduser().resolve()
    aruco_yaml = aruco_yaml.expanduser().resolve()
    assert camera_intrinsics.is_file()
    assert aruco_yaml.is_file()

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path(__file__).parent.joinpath("detect_aruco.py")

    calib_dirs = [x for x in input_video_dirs if x.name.startswith("gripper_calibration_")]
    demo_dirs = [x for x in input_video_dirs if x not in calib_dirs]

    logger.info(f"Found {len(calib_dirs)} gripper calibration videos")
    _run_detection_jobs(
        video_dirs=calib_dirs,
        script_path=script_path,
        camera_intrinsics=camera_intrinsics,
        aruco_yaml=aruco_yaml,
        num_workers=num_workers,
        run_again=run_again,
        desc="Detecting aruco (gripper calibration)",
    )

    serial_to_crop_box = _infer_serial_to_crop_box(
        calib_dirs=calib_dirs,
        crop_margin_px=crop_margin_px,
    )
    if serial_to_crop_box:
        logger.info(f"Inferred serial to crop box map: {serial_to_crop_box}")

    demo_job_configs: list[tuple[pathlib.Path, tuple[int, int, int, int] | None]] = []
    for video_dir in demo_dirs:
        serial = _extract_camera_serial(video_dir)
        crop_box = serial_to_crop_box.get(serial)
        if crop_box is None:
            logger.warning(
                f"Missing calibration crop box for {video_dir.name}, falling back to downsampled full-frame detection."
            )
        demo_job_configs.append((video_dir, crop_box))

    _run_detection_jobs(
        video_dirs=[x[0] for x in demo_job_configs],
        script_path=script_path,
        camera_intrinsics=camera_intrinsics,
        aruco_yaml=aruco_yaml,
        num_workers=num_workers,
        run_again=run_again,
        desc="Detecting aruco (demos)",
        demo_detect_fps=demo_detect_fps,
        crop_boxes={
            video_dir.absolute(): crop_box
            for video_dir, crop_box in demo_job_configs
            if crop_box is not None
        },
    )


def _extract_camera_serial(video_dir: pathlib.Path) -> str:
    name = video_dir.name
    if name.startswith("gripper_calibration_"):
        match = re.match(r"gripper_calibration_([^_]+)_", name)
        if match is None:
            raise ValueError(f"Cannot parse camera serial from {name}")
        return match.group(1)
    return name.split("_", maxsplit=1)[0]


def _infer_serial_to_crop_box(
    calib_dirs: list[pathlib.Path], crop_margin_px: int
) -> dict[str, tuple[int, int, int, int]]:
    serial_to_crop_box: dict[str, tuple[int, int, int, int]] = {}
    for video_dir in calib_dirs:
        tag_path = video_dir / "tag_detection.pkl"
        if not tag_path.is_file():
            logger.warning(
                f"Missing calibration detection result {tag_path}, cannot infer crop box."
            )
            continue
        with tag_path.open("rb") as f:
            tag_detection_results = pickle.load(f)
        all_corners = [
            np.asarray(tag_pose["corners"], dtype=np.float64).reshape(-1, 2)
            for frame in tag_detection_results
            for tag_pose in frame["tag_dict"].values()
        ]
        if not all_corners:
            logger.warning(
                f"No calibration tag corners found in {tag_path}, cannot infer crop box."
            )
            continue
        corners = np.concatenate(all_corners, axis=0)
        x0 = int(np.floor(np.min(corners[:, 0]))) - crop_margin_px
        y0 = int(np.floor(np.min(corners[:, 1]))) - crop_margin_px
        x1 = int(np.ceil(np.max(corners[:, 0]))) + crop_margin_px + 1
        y1 = int(np.ceil(np.max(corners[:, 1]))) + crop_margin_px + 1
        serial = _extract_camera_serial(video_dir)
        serial_to_crop_box[serial] = (x0, y0, x1, y1)
    return serial_to_crop_box


def _build_detect_cmd(
    script_path: pathlib.Path,
    video_dir: pathlib.Path,
    camera_intrinsics: pathlib.Path,
    aruco_yaml: pathlib.Path,
    *,
    detect_fps: float | None = None,
    crop_box: tuple[int, int, int, int] | None = None,
) -> list[str]:
    video_path = video_dir.joinpath("raw_video.mp4")
    pkl_path = video_dir.joinpath("tag_detection.pkl")
    cmd = [
        "python",
        str(script_path),
        "--input",
        str(video_path),
        "--output",
        str(pkl_path),
        "--intrinsics_json",
        str(camera_intrinsics),
        "--aruco_yaml",
        str(aruco_yaml),
        "--num_workers",
        "1",
    ]
    if detect_fps is not None:
        cmd += ["--detect-fps", str(detect_fps)]
    if crop_box is not None:
        cmd += ["--crop-box", *(str(v) for v in crop_box)]
    return cmd


def _run_detection_jobs(
    *,
    video_dirs: list[pathlib.Path],
    script_path: pathlib.Path,
    camera_intrinsics: pathlib.Path,
    aruco_yaml: pathlib.Path,
    num_workers: int,
    run_again: bool,
    desc: str,
    demo_detect_fps: float | None = None,
    crop_boxes: dict[pathlib.Path, tuple[int, int, int, int]] | None = None,
) -> None:
    # one chunk per thread, therefore no synchronization needed
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        futures: set[
            concurrent.futures.Future[subprocess.CompletedProcess]
        ] = set()
        for video_dir in video_dirs:
            video_dir = video_dir.absolute()
            pkl_path = video_dir.joinpath("tag_detection.pkl")
            if pkl_path.is_file() and not run_again:
                logger.info(
                    f"tag_detection.pkl already exists, skipping {video_dir.name}"
                )
                continue

            crop_box = None
            if crop_boxes is not None and video_dir in crop_boxes:
                crop_box = crop_boxes[video_dir]

            cmd = _build_detect_cmd(
                script_path=script_path,
                video_dir=video_dir,
                camera_intrinsics=camera_intrinsics,
                aruco_yaml=aruco_yaml,
                detect_fps=demo_detect_fps,
                crop_box=crop_box,
            )
            futures.add(
                executor.submit(
                    lambda x: subprocess.run(x, capture_output=True), cmd
                )
            )

        with tqdm(
            total=len(futures),
            desc=desc,
            dynamic_ncols=True,
        ) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                pbar.update(1)
                res: subprocess.CompletedProcess = fut.result()
                if res.returncode != 0:
                    logger.error(
                        f"Error in aruco detection:\n{res.stderr.decode()}"
                    )


# %%
if __name__ == "__main__":
    tyro.cli(main)
