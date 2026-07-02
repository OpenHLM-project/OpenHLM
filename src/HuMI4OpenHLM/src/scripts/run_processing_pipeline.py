import pathlib
import subprocess
from typing import Annotated

import tyro


def main(
    session_dirs: tuple[pathlib.Path, ...],
    /,
    calibration_dir: Annotated[
        pathlib.Path, tyro.conf.arg(aliases=["-c"])
    ] = pathlib.Path(__file__).parents[2] / "configs" / "calibration",
    htc_config_dir: Annotated[
        pathlib.Path, tyro.conf.arg(aliases=["-hc"])
    ] = pathlib.Path(__file__).parents[2] / "configs" / "htc",
    run_again: Annotated[bool, tyro.conf.arg(aliases=["-ra"])] = False,
    gopro_timezone: Annotated[
        str | None, tyro.conf.arg(aliases=["-tz"])
    ] = None,
) -> None:
    """
    Run processing pipeline for given session directories.

    Args:
        session_dirs: Tuple of paths to session directories to process.
        calibration_dir: Path to the calibration directory.
        htc_config_dir: Path to the HTC configuration directory.
        run_again: If True, pass the run_again flag to pipeline scripts to overwrite existing results.
        gopro_timezone: Optional timezone for GoPro cameras (e.g., 'Asia/Shanghai' or '+08:00'). Defaults to system timezone.
    """
    script_dir = pathlib.Path(__file__).parent.joinpath("processing_pipeline")
    calibration_dir_path = pathlib.Path(calibration_dir)
    htc_config_dir_path = pathlib.Path(htc_config_dir)
    assert calibration_dir_path.is_dir()
    assert htc_config_dir_path.is_dir()
    htc_config_path = htc_config_dir_path.joinpath("pose_mapping.yaml")
    assert htc_config_path.is_file()

    for session in session_dirs:
        session = pathlib.Path(session)

        print("############## 00_group_demos #############")
        script_path = script_dir.joinpath("group_demos.py")
        assert script_path.is_file()
        cmd = [
            "python",
            str(script_path),
            "--htc_config",
            str(htc_config_path),
        ]
        if gopro_timezone is not None:
            cmd.extend(["--gopro_timezone", gopro_timezone])
        cmd.append(str(session))
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############## 01_export_to_poselib #############")
        script_path = script_dir.joinpath("export_trajectories_to_pkl.py")
        assert script_path.is_file()
        trimmed_traj_dir = session.joinpath("trimmed_trajectories")
        cmd = [
            "python",
            str(script_path),
            "-i",
            str(trimmed_traj_dir),
            "-o",
            str(session.joinpath(f"{session.name}.pkl")),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############## 02_extract_gopro_gyro #############")
        script_path = script_dir.joinpath("extract_gopro_gyro.py")
        assert script_path.is_file()
        cmd = ["python", str(script_path), str(session)]
        if run_again:
            cmd += ["--run_again"]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 01_detect_aruco ###########")
        script_path = script_dir.joinpath("batch_detect_aruco.py")
        assert script_path.is_file()
        demo_dir = session.joinpath("demos")
        camera_intrinsics = calibration_dir_path.joinpath(
            "gopro_intrinsics_2_7k.json"
        )
        aruco_config = calibration_dir_path.joinpath("aruco_config.yaml")
        assert camera_intrinsics.is_file()
        assert aruco_config.is_file()

        cmd = [
            "python",
            str(script_path),
            "--input_dir",
            str(demo_dir),
            "--camera_intrinsics",
            str(camera_intrinsics),
            "--aruco_yaml",
            str(aruco_config),
        ]
        if run_again:
            cmd += ["--run_again"]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 02_run_calibrations ###########")
        script_path = script_dir.joinpath("run_calibrations.py")
        assert script_path.is_file()
        cmd = ["python", str(script_path), str(session)]
        if run_again:
            cmd += ["--run_again"]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 03_match_camera_and_trajectory ###########")
        script_path = script_dir.joinpath("match_camera_and_trajectory.py")
        assert script_path.is_file()
        cmd = [
            "python",
            str(script_path),
            "--htc_config",
            str(htc_config_path),
            str(session),
        ]
        if run_again:
            cmd += ["--run_again"]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print(
            "############# 04_calibrate_tracker_camera_timestamp ###########"
        )
        script_path = script_dir.joinpath(
            "batch_calibrate_tracker_camera_timestamp.py"
        )
        assert script_path.is_file()
        cmd = ["python", str(script_path)]
        if run_again:
            cmd += ["--run_again"]
        if gopro_timezone is not None:
            cmd.extend(["--gopro_timezone", gopro_timezone])
        cmd.append(str(session))
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 05_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            "python",
            str(script_path),
            "-i",
            str(session),
            "--htc_config",
            str(htc_config_path),
        ]
        if gopro_timezone is not None:
            cmd.extend(["--gopro_timezone", gopro_timezone])
        result = subprocess.run(cmd)
        assert result.returncode == 0


## %%
def cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    cli()
