import json
import pathlib
import pickle
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from typing import Annotated, Optional, TypedDict

import numpy as np
import numpy.typing as npt
import tyro
from tqdm import tqdm

from scripts.processing_pipeline.generate_final import (
    DEFAULT_FINAL_IMAGE_SIZE,
    DEFAULT_FINAL_OUT_FOV,
    DEFAULT_FINAL_VIDEO_CODEC,
    DEFAULT_FINAL_VIDEO_EXT,
    _build_fisheye_converter,
    _iter_selected_rgb_frames,
    _open_video_writer,
    _resolve_default_intrinsics_path,
    _undistort_rgb,
    _write_rgb_video_frame,
)


class CameraPlan(TypedDict):
    gripper_id: int
    video_path: str
    video_start_end: tuple[int, int]


class GripperPlan(TypedDict):
    gripper_id: int
    tcp_pose: npt.NDArray[np.float64]
    gripper_width: Optional[npt.NDArray[np.float64]]
    demo_start_pose: npt.NDArray[np.float64]
    demo_end_pose: npt.NDArray[np.float64]


class DemoPlan(TypedDict):
    episode_timestamps: npt.NDArray[np.float64]
    grippers: list[GripperPlan]
    cameras: list[CameraPlan]
    joint_pos: npt.NDArray[np.float64]


def _overwrite_episode_wrist_videos(
    ep_idx: int,
    ep: DemoPlan,
    final_dir: pathlib.Path,
    downsample_stride: int,
    teleop_delay_sec: float,
    camera_intrinsics: pathlib.Path,
    out_size: tuple[int, int],
    out_fov: float,
    video_codec: str,
    video_ext: str,
) -> int:
    fisheye_converter = _build_fisheye_converter(
        camera_intrinsics=camera_intrinsics,
        out_size=out_size,
        out_fov=out_fov,
    )
    episode_dir = final_dir / f"episode_{ep_idx:04d}"
    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"data.json not found: {data_path}")

    with data_path.open("r") as f:
        payload = json.load(f)
    entries: list[dict] = payload["data"]

    timestamps = np.asarray(ep["episode_timestamps"], dtype=np.float64)
    if len(timestamps) >= 2:
        sampled_ts = timestamps[::downsample_stride]
        sampled_fps = float(1.0 / np.median(np.diff(sampled_ts)))
    else:
        sampled_fps = 0.0
    delay_frames = int(round(max(0.0, teleop_delay_sec) * sampled_fps))
    sampled_local = np.arange(0, len(timestamps), downsample_stride, dtype=np.int64)
    valid_count = max(0, len(sampled_local) - delay_frames)
    if len(entries) != valid_count:
        raise ValueError(
            f"Episode {ep_idx:04d} entry count mismatch: "
            f"data.json has {len(entries)}, expected {valid_count}"
        )

    cam_right = next((c for c in ep["cameras"] if int(c["gripper_id"]) == 0), None)
    cam_left = next((c for c in ep["cameras"] if int(c["gripper_id"]) == 1), None)
    if cam_right is None or cam_left is None:
        raise ValueError("Expected cameras for gripper_id 0 and 1 in each episode")

    left_start, _ = cam_left["video_start_end"]
    right_start, _ = cam_right["video_start_end"]
    left_video = pathlib.Path(cam_left["video_path"])
    right_video = pathlib.Path(cam_right["video_path"])

    left_abs_indices = [left_start + int(i) for i in sampled_local[:valid_count]]
    right_abs_indices = [right_start + int(i) for i in sampled_local[:valid_count]]

    left_iter = _iter_selected_rgb_frames(left_video, left_abs_indices)
    right_iter = _iter_selected_rgb_frames(right_video, right_abs_indices)
    fps_for_video = float(max(round(sampled_fps), 1))
    left_video_rel = f"videos/wrist_rgb_left{video_ext}"
    right_video_rel = f"videos/wrist_rgb_right{video_ext}"
    left_writer = None
    right_writer = None
    try:
        for k, entry in enumerate(entries):
            try:
                left_idx, left_rgb = next(left_iter)
                right_idx, right_rgb = next(right_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Failed to decode required frame pair for episode {ep_idx:04d}, frame {k}"
                ) from exc

            if left_idx != left_abs_indices[k]:
                raise RuntimeError(
                    f"Left frame index mismatch for episode {ep_idx:04d}, frame {k}: "
                    f"expected {left_abs_indices[k]}, got {left_idx}"
                )
            if right_idx != right_abs_indices[k]:
                raise RuntimeError(
                    f"Right frame index mismatch for episode {ep_idx:04d}, frame {k}: "
                    f"expected {right_abs_indices[k]}, got {right_idx}"
                )

            left_rgb = _undistort_rgb(left_rgb, fisheye_converter)
            right_rgb = _undistort_rgb(right_rgb, fisheye_converter)
            if left_writer is None or right_writer is None:
                left_writer = _open_video_writer(
                    episode_dir / left_video_rel,
                    left_rgb.shape,
                    fps_for_video,
                    video_codec,
                )
                right_writer = _open_video_writer(
                    episode_dir / right_video_rel,
                    right_rgb.shape,
                    fps_for_video,
                    video_codec,
                )
            _write_rgb_video_frame(left_writer, left_rgb)
            _write_rgb_video_frame(right_writer, right_rgb)
            entry["wrist_rgb_left"] = {
                "video_path": left_video_rel,
                "frame_index": int(k),
            }
            entry["wrist_rgb_right"] = {
                "video_path": right_video_rel,
                "frame_index": int(k),
            }
    finally:
        if left_writer is not None:
            left_writer.release()
        if right_writer is not None:
            right_writer.release()

    payload.setdefault("info", {}).setdefault("wrist_image", {})
    payload["info"]["wrist_image"]["height"] = int(out_size[1])
    payload["info"]["wrist_image"]["width"] = int(out_size[0])
    video_info = payload.setdefault("info", {}).setdefault("video", {})
    video_info["format"] = video_ext.lstrip(".")
    video_info["codec"] = video_codec
    video_info["fps"] = int(round(sampled_fps))
    video_info["cameras"] = {
        "wrist_rgb_left": {
            "height": int(out_size[1]),
            "width": int(out_size[0]),
            "channels": 3,
            "frames": int(valid_count),
            "path": left_video_rel,
        },
        "wrist_rgb_right": {
            "height": int(out_size[1]),
            "width": int(out_size[0]),
            "channels": 3,
            "frames": int(valid_count),
            "path": right_video_rel,
        },
    }
    with data_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return valid_count


def main(
    input_dir: Annotated[pathlib.Path, tyro.conf.arg(aliases=["-i"])],
    final_dirname: str = "final_data",
    downsample_stride: int = 2,
    teleop_delay_sec: float = 0.2,
    camera_intrinsics: pathlib.Path = _resolve_default_intrinsics_path(),
    out_size: tuple[int, int] = DEFAULT_FINAL_IMAGE_SIZE,
    out_fov: float = DEFAULT_FINAL_OUT_FOV,
    video_codec: str = DEFAULT_FINAL_VIDEO_CODEC,
    video_ext: str = DEFAULT_FINAL_VIDEO_EXT,
    max_workers: int = 4,
) -> None:
    """Overwrite existing final_data wrist media with undistorted videos."""
    input_dir = input_dir.expanduser().resolve()
    camera_intrinsics = camera_intrinsics.expanduser().resolve()
    plan_path = input_dir / "dataset_plan.pkl"
    final_dir = input_dir / final_dirname
    if not plan_path.is_file():
        raise FileNotFoundError(f"dataset_plan.pkl not found: {plan_path}")
    if not final_dir.is_dir():
        raise FileNotFoundError(f"final_data folder not found: {final_dir}")
    if not camera_intrinsics.is_file():
        raise FileNotFoundError(
            f"camera intrinsics not found: {camera_intrinsics}"
        )

    with plan_path.open("rb") as f:
        plan: list[DemoPlan] = pickle.load(f)

    total_saved = 0
    if max_workers <= 1:
        for ep_idx, ep in enumerate(tqdm(plan, desc="Episodes", dynamic_ncols=True)):
            total_saved += _overwrite_episode_wrist_videos(
                ep_idx=ep_idx,
                ep=ep,
                final_dir=final_dir,
                downsample_stride=downsample_stride,
                teleop_delay_sec=teleop_delay_sec,
                camera_intrinsics=camera_intrinsics,
                out_size=out_size,
                out_fov=out_fov,
                video_codec=video_codec,
                video_ext=video_ext,
            )
    else:
        in_flight: dict[object, int] = {}
        ep_iter = iter(enumerate(plan))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for _ in range(max_workers):
                try:
                    ep_idx, ep = next(ep_iter)
                except StopIteration:
                    break
                fut = ex.submit(
                    _overwrite_episode_wrist_videos,
                    ep_idx,
                    ep,
                    final_dir,
                    downsample_stride,
                    teleop_delay_sec,
                    camera_intrinsics,
                    out_size,
                    out_fov,
                    video_codec,
                    video_ext,
                )
                in_flight[fut] = ep_idx

            with tqdm(total=len(plan), desc="Episodes", dynamic_ncols=True) as pbar:
                while in_flight:
                    done, _ = wait(
                        in_flight.keys(), return_when=FIRST_COMPLETED
                    )
                    for fut in done:
                        in_flight.pop(fut)
                        total_saved += fut.result()
                        pbar.update(1)
                        try:
                            ep_idx, ep = next(ep_iter)
                        except StopIteration:
                            continue
                        nxt = ex.submit(
                            _overwrite_episode_wrist_videos,
                            ep_idx,
                            ep,
                            final_dir,
                            downsample_stride,
                            teleop_delay_sec,
                            camera_intrinsics,
                            out_size,
                            out_fov,
                            video_codec,
                            video_ext,
                        )
                        in_flight[nxt] = ep_idx

    print(f"Overwrote {total_saved} frames in {final_dir}")


def cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    cli()
