import json
import os
import pathlib
import pickle
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from datetime import date
from typing import Annotated, NotRequired, Optional, TypedDict

import av
import cv2
import numpy as np
import numpy.typing as npt
import tyro
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from processing.cv_util import (
    FisheyeRectConverter,
    parse_fisheye_intrinsics,
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
    prompt: NotRequired[str | None]


DEFAULT_FINAL_IMAGE_SIZE = (224, 224)
DEFAULT_FINAL_OUT_FOV = 90.0
DEFAULT_FINAL_VIDEO_CODEC = "mp4v"
DEFAULT_FINAL_VIDEO_EXT = ".mp4"


def _resolve_default_intrinsics_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).parents[3]
        / "configs"
        / "calibration"
        / "gopro_intrinsics_2_7k.json"
    )


def _build_fisheye_converter(
    camera_intrinsics: pathlib.Path,
    out_size: tuple[int, int],
    out_fov: float,
) -> FisheyeRectConverter:
    with camera_intrinsics.open("r") as f:
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(f))
    return FisheyeRectConverter(
        **opencv_intr_dict,
        out_size=out_size,
        out_fov=out_fov,
    )


def _load_gripper_ranges(
    demos_dir: pathlib.Path,
) -> dict[int, tuple[float, float]]:
    """Load per-gripper (min_width, max_width) from calibration folders."""
    ranges: dict[int, tuple[float, float]] = {}
    for p in demos_dir.glob("gripper_calibration_*/gripper_range.json"):
        with p.open("r") as f:
            data = json.load(f)
        gid = int(data["gripper_id"])
        wmin = float(data["min_width"])
        wmax = float(data["max_width"])
        if wmax <= wmin:
            raise ValueError(
                f"Invalid width range in {p}: min={wmin}, max={wmax}"
            )
        ranges[gid] = (wmin, wmax)
    return ranges


def _iter_selected_rgb_frames(
    video_path: pathlib.Path, frame_indices: list[int]
):
    """Yield selected RGB frames as (absolute_frame_index, rgb_frame)."""
    if len(frame_indices) == 0:
        return
    sorted_indices = sorted(frame_indices)
    ptr = 0
    last_idx = sorted_indices[-1]

    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx > last_idx or ptr >= len(sorted_indices):
                break
            target_idx = sorted_indices[ptr]
            if frame_idx == target_idx:
                yield frame_idx, frame.to_ndarray(format="rgb24")
                ptr += 1


def _extract_single_rgb_frame(
    video_path: pathlib.Path, frame_idx: int
) -> np.ndarray:
    """Decode and return a single RGB frame by absolute index."""
    for idx, rgb in _iter_selected_rgb_frames(video_path, [frame_idx]):
        if idx == frame_idx:
            return rgb
    raise RuntimeError(f"Failed to decode frame {frame_idx} from {video_path}")


def _open_video_writer(
    path: pathlib.Path,
    rgb_shape: tuple[int, ...],
    fps: float,
    codec: str = DEFAULT_FINAL_VIDEO_CODEC,
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(codec) != 4:
        raise ValueError(f"Video codec must be four characters, got {codec!r}")
    if len(rgb_shape) < 2:
        raise ValueError(f"Expected RGB frame shape, got {rgb_shape}")
    height, width = int(rgb_shape[0]), int(rgb_shape[1])
    writer = cv2.VideoWriter(
        path.as_posix(),
        cv2.VideoWriter_fourcc(*codec),
        float(max(fps, 1.0)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def _write_rgb_video_frame(writer: cv2.VideoWriter, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    writer.write(bgr)


def _undistort_rgb(
    rgb: np.ndarray, fisheye_converter: FisheyeRectConverter
) -> np.ndarray:
    return fisheye_converter.forward(rgb)


def _find_gripper(grippers: list[GripperPlan], gid: int) -> GripperPlan:
    for g in grippers:
        if int(g["gripper_id"]) == gid:
            return g
    raise ValueError(f"Cannot find gripper_id={gid} in plan grippers")


def _process_episode(
    ep_idx: int,
    ep: DemoPlan,
    out_dir: pathlib.Path,
    downsample_stride: int,
    root_gripper_id: int,
    hand_action_threshold: float,
    teleop_delay_sec: float,
    goal: str,
    gripper_ranges: dict[int, tuple[float, float]],
    camera_intrinsics: pathlib.Path,
    out_size: tuple[int, int],
    out_fov: float,
    video_codec: str,
    video_ext: str,
    show_episode_progress: bool = False,
) -> int:
    fisheye_converter = _build_fisheye_converter(
        camera_intrinsics=camera_intrinsics,
        out_size=out_size,
        out_fov=out_fov,
    )
    ep_prompt = ep.get("prompt") or goal
    episode_dir = out_dir / f"episode_{ep_idx:04d}"
    videos_dir = episode_dir / "videos"
    episode_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    timestamps = np.asarray(ep["episode_timestamps"], dtype=np.float64)
    joint_pos = np.asarray(ep["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
        raise ValueError(
            f"Expected joint_pos shape (T,29), got {joint_pos.shape}"
        )
    if len(timestamps) != joint_pos.shape[0]:
        raise ValueError(
            f"Timestamp/joint length mismatch: {len(timestamps)} vs "
            f"{joint_pos.shape[0]}"
        )

    cam_right = next(
        (c for c in ep["cameras"] if int(c["gripper_id"]) == 0), None
    )
    cam_left = next(
        (c for c in ep["cameras"] if int(c["gripper_id"]) == 1), None
    )
    if cam_right is None or cam_left is None:
        raise ValueError(
            "Expected cameras for gripper_id 0 and 1 in each episode"
        )

    n = len(timestamps)
    sampled_local = np.arange(0, n, downsample_stride, dtype=np.int64)
    sampled_ts = timestamps[sampled_local]

    root_plan = _find_gripper(ep["grippers"], gid=root_gripper_id)
    right_gripper_plan = _find_gripper(ep["grippers"], gid=0)
    left_gripper_plan = _find_gripper(ep["grippers"], gid=1)
    root_pose = np.asarray(root_plan["tcp_pose"], dtype=np.float64)
    if root_pose.shape[0] != n or root_pose.shape[1] != 6:
        raise ValueError(f"Unexpected root tcp_pose shape: {root_pose.shape}")
    right_width = right_gripper_plan["gripper_width"]
    left_width = left_gripper_plan["gripper_width"]
    if right_width is None or left_width is None:
        raise ValueError("gripper_width is missing for gripper id 0/1")
    right_width = np.asarray(right_width, dtype=np.float64)
    left_width = np.asarray(left_width, dtype=np.float64)
    if right_width.shape[0] != n or left_width.shape[0] != n:
        raise ValueError(
            "gripper_width length mismatch with episode length: "
            f"right={right_width.shape}, left={left_width.shape}, n={n}"
        )
    right_min, right_max = gripper_ranges[0]
    left_min, left_max = gripper_ranges[1]
    right_norm = np.clip(right_width / (right_max - right_min), 0.0, 1.0)
    left_norm = np.clip(left_width / (left_max - left_min), 0.0, 1.0)
    sampled_right_norm = right_norm[sampled_local]
    sampled_left_norm = left_norm[sampled_local]
    sampled_root_pose = root_pose[sampled_local]
    sampled_rotvec = sampled_root_pose[:, 3:]
    sampled_rpy = Rotation.from_rotvec(sampled_rotvec).as_euler(
        "xyz", degrees=False
    )
    root_roll = sampled_rpy[:, 0]
    root_pitch = sampled_rpy[:, 1]
    root_yaw = sampled_rpy[:, 2]

    if len(sampled_ts) >= 2:
        root_yaw_vel = np.gradient(root_yaw, sampled_ts, edge_order=1)
        sampled_fps = float(1.0 / np.median(np.diff(sampled_ts)))
    else:
        root_yaw_vel = np.zeros_like(root_yaw)
        sampled_fps = 0.0
    delay_frames = int(round(max(0.0, teleop_delay_sec) * sampled_fps))
    valid_count = max(0, len(sampled_local) - delay_frames)

    left_start, _left_end = cam_left["video_start_end"]
    right_start, _right_end = cam_right["video_start_end"]
    left_video = pathlib.Path(cam_left["video_path"])
    right_video = pathlib.Path(cam_right["video_path"])
    left_abs_write_indices = [
        left_start + int(i) for i in sampled_local[:valid_count]
    ]
    right_abs_write_indices = [
        right_start + int(i) for i in sampled_local[:valid_count]
    ]

    sampled_body: list[list[float]] = []
    for k, local_idx in enumerate(sampled_local):
        body = np.concatenate(
            [
                np.array(
                    [root_roll[k], root_pitch[k], root_yaw_vel[k]],
                    dtype=np.float64,
                ),
                joint_pos[int(local_idx)],
            ],
            axis=0,
        )
        sampled_body.append(body.tolist())

    left_iter = _iter_selected_rgb_frames(left_video, left_abs_write_indices)
    right_iter = _iter_selected_rgb_frames(
        right_video, right_abs_write_indices
    )

    first_pair: tuple[np.ndarray, np.ndarray] | None = None
    if valid_count > 0:
        try:
            left_decoded_idx, left_rgb0 = next(left_iter)
            right_decoded_idx, right_rgb0 = next(right_iter)
        except StopIteration as exc:
            raise RuntimeError(
                f"Failed to decode required first frame pair (ep={ep_idx})"
            ) from exc
        expected_left0 = left_abs_write_indices[0]
        expected_right0 = right_abs_write_indices[0]
        if left_decoded_idx != expected_left0:
            raise RuntimeError(
                "Left first frame index mismatch "
                f"(ep={ep_idx}, expected={expected_left0}, decoded={left_decoded_idx})"
            )
        if right_decoded_idx != expected_right0:
            raise RuntimeError(
                "Right first frame index mismatch "
                f"(ep={ep_idx}, expected={expected_right0}, decoded={right_decoded_idx})"
            )
        left_rgb0 = _undistort_rgb(left_rgb0, fisheye_converter)
        right_rgb0 = _undistort_rgb(right_rgb0, fisheye_converter)
        wrist_h, wrist_w = left_rgb0.shape[:2]
        first_pair = (left_rgb0, right_rgb0)
    elif len(sampled_local) > 0:
        probe_left_abs = left_start + int(sampled_local[0])
        probe_rgb = _extract_single_rgb_frame(left_video, probe_left_abs)
        probe_rgb = _undistort_rgb(probe_rgb, fisheye_converter)
        wrist_h, wrist_w = probe_rgb.shape[:2]
    else:
        wrist_h, wrist_w = 0, 0

    fps_for_video = float(max(round(sampled_fps), 1))
    left_video_rel = f"videos/wrist_rgb_left{video_ext}"
    right_video_rel = f"videos/wrist_rgb_right{video_ext}"
    left_out_video = episode_dir / left_video_rel
    right_out_video = episode_dir / right_video_rel
    left_writer: cv2.VideoWriter | None = None
    right_writer: cv2.VideoWriter | None = None

    out_json = episode_dir / "data.json"
    try:
        if valid_count > 0 and first_pair is not None:
            left_writer = _open_video_writer(
                left_out_video, first_pair[0].shape, fps_for_video, video_codec
            )
            right_writer = _open_video_writer(
                right_out_video, first_pair[1].shape, fps_for_video, video_codec
            )

        with out_json.open("w") as f:
            f.write("{\n")
            f.write('  "info": {\n')
            f.write(f'    "date": {json.dumps(date.today().isoformat())},\n')
            f.write('    "wrist_image": {\n')
            f.write(f'      "height": {int(wrist_h)},\n')
            f.write(f'      "width": {int(wrist_w)},\n')
            f.write(f'      "fps": {int(round(sampled_fps))}\n')
            f.write("    },\n")
            f.write('    "video": {\n')
            f.write(f'      "format": {json.dumps(video_ext.lstrip("."))},\n')
            f.write(f'      "codec": {json.dumps(video_codec)},\n')
            f.write(f'      "fps": {int(round(sampled_fps))},\n')
            f.write('      "cameras": {\n')
            f.write('        "wrist_rgb_left": {\n')
            f.write(f'          "height": {int(wrist_h)},\n')
            f.write(f'          "width": {int(wrist_w)},\n')
            f.write('          "channels": 3,\n')
            f.write(f'          "frames": {int(valid_count)},\n')
            f.write(f'          "path": {json.dumps(left_video_rel)}\n')
            f.write("        },\n")
            f.write('        "wrist_rgb_right": {\n')
            f.write(f'          "height": {int(wrist_h)},\n')
            f.write(f'          "width": {int(wrist_w)},\n')
            f.write('          "channels": 3,\n')
            f.write(f'          "frames": {int(valid_count)},\n')
            f.write(f'          "path": {json.dumps(right_video_rel)}\n')
            f.write("        }\n")
            f.write("      }\n")
            f.write("    }\n")
            f.write("  },\n")
            f.write(
                f'  "text": {json.dumps({"goal": ep_prompt, "desc": ep_prompt})},\n'
            )
            f.write('  "data": [\n')

            frame_iter = range(valid_count)
            if show_episode_progress:
                frame_iter = tqdm(
                    frame_iter,
                    total=valid_count,
                    desc=f"Episode {ep_idx:04d} pid={os.getpid()}",
                    dynamic_ncols=True,
                    leave=False,
                    position=1,
                )
            for k in frame_iter:
                if k == 0 and first_pair is not None:
                    left_rgb, right_rgb = first_pair
                else:
                    try:
                        left_decoded_idx, left_rgb = next(left_iter)
                        right_decoded_idx, right_rgb = next(right_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"Failed to decode required frame pair (ep={ep_idx}, k={k})"
                        ) from exc

                    expected_left_idx = left_abs_write_indices[k]
                    expected_right_idx = right_abs_write_indices[k]
                    if left_decoded_idx != expected_left_idx:
                        raise RuntimeError(
                            "Left frame index mismatch "
                            f"(ep={ep_idx}, k={k}, expected={expected_left_idx}, "
                            f"decoded={left_decoded_idx})"
                        )
                    if right_decoded_idx != expected_right_idx:
                        raise RuntimeError(
                            "Right frame index mismatch "
                            f"(ep={ep_idx}, k={k}, expected={expected_right_idx}, "
                            f"decoded={right_decoded_idx})"
                        )
                    left_rgb = _undistort_rgb(left_rgb, fisheye_converter)
                    right_rgb = _undistort_rgb(right_rgb, fisheye_converter)

                if left_writer is None or right_writer is None:
                    left_writer = _open_video_writer(
                        left_out_video, left_rgb.shape, fps_for_video, video_codec
                    )
                    right_writer = _open_video_writer(
                        right_out_video, right_rgb.shape, fps_for_video, video_codec
                    )
                _write_rgb_video_frame(left_writer, left_rgb)
                _write_rgb_video_frame(right_writer, right_rgb)

                frame_idx = int(k)
                action_k = k + delay_frames
                entry = {
                    "idx": frame_idx,
                    "wrist_rgb_left": {
                        "video_path": left_video_rel,
                        "frame_index": frame_idx,
                    },
                    "wrist_rgb_right": {
                        "video_path": right_video_rel,
                        "frame_index": frame_idx,
                    },
                    "state_body": sampled_body[k],
                    "action_body": sampled_body[action_k],
                    "state_hand_left": float(sampled_left_norm[k]),
                    "state_hand_right": float(sampled_right_norm[k]),
                    "action_hand_left": float(
                        sampled_left_norm[action_k] > hand_action_threshold
                    ),
                    "action_hand_right": float(
                        sampled_right_norm[action_k] > hand_action_threshold
                    ),
                }
                if k > 0:
                    f.write(",\n")
                f.write("    ")
                f.write(json.dumps(entry))

            f.write("\n  ]\n")
            f.write("}\n")
    finally:
        if left_writer is not None:
            left_writer.release()
        if right_writer is not None:
            right_writer.release()
    return valid_count


def main(
    input_dir: Annotated[pathlib.Path, tyro.conf.arg(aliases=["-i"])],
    output_dirname: str = "final_data",
    downsample_stride: int = 2,
    root_gripper_id: int = 2,
    hand_action_threshold: float = 0.7,
    teleop_delay_sec: float = 0.2,
    camera_intrinsics: pathlib.Path = _resolve_default_intrinsics_path(),
    out_size: tuple[int, int] = DEFAULT_FINAL_IMAGE_SIZE,
    out_fov: float = DEFAULT_FINAL_OUT_FOV,
    video_codec: str = DEFAULT_FINAL_VIDEO_CODEC,
    video_ext: str = DEFAULT_FINAL_VIDEO_EXT,
    max_workers: int = 4,
    show_episode_progress: bool = True,
    goal: str = "Walk forward to the table and then put the bottle on the mouse pad.",
) -> None:
    """
    Generate final_data folder from dataset_plan.pkl.

    Output structure under <input_dir>/<output_dirname>:
      - episode_0000/
        - data.json
        - videos/
          - wrist_rgb_left.mp4
          - wrist_rgb_right.mp4
      - episode_0001/
        - ...
    """
    input_dir = input_dir.expanduser().resolve()
    camera_intrinsics = camera_intrinsics.expanduser().resolve()
    plan_path = input_dir / "dataset_plan.pkl"
    if not plan_path.is_file():
        raise FileNotFoundError(f"dataset_plan.pkl not found: {plan_path}")
    if not camera_intrinsics.is_file():
        raise FileNotFoundError(
            f"camera intrinsics not found: {camera_intrinsics}"
        )

    with plan_path.open("rb") as f:
        plan: list[DemoPlan] = pickle.load(f)
    gripper_ranges = _load_gripper_ranges(input_dir / "demos")
    if 0 not in gripper_ranges or 1 not in gripper_ranges:
        raise ValueError(
            "Need gripper ranges for id 0 and 1 under demos/gripper_calibration_*/gripper_range.json"
        )

    out_dir = input_dir / output_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    if max_workers <= 1:
        for ep_idx, ep in enumerate(
            tqdm(plan, desc="Episodes", dynamic_ncols=True)
        ):
            total_saved += _process_episode(
                ep_idx=ep_idx,
                ep=ep,
                out_dir=out_dir,
                downsample_stride=downsample_stride,
                root_gripper_id=root_gripper_id,
                hand_action_threshold=hand_action_threshold,
                teleop_delay_sec=teleop_delay_sec,
                goal=goal,
                gripper_ranges=gripper_ranges,
                camera_intrinsics=camera_intrinsics,
                out_size=out_size,
                out_fov=out_fov,
                video_codec=video_codec,
                video_ext=video_ext,
                show_episode_progress=show_episode_progress,
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
                    _process_episode,
                    ep_idx,
                    ep,
                    out_dir,
                    downsample_stride,
                    root_gripper_id,
                    hand_action_threshold,
                    teleop_delay_sec,
                    goal,
                    gripper_ranges,
                    camera_intrinsics,
                    out_size,
                    out_fov,
                    video_codec,
                    video_ext,
                    show_episode_progress,
                )
                in_flight[fut] = ep_idx

            with tqdm(
                total=len(plan), desc="Episodes", dynamic_ncols=True
            ) as pbar:
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
                            _process_episode,
                            ep_idx,
                            ep,
                            out_dir,
                            downsample_stride,
                            root_gripper_id,
                            hand_action_threshold,
                            teleop_delay_sec,
                            goal,
                            gripper_ranges,
                            camera_intrinsics,
                            out_size,
                            out_fov,
                            video_codec,
                            video_ext,
                            show_episode_progress,
                        )
                        in_flight[nxt] = ep_idx

    print(f"Saved {total_saved} frames to {out_dir}")


def cli() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    cli()
