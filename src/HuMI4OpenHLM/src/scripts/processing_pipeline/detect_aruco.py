import json
import os
import pickle
from typing import Annotated

import av
import cv2
import numpy as np
import tyro
import yaml
from tqdm import tqdm

from processing.cv_util import (
    TagDetectionResultDict,
    convert_fisheye_intrinsics_resolution,
    crop_fisheye_intrinsics,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


# %%
def main(
    input: Annotated[str, tyro.conf.arg(aliases=["-i"])],
    output: Annotated[str, tyro.conf.arg(aliases=["-o"])],
    intrinsics_json: Annotated[str, tyro.conf.arg(aliases=["-ij"])],
    aruco_yaml: Annotated[str, tyro.conf.arg(aliases=["-ay"])],
    num_workers: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 4,
    detect_fps: Annotated[float | None, tyro.conf.arg(aliases=["-df"])] = None,
    crop_box: Annotated[
        tuple[int, int, int, int] | None, tyro.conf.arg(aliases=["-cb"])
    ] = None,
):
    """Detect and localize ArUco tags in a video file.

    Args:
        input: Path to the input video file.
        output: Path to the output pickle file for storing detection results.
        intrinsics_json: Path to the JSON file containing camera intrinsics.
        aruco_yaml: Path to the YAML file containing ArUco configuration.
        num_workers: Number of workers for parallel processing.
        detect_fps: If set, only run detection at this sampling rate in Hz.
        crop_box: Optional crop box `(x0, y0, x1, y1)` in full-frame pixels.
    """
    cv2.setNumThreads(num_workers)
    if detect_fps is not None and detect_fps <= 0:
        raise ValueError(f"detect_fps must be positive, got {detect_fps}")

    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, "r")))
    aruco_dict = aruco_config["aruco_dict"]
    marker_size_map = aruco_config["marker_size_map"]

    # load intrinsics
    raw_fisheye_intr = parse_fisheye_intrinsics(
        json.load(open(intrinsics_json, "r"))
    )

    results: list[TagDetectionResultDict] = []
    sample_period = None if detect_fps is None else 1.0 / detect_fps
    next_sample_time = 0.0
    with av.open(os.path.expanduser(input)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = (in_stream.width, in_stream.height)
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res
        )
        effective_crop_box = None
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            x0 = max(0, min(x0, in_res[0] - 1))
            y0 = max(0, min(y0, in_res[1] - 1))
            x1 = max(x0 + 1, min(x1, in_res[0]))
            y1 = max(y0 + 1, min(y1, in_res[1]))
            effective_crop_box = (x0, y0, x1, y1)
            fisheye_intr = crop_fisheye_intrinsics(
                opencv_intr_dict=fisheye_intr,
                crop_box=effective_crop_box,
            )

        for i, frame in tqdm(
            enumerate(in_container.decode(in_stream)), total=in_stream.frames
        ):
            assert frame.pts is not None and in_stream.time_base is not None
            frame_cts_sec = frame.pts * in_stream.time_base
            should_detect = (
                sample_period is None
                or i == 0
                or float(frame_cts_sec) + 1e-9 >= next_sample_time
            )
            tag_dict = {}
            if should_detect:
                img = frame.to_ndarray(format="rgb24")
                pixel_offset = (0, 0)
                if effective_crop_box is not None:
                    x0, y0, x1, y1 = effective_crop_box
                    img = img[y0:y1, x0:x1]
                    pixel_offset = (x0, y0)
                tag_dict = detect_localize_aruco_tags(
                    img=img,
                    aruco_dict=aruco_dict,
                    marker_size_map=marker_size_map,
                    fisheye_intr_dict=fisheye_intr,
                    pixel_offset=pixel_offset,
                    refine_subpix=True,
                )
                if sample_period is not None:
                    while next_sample_time <= float(frame_cts_sec) + 1e-9:
                        next_sample_time += sample_period
            result = TagDetectionResultDict(
                frame_idx=i, time=float(frame_cts_sec), tag_dict=tag_dict
            )
            results.append(result)

    if sample_period is not None:
        results = _interpolate_tag_detections(
            results=results,
            allowed_tag_ids=sorted(marker_size_map.keys()),
            max_gap_sec=sample_period * 2.5,
        )

    # dump
    pickle.dump(results, open(os.path.expanduser(output), "wb"))


def _interpolate_tag_detections(
    results: list[TagDetectionResultDict],
    allowed_tag_ids: list[int],
    max_gap_sec: float,
) -> list[TagDetectionResultDict]:
    dense_results: list[TagDetectionResultDict] = []
    samples_by_tag: dict[
        int, list[tuple[int, float, np.ndarray, np.ndarray, np.ndarray]]
    ] = {tag_id: [] for tag_id in allowed_tag_ids}

    for frame in results:
        dense_results.append(
            TagDetectionResultDict(
                frame_idx=frame["frame_idx"],
                time=frame["time"],
                tag_dict=dict(frame["tag_dict"]),
            )
        )
        for tag_id, pose in frame["tag_dict"].items():
            if tag_id not in samples_by_tag:
                continue
            samples_by_tag[tag_id].append(
                (
                    frame["frame_idx"],
                    frame["time"],
                    np.asarray(pose["rvec"], dtype=np.float64),
                    np.asarray(pose["tvec"], dtype=np.float64),
                    np.asarray(pose["corners"], dtype=np.float64),
                )
            )

    for tag_id, samples in samples_by_tag.items():
        if len(samples) < 2:
            continue
        for (frame0, time0, rvec0, tvec0, corners0), (
            frame1,
            time1,
            rvec1,
            tvec1,
            corners1,
        ) in zip(samples[:-1], samples[1:], strict=False):
            gap_sec = time1 - time0
            if gap_sec <= 0 or gap_sec > max_gap_sec:
                continue
            for frame_idx in range(frame0 + 1, frame1):
                target_time = dense_results[frame_idx]["time"]
                alpha = (target_time - time0) / gap_sec
                if not (0.0 < alpha < 1.0):
                    continue
                dense_results[frame_idx]["tag_dict"][tag_id] = {
                    "rvec": ((1.0 - alpha) * rvec0 + alpha * rvec1).astype(
                        np.float64
                    ),
                    "tvec": ((1.0 - alpha) * tvec0 + alpha * tvec1).astype(
                        np.float64
                    ),
                    "corners": (
                        (1.0 - alpha) * corners0 + alpha * corners1
                    ).astype(np.float64),
                }
    return dense_results


if __name__ == "__main__":
    tyro.cli(main)
