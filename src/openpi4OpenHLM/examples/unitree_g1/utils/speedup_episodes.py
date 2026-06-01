"""
Speed up every episode in a dataset by a configurable factor (must be > 1.0).

Each episode contains:
  - data.json         : frame-by-frame state/action records at a fixed fps
  - wrist_rgb_left/   : left-wrist camera images  (000000.jpg, 000001.jpg, ...)
  - wrist_rgb_right/  : right-wrist camera images (000000.jpg, 000001.jpg, ...)

How speed-up works
------------------
Given a speed factor  s > 1.0  and an episode with  N  original frames:

  1. The new episode contains  N' = ceil(N / s)  frames.
  2. New frame  j  is sourced from original frame  round(j * s),  clamped to
     [0, N-1].  This gives uniform temporal sub-sampling across the episode.
  3. The fps field in data.json is kept unchanged.  Each new timestep now
     corresponds to  s  original timesteps, so the robot covers  s×  more
     distance per step → the motion is  s×  faster.

Action recalculation
--------------------
For position-control datasets where  action[t] ≈ state[t+1],  the action at
new frame  j  should target the state at new frame  j+1  (not j+1 in the
original sequence).  Concretely:

    src[j]         = round(j * s)          # source index for frame j
    new_state[j]   = old_state[ src[j] ]
    new_action[j]  = old_state[ src[j+1] ] # state the robot must reach next
                                            # (or old_state[N-1] at the last frame)

This preserves the correct per-step displacement for any speed factor.

Usage
-----
python speedup_episodes.py /home/hyd/codebase/openpi/data/0316-pick-laundry \
--speed 2.0 --output /home/hyd/codebase/openpi/data/0316-pick-laundry-2x

python speedup_episodes.py /home/hyd/codebase/openpi/data/0316-pick-cola \
--speed 2.0 --output /home/hyd/codebase/openpi/data/0316-pick-cola-2x --workers 8

"""

import argparse
import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Frame index computation
# ---------------------------------------------------------------------------


def compute_src_indices(n_orig: int, speed: float) -> list[int]:
    """Return the list of original frame indices to use for the sped-up episode.

    New episode length:  N' = ceil(N / speed)
    New frame j maps to: round(j * speed), clamped to [0, N-1].

    Example (n_orig=10, speed=1.5):
        N' = ceil(10/1.5) = 7
        indices = [0, 2, 3, 5, 6, 8, 9]
    """
    n_new = max(1, math.ceil(n_orig / speed))
    return [min(round(j * speed), n_orig - 1) for j in range(n_new)]


# ---------------------------------------------------------------------------
# Core per-episode logic
# ---------------------------------------------------------------------------


def resample_json(src_json: Path, dst_json: Path, speed: float) -> int:
    """Read src data.json, resample at the given speed factor, write to dst_json.

    For each new frame j:
      - state  = old_state[ src_indices[j] ]
      - action = old_state[ src_indices[j+1] ]  (next new frame's state)
                 old_state[ N-1 ]               (at the last frame)

    Returns the number of frames in the new episode.
    """
    with open(src_json) as f:
        data = json.load(f)

    frames = data["data"]
    src_indices = compute_src_indices(len(frames), speed)
    n_new = len(src_indices)

    # Keys that are rebuilt explicitly for each new frame.
    rebuilt_keys = {
        "idx",
        "wrist_rgb_left",
        "wrist_rgb_right",
        "state_body",
        "action_body",
        "state_hand_left",
        "state_hand_right",
        "action_hand_left",
        "action_hand_right",
    }

    new_frames: list[dict] = []
    for new_idx, old_idx in enumerate(src_indices):
        frame = frames[old_idx]
        # The action target is the state of the *next* new frame.
        # At the last frame the action repeats the final state (zero velocity).
        next_old_idx = src_indices[new_idx + 1] if new_idx + 1 < n_new else old_idx
        next_frame = frames[next_old_idx]

        new_frame: dict = {
            "idx": new_idx,
            "wrist_rgb_left":  f"wrist_rgb_left/{new_idx:06d}.jpg",
            "wrist_rgb_right": f"wrist_rgb_right/{new_idx:06d}.jpg",
            "state_body":       frame["state_body"],
            "action_body":      next_frame["state_body"],
            "state_hand_left":  frame["state_hand_left"],
            "state_hand_right": frame["state_hand_right"],
            "action_hand_left":  next_frame["state_hand_left"],
            "action_hand_right": next_frame["state_hand_right"],
        }

        # Forward any extra keys present in the source frame so the script
        # stays compatible with future dataset schema extensions.
        for key, val in frame.items():
            if key not in rebuilt_keys:
                new_frame[key] = val

        new_frames.append(new_frame)

    new_data = {
        "info": data["info"],  # fps unchanged; each step covers speed× more movement
        "text": data["text"],
        "data": new_frames,
    }

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    return n_new


def resample_camera(src_cam_dir: Path, dst_cam_dir: Path, src_indices: list[int]) -> int:
    """Copy the selected source frames to dst_cam_dir with sequential names.

    src_indices[j] is the original frame number to use for new frame j.
    Source images are assumed to be named 000000.jpg, 000001.jpg, ...

    Returns the number of images written.
    """
    dst_cam_dir.mkdir(parents=True, exist_ok=True)
    src_images = sorted(src_cam_dir.glob("*.jpg"))

    count = 0
    for new_idx, old_idx in enumerate(src_indices):
        if old_idx >= len(src_images):
            break
        dst_img = dst_cam_dir / f"{new_idx:06d}.jpg"
        shutil.copy2(src_images[old_idx], dst_img)
        count += 1

    return count


CAMERA_NAMES = ["wrist_rgb_left", "wrist_rgb_right"]


def process_episode(src_ep_dir: Path, dst_ep_dir: Path, speed: float) -> str:
    """Process a single episode: resample JSON and both camera streams.

    Returns a short status string for progress logging.
    """
    try:
        src_json = src_ep_dir / "data.json"
        with open(src_json) as f:
            n_orig = len(json.load(f)["data"])
        src_indices = compute_src_indices(n_orig, speed)

        n_frames = resample_json(src_json, dst_ep_dir / "data.json", speed)

        for cam in CAMERA_NAMES:
            src_cam = src_ep_dir / cam
            if not src_cam.is_dir():
                continue
            resample_camera(src_cam, dst_ep_dir / cam, src_indices)

        return f"OK  {src_ep_dir.name}: {n_orig} → {n_frames} frames"

    except Exception as exc:
        return f"ERR {src_ep_dir.name}: {exc}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def collect_episodes(root: Path) -> list[Path]:
    """Return a sorted list of episode_XXXX directories under root."""
    return sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Speed up every episode in a dataset by a configurable factor.\n"
            "Each new frame is sourced from the corresponding position in the\n"
            "original timeline, and action targets are recalculated to span the\n"
            "correct number of original frames so per-step displacement is preserved."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Root dataset directory containing episode_XXXX sub-folders.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        required=True,
        metavar="FACTOR",
        help="Speed-up factor (must be > 1.0).  E.g. 1.5, 1.8, 2.0.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="DIR",
        help="Output directory for the sped-up dataset (created if absent).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, (os.cpu_count() or 4)),
        help="Number of parallel worker processes (one per episode). Default: min(16, cpu_count).",
    )
    args = parser.parse_args()

    if args.speed <= 1.0:
        print(
            f"[ERROR] --speed must be > 1.0 (got {args.speed}).  "
            "This script only supports speed-up, not slow-down.",
            file=sys.stderr,
        )
        sys.exit(1)

    src_root = Path(args.dataset_dir)
    if not src_root.is_dir():
        print(f"[ERROR] Not a directory: {src_root}", file=sys.stderr)
        sys.exit(1)

    dst_root = Path(args.output)
    if dst_root.exists() and any(dst_root.iterdir()):
        print(
            f"[ERROR] Output directory already exists and is non-empty: {dst_root}\n"
            "        Remove it or choose a different path.",
            file=sys.stderr,
        )
        sys.exit(1)
    dst_root.mkdir(parents=True, exist_ok=True)

    episodes = collect_episodes(src_root)
    if not episodes:
        print("[WARNING] No episode_XXXX directories found. Exiting.")
        sys.exit(0)

    print(f"[INFO] Source dataset  : {src_root}")
    print(f"[INFO] Output dataset  : {dst_root}")
    print(f"[INFO] Speed factor    : {args.speed}×")
    print(f"[INFO] Episodes found  : {len(episodes)}")
    print(f"[INFO] Camera streams  : {CAMERA_NAMES}")
    print(f"[INFO] Workers         : {args.workers}")
    print()

    total = len(episodes)
    errors: list[str] = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_name = {
            executor.submit(
                process_episode,
                ep,
                dst_root / ep.name,
                args.speed,
            ): ep.name
            for ep in episodes
        }
        for future in as_completed(future_to_name):
            result = future.result()
            done += 1
            print(f"  [{done:>4}/{total}] {result}", flush=True)
            if result.startswith("ERR"):
                errors.append(result)

    print()
    if errors:
        print(f"[DONE] Finished with {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"[DONE] Successfully processed all {total} episodes.")
        print(f"       {args.speed}×-speed dataset written to: {dst_root}")


if __name__ == "__main__":
    main()
