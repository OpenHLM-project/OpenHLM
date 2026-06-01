"""
Crop and resize all JPG images under a dataset directory in-place.

Processing pipeline per image:
  1. Assert shape is exactly 2704 (W) x 2028 (H).
  2. Center-crop horizontally: remove 338 px from left and right → 2028 x 2028.
  3. Resize to 224 x 224 using Lanczos resampling.

Usage:
    python crop_and_resize_images.py /path/to/dataset_folder
    python crop_and_resize_images.py /path/to/dataset_folder --quality 95 --workers 16
"""

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# Source dimensions (must match exactly)
SRC_W, SRC_H = 2704, 2028
# After center-crop: remove (2704-2028)//2 = 338 px from each side
CROP_X0 = (SRC_W - SRC_H) // 2   # 338
CROP_X1 = SRC_W - CROP_X0        # 2366
# Target dimensions after resize
DST_SIZE = (224, 224)


def check_image_size(jpg_path: str) -> tuple[str, int, int]:
    """Return (path, width, height) for a single image without decoding pixels."""
    with Image.open(jpg_path) as img:
        w, h = img.size
    return jpg_path, w, h


def check_episode_sizes(episode_dir: Path, workers: int) -> list[tuple[str, int, int]]:
    """Check every JPG in one episode directory in parallel.

    Returns a list of (path, w, h) tuples for images that do NOT match SRC_W x SRC_H.
    """
    jpgs = [str(p) for p in sorted(episode_dir.rglob("*.jpg"))]
    bad: list[tuple[str, int, int]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for path, w, h in executor.map(check_image_size, jpgs, chunksize=64):
            if w != SRC_W or h != SRC_H:
                bad.append((path, w, h))
    return bad


def process_image(jpg_path: str, quality: int) -> str:
    """Crop and resize a single JPG image in-place.

    Returns the path on success, or an error string prefixed with 'ERROR:'.
    """
    try:
        with Image.open(jpg_path) as img:
            w, h = img.size
            if w != SRC_W or h != SRC_H:
                return f"ERROR: unexpected size {w}x{h} — {jpg_path}"
            # Center-crop horizontally (height unchanged)
            cropped = img.crop((CROP_X0, 0, CROP_X1, SRC_H))
            # Resize to 224x224
            resized = cropped.resize(DST_SIZE, Image.LANCZOS)
            resized.save(jpg_path, "JPEG", quality=quality, optimize=True)
        return jpg_path
    except Exception as exc:
        return f"ERROR: {exc} — {jpg_path}"


def collect_jpg_files(root: Path) -> dict[str, list[Path]]:
    """Walk root and group JPG files by their episode directory.

    Returns a dict mapping episode_dir_name → sorted list of Path objects.
    """
    episode_files: dict[str, list[Path]] = defaultdict(list)
    for episode_dir in sorted(root.iterdir()):
        if not episode_dir.is_dir() or not episode_dir.name.startswith("episode_"):
            continue
        jpgs = sorted(episode_dir.rglob("*.jpg"))
        if jpgs:
            episode_files[episode_dir.name] = jpgs
    return episode_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Center-crop (2704→2028) and resize (→224×224) all JPG images in a dataset directory."
    )
    parser.add_argument("dataset_dir", type=str, help="Root dataset directory containing episode_XXXX sub-folders.")
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG save quality (1-95). Default: 95.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 4) * 2),
        help="Number of parallel worker processes. Default: min(32, cpu_count*2).",
    )
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Dataset directory : {root}")
    print(f"[INFO] Workers           : {args.workers}")
    print(f"[INFO] JPEG quality      : {args.quality}")
    print(f"[INFO] Crop box          : x=[{CROP_X0}, {CROP_X1}), y=[0, {SRC_H})  →  {SRC_H}x{SRC_H}")
    print(f"[INFO] Resize target     : {DST_SIZE[0]}x{DST_SIZE[1]}")
    print()

    # --- Collect files and report per-episode counts ---
    episode_files = collect_jpg_files(root)
    if not episode_files:
        print("[WARNING] No episodes with JPG files found. Exiting.")
        sys.exit(0)

    all_files: list[Path] = []
    for ep_name, files in sorted(episode_files.items()):
        print(f"  {ep_name}: {len(files):>6} image frames")
        all_files.extend(files)

    total = len(all_files)
    print(f"\n[INFO] Total images to process: {total}")

    # --- Pre-flight: verify every image is SRC_W x SRC_H before touching anything ---
    print(f"[INFO] Pre-flight size check on all {total} images...\n")
    bad_episodes: dict[str, list[tuple[str, int, int]]] = {}
    for episode_dir in sorted(root.iterdir()):
        if not episode_dir.is_dir() or not episode_dir.name.startswith("episode_"):
            continue
        if episode_dir.name not in episode_files:
            continue
        bad = check_episode_sizes(episode_dir, args.workers)
        if bad:
            bad_episodes[str(episode_dir)] = bad

    if bad_episodes:
        print("[ERROR] The following episodes contain images with unexpected dimensions (expected 2704x2028):\n")
        for ep_path, bad_list in sorted(bad_episodes.items()):
            print(f"  {ep_path}  ({len(bad_list)} bad image(s))")
            for img_path, w, h in bad_list[:5]:  # show up to 5 examples per episode
                print(f"    {img_path}  →  {w}x{h}")
            if len(bad_list) > 5:
                print(f"    ... and {len(bad_list) - 5} more")
        print("\n[ABORT] Fix the above images before running this script. Exiting.")
        sys.exit(1)

    print("[INFO] Pre-flight check passed. All images are 2704x2028.")
    print("[INFO] Starting parallel processing...\n")

    # --- Parallel processing ---
    errors: list[str] = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, str(p), args.quality): p for p in all_files}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result.startswith("ERROR:"):
                errors.append(result)
                print(f"[WARN] {result}", flush=True)
            # Progress reporting every 500 images
            if done % 500 == 0 or done == total:
                pct = done / total * 100
                print(f"  Progress: {done}/{total} ({pct:.1f}%)", flush=True)

    print()
    if errors:
        print(f"[DONE] Finished with {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"[DONE] Successfully processed all {total} images.")


if __name__ == "__main__":
    main()
