"""
Merge one episode dataset into another by moving and renumbering episodes.

This script moves every `episode_XXXX` directory from a source dataset root into
a target dataset root. The moved episodes are renamed so their indices continue
from the largest episode index already present in the target dataset.

Example:
    Target: /home/hyd/codebase/openpi/data/20260319_2046_pick_place_high
    Source: /home/hyd/codebase/openpi/data/20260320_1549_pick_place_high

If the largest target episode is `episode_0021`, then the first moved source
episode becomes `episode_0022`, the next becomes `episode_0023`, and so on.

Usage:
    python merge_datasets.py /home/hyd/codebase/openpi/data/20260318_2046_long_task /home/hyd/codebase/openpi/data/20260320_1549_long_task
    python merge_datasets.py /path/to/target_dataset /path/to/source_dataset --dry-run
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


EPISODE_NAME_RE = re.compile(r"^episode_(\d+)$")


def parse_episode_index(path: Path) -> int | None:
    """Return the integer index for an episode directory name, or None."""
    match = EPISODE_NAME_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def collect_episode_dirs(root: Path) -> list[tuple[int, Path]]:
    """Return all episode directories under root, sorted by episode index."""
    episodes: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        index = parse_episode_index(child)
        if index is not None:
            episodes.append((index, child))
    episodes.sort(key=lambda item: item[0])
    return episodes


def get_next_episode_index(root: Path) -> int:
    """Return the next available episode index under the target dataset root."""
    episodes = collect_episode_dirs(root)
    if not episodes:
        return 0
    return episodes[-1][0] + 1


def build_move_plan(target_root: Path, source_root: Path) -> list[tuple[Path, Path]]:
    """Build the source→destination move plan for all source episodes."""
    source_episodes = collect_episode_dirs(source_root)
    if not source_episodes:
        return []

    next_index = get_next_episode_index(target_root)
    move_plan: list[tuple[Path, Path]] = []
    for offset, (_, src_episode_dir) in enumerate(source_episodes):
        dst_episode_dir = target_root / f"episode_{next_index + offset:04d}"
        move_plan.append((src_episode_dir, dst_episode_dir))
    return move_plan


def validate_inputs(target_root: Path, source_root: Path) -> None:
    """Validate dataset paths before building the move plan."""
    if not target_root.is_dir():
        raise ValueError(f"Target dataset directory does not exist: {target_root}")
    if not source_root.is_dir():
        raise ValueError(f"Source dataset directory does not exist: {source_root}")
    if target_root.resolve() == source_root.resolve():
        raise ValueError("Target and source dataset directories must be different.")


def print_plan(move_plan: list[tuple[Path, Path]], dry_run: bool) -> None:
    """Print the planned rename/move operations."""
    action = "Would move" if dry_run else "Will move"
    print(f"[INFO] {action} {len(move_plan)} episode(s):")
    for src_path, dst_path in move_plan:
        print(f"  {src_path.name}  ->  {dst_path.name}")


def execute_move_plan(move_plan: list[tuple[Path, Path]]) -> None:
    """Execute the move plan in order."""
    for src_path, dst_path in move_plan:
        if dst_path.exists():
            raise FileExistsError(f"Destination already exists: {dst_path}")
        shutil.move(str(src_path), str(dst_path))
        print(f"[DONE] Moved {src_path.name} -> {dst_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move every episode_XXXX directory from a source dataset into a "
            "target dataset, renumbering them to continue after the target's "
            "largest existing episode index."
        )
    )
    parser.add_argument(
        "target_dataset",
        type=str,
        help="Dataset root that will receive the moved episodes.",
    )
    parser.add_argument(
        "source_dataset",
        type=str,
        help="Dataset root whose episode_XXXX directories will be moved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned moves without modifying anything.",
    )
    args = parser.parse_args()

    target_root = Path(args.target_dataset)
    source_root = Path(args.source_dataset)

    try:
        validate_inputs(target_root, source_root)
        move_plan = build_move_plan(target_root, source_root)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    if not move_plan:
        print("[WARNING] No episode_XXXX directories found in the source dataset. Nothing to do.")
        sys.exit(0)

    print(f"[INFO] Target dataset : {target_root}")
    print(f"[INFO] Source dataset : {source_root}")
    print(f"[INFO] Dry run        : {args.dry_run}")
    print()
    print_plan(move_plan, args.dry_run)

    if args.dry_run:
        print("\n[DONE] Dry run finished. No files were moved.")
        sys.exit(0)

    try:
        execute_move_plan(move_plan)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"[DONE] Successfully moved {len(move_plan)} episode(s) into {target_root}.")


if __name__ == "__main__":
    main()
