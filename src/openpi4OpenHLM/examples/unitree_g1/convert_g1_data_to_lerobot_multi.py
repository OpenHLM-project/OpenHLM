"""
Script for converting multiple Unitree G1 datasets to LeRobot format.

Usage:
uv run examples/unitree_g1/convert_g1_data_to_lerobot_multi.py \
    --parent_dir /home/hyd/codebase/openpi/data \
    --dataset_folders 20260309_2054 20260307_1657 \
    --repo_name my_org/g1_combined

Each dataset folder (e.g. 20260309_2054) must contain episode_* subdirectories,
where each episode_* directory holds a data.json and camera videos.

If you want to push the resulting dataset to the Hugging Face Hub, add --push_to_hub.

Use --episode_sample_ratio to randomly keep only a fraction of episodes from each folder.
For example, --episode_sample_ratio 0.5 will randomly sample 50% of episodes per folder.

Use --max_episodes_per_folder to cap the number of episodes used from each folder.
For example, --max_episodes_per_folder 40 will randomly sample at most 40 episodes per folder.
When both --episode_sample_ratio and --max_episodes_per_folder are set, the stricter limit applies.

Use --extra_dataset_folders to add extra dataset folders from which exactly ONE episode is
randomly sampled per folder. This is useful for adding a small amount of auxiliary data
from many different tasks. For example:
    --extra_dataset_folders 20260408_1546_pp_cola 20260402_1909_shelf_cup
will randomly pick 1 episode from each of those two folders.

Use --first_k_episodes_per_folder to deterministically select the first k episodes (by sorted
name) from each folder in --dataset_folders. The list must have the same length as
--dataset_folders. For example, with three dataset folders:
    --first_k_episodes_per_folder 36 80 80
will take the first 36 episodes from the first folder and the first 80 episodes from each of
the second and third folders. Unlike --max_episodes_per_folder, this option is deterministic
(no random sampling) and allows a different cap per folder. When this option is set,
--episode_sample_ratio and --max_episodes_per_folder are ignored for --dataset_folders.

The resulting dataset will be saved to the $LEROBOT_HOME directory.
"""

import json
from pathlib import Path
import random
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

REPO_NAME = "Yingdong-Hu/g1_walk_forward"  # Default output dataset name


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL.

    If the input images have dimensions (height=2028, width=2704), a center crop is first applied
    to produce (height=2028, width=2028) before resizing. The crop removes equal margins from the
    left and right sides only; the height is left intact.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # Center-crop 2704x2028 images to 2028x2028 before resizing.
    if images.shape[-3:-1] == (2028, 2704):
        crop_h, crop_w = 2028, 2028
        h_start = (2028 - crop_h) // 2  # 0 — no vertical crop
        w_start = (2704 - crop_w) // 2  # 338 pixels removed from each side
        images = images[..., h_start : h_start + crop_h, w_start : w_start + crop_w, :]  # (2028, 2028, 3)
    
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for one image using PIL.

    Resizes an image to a target height and width without distortion by padding with zeros.
    Note that PIL uses [width, height] ordering instead of [height, width].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return np.array(zero_image)


def load_episode_data(json_path: str) -> dict:
    """Load episode data from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def _resolve_video_ref(episode_dir: Path, ref: object) -> tuple[Path, int]:
    if not isinstance(ref, dict):
        raise ValueError(f"Expected video frame reference dict, got {type(ref).__name__}")
    video_path = ref.get("video_path")
    frame_index = ref.get("frame_index")
    if video_path is None or frame_index is None:
        raise ValueError(f"Invalid video frame reference: {ref}")
    return episode_dir / str(video_path), int(frame_index)


def _has_video_ref(episode_dir: Path, frame: dict, camera_key: str) -> bool:
    try:
        video_path, _ = _resolve_video_ref(episode_dir, frame[camera_key])
    except (KeyError, ValueError):
        return False
    return video_path.exists()


class EpisodeVideoReader:
    """Read RGB frames from an episode's per-camera videos."""

    def __init__(self, episode_dir: Path) -> None:
        self.episode_dir = episode_dir
        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._positions: dict[Path, int] = {}

    def read_rgb(self, frame: dict, camera_key: str) -> np.ndarray:
        video_path, frame_index = _resolve_video_ref(self.episode_dir, frame[camera_key])
        if not video_path.exists():
            raise ValueError(f"Missing video file for {camera_key}: {video_path}")

        cap = self._captures.get(video_path)
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            self._captures[video_path] = cap
            self._positions[video_path] = 0

        if self._positions.get(video_path) != frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise ValueError(f"Failed to read frame {frame_index} from {video_path}")

        self._positions[video_path] = frame_index + 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()
        self._positions.clear()


def get_video_frame_shape(data_dir: Path, camera_key: str = "rgb_left") -> tuple[int, int, int]:
    """Get shape by inspecting the first valid frame in a camera video.

    Args:
        data_dir: Path to a dataset directory containing episode_* subdirectories.
        camera_key: Camera key to inspect (default: "rgb_left").

    Returns:
        Tuple of (height, width, channels).
    """
    episode_dirs = sorted(data_dir.glob("episode_*"))
    if not episode_dirs:
        raise ValueError(f"No episode directories found in {data_dir}")

    for episode_dir in episode_dirs:
        json_path = episode_dir / "data.json"
        if not json_path.exists():
            continue
        episode_data = load_episode_data(str(json_path))
        for frame in episode_data.get("data", []):
            if camera_key not in frame:
                continue
            reader = EpisodeVideoReader(episode_dir)
            try:
                return reader.read_rgb(frame, camera_key).shape
            finally:
                reader.close()

    raise ValueError(f"Could not find any valid video frames for '{camera_key}' under {data_dir}")


def collect_all_episode_dirs(
    parent_dir: Path,
    dataset_folders: list[str],
    episode_sample_ratio: float = 1.0,
    max_episodes_per_folder: int | None = None,
    seed: int = 42,
    first_k_per_folder: list[int] | None = None,
) -> list[tuple[Path, str]]:
    """Collect all (or a subset of) episode directories from multiple dataset folders.

    Two selection modes are available, controlled by ``first_k_per_folder``:

    **Random-sampling mode** (``first_k_per_folder`` is None, default):
        Each folder is shuffled with a per-folder RNG seeded by ``seed`` and the folder
        name, then the first ``n_select`` episodes are taken.  This guarantees that the
        episodes selected at a lower ratio are always a *subset* of those selected at a
        higher ratio (given the same seed).

    **Deterministic first-k mode** (``first_k_per_folder`` is provided):
        For each folder, the sorted episode directories are taken in order and only the
        first k are kept, where k is the corresponding element of ``first_k_per_folder``.
        No shuffling is performed.  ``episode_sample_ratio`` and
        ``max_episodes_per_folder`` are ignored in this mode.

    Args:
        parent_dir: Root directory containing all dataset folders.
        dataset_folders: List of dataset folder names (subdirectories of parent_dir).
        episode_sample_ratio: Fraction of episodes to randomly select from each folder.
            Must be in (0, 1]. 1.0 means use all episodes (default). Ignored when
            ``first_k_per_folder`` is provided.
        max_episodes_per_folder: Maximum number of episodes to use from each folder.
            If None (default), no cap is applied. Ignored when ``first_k_per_folder``
            is provided.
        seed: Integer seed used to make episode sampling deterministic and reproducible.
            The same seed guarantees that episodes sampled at a smaller ratio are always
            a subset of those sampled at a larger ratio. Ignored when
            ``first_k_per_folder`` is provided.
        first_k_per_folder: If provided, must be a list of the same length as
            ``dataset_folders``. Each element specifies how many episodes to take from
            the front of the corresponding folder (sorted by name). Overrides
            ``episode_sample_ratio`` and ``max_episodes_per_folder``.

    Returns:
        List of (episode_dir, dataset_name) tuples sorted by dataset then episode name.
    """
    all_episodes: list[tuple[Path, str]] = []
    for i, folder_name in enumerate(dataset_folders):
        dataset_path = parent_dir / folder_name
        if not dataset_path.exists():
            print(f"Warning: dataset folder '{dataset_path}' does not exist, skipping...")
            continue
        episode_dirs = sorted(dataset_path.glob("episode_*"))
        if not episode_dirs:
            print(f"Warning: no episode_* directories found in '{dataset_path}', skipping...")
            continue

        total = len(episode_dirs)

        if first_k_per_folder is not None:
            # Deterministic first-k mode: take the first k sorted episodes, no shuffling.
            k = first_k_per_folder[i]
            episode_dirs = episode_dirs[:k]
            n_selected = len(episode_dirs)
            print(f"  Found {total} episodes in '{folder_name}', taking first {n_selected} (deterministic)")
        else:
            # Random-sampling mode: apply ratio and/or absolute cap, then shuffle+slice.
            n_select = total
            if episode_sample_ratio < 1.0:
                n_select = min(n_select, max(1, round(total * episode_sample_ratio)))
            if max_episodes_per_folder is not None:
                n_select = min(n_select, max_episodes_per_folder)
            # Clamp to at least 1 to avoid empty selections.
            n_select = max(1, n_select)

            if n_select < total:
                # Use a per-folder RNG so that folders don't interfere with each other's
                # sampling order.  Shuffling then slicing guarantees subset nesting: the
                # episodes picked at ratio r1 < r2 are always a subset of those at ratio r2
                # when the same seed is used, because both runs share the same shuffle order
                # and only differ in how many leading elements they keep.
                rng = random.Random(f"{seed}:{folder_name}")
                shuffled = list(episode_dirs)
                rng.shuffle(shuffled)
                episode_dirs = sorted(shuffled[:n_select])
                print(f"  Found {total} episodes in '{folder_name}', sampled {n_select} (seed={seed})")
            else:
                print(f"  Found {total} episodes in '{folder_name}'")

        for ep_dir in episode_dirs:
            all_episodes.append((ep_dir, folder_name))
    return all_episodes


def print_episode_summary(all_episodes: list[tuple[Path, str]]) -> None:
    """Print a summary of episode lengths for all episodes."""
    print("\nEpisode lengths:")
    for episode_dir, dataset_name in all_episodes:
        json_path = episode_dir / "data.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    episode_data = json.load(f)
                episode_length = len(episode_data.get("data", []))
                print(f"  [{dataset_name}] {episode_dir.name}: {episode_length} frames")
            except Exception as e:
                print(f"  [{dataset_name}] {episode_dir.name}: error reading data.json - {e}")
        else:
            print(f"  [{dataset_name}] {episode_dir.name}: data.json not found")
    print()


def _load_frame_images(
    frame: dict,
    video_reader: EpisodeVideoReader,
    target_height: int,
    target_width: int,
    has_head_camera: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load and resize all video-backed images for a single frame.

    Returns (head_image_left, left_wrist_image, right_wrist_image), or None if
    required wrist video frames are missing.
    """
    try:
        left_wrist = resize_with_pad(video_reader.read_rgb(frame, "wrist_rgb_left"), target_height, target_width)
        right_wrist = resize_with_pad(video_reader.read_rgb(frame, "wrist_rgb_right"), target_height, target_width)
    except (KeyError, ValueError) as exc:
        print(f"Warning: failed to load wrist video frames for frame {frame.get('idx')}: {exc}")
        return None

    if has_head_camera:
        try:
            head_img = resize_with_pad(video_reader.read_rgb(frame, "rgb_left"), target_height, target_width)
        except (KeyError, ValueError):
            head_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    else:
        head_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    return head_img, left_wrist, right_wrist


def process_episode(
    dataset: LeRobotDataset,
    episode_dir: Path,
    target_height: int,
    target_width: int,
) -> tuple[bool, str | None]:
    """Process a single episode and add all its frames to the dataset.

    Video frames are decoded in episode order, then resized before being added
    to the dataset. dataset.add_frame() is called sequentially to preserve
    episode ordering.

    Args:
        dataset: The LeRobotDataset instance to add frames to.
        episode_dir: Path to the episode directory.
        target_height: Target image height after resizing.
        target_width: Target image width after resizing.

    Returns:
        (success, task) where success is True if the episode was processed
        successfully and task is the instruction string (or None on failure).
    """
    json_path = episode_dir / "data.json"
    if not json_path.exists():
        print(f"Warning: {json_path} does not exist, skipping...")
        return False, None

    episode_data = load_episode_data(str(json_path))
    data = episode_data["data"]

    task = episode_data.get("text", {}).get("goal", "perform the task")

    # Detect if this episode has a head camera by checking the first frame.
    has_head_camera = (
        bool(data)
        and _has_video_ref(episode_dir, data[0], "rgb_left")
    )

    video_reader = EpisodeVideoReader(episode_dir)
    try:
        for frame in data:
            images = _load_frame_images(frame, video_reader, target_height, target_width, has_head_camera)
            if images is None:
                print(f"Warning: missing wrist video frames for frame {frame['idx']} in {episode_dir.name}, skipping frame...")
                continue

            head_image_left, left_wrist_image, right_wrist_image = images

            # state_body layout (32 dims):
            #   [0:3]   root:      roll, pitch, yaw_angular_velocity
            #   [3:9]   leg_left:  6 joints
            #   [9:15]  leg_right: 6 joints
            #   [15:18] waist:     3 joints
            #   [18:25] arm_left:  7 joints
            #   [25:32] arm_right: 7 joints
            state_body = np.array(frame["state_body"], dtype=np.float32)
            state_root = state_body[0:3]
            state_leg_left = state_body[3:9]
            state_leg_right = state_body[9:15]
            state_waist = state_body[15:18]
            state_arm_left = state_body[18:25]
            state_arm_right = state_body[25:32]
            state_hand_left = np.array([frame["state_hand_left"]], dtype=np.float32)
            state_hand_right = np.array([frame["state_hand_right"]], dtype=np.float32)

            # OpenPI state format (34 dims):
            #   arm_left(7), hand_left(1), arm_right(7), hand_right(1),
            #   leg_left(6), leg_right(6), waist(3), root(3)
            state = np.concatenate([
                state_arm_left, state_hand_left,
                state_arm_right, state_hand_right,
                state_leg_left, state_leg_right,
                state_waist, state_root,
            ])

            # action_body layout mirrors state_body (32 dims)
            action_body = np.array(frame["action_body"], dtype=np.float32)
            action_root = action_body[0:3]
            action_leg_left = action_body[3:9]
            action_leg_right = action_body[9:15]
            action_waist = action_body[15:18]
            action_arm_left = action_body[18:25]
            action_arm_right = action_body[25:32]
            action_hand_left = np.array([frame["action_hand_left"]], dtype=np.float32)
            action_hand_right = np.array([frame["action_hand_right"]], dtype=np.float32)

            # OpenPI action format (34 dims):
            #   arm_left(7), hand_left(1), arm_right(7), hand_right(1),
            #   leg_left(6), leg_right(6), waist(3), root(3)
            actions = np.concatenate([
                action_arm_left, action_hand_left,
                action_arm_right, action_hand_right,
                action_leg_left, action_leg_right,
                action_waist, action_root,
            ])

            # The inputs are decoded RGB arrays; LeRobot still stores them as
            # feature dtype "image".
            dataset.add_frame(
                {
                    "head_image_left": head_image_left,
                    "left_wrist_image": left_wrist_image,
                    "right_wrist_image": right_wrist_image,
                    "state": state,
                    "actions": actions,
                    "task": task,
                }
            )
    finally:
        video_reader.close()

    dataset.save_episode()
    return True, task


def main(
    parent_dir: str,
    dataset_folders: list[str],
    *,
    repo_name: str = REPO_NAME,
    push_to_hub: bool = False,
    fps: int = 30,
    episode_sample_ratio: float = 1.0,
    max_episodes_per_folder: int | None = None,
    first_k_episodes_per_folder: list[int] | None = None,
    extra_dataset_folders: list[str] | None = None,
    humi_dataset_folders: list[str] | None = None,
    humi_max_episodes_per_folder: int | None = None,
    seed: int = 42,
) -> None:
    """
    Convert multiple Unitree G1 datasets to LeRobot format.

    Args:
        parent_dir: Parent directory containing all dataset folders.
        dataset_folders: List of dataset folder names inside parent_dir.
        repo_name: Name of the output LeRobot dataset (also used as the HuggingFace Hub repo id).
        push_to_hub: Whether to push the resulting dataset to HuggingFace Hub.
        fps: Frames per second of the dataset (default: 30).
        episode_sample_ratio: Fraction of episodes to randomly select from each dataset folder.
            Must be in (0, 1]. 1.0 uses all episodes. For example, 0.5 keeps a random 50% of
            episodes from each folder. Ignored when first_k_episodes_per_folder is set.
        max_episodes_per_folder: Maximum number of episodes to use from each dataset folder.
            Defaults to None (use all episodes). If set, at most this many episodes are randomly
            sampled from each folder. When combined with episode_sample_ratio, the stricter
            limit applies. Ignored when first_k_episodes_per_folder is set.
        first_k_episodes_per_folder: If provided, must be a list with the same length as
            dataset_folders. Each element specifies how many episodes to take from the *front*
            of the corresponding folder (sorted by episode name) without any random sampling.
            For example, [36, 80, 80] takes the first 36 episodes from the first folder and
            the first 80 episodes from each of the second and third folders. When this option
            is set, episode_sample_ratio and max_episodes_per_folder are ignored for
            dataset_folders.
        extra_dataset_folders: Optional list of additional dataset folder names inside parent_dir.
            From each of these folders, exactly ONE episode is randomly sampled and added to the
            output dataset. These are processed independently of dataset_folders and are not
            affected by episode_sample_ratio or max_episodes_per_folder.
        humi_dataset_folders: Optional list of dataset folder names inside parent_dir that should
            be sampled with a dedicated per-folder cap (humi_max_episodes_per_folder). These folders
            are processed independently of dataset_folders and extra_dataset_folders, and are not
            affected by episode_sample_ratio or max_episodes_per_folder.
        humi_max_episodes_per_folder: Maximum number of episodes to randomly sample from each
            folder listed in humi_dataset_folders. Defaults to None (use all episodes in those
            folders). Has no effect when humi_dataset_folders is not provided.
        seed: Random seed for deterministic episode sampling. Using the same seed guarantees
            that episodes sampled at a smaller ratio are always a subset of those sampled at a
            larger ratio. For example, with seed=42, ratio=0.25 will always pick a subset of
            the episodes that ratio=0.5 would pick. Has no effect when first_k_episodes_per_folder
            is set (for dataset_folders).
    """
    # import debugpy
    # debugpy.listen(("127.0.0.1", 5678))
    # print("Debug server started on 127.0.0.1:5678")
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, continuing execution...")

    if not dataset_folders:
        raise ValueError("--dataset_folders must contain at least one folder name.")

    if not (0 < episode_sample_ratio <= 1.0):
        raise ValueError(f"--episode_sample_ratio must be in (0, 1], got {episode_sample_ratio}")

    if max_episodes_per_folder is not None and max_episodes_per_folder < 1:
        raise ValueError(f"--max_episodes_per_folder must be >= 1, got {max_episodes_per_folder}")

    if humi_max_episodes_per_folder is not None and humi_max_episodes_per_folder < 1:
        raise ValueError(f"--humi_max_episodes_per_folder must be >= 1, got {humi_max_episodes_per_folder}")

    if first_k_episodes_per_folder is not None:
        if len(first_k_episodes_per_folder) != len(dataset_folders):
            raise ValueError(
                f"--first_k_episodes_per_folder must have the same length as --dataset_folders "
                f"(got {len(first_k_episodes_per_folder)} vs {len(dataset_folders)})"
            )
        if any(k < 1 for k in first_k_episodes_per_folder):
            raise ValueError(f"All values in --first_k_episodes_per_folder must be >= 1, got {first_k_episodes_per_folder}")

    parent_path = Path(parent_dir)
    if not parent_path.exists():
        raise ValueError(f"parent_dir does not exist: {parent_path}")

    print(f"Parent directory         : {parent_path}")
    print(f"Dataset folders          : {dataset_folders}")
    if first_k_episodes_per_folder is not None:
        print(f"First-k episodes/folder  : {first_k_episodes_per_folder} (deterministic, ignores ratio/max)")
    else:
        print(f"Episode sample ratio     : {episode_sample_ratio:.1%}")
        print(f"Sampling seed            : {seed}")
        if max_episodes_per_folder is not None:
            print(f"Max episodes per folder  : {max_episodes_per_folder}")
        else:
            print(f"Max episodes per folder  : (all)")
    if extra_dataset_folders:
        print(f"Extra dataset folders    : {extra_dataset_folders} (1 random episode each)")
    if humi_dataset_folders:
        cap_str = str(humi_max_episodes_per_folder) if humi_max_episodes_per_folder is not None else "(all)"
        print(f"Humi dataset folders     : {humi_dataset_folders} ({cap_str} random episodes each)")

    # Collect all episodes across all datasets
    print("\nScanning dataset folders...")
    all_episodes = collect_all_episode_dirs(
        parent_path,
        dataset_folders,
        episode_sample_ratio,
        max_episodes_per_folder,
        seed,
        first_k_per_folder=first_k_episodes_per_folder,
    )

    # Additionally, sample exactly one random episode from each extra dataset folder.
    if extra_dataset_folders:
        print("\nScanning extra dataset folders (sampling 1 episode per folder)...")
        extra_episodes = collect_all_episode_dirs(
            parent_path,
            extra_dataset_folders,
            episode_sample_ratio=1.0,
            max_episodes_per_folder=1,
            seed=seed,
        )
        all_episodes.extend(extra_episodes)

    # Additionally, sample humi_max_episodes_per_folder random episodes from each humi dataset folder.
    if humi_dataset_folders:
        cap_str = str(humi_max_episodes_per_folder) if humi_max_episodes_per_folder is not None else "all"
        print(f"\nScanning humi dataset folders (sampling {cap_str} episodes per folder)...")
        humi_episodes = collect_all_episode_dirs(
            parent_path,
            humi_dataset_folders,
            episode_sample_ratio=1.0,
            max_episodes_per_folder=humi_max_episodes_per_folder,
            seed=seed,
        )
        all_episodes.extend(humi_episodes)

    if not all_episodes:
        raise ValueError("No valid episodes found across all specified dataset folders.")

    print(f"\nTotal episodes collected: {len(all_episodes)}")

    print_episode_summary(all_episodes)

    # Print video frame shapes for each dataset folder that actually contributed episodes
    # (this includes extra folders). Preserve insertion order and remove duplicates.
    contributing_folders = list(dict.fromkeys(name for _, name in all_episodes))
    target_height, target_width = 224, 224
    print("\nVideo frame shapes per dataset:")
    for folder in contributing_folders:
        dataset_path = parent_path / folder
        print(f"  [{folder}]")
        try:
            h_h, h_w, h_c = get_video_frame_shape(dataset_path, "rgb_left")
            print(f"    rgb_left        : {h_w}x{h_h}x{h_c}")
        except ValueError:
            print(f"    rgb_left        : (not present, black placeholder will be used)")
        lw_h, lw_w, lw_c = get_video_frame_shape(dataset_path, "wrist_rgb_left")
        rw_h, rw_w, rw_c = get_video_frame_shape(dataset_path, "wrist_rgb_right")
        print(f"    wrist_rgb_left  : {lw_w}x{lw_h}x{lw_c}")
        print(f"    wrist_rgb_right : {rw_w}x{rw_h}x{rw_c}")

    num_channels = 3  # all cameras produce RGB images
    head_c = lw_c = rw_c = num_channels
    print(f"\nAll images will be resized to   : {target_width}x{target_height}")
    print(f"Cropping 2704x2028 images to 2028x2028 before resizing.")
    print()

    # State/action dimensions (OpenPI format, 34 dims total):
    #   0-6:   left arm (7)
    #   7:     left gripper (1)
    #   8-14:  right arm (7)
    #   15:    right gripper (1)
    #   16-21: left leg (6)
    #   22-27: right leg (6)
    #   28-30: waist (3)
    #   31-33: root (3) -- roll, pitch, yaw angular velocity
    state_action_dim = 34

    features = {
        "head_image_left": {
            "dtype": "image",
            "shape": (target_height, target_width, head_c),
            "names": ["height", "width", "channel"],
        },
        "left_wrist_image": {
            "dtype": "image",
            "shape": (target_height, target_width, lw_c),
            "names": ["height", "width", "channel"],
        },
        "right_wrist_image": {
            "dtype": "image",
            "shape": (target_height, target_width, rw_c),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_action_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (state_action_dim,),
            "names": ["actions"],
        },
    }

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset: {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="unitree_g1",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    success_count = 0
    unique_tasks: set[str] = set()
    for episode_dir, dataset_name in tqdm(all_episodes, desc="Converting episodes"):
        try:
            ok, task = process_episode(dataset, episode_dir, target_height, target_width)
            if ok:
                success_count += 1
                if task is not None:
                    unique_tasks.add(task)
        except Exception as e:
            print(f"Error processing [{dataset_name}] {episode_dir.name}: {e}")
            continue

    print(f"\nDataset saved to  : {output_path}")
    print(f"Episodes converted: {success_count} / {len(all_episodes)}")
    print(f"\nUnique task instructions ({len(unique_tasks)}):")
    for i, t in enumerate(sorted(unique_tasks), 1):
        print(f"  {i}. {t}")
    print(f"Total episodes    : {dataset.meta.total_episodes}")
    print(f"Total frames      : {dataset.meta.total_frames}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["unitree_g1", "bimanual", "humanoid"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Dataset pushed to HuggingFace Hub: {repo_name}")


if __name__ == "__main__":
    tyro.cli(main)
