"""
Script for converting Unitree G1 dataset to LeRobot format.

Usage:
uv run examples/unitree_g1/convert_g1_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/unitree_g1/convert_g1_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

REPO_NAME = "Yingdong-Hu/g1_walk_forward"  # Name of the output dataset


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)  # No need to resize if the image is already the correct size.

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
    """Get frame shape by reading the first valid frame from a camera video.
    
    Args:
        data_dir: Path to the dataset directory.
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

    raise ValueError(f"Could not find any valid video frames for {camera_key} to determine shape")


def _load_frame_images(
    frame: dict,
    video_reader: EpisodeVideoReader,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load and resize all video-backed images for a single frame.

    Returns (head_image_left, left_wrist_image, right_wrist_image), or None if
    required video frames are missing.
    """
    try:
        head_img = resize_with_pad(video_reader.read_rgb(frame, "rgb_left"), target_height, target_width)
        left_wrist = resize_with_pad(video_reader.read_rgb(frame, "wrist_rgb_left"), target_height, target_width)
        right_wrist = resize_with_pad(video_reader.read_rgb(frame, "wrist_rgb_right"), target_height, target_width)
    except (KeyError, ValueError) as exc:
        print(f"Warning: failed to load video-backed images for frame {frame.get('idx')}: {exc}")
        return None

    return head_img, left_wrist, right_wrist


def main(
    data_dir: str,
    *,
    repo_name: str = REPO_NAME,
    push_to_hub: bool = False,
    fps: int = 30,
    num_episodes: int | None = None,
):
    """
    Convert Unitree G1 dataset to LeRobot format.

    Args:
        data_dir: Path to the dataset directory containing episode folders.
        repo_name: Name of the output dataset (also used for HuggingFace Hub).
        push_to_hub: Whether to push the dataset to HuggingFace Hub.
        fps: Frames per second of the dataset (default: 30).
        num_episodes: If set, only use the first N episodes (e.g. 36 means episode_0000
            through episode_0035). If None, all episodes are used.
    """
    # import debugpy
    # debugpy.listen(("127.0.0.1", 5678))
    # print("Debug server started on 127.0.0.1:5678")
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, continuing execution...")

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset: {output_path}")
        shutil.rmtree(output_path)

    data_path = Path(data_dir)
    
    # Print episode lengths from data.json files
    episode_dirs_preview = sorted(data_path.glob("episode_*"))
    if num_episodes is not None:
        episode_dirs_preview = episode_dirs_preview[:num_episodes]
    print(f"\nEpisode lengths:")
    for episode_dir in episode_dirs_preview:
        json_path = episode_dir / "data.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    episode_data = json.load(f)
                    episode_length = len(episode_data.get("data", []))
                    print(f"  {episode_dir.name}: {episode_length} frames")
            except Exception as e:
                print(f"  {episode_dir.name}: Error reading file - {e}")
        else:
            print(f"  {episode_dir.name}: data.json not found")
    print()

    # Get frame shapes from the first episode videos
    # All images will be resized to 224x224
    target_height, target_width = 224, 224

    head_image_shape = get_video_frame_shape(data_path, "rgb_left")
    left_wrist_image_shape = get_video_frame_shape(data_path, "wrist_rgb_left")
    right_wrist_image_shape = get_video_frame_shape(data_path, "wrist_rgb_right")

    head_height, head_width, head_channels = head_image_shape
    left_wrist_height, left_wrist_width, left_wrist_channels = left_wrist_image_shape
    right_wrist_height, right_wrist_width, right_wrist_channels = right_wrist_image_shape

    print(f"Detected head video frame shape: {head_width}x{head_height}x{head_channels}")
    print(f"Detected left wrist video frame shape: {left_wrist_width}x{left_wrist_height}x{left_wrist_channels}")
    print(f"Detected right wrist video frame shape: {right_wrist_width}x{right_wrist_height}x{right_wrist_channels}")
    print(f"All images will be resized to {target_width}x{target_height}")

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

    # Create LeRobot dataset with features
    features = {
        "head_image_left": {
            "dtype": "image",
            "shape": (target_height, target_width, head_channels),
            "names": ["height", "width", "channel"],
        },
        "left_wrist_image": {
            "dtype": "image",
            "shape": (target_height, target_width, left_wrist_channels),
            "names": ["height", "width", "channel"],
        },
        "right_wrist_image": {
            "dtype": "image",
            "shape": (target_height, target_width, right_wrist_channels),
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
    
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="unitree_g1",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Find all episode directories
    episode_dirs = sorted(data_path.glob("episode_*"))
    if num_episodes is not None:
        episode_dirs = episode_dirs[:num_episodes]
    print(f"Found {len(episode_dirs)} episodes for conversion")

    # Process each episode
    for episode_dir in tqdm(episode_dirs, desc="Converting episodes"):
        json_path = episode_dir / "data.json"
        if not json_path.exists():
            print(f"Warning: {json_path} does not exist, skipping...")
            continue

        try:
            # Load episode data
            episode_data = load_episode_data(str(json_path))
            data = episode_data["data"]

            # Get language instruction (task description)
            task = episode_data.get("text", {}).get("goal", "perform the task")
            # task = 'Pick up the purple soft finger on the table and place it on the mouse pad.'

            video_reader = EpisodeVideoReader(episode_dir)
            try:
                for frame in data:
                    images = _load_frame_images(frame, video_reader, target_height, target_width)
                    if images is None:
                        print(f"Warning: Missing video frames for frame {frame['idx']} in {episode_dir.name}, skipping frame...")
                        continue

                    head_image_left, left_wrist_image, right_wrist_image = images

                    # Get state
                    # state_body layout (32 dims):
                    #   [0:3]   root:      roll, pitch, yaw_angular_velocity
                    #   [3:9]   leg_left:  6 joints
                    #   [9:15]  leg_right: 6 joints
                    #   [15:18] waist:     3 joints
                    #   [18:25] arm_left:  7 joints
                    #   [25:32] arm_right: 7 joints
                    state_body = np.array(frame["state_body"], dtype=np.float32)
                    state_root = state_body[0:3]    # [roll, pitch, yaw_vel]
                    state_leg_left = state_body[3:9]
                    state_leg_right = state_body[9:15]
                    state_waist = state_body[15:18]
                    state_arm_left = state_body[18:25]
                    state_arm_right = state_body[25:32]
                    # state_hand_left/right are scalars in the JSON
                    state_hand_left = np.array([frame["state_hand_left"]], dtype=np.float32)
                    state_hand_right = np.array([frame["state_hand_right"]], dtype=np.float32)

                    # Assemble OpenPI state format (34 dims):
                    #   arm_left(7), hand_left(1), arm_right(7), hand_right(1),
                    #   leg_left(6), leg_right(6), waist(3), root(3)
                    state = np.concatenate([
                        state_arm_left,
                        state_hand_left,
                        state_arm_right,
                        state_hand_right,
                        state_leg_left,
                        state_leg_right,
                        state_waist,
                        state_root,
                    ])

                    # Get action
                    # action_body layout (32 dims) mirrors state_body:
                    #   [0:3]   root:      roll, pitch, yaw_angular_velocity
                    #   [3:9]   leg_left:  6 joints
                    #   [9:15]  leg_right: 6 joints
                    #   [15:18] waist:     3 joints
                    #   [18:25] arm_left:  7 joints
                    #   [25:32] arm_right: 7 joints
                    action_body = np.array(frame["action_body"], dtype=np.float32)
                    action_root = action_body[0:3]    # [roll, pitch, yaw_vel]
                    action_leg_left = action_body[3:9]
                    action_leg_right = action_body[9:15]
                    action_waist = action_body[15:18]
                    action_arm_left = action_body[18:25]
                    action_arm_right = action_body[25:32]
                    # action_hand_left/right are scalars in the JSON
                    action_hand_left = np.array([frame["action_hand_left"]], dtype=np.float32)
                    action_hand_right = np.array([frame["action_hand_right"]], dtype=np.float32)

                    # Assemble OpenPI action format (34 dims):
                    #   arm_left(7), hand_left(1), arm_right(7), hand_right(1),
                    #   leg_left(6), leg_right(6), waist(3), root(3)
                    actions = np.concatenate([
                        action_arm_left,
                        action_hand_left,
                        action_arm_right,
                        action_hand_right,
                        action_leg_left,
                        action_leg_right,
                        action_waist,
                        action_root,
                    ])

                    # Add frame to dataset. These are decoded RGB arrays; the
                    # LeRobot feature dtype stays "image".
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

            # Save the episode
            dataset.save_episode()

        except Exception as e:
            print(f"Error processing {episode_dir.name}: {e}")
            continue

    print(f"\nDataset saved to: {output_path}")
    print(f"Total episodes: {dataset.meta.total_episodes}")
    print(f"Total frames: {dataset.meta.total_frames}")

    # Optionally push to HuggingFace Hub
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
