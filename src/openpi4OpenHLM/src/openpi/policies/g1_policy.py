"""
Policy transforms for Unitree G1 humanoid robot.

This module provides input and output transforms for the Unitree G1 robot dataset.
The G1 robot uses 34-dimensional state and action spaces (OpenPI format):
  0-6:   left arm (7)
  7:     left gripper (1)
  8-14:  right arm (7)
  15:    right gripper (1)
  16-21: left leg (6)
  22-27: right leg (6)
  28-30: waist (3)
  31-33: root (3) -- roll, pitch, yaw angular velocity

Up to three camera views are used: one optional head camera and two wrist cameras.
When a dataset has no head camera, head_image_left is stored as a black placeholder.
G1Inputs detects this by checking whether the image is all zeros and masks it out accordingly.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_g1_example() -> dict:
    """Creates a random input example for the G1 policy.

    Generates a 34-dimensional state with three camera views (head and wrist cameras).
    """
    return {
        "observation/head_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(34).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """
    Transform inputs to the model format for Unitree G1 robot.

    This class converts inputs from the dataset format to the format expected by the model.
    It handles image parsing and state extraction for both training and inference.

    Expected inputs:
    - observation/head_image_left: Head-mounted camera image [height, width, channel] or [channel, height, width].
      For 2-camera datasets this is stored as an all-zero (black) placeholder; it will be masked out automatically.
    - observation/left_wrist_image: Left wrist camera image
    - observation/right_wrist_image: Right wrist camera image
    - observation/state: Robot state [34]
    - actions: Action sequence [action_horizon, 34] (only during training)
    - prompt: Language instruction string

    State and actions are 34-dimensional (OpenPI format):
      0-6: left arm, 7: left gripper, 8-14: right arm, 15: right gripper,
      16-21: left leg, 22-27: right leg, 28-30: waist, 31-33: root (roll/pitch/yaw_vel).
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H, W, C) format since LeRobot automatically
        # stores as float32 (C, H, W), gets skipped for policy inference.
        base_image = _parse_image(data["observation/head_image_left"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        # A black (all-zero) head image indicates a 2-camera episode with no head camera.
        # Mask it out so the model ignores the placeholder entirely.
        has_head_camera = np.any(base_image > 0)

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.bool_(has_head_camera),
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # Pass the prompt (language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Forward last_action from previous inference chunk if provided.
        # Used by AbsoluteActions(use_first_action=True) to recover absolute actions.
        if "observation/last_action" in data:
            inputs["last_action"] = np.asarray(data["observation/last_action"])

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """
    Transform outputs from the model back to the dataset format for Unitree G1 robot.

    This class is used for inference only. It extracts the relevant action dimensions
    from the model output, since the model may output padded actions.

    The G1 robot has 34-dimensional actions (OpenPI format).
    """

    def __call__(self, data: dict) -> dict:
        # Return the first 34 action dimensions since the model may output padded actions.
        return {"actions": np.asarray(data["actions"][:, :34])}
