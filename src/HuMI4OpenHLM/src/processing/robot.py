import pathlib
from functools import cache

import mujoco
from loguru import logger


@cache
def get_recording_joint_idx_to_poselib(
    path: pathlib.Path,
) -> dict[int, int]:
    """Extract joint ordering mapping from recording json files to PoseLib expected format.
    Args:
        path: Path to MJCF XML file.
    Returns:
        mapping: Dictionary mapping recording joint indices to PoseLib expected joint indices.

    """
    if not path.exists():
        # Fallback to local project directory if the absolute path is from another machine
        parts = path.parts
        project_root = pathlib.Path(__file__).resolve().parents[2]

        # Try to match from "g1_description" onwards
        if "g1_description" in parts:
            idx = parts.index("g1_description")
            rel_path = pathlib.Path(*parts[idx:])
            new_path = project_root / rel_path
            if new_path.exists():
                logger.info(
                    f"MJCF path {path} not found. Using fallback {new_path}"
                )
                path = new_path

        # If still not found, try just the filename in g1_description
        if not path.exists():
            fallback_path = project_root / "g1_description" / path.name
            if fallback_path.exists():
                logger.info(
                    f"MJCF path {path} not found. Using fallback {fallback_path}"
                )
                path = fallback_path

    model = mujoco.MjModel.from_xml_path(path.as_posix())
    joint_order: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        joint_order.append(joint_name)
    mapping: dict[int, int] = {}
    unexpected: list[str] = []
    for actual_idx, joint_name in enumerate(joint_order):
        expected_idx = EXPECTED_ACTUATED_INDEX.get(joint_name)
        if expected_idx is None:
            unexpected.append(joint_name)
            continue
        mapping[actual_idx] = expected_idx
    missing_expected = [
        joint_name
        for joint_name in EXPECTED_ACTUATED_JOINTS
        if joint_name not in joint_order
    ]
    if unexpected:
        logger.warning(
            f"Joints present in {path.as_posix()} but not in expected schema: {', '.join(unexpected)}"
        )
    if missing_expected:
        logger.warning(
            f"Expected joints missing from {path.as_posix()}: {', '.join(missing_expected)}, use default zero values."
        )
    return mapping


EXPECTED_ACTUATED_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
EXPECTED_ACTUATED_INDEX = {
    name: idx for idx, name in enumerate(EXPECTED_ACTUATED_JOINTS)
}

__all__ = [
    "EXPECTED_ACTUATED_JOINTS",
    "EXPECTED_ACTUATED_INDEX",
    "get_recording_joint_idx_to_poselib",
]
