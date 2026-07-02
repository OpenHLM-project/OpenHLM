from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class IKConfig:
    """Parameters that configure Mink IK optimization settings and task weights."""

    mjcf_path: Path = Path("g1_description/g1_29dof_rev_1_0.xml")
    """Path to the MJCF file of G1 robot model to use for IK."""

    dt: float = 0.02
    """Outer integration time step (seconds) used per IK solve loop."""
    damping: float = 1e-6
    """Global Levenberg-Marquardt damping applied inside Mink's solver."""
    inner_steps: int = 10
    """Number of mink.solve_ik substeps performed per outer solve invocation."""
    solver: Literal["daqp"] = "daqp"
    """QP backend identifier passed to Mink (currently only `daqp` supported)."""

    # Per-task gains and LM damping
    hand_gain: float = 1.0
    """Task gain for hand frame targets."""
    hand_damping: float = 1e-12
    """LM damping for the hand tasks."""
    hand_position_cost: float = 20.0
    """Cost weight applied to hand position errors."""
    hand_orientation_cost: float = 20.0
    """Cost weight applied to hand orientation errors."""

    foot_gain: float = 0.5
    """Task gain for the ankle/foot frames."""
    foot_damping: float = 1e-12
    """LM damping for the foot tasks."""
    foot_position_cost: float = 30.0
    """Cost weight applied to foot translation errors."""
    foot_orientation_cost: float = 3.0
    """Cost weight applied to foot orientation errors."""
    suppress_foot_pitch: bool = False
    """If True, ensure foot parallelism with ground by suppressing foot pitch."""

    pelvis_gain: float = 0.5
    """Task gain for the pelvis frame when tracked."""
    pelvis_damping: float = 1e-12
    """LM damping for the pelvis task."""
    pelvis_xy_cost: float = 8.0
    """Cost weight applied to pelvis X/Y translation errors."""
    pelvis_z_cost: float = 8.0
    """Cost weight applied to pelvis Z translation errors."""
    pelvis_orientation_cost: float = 3.0
    """Cost weight applied to pelvis orientation errors."""
    torso_upright_gain: float = 0.5
    """Task gain for keeping `torso_link` upright."""
    torso_upright_damping: float = 1e-12
    """LM damping for the torso upright task."""
    torso_upright_orientation_cost: float = 0.0
    """Orientation cost for the torso upright task. Set > 0 to enable it."""

    posture_gain: float = 0.05
    """Task gain for the whole-body posture regularization."""
    posture_damping: float = 1e-12
    """LM damping for the posture task."""
    posture_cost: float = 1.0
    """Uniform cost weight per joint DoF in the posture task."""
    waist_posture_cost: float | None = None
    """Optional shared posture cost override for waist roll/pitch joints."""

    com_gain: float = 0.3
    """Task gain for COM regulation (hand/foot mode)."""
    com_damping: float = 1e-12
    """LM damping for the COM task."""
    com_xy_cost: float = 10.0
    """Cost weight applied to COM X/Y error."""
    com_z_cost: float = 10.0
    """Cost weight applied to COM Z error."""

    joint_limit_gain: float = 0.95
    """Gain factor used by configuration (joint) limits."""
    joint_limit_margin: float = 0.03
    """Minimum buffer maintained from joint limits (radians/meters)."""

    collision_gain: float = 0.85
    """Gain applied to collision avoidance constraints."""
    minimum_distance_from_collisions: float = 0.05
    """Minimum allowed distance between any two geoms."""
    collision_detection_distance: float = 0.01
    """Distance threshold for enabling collision checks."""
    disable_collision: bool = False
    """Disable collision avoidance constraints if True."""

    velocity_limit_gain: float = 0.3
    """Gain applied to velocity-limiting constraints."""
    max_joint_velocity: float = 2.0
    """Max joint-speed magnitude allowed for the lower-limb joints (rad/s)."""

    standing_pelvis_height: float = 0.763
    """Standing pelvis height above ground (meters). Used for reference transform computation."""
    standing_pelvis_pitch: float = 0.0
    """Standing pelvis pitch (radians) for stability. Used for reference transform computation."""
    foot_tracker_x_offset: float = 0.10
    """Translation offset along X from ankle joint to mounted tracker (meters)."""
    root_tracker_x_offset: float = 0.10
    """Translation offset along X from pelvis joint to mounted tracker (meters)."""
    use_root_tracker: bool = True
    """If False, ignore the pelvis/root tracker and synthesize pelvis targets from the feet."""
    home_keyframe: Literal["stand"] | None = None
    """Keyframe name to use for the home configuration. If None, use the mjcf default pose."""
    posture_keyframe: Literal["stand"] | None = "stand"
    """Keyframe name to use for the posture regularization task. If None, use the mjcf default pose."""

    def create(self):
        """Create an IKSolver instance using this configuration."""
        from .solver import IKSolver

        return IKSolver(config=self)

    def save_as_yaml(self, output_path: Path):
        """Save this IKConfig as a YAML file."""
        import yaml

        # Serialize fields
        mjcf_path_str = str(self.mjcf_path.absolute())
        data_dict = asdict(self)
        data_dict["mjcf_path"] = mjcf_path_str

        with output_path.open("w") as f:
            yaml.dump(data_dict, f)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "IKConfig":
        """Load an IKConfig from a YAML file."""
        import yaml

        with yaml_path.open("r") as f:
            data = yaml.safe_load(f)

        # Parse fields
        data["mjcf_path"] = Path(data["mjcf_path"])

        # Backward compatible for pelvis_position_cost
        if "pelvis_position_cost" in data:
            cost = data.pop("pelvis_position_cost")
            data["pelvis_xy_cost"] = cost
            data["pelvis_z_cost"] = cost

        if "mode" in data:
            data.pop("mode")

        return cls(**data)


_config_dict = {
    "proposal": IKConfig(
        inner_steps=20,
        standing_pelvis_height=0.763,
        home_keyframe="stand",
        posture_keyframe="stand",
        standing_pelvis_pitch=0.0,
        com_xy_cost=0.0,
        com_z_cost=0.0,
        hand_position_cost=30.0,
    ),
    "walk": IKConfig(
        inner_steps=30,
        standing_pelvis_height=0.793,
        posture_keyframe="stand",
        home_keyframe=None,
        root_tracker_x_offset=0.10,
        suppress_foot_pitch=True,
        com_xy_cost=0.0,
        com_z_cost=0.0,
        pelvis_xy_cost=20.0,
        pelvis_z_cost=20.0,
        pelvis_orientation_cost=10.0,
        hand_position_cost=80.0,
    ),
    "squat_pick_ground": IKConfig(
        inner_steps=20,
        standing_pelvis_height=0.763,
        posture_keyframe="stand",
        home_keyframe="stand",
        com_xy_cost=20.0,
        com_z_cost=0.0,
        hand_position_cost=50.0,
        posture_cost=1.0,
        standing_pelvis_pitch=0.1,
        suppress_foot_pitch=True,
    ),
    "pick-low": IKConfig(
        inner_steps=20,
        standing_pelvis_height=0.6,
        posture_keyframe="stand",
        home_keyframe="stand",
        com_xy_cost=20.0,
        com_z_cost=0.0,
        hand_position_cost=50.0,
        posture_cost=5.0,
        standing_pelvis_pitch=0.1,
        suppress_foot_pitch=True,
        pelvis_z_cost=50.0,
        root_tracker_x_offset=-0.3,
        waist_posture_cost=20.0,
        torso_upright_orientation_cost=10.0,
    ),
    "toss": IKConfig(
        inner_steps=20,
        standing_pelvis_height=0.763,
        posture_keyframe="stand",
        home_keyframe="stand",
        com_xy_cost=20.0,
        com_z_cost=0.0,
        hand_position_cost=50.0,
        posture_cost=1.5,
        standing_pelvis_pitch=0.0,
        suppress_foot_pitch=True,
    ),
    "unsheathe": IKConfig(
        inner_steps=20,
        standing_pelvis_height=0.793,
        posture_keyframe=None,
        com_xy_cost=30.0,
        com_z_cost=0.0,
        hand_position_cost=50.0,
        posture_cost=2.0,
        pelvis_orientation_cost=2.0,
    ),
    "pick": IKConfig(
        inner_steps=20,
        standing_pelvis_height=0.793,
        posture_keyframe=None,
        com_xy_cost=30.0,
        com_z_cost=0.0,
        hand_position_cost=50.0,
        pelvis_orientation_cost=5.0,
        root_tracker_x_offset=-0.22,
        foot_tracker_x_offset=0.18,
        posture_cost=8,
        standing_pelvis_pitch=0,
        pelvis_xy_cost=20,
        torso_upright_orientation_cost=10.0,
    ),
}

_config_dict["shelf"] = replace(
    _config_dict["pick"],
    pelvis_z_cost=0.0,
    pelvis_xy_cost=0.0,
    posture_cost=2.0,
    hand_position_cost=100.0,
    use_root_tracker=True,
)


def get_config(name: str) -> IKConfig:
    """Retrieve a predefined IKConfig by name."""
    if name not in _config_dict:
        raise ValueError(f"Unknown config name: {name}")
    return _config_dict[name]


__all__ = ["IKConfig", "get_config"]
