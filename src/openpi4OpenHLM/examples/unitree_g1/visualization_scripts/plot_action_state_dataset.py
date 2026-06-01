"""
Visualize action and state data from dataset JSON files.

This script reads state_body and action_body data from collected dataset files and generates
matplotlib plots for each body part (left arm, right arm, left leg, right leg, waist, root).
Each plot contains subplots for individual dimensions, showing both state and action values
to help analyze the correspondence between observations and actions.

Arguments:
  --summary-only  Skip generating matplotlib plot files and only print the delay summary
                  (per-joint state-action delay statistics) to the console. Useful for a
                  quick diagnostic without writing any image files to disk.
  --max-frames N  Limit analysis to the first N frames of the episode.

Example usage:
  python examples/unitree_g1/visualization_scripts/plot_action_state_dataset.py \
    --json-path /home/hyd/codebase/openpi/data/0316-pick-cola-2x/episode_0002/data.json \
    --summary-only \
    --max-frames 500
"""

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import tyro


# Body part configuration with indices and dimension labels
# Based on the 32-dim state_body/action_body layout used in dataset JSON files.
STEP_DURATION_SECONDS = 1.0 / 30.0

FULL_BODY_DIMENSION_LABELS = [
    "root.roll",
    "root.pitch",
    "root.yaw_velocity",
    "left_leg.joint_0",
    "left_leg.joint_1",
    "left_leg.joint_2",
    "left_leg.joint_3",
    "left_leg.joint_4",
    "left_leg.joint_5",
    "right_leg.joint_0",
    "right_leg.joint_1",
    "right_leg.joint_2",
    "right_leg.joint_3",
    "right_leg.joint_4",
    "right_leg.joint_5",
    "waist.joint_0",
    "waist.joint_1",
    "waist.joint_2",
    "left_arm.joint_0",
    "left_arm.joint_1",
    "left_arm.joint_2",
    "left_arm.joint_3",
    "left_arm.joint_4",
    "left_arm.joint_5",
    "left_arm.joint_6",
    "right_arm.joint_0",
    "right_arm.joint_1",
    "right_arm.joint_2",
    "right_arm.joint_3",
    "right_arm.joint_4",
    "right_arm.joint_5",
    "right_arm.joint_6",
]

BODY_PARTS = {
    "left_arm": {
        "state_indices": range(18, 25),  # 7 dims
        "action_indices": range(18, 25),  # 7 dims
        "labels": [
            "joint_0", "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6"
        ]
    },
    "right_arm": {
        "state_indices": range(25, 32),  # 7 dims
        "action_indices": range(25, 32),  # 7 dims
        "labels": [
            "joint_0", "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6"
        ]
    },
    "left_leg": {
        "state_indices": range(3, 9),  # 6 dims
        "action_indices": range(3, 9),  # 6 dims
        "labels": [
            "joint_0", "joint_1", "joint_2",
            "joint_3", "joint_4", "joint_5"
        ]
    },
    "right_leg": {
        "state_indices": range(9, 15),  # 6 dims
        "action_indices": range(9, 15),  # 6 dims
        "labels": [
            "joint_0", "joint_1", "joint_2",
            "joint_3", "joint_4", "joint_5"
        ]
    },
    "waist": {
        "state_indices": range(15, 18),  # 3 dims
        "action_indices": range(15, 18),  # 3 dims
        "labels": ["joint_0", "joint_1", "joint_2"]
    },
    "root": {
        "state_indices": range(0, 3),  # 3 dims: roll, pitch, yaw angular velocity
        "action_indices": range(0, 3),  # 3 dims: roll, pitch, yaw angular velocity
        "labels": ["roll", "pitch", "yaw_velocity"]
    }
}


@dataclasses.dataclass
class Args:
    json_path: str  # Path to input JSON file containing state_body and action_body data
    output_dir: str = None  # Optional output directory override (default: same folder as json_path)
    max_frames: int = None  # Maximum number of frames to plot (default: None, plot all frames)
    summary_only: bool = False  # If True, only print the delay summary and skip plot generation


def load_dataset(json_path: str) -> Dict[str, Any]:
    """
    Load dataset from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary containing:
            - info: Metadata about the dataset
            - text: Task descriptions
            - data: List of frames with state_body and action_body
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Validate required keys
    if "data" not in data:
        raise ValueError("Missing required key 'data' in JSON file")
    
    return data


def extract_body_part_data(
    frames: List[Dict[str, Any]],
    body_part: str,
    config: Dict[str, Any],
    max_frames: int = None
) -> Dict[str, np.ndarray]:
    """
    Extract state and action data for a specific body part from all frames.
    
    Args:
        frames: List of frame dictionaries with state_body and action_body
        body_part: Name of the body part (e.g., "left_arm")
        config: Configuration dict with 'state_indices' and 'action_indices'
        max_frames: Maximum number of frames to extract (None = all)
        
    Returns:
        Dictionary with keys:
            - states: np.ndarray of shape [n_frames, n_dims]
            - actions: np.ndarray of shape [n_frames, n_dims]
            - frame_indices: np.ndarray of shape [n_frames]
    """
    state_indices = list(config["state_indices"])
    action_indices = list(config["action_indices"])
    n_state_dims = len(state_indices)
    n_action_dims = len(action_indices)
    
    # Limit frames if max_frames is specified
    if max_frames is not None:
        frames = frames[:max_frames]
    
    n_frames = len(frames)
    states = np.zeros((n_frames, n_state_dims), dtype=np.float32)
    actions = np.zeros((n_frames, n_action_dims), dtype=np.float32)
    frame_indices = np.arange(n_frames)
    
    for i, frame in enumerate(frames):
        state_body = np.array(frame["state_body"], dtype=np.float32)
        action_body = np.array(frame["action_body"], dtype=np.float32)
        
        # Extract state data
        states[i, :] = state_body[state_indices]
        
        # Extract action data
        actions[i, :] = action_body[action_indices]
    
    return {
        "states": states,
        "actions": actions,
        "frame_indices": frame_indices
    }


def create_body_part_plot(
    body_part: str,
    config: Dict[str, Any],
    data: Dict[str, np.ndarray]
) -> plt.Figure:
    """
    Create a figure with subplots for all dimensions of a body part.
    Each subplot shows both state and action values over time.
    
    Args:
        body_part: Name of the body part (e.g., "left_arm")
        config: Configuration dict with 'labels'
        data: Dictionary with 'states', 'actions', and 'frame_indices'
        
    Returns:
        Matplotlib figure object
    """
    labels = config["labels"]
    states = data["states"]
    actions = data["actions"]
    frame_indices = data["frame_indices"]
    
    n_state_dims = states.shape[1]
    n_action_dims = actions.shape[1]
    
    # Determine number of subplots (max of state and action dimensions)
    n_dims = max(n_state_dims, n_action_dims)
    
    # Create figure with subplots (one per dimension)
    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims))
    if n_dims == 1:
        axes = [axes]  # Ensure axes is always a list
    
    fig.suptitle(f"{body_part.replace('_', ' ').title()} - State vs Action", fontsize=16, y=0.995)
    
    # Plot each dimension
    for dim_idx in range(n_dims):
        ax = axes[dim_idx]
        dim_label = labels[dim_idx] if dim_idx < len(labels) else f"dim_{dim_idx}"
        
        # Plot state (blue line)
        if dim_idx < n_state_dims:
            ax.plot(frame_indices, states[:, dim_idx], 
                   color='C0', alpha=0.8, linewidth=2.0, label='State')
        
        # Plot action (orange line)
        if dim_idx < n_action_dims:
            ax.plot(frame_indices, actions[:, dim_idx], 
                   color='C1', alpha=0.8, linewidth=2.0, label='Action')
        
        # Styling
        ax.set_ylabel(dim_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # X-label only on last subplot
        if dim_idx == n_dims - 1:
            ax.set_xlabel("Frame Index", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space at top for title
    return fig


def extract_full_body_data(frames: List[Dict[str, Any]], max_frames: int = None) -> Dict[str, np.ndarray]:
    """
    Extract all 32 state_body and action_body dimensions from the dataset.

    Args:
        frames: List of frame dictionaries with state_body and action_body
        max_frames: Maximum number of frames to extract (None = all)

    Returns:
        Dictionary with keys:
            - states: np.ndarray of shape [n_frames, 32]
            - actions: np.ndarray of shape [n_frames, 32]
    """
    if max_frames is not None:
        frames = frames[:max_frames]

    n_frames = len(frames)
    states = np.zeros((n_frames, 32), dtype=np.float32)
    actions = np.zeros((n_frames, 32), dtype=np.float32)

    for i, frame in enumerate(frames):
        states[i, :] = np.array(frame["state_body"], dtype=np.float32)
        actions[i, :] = np.array(frame["action_body"], dtype=np.float32)

    return {
        "states": states,
        "actions": actions,
    }


def estimate_delay_steps(action_series: np.ndarray, state_series: np.ndarray, max_lag_steps: int) -> tuple[int, float]:
    """
    Estimate how many steps the state lags behind the action.

    A positive lag means the state is delayed relative to the action.

    Args:
        action_series: Action values for one dimension, shape [n_frames]
        state_series: State values for one dimension, shape [n_frames]
        max_lag_steps: Maximum absolute lag to search

    Returns:
        Tuple of:
            - best_lag: Estimated lag in steps
            - best_score: Normalized correlation score for that lag
    """
    if len(action_series) != len(state_series):
        raise ValueError("Action and state series must have the same length")

    if len(action_series) < 3:
        return 0, float("nan")

    # Use first-order differences to focus on motion timing instead of slow trajectory drift.
    action_signal = np.diff(action_series.astype(np.float64))
    state_signal = np.diff(state_series.astype(np.float64))

    # Fall back to raw values when the motion signal is nearly constant.
    if np.std(action_signal) < 1e-8 or np.std(state_signal) < 1e-8:
        action_signal = action_series.astype(np.float64)
        state_signal = state_series.astype(np.float64)

    best_lag = 0
    best_score = -np.inf
    clipped_max_lag = min(max_lag_steps, len(action_signal) - 2)

    for lag in range(-clipped_max_lag, clipped_max_lag + 1):
        if lag > 0:
            aligned_action = action_signal[:-lag]
            aligned_state = state_signal[lag:]
        elif lag < 0:
            aligned_action = action_signal[-lag:]
            aligned_state = state_signal[:lag]
        else:
            aligned_action = action_signal
            aligned_state = state_signal

        if len(aligned_action) < 3:
            continue

        aligned_action = aligned_action - np.mean(aligned_action)
        aligned_state = aligned_state - np.mean(aligned_state)
        denominator = np.linalg.norm(aligned_action) * np.linalg.norm(aligned_state)

        if denominator < 1e-8:
            continue

        score = float(np.dot(aligned_action, aligned_state) / denominator)
        if score > best_score:
            best_score = score
            best_lag = lag

    if not np.isfinite(best_score):
        return 0, float("nan")

    return best_lag, best_score


def print_full_body_delay_summary(frames: List[Dict[str, Any]], max_frames: int = None) -> None:
    """
    Print an estimated per-dimension delay between action_body and state_body.

    Args:
        frames: List of frame dictionaries
        max_frames: Maximum number of frames to analyze (None = all)
    """
    full_body_data = extract_full_body_data(frames, max_frames)
    states = full_body_data["states"]
    actions = full_body_data["actions"]

    max_lag_steps = min(60, max(1, len(states) // 4))

    print("\nEstimated state delay relative to action for all 32 dimensions:")
    print(f"Search window: +/-{max_lag_steps} steps")
    print(f"Assumed step duration: {STEP_DURATION_SECONDS:.6f} s (30 Hz)")
    print()

    rows = []
    for dim_idx, dim_label in enumerate(FULL_BODY_DIMENSION_LABELS):
        lag_steps, score = estimate_delay_steps(actions[:, dim_idx], states[:, dim_idx], max_lag_steps)

        if lag_steps > 0:
            relation = "state lags"
        elif lag_steps < 0:
            relation = "state leads"
        else:
            relation = "aligned"

        rows.append(
            {
                "index": f"{dim_idx:02d}",
                "dimension": dim_label,
                "relation": relation,
                "steps": str(abs(lag_steps)),
                "seconds": f"{abs(lag_steps) * STEP_DURATION_SECONDS:.3f}",
                "score": f"{score:.3f}" if np.isfinite(score) else "nan",
            }
        )

    columns = [
        ("index", "idx"),
        ("dimension", "dimension"),
        ("relation", "relation"),
        ("steps", "steps"),
        ("seconds", "seconds"),
        ("score", "score"),
    ]
    column_widths = {
        key: max(len(header), max(len(row[key]) for row in rows))
        for key, header in columns
    }

    header = " | ".join(header.ljust(column_widths[key]) for key, header in columns)
    separator = "-+-".join("-" * column_widths[key] for key, _ in columns)
    print(header)
    print(separator)

    for row in rows:
        line = " | ".join(row[key].ljust(column_widths[key]) for key, _ in columns)
        print(line)


def main(args: Args):
    """Main function to load data and generate plots."""
    # import debugpy
    # debugpy.listen(("127.0.0.1", 5678))
    # print("\033[91mDebug server started on 127.0.0.1:5678\033[0m")
    # print("\033[91mWaiting for debugger to attach...\033[0m")
    # debugpy.wait_for_client()
    # print("Debugger attached, continuing execution...")
    
    # 1. Load JSON data
    print(f"Loading data from: {args.json_path}")
    dataset = load_dataset(args.json_path)
    
    frames = dataset["data"]
    total_frames = len(frames)
    
    print(f"Total frames: {total_frames}")
    if args.max_frames is not None:
        print(f"Plotting first {args.max_frames} frames")
        frames_to_plot = min(args.max_frames, total_frames)
    else:
        frames_to_plot = total_frames
    
    if args.summary_only:
        print("\nSummary-only mode enabled. Skipping plot generation.")
    else:
        # 2. Determine output directory
        json_path = Path(args.json_path)
        if args.output_dir is None:
            output_dir = json_path.parent  # Save in same folder as data.json
        else:
            output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # 3. Plot each body part
        print("\nGenerating plots...")
        for body_part, config in BODY_PARTS.items():
            print(f"  Creating {body_part} plot...")

            # Extract data for this body part
            data = extract_body_part_data(frames, body_part, config, args.max_frames)

            # Create plot
            fig = create_body_part_plot(body_part, config, data)

            # Save figure
            save_path = output_dir / f"state_action_{body_part}.jpg"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {save_path}")
    
    print_full_body_delay_summary(frames, args.max_frames)
    if args.summary_only:
        print("\nDelay summary generated successfully!")
    else:
        print("\nAll plots generated successfully!")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
