"""
Check action and state maximum absolute values across all episodes in the dataset.

This script loads all episodes from a dataset directory, reorders the state and action
data according to the LeRobot conversion logic, optionally applies normalization (quantile
or z-score), and analyzes the maximum absolute values in the reordered action and state 
data for each trajectory. 

The reordering reorganizes components to:
[arm_left, hand_left, arm_right, hand_right, leg_left, leg_right, waist, neck (optional), root]

Normalization options:
- quantile: Maps [q01, q99] to [-1, 1] using (x - q01) / (q99 - q01) * 2 - 1
- z-score: Standardizes using (x - mean) / std

It reports the maximum absolute value (with original sign), dimension, and timestep for 
both actions and states in each episode, and identifies the global maximum absolute values 
across all episodes.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np


def load_episode_data(episode_path: Path) -> Dict[str, Any]:
    """
    Load data from a single episode's data.json file.
    
    Args:
        episode_path: Path to the episode directory
        
    Returns:
        Dictionary containing the parsed JSON data
    """
    data_file = episode_path / "data.json"
    with open(data_file, 'r') as f:
        return json.load(f)


def extract_trajectories(episode_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract state and action trajectories from episode data.
    
    Args:
        episode_data: Parsed episode JSON data
        
    Returns:
        Dictionary with keys for each trajectory type and numpy arrays as values
    """
    data_points = episode_data['data']
    
    # Check what fields are available in the first data point
    first_point = data_points[0] if data_points else {}
    has_neck_data = 'state_neck' in first_point and 'action_neck' in first_point
    
    # Initialize lists to collect data
    trajectories = {
        'idx': [],
        'state_body': [],
        'state_hand_left': [],
        'state_hand_right': [],
        'action_body': [],
        'action_hand_left': [],
        'action_hand_right': [],
    }
    
    if has_neck_data:
        trajectories['state_neck'] = []
        trajectories['action_neck'] = []
    
    # Extract data from each timestep
    for point in data_points:
        trajectories['idx'].append(point['idx'])
        trajectories['state_body'].append(point['state_body'])
        trajectories['state_hand_left'].append(point['state_hand_left'])
        trajectories['state_hand_right'].append(point['state_hand_right'])
        trajectories['action_body'].append(point['action_body'])
        trajectories['action_hand_left'].append(point['action_hand_left'])
        trajectories['action_hand_right'].append(point['action_hand_right'])
        
        if has_neck_data:
            trajectories['state_neck'].append(point['state_neck'])
            trajectories['action_neck'].append(point['action_neck'])
    
    # Convert lists to numpy arrays
    for key in trajectories:
        trajectories[key] = np.array(trajectories[key])
    
    return trajectories


def load_norm_stats(norm_stats_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load normalization statistics from a JSON file.
    
    Args:
        norm_stats_path: Path to the norm_stats.json file
        
    Returns:
        Dictionary containing norm stats for state and actions
    """
    with open(norm_stats_path, 'r') as f:
        data = json.load(f)
    
    norm_stats = data['norm_stats']
    
    # Convert lists to numpy arrays
    result = {}
    for key in ['state', 'actions']:
        if key in norm_stats:
            result[key] = {
                'mean': np.array(norm_stats[key]['mean'], dtype=np.float32),
                'std': np.array(norm_stats[key]['std'], dtype=np.float32),
                'q01': np.array(norm_stats[key]['q01'], dtype=np.float32),
                'q99': np.array(norm_stats[key]['q99'], dtype=np.float32),
            }
    
    return result


def normalize_z_score(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization: (x - mean) / std
    
    Args:
        data: Data to normalize (timesteps, dimensions)
        mean: Mean values for each dimension
        std: Standard deviation for each dimension
        
    Returns:
        Normalized data
    """
    # Ensure mean and std match data dimensions
    mean = mean[: data.shape[-1]]
    std = std[: data.shape[-1]]
    return (data - mean) / (std + 1e-6)


def normalize_quantile(data: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """
    Apply quantile normalization: (x - q01) / (q99 - q01) * 2 - 1
    Maps [q01, q99] to [-1, 1]
    
    Args:
        data: Data to normalize (timesteps, dimensions)
        q01: 1st percentile values for each dimension
        q99: 99th percentile values for each dimension
        
    Returns:
        Normalized data
    """
    # Ensure q01 and q99 match data dimensions
    q01 = q01[: data.shape[-1]]
    q99 = q99[: data.shape[-1]]
    return (data - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def apply_normalization(
    trajectories: Dict[str, np.ndarray],
    norm_stats: Dict[str, Dict[str, np.ndarray]],
    method: str = 'quantile'
) -> Dict[str, np.ndarray]:
    """
    Apply normalization to state and action data.
    
    Args:
        trajectories: Dictionary containing 'state' and 'action' arrays
        norm_stats: Dictionary containing normalization statistics
        method: Normalization method ('quantile' or 'z-score')
        
    Returns:
        Dictionary with normalized state and action arrays
    """
    if method == 'quantile':
        if 'state' in norm_stats:
            trajectories['state'] = normalize_quantile(
                trajectories['state'],
                norm_stats['state']['q01'],
                norm_stats['state']['q99']
            )
        if 'actions' in norm_stats:
            trajectories['action'] = normalize_quantile(
                trajectories['action'],
                norm_stats['actions']['q01'],
                norm_stats['actions']['q99']
            )
    elif method == 'z-score':
        if 'state' in norm_stats:
            trajectories['state'] = normalize_z_score(
                trajectories['state'],
                norm_stats['state']['mean'],
                norm_stats['state']['std']
            )
        if 'actions' in norm_stats:
            trajectories['action'] = normalize_z_score(
                trajectories['action'],
                norm_stats['actions']['mean'],
                norm_stats['actions']['std']
            )
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'quantile' or 'z-score'")
    
    return trajectories


def get_body_component_name(dimension: int) -> str:
    """
    Get the body component name for a given dimension after reordering.

    Args:
        dimension: The dimension index (0-based, 37-dimensional layout without neck)

    Returns:
        String describing which body component this dimension belongs to
    """
    if dimension < 7:
        return f"arm_left[{dimension}]"
    elif dimension < 8:
        return "hand_left[0]"
    elif dimension < 15:
        return f"arm_right[{dimension - 8}]"
    elif dimension < 16:
        return "hand_right[0]"
    elif dimension < 22:
        return f"leg_left[{dimension - 16}]"
    elif dimension < 28:
        return f"leg_right[{dimension - 22}]"
    elif dimension < 31:
        return f"waist[{dimension - 28}]"
    elif dimension < 37:
        return f"root[{dimension - 31}]"
    else:
        return f"unknown[{dimension}]"


def get_dimension_mapping() -> str:
    """
    Get a string describing the dimension mapping after reordering.

    Returns:
        String describing dimension ranges for each body component (37-dimensional, without neck)
    """
    mapping = [
        "Dimension mapping after reordering:",
        "  [0:7]    arm_left (7 dims)",
        "  [7:8]    hand_left (1 dim)",
        "  [8:15]   arm_right (7 dims)",
        "  [15:16]  hand_right (1 dim)",
        "  [16:22]  leg_left (6 dims)",
        "  [22:28]  leg_right (6 dims)",
        "  [28:31]  waist (3 dims)",
        "  [31:37]  root (6 dims)",
        "  Total: 37 dimensions",
    ]
    return "\n".join(mapping)


def reorder_data(trajectories: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Reorder state and action data according to the LeRobot conversion logic.

    This function reorganizes the body components from the original order to:
    [arm_left, hand_left, arm_right, hand_right, leg_left, leg_right, waist, root]

    Args:
        trajectories: Dictionary containing trajectory data with state_body and action_body

    Returns:
        Dictionary with reordered state and action arrays (37-dimensional, without neck)
    """
    state_body = trajectories['state_body']
    action_body = trajectories['action_body']
    state_hand_left = trajectories['state_hand_left']
    state_hand_right = trajectories['state_hand_right']
    action_hand_left = trajectories['action_hand_left']
    action_hand_right = trajectories['action_hand_right']
    
    num_timesteps = state_body.shape[0]
    
    # Initialize reordered arrays
    reordered_state = []
    reordered_action = []
    
    # Process each timestep
    for t in range(num_timesteps):
        # Extract state components from state_body
        state_root = state_body[t, 0:5]  # [angular velocity (3), roll (1), pitch (1)]
        state_root = np.concatenate([state_root, np.zeros(1, dtype=np.float32)])  # pad to 6 dimensions
        state_leg_left = state_body[t, 5:11]
        state_leg_right = state_body[t, 11:17]
        state_waist = state_body[t, 17:20]
        state_arm_left = state_body[t, 20:27]
        state_arm_right = state_body[t, 27:34]
        state_hand_left_t = state_hand_left[t, :1]
        state_hand_right_t = state_hand_right[t, :1]
        
        # Reorder state components
        state_components = [
            state_arm_left,
            state_hand_left_t,
            state_arm_right,
            state_hand_right_t,
            state_leg_left,
            state_leg_right,
            state_waist,
            state_root,
        ]
        reordered_state.append(np.concatenate(state_components))

        # Extract action components from action_body
        action_root = action_body[t, 0:6]
        action_leg_left = action_body[t, 6:12]
        action_leg_right = action_body[t, 12:18]
        action_waist = action_body[t, 18:21]
        action_arm_left = action_body[t, 21:28]
        action_arm_right = action_body[t, 28:35]
        action_hand_left_t = action_hand_left[t, :1]
        action_hand_right_t = action_hand_right[t, :1]

        # Reorder action components
        action_components = [
            action_arm_left,
            action_hand_left_t,
            action_arm_right,
            action_hand_right_t,
            action_leg_left,
            action_leg_right,
            action_waist,
            action_root,
        ]
        reordered_action.append(np.concatenate(action_components))
    
    # Update trajectories with reordered data
    trajectories['state'] = np.array(reordered_state)
    trajectories['action'] = np.array(reordered_action)
    
    return trajectories


def analyze_action_body_max(action_body: np.ndarray) -> Tuple[float, int, int]:
    """
    Find the maximum absolute value in action_body and its location.
    
    Args:
        action_body: numpy array of shape (timesteps, dimensions)
        
    Returns:
        Tuple of (max_abs_value, dimension, timestep)
        Note: max_abs_value retains its original sign (not absolute value)
    """
    abs_action_body = np.abs(action_body)
    max_abs_position = np.unravel_index(np.argmax(abs_action_body), abs_action_body.shape)
    max_timestep = max_abs_position[0]
    max_dimension = max_abs_position[1]
    # Return the original value (with sign), not the absolute value
    max_value = action_body[max_timestep, max_dimension]
    
    return max_value, max_dimension, max_timestep


def analyze_state_max(state: np.ndarray) -> Tuple[float, int, int]:
    """
    Find the maximum absolute value in state and its location.
    
    Args:
        state: numpy array of shape (timesteps, dimensions)
        
    Returns:
        Tuple of (max_abs_value, dimension, timestep)
        Note: max_abs_value retains its original sign (not absolute value)
    """
    abs_state = np.abs(state)
    max_abs_position = np.unravel_index(np.argmax(abs_state), abs_state.shape)
    max_timestep = max_abs_position[0]
    max_dimension = max_abs_position[1]
    # Return the original value (with sign), not the absolute value
    max_value = state[max_timestep, max_dimension]
    
    return max_value, max_dimension, max_timestep


def main():
    """Main function to check action and state maximum absolute values across all episodes."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Check action and state maximum absolute values in dataset (after reordering and optional normalization)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/hyd/codebase/openpi/data/20260126_1624",
        help="Path to the data root directory containing episode folders"
    )
    parser.add_argument(
        "--norm_stats",
        type=str,
        default=None,
        help="Path to norm_stats.json file for normalization (default: None, no normalization)"
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default="quantile",
        choices=["quantile", "z-score"],
        help="Normalization method: 'quantile' or 'z-score' (default: quantile)"
    )

    # import debugpy
    # debugpy.listen(("127.0.0.1", 5678))
    # print("\033[91mDebug server started on 127.0.0.1:5678\033[0m")
    # print("\033[91mWaiting for debugger to attach...\033[0m")
    # debugpy.wait_for_client()
    # print("Debugger attached, continuing execution...")
    
    args = parser.parse_args()

    # Convert path to Path object
    data_root = Path(args.data_root)

    # Load normalization statistics if provided
    norm_stats = None
    if args.norm_stats:
        norm_stats_path = Path(args.norm_stats)
        if not norm_stats_path.exists():
            print(f"Error: norm_stats file not found at {norm_stats_path}")
            return
        norm_stats = load_norm_stats(norm_stats_path)
        print(f"Loaded norm_stats from: {norm_stats_path}")
        print(f"Normalization method: {args.norm_method}")
    
    # Get all episode directories
    episode_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    
    print(f"Found {len(episode_dirs)} episodes in {data_root}")
    print(f"Apply normalization: {norm_stats is not None}")
    print("\n" + get_dimension_mapping())
    print("=" * 80)
    
    # Track global maximum absolute value for actions
    global_max_abs_value = -np.inf
    global_max_value = None
    global_max_episode = None
    global_max_dimension = None
    global_max_timestep = None
    
    # Track global maximum absolute value for states
    global_state_max_abs_value = -np.inf
    global_state_max_value = None
    global_state_max_episode = None
    global_state_max_dimension = None
    global_state_max_timestep = None
    
    # Process all episodes
    for episode_dir in episode_dirs:
        episode_name = episode_dir.name
        
        # Load episode data
        try:
            episode_data = load_episode_data(episode_dir)
        except Exception as e:
            print(f"Error loading {episode_name}: {e}")
            continue
        
        # Extract trajectories
        trajectories = extract_trajectories(episode_data)
        
        # Reorder data according to LeRobot conversion logic
        trajectories = reorder_data(trajectories)

        # Apply normalization if norm_stats is provided
        if norm_stats is not None:
            trajectories = apply_normalization(trajectories, norm_stats, method=args.norm_method)
        
        action = trajectories['action']
        state = trajectories['state']
        
        # Analyze action maximum absolute value
        max_value, max_dimension, max_timestep = analyze_action_body_max(action)
        body_component = get_body_component_name(max_dimension)

        # Analyze state maximum absolute value
        state_max_value, state_max_dimension, state_max_timestep = analyze_state_max(state)
        state_body_component = get_body_component_name(state_max_dimension)
        
        # Print results for this episode
        print(f"\n{episode_name}:")
        print(f"  Action max absolute value: {max_value:.6f} (abs: {abs(max_value):.6f})")
        print(f"    Dimension: {max_dimension} (out of {action.shape[1]} dimensions) -> {body_component}")
        print(f"    Timestep: {max_timestep} (out of {action.shape[0]} timesteps)")
        print(f"  State max absolute value: {state_max_value:.6f} (abs: {abs(state_max_value):.6f})")
        print(f"    Dimension: {state_max_dimension} (out of {state.shape[1]} dimensions) -> {state_body_component}")
        print(f"    Timestep: {state_max_timestep} (out of {state.shape[0]} timesteps)")
        
        # Update global maximum if this episode has a larger absolute value
        if abs(max_value) > global_max_abs_value:
            global_max_abs_value = abs(max_value)
            global_max_value = max_value
            global_max_episode = episode_name
            global_max_dimension = max_dimension
            global_max_timestep = max_timestep
        
        # Update global state maximum if this episode has a larger absolute value
        if abs(state_max_value) > global_state_max_abs_value:
            global_state_max_abs_value = abs(state_max_value)
            global_state_max_value = state_max_value
            global_state_max_episode = episode_name
            global_state_max_dimension = state_max_dimension
            global_state_max_timestep = state_max_timestep
    
    # Print global maximum summary
    if global_max_episode is not None:
        global_body_component = get_body_component_name(global_max_dimension)
        print("\n" + "=" * 80)
        print("\nGlobal Maximum Absolute Value Summary (after reordering):")
        print("\nAction:")
        print(f"  Episode: {global_max_episode}")
        print(f"  Maximum absolute value: {global_max_value:.6f} (abs: {global_max_abs_value:.6f})")
        print(f"  Dimension: {global_max_dimension} -> {global_body_component}")
        print(f"  Timestep: {global_max_timestep}")
        
    if global_state_max_episode is not None:
        global_state_body_component = get_body_component_name(global_state_max_dimension)
        if global_max_episode is None:
            print("\n" + "=" * 80)
            print("\nGlobal Maximum Absolute Value Summary (after reordering):")
        print("\nState:")
        print(f"  Episode: {global_state_max_episode}")
        print(f"  Maximum absolute value: {global_state_max_value:.6f} (abs: {global_state_max_abs_value:.6f})")
        print(f"  Dimension: {global_state_max_dimension} -> {global_state_body_component}")
        print(f"  Timestep: {global_state_max_timestep}")
        print("\n" + "=" * 80)
    elif global_max_episode is not None:
        print("\n" + "=" * 80)
    else:
        print("\nNo valid episodes found.")
        print("=" * 80)


if __name__ == "__main__":
    main()
