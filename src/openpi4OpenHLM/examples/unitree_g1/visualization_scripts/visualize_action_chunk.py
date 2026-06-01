"""
Visualize action chunks from saved JSON data.

This script reads action chunk data saved during evaluation episodes and generates
matplotlib plots for each body part (left arm, right arm, left leg, right leg, waist, root).
Each plot contains subplots for individual dimensions, with alternating colors for
consecutive action chunks to distinguish policy predictions.

Example usage:
  python examples/unitree_g1/visualization_scripts/visualize_action_chunk.py \
    --json-path /home/hyd/codebase/openpi/visualization/raw_action_chunk/20260311_1717/episode_1.json \
    --chunk-plot-length 30 \
    --max-action-steps 800
"""

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import tyro


# Body part configuration with indices and dimension labels
# Based on sonic_g1_env.py action format (34 dims)
BODY_PARTS = {
    "left_arm": {
        "indices": range(0, 8),
        "labels": [
            "joint_0", "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6", "gripper"
        ]
    },
    "right_arm": {
        "indices": range(8, 16),
        "labels": [
            "joint_0", "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6", "gripper"
        ]
    },
    "left_leg": {
        "indices": range(16, 22),
        "labels": [
            "joint_0", "joint_1", "joint_2",
            "joint_3", "joint_4", "joint_5"
        ]
    },
    "right_leg": {
        "indices": range(22, 28),
        "labels": [
            "joint_0", "joint_1", "joint_2",
            "joint_3", "joint_4", "joint_5"
        ]
    },
    "waist": {
        "indices": range(28, 31),
        "labels": ["joint_0", "joint_1", "joint_2"]
    },
    "root": {
        "indices": range(31, 34),
        "labels": ["roll", "pitch", "yaw_velocity"]
    }
}


@dataclasses.dataclass
class Args:
    json_path: str  # Path to input JSON file containing action chunk data
    output_dir: str = None  # Optional output directory override (default: visualization/plot_action_chunk/<timestamp>)
    chunk_plot_length: int = 50  # Number of points to plot from each action chunk (default: 50)
    max_action_steps: int = None  # Maximum action step index to display on x-axis (default: None, show all)


def load_action_chunk_data(json_path: str) -> Dict[str, Any]:
    """
    Load action chunk data from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary containing episode data with keys:
            - episode_idx: Episode number
            - timestamp: Timestamp string
            - total_steps: Total action steps
            - action_chunk_shape: [chunk_length, action_dim]
            - action_chunks: List of dicts with 'step_index_before_inference' and 'action_chunk'
            - states: (optional) List of dicts with 'step_index' and 'state'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Validate required keys
    required_keys = ["episode_idx", "timestamp", "action_chunks", "action_chunk_shape"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in JSON file")
    
    return data


def create_body_part_plot(
    body_part: str,
    config: Dict[str, Any],
    action_chunks: List[Dict[str, Any]],
    chunk_plot_length: int,
    max_action_steps: int = None,
    states: List[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Create a figure with subplots for all dimensions of a body part.
    
    Args:
        body_part: Name of the body part (e.g., "left_arm")
        config: Configuration dict with 'indices' and 'labels'
        action_chunks: List of action chunk dicts from JSON
        chunk_plot_length: Number of points to plot from each action chunk
        max_action_steps: Maximum action step index to display (None = show all)
        states: (optional) List of state dicts with 'step_index' and 'state'
        
    Returns:
        Matplotlib figure object
    """
    indices = list(config["indices"])
    labels = config["labels"]
    n_dims = len(indices)
    
    # Create figure with subplots (one per dimension)
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims))
    if n_dims == 1:
        axes = [axes]  # Ensure axes is always a list
    
    fig.suptitle(f"{body_part.replace('_', ' ').title()} Action Chunks", fontsize=16, y=0.995)
    
    # Precompute policy query steps for x ticks
    step_indices = [chunk_data["step_index_before_inference"] for chunk_data in action_chunks]
    step_indices = sorted({idx for idx in step_indices if idx is not None})
    if max_action_steps is not None:
        step_indices = [idx for idx in step_indices if idx <= max_action_steps]

    # Plot each dimension
    for dim_idx, (action_dim, dim_label) in enumerate(zip(indices, labels)):
        ax = axes[dim_idx]
        
        # Plot each action chunk
        for chunk_idx, chunk_data in enumerate(action_chunks):
            # Get step index
            step_index = chunk_data["step_index_before_inference"]
            action_chunk = np.array(chunk_data["action_chunk"])  # Shape: [chunk_length, action_dim]
            
            # Extract data for this dimension, limiting to chunk_plot_length
            dim_values = action_chunk[:chunk_plot_length, action_dim]
            
            # Calculate x-axis values (action step indices)
            x_values = step_index + np.arange(len(dim_values))
            
            # Determine color based on chunk parity (odd=0,2,4... even=1,3,5...)
            is_odd_chunk = (chunk_idx % 2 == 0)
            color = 'C0' if is_odd_chunk else 'C1'  # C0=blue, C1=orange
            
            # Plot the action chunk
            ax.plot(x_values, dim_values, color=color, alpha=0.7, linewidth=2)
            
            # Plot action_prefix if available (in green)
            if "action_prefix" in chunk_data:
                action_prefix = np.array(chunk_data["action_prefix"])  # Shape: [prefix_length, action_dim]
                if len(action_prefix.shape) == 2 and action_prefix.shape[1] > action_dim:
                    # Extract data for this dimension
                    prefix_values = action_prefix[:, action_dim]
                    # X-axis values start from step_index_before_inference
                    prefix_x_values = step_index + np.arange(len(prefix_values))
                    # Plot with green color
                    ax.plot(prefix_x_values, prefix_values, color='red', alpha=0.9, linewidth=2)
            
            # Add vertical line at policy query point
            after_index = chunk_data.get("step_index_after_inference")
            if after_index is not None:
                ax.axvline(x=after_index, color='gray', linestyle='--', alpha=0.5, linewidth=1.0)
        
        # Plot states if available
        if states is not None and len(states) > 0:
            # Extract state values for this dimension
            state_step_indices = []
            state_values = []
            for state_data in states:
                step_idx = state_data["step_index"]
                state_array = np.array(state_data["state"])
                
                # Check if this dimension exists in the state
                if len(state_array) > action_dim:
                    state_step_indices.append(step_idx)
                    state_values.append(state_array[action_dim])
            
            # Plot states as a green line
            if len(state_step_indices) > 0:
                # Filter by max_action_steps if specified
                if max_action_steps is not None:
                    filtered_indices = []
                    filtered_values = []
                    for idx, val in zip(state_step_indices, state_values):
                        if idx <= max_action_steps:
                            filtered_indices.append(idx)
                            filtered_values.append(val)
                    state_step_indices = filtered_indices
                    state_values = filtered_values
                
                if len(state_step_indices) > 0:
                    ax.plot(state_step_indices, state_values, color='green', 
                           alpha=0.8, linewidth=1.5, label='State')
        
        # Styling
        ax.set_ylabel(dim_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limit and ticks
        if max_action_steps is not None:
            ax.set_xlim(0, max_action_steps)
            tick_positions = step_indices
            ax.set_xticks(tick_positions)
        else:
            # Auto-determine x-axis range from data
            xlim = ax.get_xlim()
            tick_positions = [idx for idx in step_indices if idx <= xlim[1]]
            ax.set_xticks(tick_positions)
        for tick_pos in tick_positions:
            ax.axvline(x=tick_pos, color='gray', linestyle='-', alpha=0.5, linewidth=1.0)
        
        # X-label only on last subplot
        if dim_idx == n_dims - 1:
            ax.set_xlabel("Action Step Index", fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space at top for title
    return fig


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
    data = load_action_chunk_data(args.json_path)
    
    # 2. Extract metadata
    episode_idx = data["episode_idx"]
    timestamp = data["timestamp"]
    action_chunks = data["action_chunks"]
    chunk_length = data["action_chunk_shape"][0]
    action_dim = data["action_chunk_shape"][1]
    states = data.get("states", None)  # Optional states data
    
    print(f"Episode: {episode_idx}")
    print(f"Timestamp: {timestamp}")
    print(f"Total chunks: {len(action_chunks)}")
    print(f"Chunk shape: [{chunk_length}, {action_dim}]")
    print(f"Plotting {args.chunk_plot_length} points per chunk")
    if states is not None:
        print(f"States available: {len(states)} state records")
    if args.max_action_steps is not None:
        print(f"X-axis range: 0 to {args.max_action_steps}")
    
    # 3. Determine output directory
    if args.output_dir is None:
        output_dir = Path("visualization/plot_action_chunk") / timestamp
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 4. Plot each body part
    print("\nGenerating plots...")
    for body_part, config in BODY_PARTS.items():
        print(f"  Creating {body_part} plot...")
        fig = create_body_part_plot(body_part, config, action_chunks, args.chunk_plot_length, args.max_action_steps, states)
        
        # Save figure
        save_path = output_dir / f"episode_{episode_idx}_{body_part}.jpg"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {save_path}")
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
