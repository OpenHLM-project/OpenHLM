"""
Visualization script for robotics data from 0908_bidex_v5 dataset.

This script visualizes all episodes in a folder and plots the state and action trajectories
over time. States and actions are plotted separately in different images.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
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


def plot_state_trajectories(trajectories: Dict[str, np.ndarray], 
                            episode_name: str, 
                            output_path: Path,
                            visualize_neck: bool = False) -> None:
    """
    Plot all state trajectories (state_body, state_hand_left, state_hand_right, and optionally state_neck).
    
    Args:
        trajectories: Dictionary containing trajectory data
        episode_name: Name of the episode for the title
        output_path: Path where the figure will be saved
        visualize_neck: Whether to include state_neck in the visualization (default: False)
    """
    idx = trajectories['idx']
    state_body = trajectories['state_body']
    state_hand_left = trajectories['state_hand_left']
    state_hand_right = trajectories['state_hand_right']
    
    # Check if neck data exists in trajectories
    has_neck_data = 'state_neck' in trajectories
    state_neck = trajectories.get('state_neck', None)
    
    # Calculate number of subplots needed for state_body
    body_dim = state_body.shape[1]
    num_body_plots = (body_dim + 9) // 10  # Ceiling division
    
    # Total subplots: body_plots + 2 (left_hand, right_hand) + 1 (neck if enabled and available)
    total_plots = num_body_plots + 2 + (1 if (visualize_neck and has_neck_data) else 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 4 * total_plots))
    if total_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'State Trajectories - {episode_name}', fontsize=16, y=0.995)
    
    # Plot state_body (every 10 dimensions in one subplot)
    for i in range(num_body_plots):
        start_dim = i * 10
        end_dim = min((i + 1) * 10, body_dim)
        
        for dim in range(start_dim, end_dim):
            axes[i].plot(idx, state_body[:, dim], label=f'dim {dim}', alpha=0.7)
        
        axes[i].set_xlabel('Time Step (idx)')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'state_body [dim {start_dim}:{end_dim}]')
        axes[i].set_ylim(-0.4, 0.8)  # Set y-axis range for state_body
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Plot state_hand_left (all dimensions in one subplot)
    ax_idx = num_body_plots
    for dim in range(state_hand_left.shape[1]):
        axes[ax_idx].plot(idx, state_hand_left[:, dim], label=f'dim {dim}', alpha=0.7)
    axes[ax_idx].set_xlabel('Time Step (idx)')
    axes[ax_idx].set_ylabel('Value')
    axes[ax_idx].set_title('state_hand_left (all dimensions)')
    axes[ax_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[ax_idx].grid(True, alpha=0.3)
    
    # Plot state_hand_right (all dimensions in one subplot)
    ax_idx += 1
    for dim in range(state_hand_right.shape[1]):
        axes[ax_idx].plot(idx, state_hand_right[:, dim], label=f'dim {dim}', alpha=0.7)
    axes[ax_idx].set_xlabel('Time Step (idx)')
    axes[ax_idx].set_ylabel('Value')
    axes[ax_idx].set_title('state_hand_right (all dimensions)')
    axes[ax_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[ax_idx].grid(True, alpha=0.3)
    
    # Plot state_neck (all dimensions in one subplot) - only if visualize_neck is True and data exists
    if visualize_neck and has_neck_data:
        ax_idx += 1
        for dim in range(state_neck.shape[1]):
            axes[ax_idx].plot(idx, state_neck[:, dim], label=f'dim {dim}', alpha=0.7)
        axes[ax_idx].set_xlabel('Time Step (idx)')
        axes[ax_idx].set_ylabel('Value')
        axes[ax_idx].set_title('state_neck (all dimensions)')
        axes[ax_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[ax_idx].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    # print(f"Saved state trajectories to {output_path}")


def plot_state_body_first5(trajectories: Dict[str, np.ndarray], 
                           episode_name: str, 
                           output_path: Path) -> None:
    """
    Plot the first 5 dimensions of state_body in separate subplots.
    
    Args:
        trajectories: Dictionary containing trajectory data
        episode_name: Name of the episode for the title
        output_path: Path where the figure will be saved
    """
    idx = trajectories['idx']
    state_body = trajectories['state_body']
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(12, 16))
    
    fig.suptitle(f'State Body First 5 Dimensions - {episode_name}', fontsize=16, y=0.995)
    
    # Plot each of the first 5 dimensions in separate subplots
    for dim in range(5):
        axes[dim].plot(idx, state_body[:, dim], linewidth=2, color=f'C{dim}')
        axes[dim].set_xlabel('Time Step (idx)')
        axes[dim].set_ylabel('Value')
        axes[dim].set_title(f'state_body dimension {dim}')
        axes[dim].set_ylim(-0.4, 0.8)  # Set y-axis range
        axes[dim].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    # print(f"Saved state_body first 5 dimensions to {output_path}")


def plot_action_trajectories(trajectories: Dict[str, np.ndarray], 
                             episode_name: str, 
                             output_path: Path,
                             visualize_neck: bool = False) -> None:
    """
    Plot all action trajectories (action_body, action_hand_left, action_hand_right, and optionally action_neck).
    
    Args:
        trajectories: Dictionary containing trajectory data
        episode_name: Name of the episode for the title
        output_path: Path where the figure will be saved
        visualize_neck: Whether to include action_neck in the visualization (default: False)
    """
    idx = trajectories['idx']
    action_body = trajectories['action_body']
    action_hand_left = trajectories['action_hand_left']
    action_hand_right = trajectories['action_hand_right']
    
    # Check if neck data exists in trajectories
    has_neck_data = 'action_neck' in trajectories
    action_neck = trajectories.get('action_neck', None)
    
    # Calculate number of subplots needed for action_body
    body_dim = action_body.shape[1]
    num_body_plots = (body_dim + 9) // 10  # Ceiling division
    
    # Total subplots: body_plots + 2 (left_hand, right_hand) + 1 (neck if enabled and available)
    total_plots = num_body_plots + 2 + (1 if (visualize_neck and has_neck_data) else 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 4 * total_plots))
    if total_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'Action Trajectories - {episode_name}', fontsize=16, y=0.995)
    
    # Plot action_body (every 10 dimensions in one subplot)
    for i in range(num_body_plots):
        start_dim = i * 10
        end_dim = min((i + 1) * 10, body_dim)
        
        for dim in range(start_dim, end_dim):
            axes[i].plot(idx, action_body[:, dim], label=f'dim {dim}', alpha=0.7)
        
        axes[i].set_xlabel('Time Step (idx)')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'action_body [dim {start_dim}:{end_dim}]')
        axes[i].set_ylim(-0.3, 0.9)  # Set y-axis range for action_body
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Plot action_hand_left (all dimensions in one subplot)
    ax_idx = num_body_plots
    for dim in range(action_hand_left.shape[1]):
        axes[ax_idx].plot(idx, action_hand_left[:, dim], label=f'dim {dim}', alpha=0.7)
    axes[ax_idx].set_xlabel('Time Step (idx)')
    axes[ax_idx].set_ylabel('Value')
    axes[ax_idx].set_title('action_hand_left (all dimensions)')
    axes[ax_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[ax_idx].grid(True, alpha=0.3)
    
    # Plot action_hand_right (all dimensions in one subplot)
    ax_idx += 1
    for dim in range(action_hand_right.shape[1]):
        axes[ax_idx].plot(idx, action_hand_right[:, dim], label=f'dim {dim}', alpha=0.7)
    axes[ax_idx].set_xlabel('Time Step (idx)')
    axes[ax_idx].set_ylabel('Value')
    axes[ax_idx].set_title('action_hand_right (all dimensions)')
    axes[ax_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[ax_idx].grid(True, alpha=0.3)
    
    # Plot action_neck (all dimensions in one subplot) - only if visualize_neck is True and data exists
    if visualize_neck and has_neck_data:
        ax_idx += 1
        for dim in range(action_neck.shape[1]):
            axes[ax_idx].plot(idx, action_neck[:, dim], label=f'dim {dim}', alpha=0.7)
        axes[ax_idx].set_xlabel('Time Step (idx)')
        axes[ax_idx].set_ylabel('Value')
        axes[ax_idx].set_title('action_neck (all dimensions)')
        axes[ax_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[ax_idx].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    # print(f"Saved action trajectories to {output_path}")


def main():
    """Main function to run the visualization pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Visualize all trajectories in a dataset folder"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/hyd/codebase/openpi/data/20260126_1624",
        help="Path to the data root directory containing episode folders"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/hyd/codebase/openpi/data/visualization/20260126_1624",
        help="Path to save the visualization outputs"
    )
    parser.add_argument(
        "--visualize_neck",
        action="store_true",
        default=False,
        help="Whether to visualize state_neck and action_neck (default: False)"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    visualize_neck = args.visualize_neck
    
    # Create output directory if it doesn't exist
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get all episode directories
    episode_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    
    print(f"Found {len(episode_dirs)} episodes in total")
    print(f"Visualize neck: {visualize_neck}")
    print(f"Output directory: {output_root}")
    
    # Process all episodes
    for episode_dir in episode_dirs:
        episode_name = episode_dir.name
        print(f"\nProcessing {episode_name}...")
        
        # Load episode data
        try:
            episode_data = load_episode_data(episode_dir)
        except Exception as e:
            print(f"Error loading {episode_name}: {e}")
            continue
        
        # Extract trajectories
        trajectories = extract_trajectories(episode_data)
        
        # Analyze action_body to find maximum value
        action_body = trajectories['action_body']
        max_value = np.max(action_body)
        max_position = np.unravel_index(np.argmax(action_body), action_body.shape)
        max_timestep = max_position[0]
        max_dimension = max_position[1]
        
        print(f"  action_body max value: {max_value:.6f}")
        print(f"  - Dimension: {max_dimension} (out of {action_body.shape[1]} dimensions)")
        print(f"  - Timestep: {max_timestep} (out of {action_body.shape[0]} timesteps)")
        
        # Plot and save state trajectories
        state_output_path = output_root / f"{episode_name}_state.png"
        plot_state_trajectories(trajectories, episode_name, state_output_path, visualize_neck)
        
        # Plot and save action trajectories
        action_output_path = output_root / f"{episode_name}_action.png"
        plot_action_trajectories(trajectories, episode_name, action_output_path, visualize_neck)
        
        # Plot and save first 5 dimensions of state_body
        state_body_first5_path = output_root / f"{episode_name}_state_body_first5.png"
        plot_state_body_first5(trajectories, episode_name, state_body_first5_path)
    
    print(f"\n✓ Visualization complete! All figures saved to {output_root}")


if __name__ == "__main__":
    main()

