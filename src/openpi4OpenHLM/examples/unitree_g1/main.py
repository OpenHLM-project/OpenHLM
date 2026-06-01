# flake8: noqa: E402
import sys
import os

# Add the project root to sys.path to allow imports from examples
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import contextlib
import dataclasses
import json
import signal
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from sonic_g1_env import SonicG1Env, DEFAULT_MIMIC_OBS_G1, DEFAULT_MIMIC_OBS_G1_RAISED

########################################################
# copy and paste the following code to SONIC codebase
########################################################

# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


@dataclasses.dataclass
class Args:
    # Network parameters
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    # Evaluation parameters
    max_steps: int = 1000
    # Fixed instruction as requested
    instruction: str = "Pick up the purple soft finger on the table and place it on the mouse pad."
    
    # Control parameters
    control_hz: int = 30
    # Policy action horizon is 50, so we can run open loop for up to 25 steps
    open_loop_horizon: int = 25

    # Robot network parameters (SonicG1Env)
    vision_server_address: str = "192.168.123.164"
    vision_server_port: int = 5555
    wrist_server_port: int = 5554
    action_zmq_port: int = 5556
    body_state_address: str = "192.168.123.222"
    body_state_port: int = 5557
    mock: bool = False
    num_frames_to_send: int = 2  # Number of frames to buffer before sending a single ZMQ message

    # Visualization
    opencv_visualize: bool = True

    # Video recording: save combined_image_bgr frames from each episode as an MP4 file.
    save_video: bool = False
    video_save_dir: str = "/data/eval_videos"  # Directory where episode videos are saved
    exp_name: str = "exp"  # Short experiment label appended to each video filename

    # Action chunk saving
    save_action_chunk: bool = False  # If True, save predicted action chunks to disk
    
    # Debug parameters
    use_fake_policy: bool = False
    use_default_pose_policy: bool = False
    debug: bool = False


class FakePolicyClient:
    def __init__(self, action_dim=34, chunk_size=50):
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def infer(self, request_data):
        time.sleep(0.1)  # Simulate 100ms inference time
        # Return bounded random actions for testing.
        # Shape: (action_horizon, action_dim)
        actions = np.random.uniform(-1.0, 1.0, size=(self.chunk_size, self.action_dim)).astype(np.float32)
        return {"actions": actions}


class DefaultPosePolicyClient:
    def __init__(self, action_dim=34, chunk_size=50):
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # Construct the default action vector based on DEFAULT_MIMIC_OBS_G1.
        # DEFAULT_MIMIC_OBS_G1 layout (32 dims):
        #   0-2:   root (3): roll, pitch, yaw angular velocity
        #   3-8:   left leg (6)
        #   9-14:  right leg (6)
        #   15-17: waist (3)
        #   18-24: left arm (7)
        #   25-31: right arm (7)
        mimic_obs = DEFAULT_MIMIC_OBS_G1

        self.default_action = np.zeros(action_dim, dtype=np.float32)

        # OpenPI action format (34 dims):
        # 0-6: left arm (7)
        self.default_action[0:7] = mimic_obs[18:25]
        # 7: left gripper (1) - default 1.0 (open)
        self.default_action[7] = 1.0
        # 8-14: right arm (7)
        self.default_action[8:15] = mimic_obs[25:32]
        # 15: right gripper (1) - default 1.0 (open)
        self.default_action[15] = 1.0
        # 16-21: left leg (6)
        self.default_action[16:22] = mimic_obs[3:9]
        # 22-27: right leg (6)
        self.default_action[22:28] = mimic_obs[9:15]
        # 28-30: waist (3)
        self.default_action[28:31] = mimic_obs[15:18]
        # 31-33: root (3): roll, pitch, yaw angular velocity
        self.default_action[31:34] = mimic_obs[0:3]

    def infer(self, request_data):
        time.sleep(0.1)  # Simulate 100ms inference time
        # Return a chunk of default actions
        return {"actions": np.tile(self.default_action, (self.chunk_size, 1))}


def main(args: Args):
    if args.debug:
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        print("\033[91mDebug server started on 127.0.0.1:5678\033[0m")
        print("\033[91mWaiting for debugger to attach...\033[0m")
        debugpy.wait_for_client()
        print("Debugger attached, continuing execution...")

    action_dim = 34
    state_dim = 34
    print(f"Using action_dim={action_dim}, state_dim={state_dim}")

    # Initialize the environment
    env = SonicG1Env(
        control_hz=args.control_hz,
        mock=args.mock,
        vision_server_address=args.vision_server_address,
        vision_server_port=args.vision_server_port,
        wrist_server_port=args.wrist_server_port,
        action_zmq_port=args.action_zmq_port,
        body_state_address=args.body_state_address,
        body_state_port=args.body_state_port,
        num_frames_to_send=args.num_frames_to_send,
    )
    print("Initialized SonicG1Env")

    print("\nResetting robot to default pose (arms down)...")
    obs = env.reset(default_pose=DEFAULT_MIMIC_OBS_G1)
    time.sleep(2.5)  # Wait for 2.5 seconds to ensure the robot is in the default pose
    print("Reset to arms down complete.")
    print("\nResetting robot to raised pose (arms up)...")
    obs = env.reset(default_pose=DEFAULT_MIMIC_OBS_G1_RAISED)
    time.sleep(2.5)  # Wait for 2.5 seconds to ensure the robot is in the raised pose
    print("Reset to arms up complete.")

    # Connect to the policy server
    # We assume a server is running with a G1-compatible policy
    if args.use_fake_policy:
        policy_client = FakePolicyClient(action_dim=action_dim)
        print("\033[93m\nUsing FakePolicyClient (simulated inference)\033[0m")
    elif args.use_default_pose_policy:
        policy_client = DefaultPosePolicyClient(action_dim=action_dim)
        print("\033[93m\nUsing DefaultPosePolicyClient (sending default pose)\033[0m")
    else:
        policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
        print(f"\033[93m\nConnected to policy server at {args.remote_host}:{args.remote_port}\033[0m")

    # warmup the policy server
    print("\nWarming up policy server...")
    start_time = time.time()
    for _ in range(10):
        result = policy_client.infer({
            "observation/head_image_left": image_tools.resize_with_pad(obs["head_image_left"], 224, 224),
            "observation/left_wrist_image": image_tools.resize_with_pad(obs["left_wrist_image"], 224, 224),
            "observation/right_wrist_image": image_tools.resize_with_pad(obs["right_wrist_image"], 224, 224),
            "observation/state": obs["state"],
            "prompt": args.instruction,
        })
    print(f"Policy server warmed up in {time.time() - start_time:.3f} seconds.")

    # Generate timestamp once for all episodes (used for saving action chunks)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M") if args.save_action_chunk else None

    episode_idx = 0
    while True:
        episode_idx += 1
        instruction = args.instruction

        # Variables for open-loop execution
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        last_executed_action = None
        
        # Variables for action chunk saving
        action_step_counter = 0
        action_chunk_records = []
        state_records = []

        # Video recording setup: one VideoWriter per episode
        video_writer = None
        video_path = None
        if args.save_video:
            episode_dt = datetime.now()
            video_dir = Path(args.video_save_dir) / episode_dt.strftime("%Y%m%d")
            video_dir.mkdir(parents=True, exist_ok=True)
            video_filename = f"{episode_dt.strftime('%H%M%S')}_{args.exp_name}.mp4"
            video_path = video_dir / video_filename
            display_size = 480
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, args.control_hz, (display_size * 3, display_size)
            )
            print(f"\nRecording video to: {video_path}")

        print(f"\nStarting evaluation with instruction: '\033[91m{instruction}\033[0m'")
        print(f"Control Frequency: {args.control_hz} Hz")
        print(f"Policy Action Horizon: {result['actions'].shape[0]} steps")
        print(f"Open Loop Horizon: {args.open_loop_horizon} steps")

        # Wait for 's' key press to start evaluation
        print("\nPress 's' to start the evaluation...")
        while True:
            key = input().strip().lower()
            if key == 's':
                print("Starting evaluation now! Press Ctrl+C to stop early.")
                break
            else:
                print("Invalid key. Please press 's' to start.")

        # Main evaluation loop
        pbar = tqdm.tqdm(range(args.max_steps))
        policy_inference_time = 0.0
        actual_control_hz = 0.0

        for step_idx in pbar:
            start_time = time.time()
            
            try:
                # Get the current observation
                obs = env.get_observation()
                
                # Record state if saving is enabled
                if args.save_action_chunk:
                    state_records.append({
                        "step_index": action_step_counter,
                        "state": obs["state"].tolist()  # Convert numpy to list for JSON
                    })

                # Build combined BGR frame when display or video recording is needed
                if args.opencv_visualize or args.save_video:
                    display_size = 480
                    head_image_resized = image_tools.resize_with_pad(obs["head_image_left"], display_size, display_size)
                    left_wrist_resized = image_tools.resize_with_pad(obs["left_wrist_image"], display_size, display_size)
                    right_wrist_resized = image_tools.resize_with_pad(obs["right_wrist_image"], display_size, display_size)
                    combined_image = np.concatenate([head_image_resized, left_wrist_resized, right_wrist_resized], axis=1)
                    combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

                    # Overlay instruction text at the bottom center of the frame
                    prompt_text = f"Prompt: {instruction}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    padding = 8
                    frame_h, frame_w = combined_image_bgr.shape[:2]
                    (text_w, text_h), baseline = cv2.getTextSize(prompt_text, font, font_scale, thickness)
                    text_x = (frame_w - text_w) // 2
                    text_y = frame_h - padding - baseline
                    # Draw a dark background rectangle for readability
                    cv2.rectangle(
                        combined_image_bgr,
                        (text_x - padding, text_y - text_h - padding),
                        (text_x + text_w + padding, text_y + baseline + padding),
                        (0, 0, 0),
                        cv2.FILLED,
                    )
                    cv2.putText(combined_image_bgr, prompt_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    if args.opencv_visualize:
                        cv2.imshow("Robot Cameras (Head | Left Wrist | Right Wrist)", combined_image_bgr)
                        cv2.waitKey(1)

                    if args.save_video and video_writer is not None:
                        video_writer.write(combined_image_bgr)

                # Check if we need to query the policy for a new chunk of actions
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    # Prepare data for the policy
                    # The keys must match what the policy input transform (G1Inputs) expects.
                    request_data = {
                        "observation/head_image_left": image_tools.resize_with_pad(obs["head_image_left"], 224, 224),
                        "observation/left_wrist_image": image_tools.resize_with_pad(obs["left_wrist_image"], 224, 224),
                        "observation/right_wrist_image": image_tools.resize_with_pad(obs["right_wrist_image"], 224, 224),
                        "observation/state": obs["state"],
                        "prompt": instruction,
                    }
                    if last_executed_action is not None:
                        request_data["observation/last_action"] = last_executed_action

                    # Query the policy server
                    # The server will handle resizing and normalization if configured in the policy config
                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    inference_start = time.time()
                    with prevent_keyboard_interrupt():
                        result = policy_client.infer(request_data)
                        pred_action_chunk = result["actions"]
                    policy_inference_time = time.time() - inference_start
                    
                    # Record action chunk if saving is enabled
                    if args.save_action_chunk:
                        action_chunk_records.append({
                            "step_index_before_inference": action_step_counter,
                            "action_chunk": pred_action_chunk.tolist()  # Convert numpy to list for JSON
                        })

                    # Reset counter for the new chunk
                    actions_from_chunk_completed = 0
                
                # Get the next action to execute from the chunk
                # pred_action_chunk shape: (action_horizon, action_dim)
                if pred_action_chunk is not None and actions_from_chunk_completed < len(pred_action_chunk):
                    action = pred_action_chunk[actions_from_chunk_completed]
                else:
                    print("Warning: Action chunk exhausted or invalid. Stopping.")
                    break

                actions_from_chunk_completed += 1
                last_executed_action = action.copy()

                # Send action to environment
                env.step(action)

                # Increment action step counter if saving is enabled
                if args.save_action_chunk:
                    action_step_counter += 1

                # Sleep to maintain control frequency
                elapsed_time = time.time() - start_time
                target_dt = 1.0 / args.control_hz
                if elapsed_time < target_dt:
                    time.sleep(target_dt - elapsed_time)
                
                # Calculate actual control frequency
                actual_loop_time = time.time() - start_time
                actual_control_hz = 1.0 / actual_loop_time if actual_loop_time > 0 else 0.0
                
                # Update tqdm progress bar with policy inference time and actual control Hz
                pbar.set_postfix({
                    "policy_infer_time": f"{policy_inference_time*1000:.1f}ms",
                    "actual_hz": f"{actual_control_hz:.1f}"
                })
            except KeyboardInterrupt:
                print("\n\nCtrl+C detected. Resetting robot to default pose...")
                if video_writer is not None:
                    video_writer.release()
                    print(f"Video saved to: {video_path}")
                    video_writer = None
                env.reset()
                time.sleep(2.0)
                print("Reset complete.")
                break

        print("Evaluation finished.")

        # Release video writer if still open (normal episode completion)
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_path}")
            video_writer = None

        # Save action chunk data if enabled
        if args.save_action_chunk and len(action_chunk_records) > 0:
            # Use session timestamp (same for all episodes)
            timestamp = session_timestamp
            
            # Create directory structure: visualization/raw_action_chunk/<timestamp>/
            save_dir = Path("visualization/raw_action_chunk") / timestamp
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data to save
            # Get action chunk shape from the first recorded chunk
            action_chunk_shape = None
            if len(action_chunk_records) > 0 and len(action_chunk_records[0]["action_chunk"]) > 0:
                action_chunk_shape = [
                    len(action_chunk_records[0]["action_chunk"]),  # action_horizon
                    len(action_chunk_records[0]["action_chunk"][0])  # action_dim
                ]
            
            save_data = {
                "episode_idx": episode_idx,
                "timestamp": timestamp,
                "total_steps": action_step_counter,
                "action_chunk_shape": action_chunk_shape,
                "action_chunks": action_chunk_records,
                "states": state_records
            }
            
            # Save to JSON file: episode_<episode_idx>.json
            save_path = save_dir / f"episode_{episode_idx}.json"
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Action chunk data saved to: {save_path}")
        
        if input("\nDo one more eval? (enter y or n) ").lower() != "y":
            break
        print("\nResetting robot to default pose...")
        env.reset()
        time.sleep(2.0)
        print("Reset complete.")

    # Final cleanup: reset robot to default pose before exiting
    print("\nProgram exiting. Resetting robot to default pose...")
    env.reset()
    time.sleep(2.0)
    print("Final reset complete.")
    
    if args.opencv_visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)