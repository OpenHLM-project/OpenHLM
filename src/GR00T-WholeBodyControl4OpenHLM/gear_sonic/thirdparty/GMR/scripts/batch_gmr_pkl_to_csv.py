import argparse
import pickle
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    args = parser.parse_args()

    out_folder = os.path.join(args.folder, "csv")
    os.makedirs(out_folder, exist_ok=True)

    for i, file in enumerate(os.listdir(args.folder)):
        if file.endswith(".pkl"):
            with open(os.path.join(args.folder, file), "rb") as f:
                motion_data = pickle.load(f)
        else:
            continue

        dof_pos = motion_data["dof_pos"]
        frame_rate = motion_data["fps"]            
        motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
        motion[:, :3] = motion_data["root_pos"]
        motion[:, 3:7] = motion_data["root_rot"]
        motion[:, 7:] = dof_pos
        
        if frame_rate > 30:
            # downsample to 30 fps
            downsample_factor = frame_rate / 30.0
            indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
            old_length = motion.shape[0]
            motion = motion[indices]
            print(f"Downsampled from {old_length} to {motion.shape[0]} frames")
        

        np.savetxt(
            os.path.join(args.folder, "csv", file.replace(".pkl", ".csv")),
            motion,
            delimiter=",",
        )
        print(f"({i}/{len(os.listdir(args.folder))}) Saved to {os.path.join(args.folder, 'csv', file.replace('.pkl', '.csv'))}")
