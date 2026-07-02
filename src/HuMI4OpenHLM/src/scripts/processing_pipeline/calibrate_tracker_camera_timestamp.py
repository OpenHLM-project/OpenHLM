import json
import os
import warnings
from typing import Annotated, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from processing import mp4_get_start_datetime

RESAMPLE_FREQ_HZ = 60.0
HIGHPASS_CUTOFF_HZ = 0.25
HIGHPASS_ORDER = 3
LOW_CORRELATION_WARNING_THRESHOLD = 0.3


def calculate_tracker_angular_velocity(
    tracker_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate angular velocity from tracker pose data."""
    # Try different possible rotation column names
    quats = tracker_df[["q_x", "q_y", "q_z", "q_w"]].to_numpy()
    rotations = R.from_quat(quats)

    # Calculate angular velocity
    timestamps = np.asarray(tracker_df["timestamp"].values)
    dt = np.diff(timestamps)

    angular_velocities = []
    for i in range(len(rotations) - 1):
        # Calculate relative rotation
        rel_rot = rotations[i + 1] * rotations[i].inv()
        # Convert to axis-angle representation
        rotvec = rel_rot.as_rotvec()
        # Divide by time difference to get angular velocity
        angular_vel = rotvec / dt[i]
        angular_velocities.append(angular_vel)

    # Add one more entry (repeat last) to match original length
    angular_velocities.append(angular_velocities[-1])

    return np.array(angular_velocities), timestamps


def transform_gopro_gyro_to_tracker(
    gopro_gyro: np.ndarray, rot_tracker_camera: np.ndarray
) -> np.ndarray:
    """Transform GoPro gyroscope data to tracker coordinate system."""
    # gopro_gyro is Nx3 array of angular velocities
    # Apply rotation: omega_tracker = ROT_TRACKER_CAMERA @ omega_gopro
    return (rot_tracker_camera @ gopro_gyro.T).T


def resample_to_60hz(
    timestamps: Union[np.ndarray, List],
    data: np.ndarray,
    start_timestamp: float,
    end_timestamp: float,
    resample_freq: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample data to 60Hz using interpolation."""
    # Create 60Hz time grid
    assert (
        start_timestamp >= timestamps[0] and end_timestamp <= timestamps[-1]
    ), "Resample range must be within original timestamp range"
    target_timestamps = np.arange(
        start_timestamp, end_timestamp, 1.0 / resample_freq
    )

    # Interpolate each axis
    resampled_data = []
    for axis in range(data.shape[1]):
        interp_func = interp1d(
            timestamps,
            data[:, axis],
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        resampled_data.append(interp_func(target_timestamps))

    return target_timestamps, np.array(resampled_data).T


def preprocess_signal_for_correlation(
    signal_values: np.ndarray, sample_freq_hz: float
) -> np.ndarray:
    """High-pass filter and z-score a 1D signal for alignment."""
    if signal_values.ndim != 1:
        raise ValueError(
            f"Expected 1D signal for preprocessing, got {signal_values.shape}"
        )
    if len(signal_values) < 8:
        raise ValueError(
            "Signal is too short to preprocess for correlation."
        )

    b, a = signal.butter(
        HIGHPASS_ORDER,
        HIGHPASS_CUTOFF_HZ / (sample_freq_hz / 2.0),
        btype="high",
    )
    filtered = signal.filtfilt(b, a, signal_values)
    std = float(np.std(filtered))
    if std < 1e-8:
        raise ValueError(
            "Signal variance is too small after high-pass filtering."
        )
    return (filtered - float(np.mean(filtered))) / std


def find_optimal_time_offset(
    tracker_angular_vel: np.ndarray,
    gopro_angular_vel: np.ndarray,
    tracker_timestamps: Union[np.ndarray, List],
    gopro_timestamps: Union[np.ndarray, List],
    max_offset: float = 1.0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Find optimal time offset using high-pass + z-score correlation."""
    # Calculate magnitudes
    tracker_magnitude = np.linalg.norm(tracker_angular_vel, axis=1)
    gopro_magnitude = np.linalg.norm(gopro_angular_vel, axis=1)

    # Resample both to 60Hz
    resample_start = max(tracker_timestamps[0], gopro_timestamps[0])
    resample_end = min(tracker_timestamps[-1], gopro_timestamps[-1])
    if resample_end <= resample_start:
        raise ValueError(
            "Tracker and GoPro timestamps do not overlap before offset search."
        )
    tracker_timestamps, tracker_mag_60hz = resample_to_60hz(
        tracker_timestamps,
        tracker_magnitude.reshape(-1, 1),
        start_timestamp=resample_start,
        end_timestamp=resample_end,
        resample_freq=RESAMPLE_FREQ_HZ,
    )
    gopro_timestamps, gopro_mag_60hz = resample_to_60hz(
        gopro_timestamps,
        gopro_magnitude.reshape(-1, 1),
        start_timestamp=resample_start,
        end_timestamp=resample_end,
        resample_freq=RESAMPLE_FREQ_HZ,
    )

    tracker_mag_60hz = tracker_mag_60hz.flatten()
    gopro_mag_60hz = gopro_mag_60hz.flatten()
    tracker_preprocessed = preprocess_signal_for_correlation(
        tracker_mag_60hz, sample_freq_hz=RESAMPLE_FREQ_HZ
    )
    gopro_preprocessed = preprocess_signal_for_correlation(
        gopro_mag_60hz, sample_freq_hz=RESAMPLE_FREQ_HZ
    )

    # Calculate cross-correlation
    correlation = signal.correlate(
        tracker_preprocessed, gopro_preprocessed, mode="full"
    ) / len(tracker_preprocessed)

    # Calculate offset values in seconds
    dt = 1.0 / RESAMPLE_FREQ_HZ
    offset_indices = np.arange(-len(gopro_mag_60hz) + 1, len(tracker_mag_60hz))
    offset_seconds = offset_indices * dt

    # Limit search to max_offset range
    valid_mask = np.abs(offset_seconds) <= max_offset
    valid_correlations = correlation[valid_mask]
    valid_offsets = offset_seconds[valid_mask]

    # Find peak correlation
    peak_idx = np.argmax(valid_correlations)
    optimal_offset = valid_offsets[peak_idx]
    peak_correlation = valid_correlations[peak_idx]

    return optimal_offset, peak_correlation, valid_offsets, valid_correlations


def visualize_alignment(
    tracker_angular_vel: np.ndarray,
    gopro_angular_vel: np.ndarray,
    tracker_timestamps: Union[np.ndarray, List],
    gopro_timestamps: Union[np.ndarray, List],
    time_offset: float,
    correlation_offsets: np.ndarray,
    correlations: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Visualize the alignment result and save to file."""
    # Calculate magnitudes
    tracker_magnitude = np.linalg.norm(tracker_angular_vel, axis=1)
    gopro_magnitude = np.linalg.norm(gopro_angular_vel, axis=1)

    # Convert timestamps to seconds for plotting
    start_timestamp = min(tracker_timestamps[0], gopro_timestamps[0])
    tracker_time_sec = tracker_timestamps - start_timestamp
    gopro_time_sec = gopro_timestamps - start_timestamp

    # Apply time offset to GoPro data
    gopro_time_aligned = gopro_time_sec + time_offset

    plt.figure(figsize=(12, 8))

    # Plot angular velocity magnitudes
    plt.subplot(2, 1, 1)
    plt.plot(
        tracker_time_sec,
        tracker_magnitude,
        label="Tracker Angular Velocity",
        alpha=0.7,
    )
    plt.plot(
        gopro_time_aligned,
        gopro_magnitude,
        label=f"GoPro Angular Velocity (offset: {time_offset:.3f}s)",
        alpha=0.7,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity Magnitude (rad/s)")
    plt.title("Angular Velocity Magnitude Alignment")
    plt.legend()
    plt.grid(True)

    # Plot correlation curve within the searched offset range.
    plt.subplot(2, 1, 2)
    plt.plot(correlation_offsets, correlations, label="Correlation")
    plt.axvline(
        x=time_offset,
        color="tab:red",
        linestyle="--",
        label=f"Best offset: {time_offset:.3f}s",
    )
    plt.xlabel("Offset (s)")
    plt.ylabel("Correlation")
    plt.title("Correlation within Search Range")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    plt.close()


def main(
    video_path: Annotated[str, tyro.conf.arg(aliases=["-v"])],
    tracker_csv: Annotated[str, tyro.conf.arg(aliases=["-t"])],
    gyro_csv: Annotated[str, tyro.conf.arg(aliases=["-g"])],
    output: str,
    max_offset: float = 1.0,
    visualize: bool = False,
    gopro_timezone: str | None = None,
) -> None:
    """
    Calibrate time offset between tracker and camera using angular velocity correlation.
    Args:
        video_path: Path to the video file.
        tracker_csv: Path to the CSV file containing tracker trajectory data with timestamps.
        gyro_csv: Path to the CSV file containing GoPro gyroscope data with timestamps.
        output: Path to save the output JSON file containing calibration results.
        max_offset: Maximum time offset range (in seconds) to search for optimal alignment.
        visualize: If True, save visualization plots of the alignment results.
        gopro_timezone: Optional timezone for GoPro cameras (e.g., 'Asia/Shanghai' or '+08:00').
    """

    # Load video start datetime
    print(f"Loading video timestamp from: {video_path}")
    video_start_datetime = mp4_get_start_datetime(video_path)
    if gopro_timezone is not None:
        import re
        import zoneinfo
        from datetime import timedelta, timezone

        match = re.match(r"^([+-])(\d{2}):?(\d{2})$", gopro_timezone)
        if match:
            sign = 1 if match.group(1) == "+" else -1
            hours = int(match.group(2))
            minutes = int(match.group(3))
            tz = timezone(timedelta(minutes=sign * (hours * 60 + minutes)))
        else:
            tz = zoneinfo.ZoneInfo(gopro_timezone)

        video_start_datetime = video_start_datetime.replace(tzinfo=tz)

    print(f"Video start datetime: {video_start_datetime}")

    # Load tracker data
    print(f"Loading tracker data from: {tracker_csv}")
    tracker_df = pd.read_csv(tracker_csv)

    # Load GoPro gyroscope data
    print(f"Loading GoPro gyro data from: {gyro_csv}")
    gopro_df = pd.read_csv(gyro_csv)

    # Convert GoPro timestamps to datetime
    # Assume the CSV has a 'timestamp' column with seconds since video start
    if "timestamp" in gopro_df.columns:
        gopro_timestamps = (
            np.asarray(gopro_df["timestamp"].values)
            + video_start_datetime.timestamp()
        )
    else:
        raise ValueError("No timestamp column found in GoPro CSV")

    # Extract gyroscope data (assume columns gx, gy, gz or similar)
    gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
    gopro_gyro = np.asarray(gopro_df[gyro_cols].values)

    # Calculate tracker angular velocity
    print("Calculating tracker angular velocity...")
    tracker_angular_vel, tracker_timestamps = (
        calculate_tracker_angular_velocity(tracker_df)
    )

    # Transform GoPro gyroscope to tracker coordinate system
    print("Transforming GoPro gyroscope to tracker coordinates...")
    ROT_TRACKER_CAMERA = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    gopro_angular_vel_tracker = transform_gopro_gyro_to_tracker(
        gopro_gyro, ROT_TRACKER_CAMERA
    )

    # Find optimal time offset
    print(f"Finding optimal time offset (max offset: ±{max_offset}s)...")
    (
        optimal_offset,
        peak_correlation,
        correlation_offsets,
        correlations,
    ) = find_optimal_time_offset(
        tracker_angular_vel,
        gopro_angular_vel_tracker,
        tracker_timestamps,
        gopro_timestamps,
        max_offset,
    )

    print(f"Optimal time offset: {optimal_offset:.6f} seconds")
    print(f"Peak correlation: {peak_correlation:.6f}")
    if peak_correlation < LOW_CORRELATION_WARNING_THRESHOLD:
        warnings.warn(
            "Low peak correlation detected "
            f"({peak_correlation:.3f} < {LOW_CORRELATION_WARNING_THRESHOLD:.3f}) "
            f"for {video_path}. Timestamp calibration may be unreliable.",
            stacklevel=1,
        )

    # Save results
    results = {
        "video_path": str(video_path),
        "tracker_csv": str(tracker_csv),
        "gyro_csv": str(gyro_csv),
        "video_start_datetime": video_start_datetime.isoformat(),
        "optimal_time_offset_seconds": float(optimal_offset),
        "peak_correlation": float(peak_correlation),
        "low_correlation_warning_threshold": (
            LOW_CORRELATION_WARNING_THRESHOLD
        ),
        "preprocessing": {
            "resample_freq_hz": RESAMPLE_FREQ_HZ,
            "filter": "highpass",
            "highpass_cutoff_hz": HIGHPASS_CUTOFF_HZ,
            "highpass_order": HIGHPASS_ORDER,
            "normalize": "zscore",
        },
        "max_offset_range": max_offset,
        "rot_tracker_camera": ROT_TRACKER_CAMERA.tolist(),
    }

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output}")

    # Visualization
    if visualize:
        print("Saving visualization...")
        output_dir = (
            os.path.dirname(output) if os.path.dirname(output) else "."
        )
        output_name = os.path.splitext(os.path.basename(output))[0]
        viz_path = os.path.join(output_dir, f"{output_name}_alignment.png")
        visualize_alignment(
            tracker_angular_vel,
            gopro_angular_vel_tracker,
            tracker_timestamps,
            gopro_timestamps,
            optimal_offset,
            correlation_offsets=correlation_offsets,
            correlations=correlations,
            save_path=viz_path,
        )


if __name__ == "__main__":
    tyro.cli(main)
