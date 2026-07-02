# HTC Interface

This package contains scripts and utilities to interface with HTC Vive trackers on Windows for the HuMI data collection project.

## Installation

This package is designed to be installed on a Windows machine.

### SteamVR Configuration

To enable "headless" mode (running SteamVR without a Head-Mounted Display), edit the `steamvr.vrsettings` file (typically located at `C:\Program Files (x86)\Steam\config\steamvr.vrsettings`). Add or merge the following keys into the `"steamvr"` section:

```json
{
  "steamvr": {
    "requireHmd": false,
    "activateMultipleDrivers": true
  }
}
```

### Setup

Install the package directly using `pip`:

```powershell
pip install .
```

If you are developing this package, you can install it in editable mode:

```powershell
pip install -e .
```

## Scripts

This package provides the following console scripts:

- `record-pose`: Records transformed HTC tracker poses to per-episode JSON files. It initializes OpenVR, reads tracker poses, transforms them into the robot frame, and can optionally serve an RPC endpoint for remote control.
- `send-pose`: A utility script for sending live tracker poses to a ZMQ socket (primarily used for debugging or separate streaming).

## Usage

Connect your HTC Vive trackers to the Vive hub and calibrate the tracking space according to the on-screen prompts. Once the setup is complete, you can start recording tracker poses.

Example usage for `record-pose`:

```powershell
record-pose --rpc.serve --output-dir data/my-tracker-recordings --config-path tracker_config.json
```

This will start recording and serve an RPC endpoint on the default port (4242) to allow a remote Linux/macOS machine to control the recording state (start, stop, delete).
