# Robot Data Collection


**Attention:** Teleoperation can be dangerous. Always ensure the robot is in a safe environment and start with low gains to prevent any potential damage or injury. Before running the data collection script, you are suggested to first teleop in simulation and then teleop in the real world following the instructions in [vr_wholebody_teleop](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html) to get familiar with the controls.

## setup the hardware on g1.
```bash
### g1 side, ssh into the g1's onboard computer and run the hardware setup script to initialize the cameras, grippers, and VR system
cd path/to/hardware4OpenHLM
uv run hardware_setup

### remember to fill in the sudo password in the gripper pane
```

## Start the data collection script on the PC side.

Requires tmux to be installed (sudo apt install tmux).

1. First change the `task_name` and `task_desc` in `data_record.sh` to specify the data folder name and the task description. Then run the script:

```bash
### PC side
cd src/GR00T-WholeBodyControl4OpenHLM/scripts
bash launch_data_collection.sh
```

The launcher splits the OpenHLM window into
four tiled panes:

| Pane | Script | Description |
|---|---|---|
| Pane 0 | `deploy_stream.sh`(upper left) | Starts the real-robot deployment stack through `gear_sonic_deploy/deploy.sh real --input-type zmq`. This is the robot-side control process that consumes the ZMQ motion stream and drives the G1. |
| Pane 1 | `pico_stream_pure.sh`(upper right) | Activates `.venv_teleop` and runs `gear_sonic/scripts/pico_thread.py --visualize --visualize_human_motion`. `pico_thread.py` is a simplified version of `pico_manager_thread_server.py` that sends human-motion commands after `GMR` retargeting. |
| Pane 2 | `data_record.sh`(lower left) | Activates `.venv_teleop` and runs `gear_sonic/scripts/run_openhlm_data_record.py`.|
| Pane 3 | `run_camera_teleop.sh`(lower right) | Starts `OrinVideoSender_SV1_realsense`, and then forwards SV1 and RealSense camera streams to `tcp://192.168.123.164:5555` and `tcp://192.168.123.164:5554`. |

We threw away the state machine in the original code and retained only the joint_based teleoperation in our `pico_thread.py`. You don't need to worry about the mode switching(like "A+X" in the original `pico_manager_thread_server.py`) and just teleop in the whole-body teleoperation mode.

![image](../../assets/pico_screenshot.png)

2. Then you need to connect the VR headset. Follow the instructions in [vr_teleop_setup](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_teleop_setup.html) to set up the VR system and connect the headset. After this documentation, you should be able to see a mujoco simulation view on your PC to prview the retargeting results.\
3. Then you need to listen to the camera streams. You can click  "`Listen`"  in the upper right corner of the ui, and enter the ip as displayed in the left side `PC Service: 192.168.x.x` to connect to the camera streams. After this, you should be able to see the camera streams in the ui. You can press the "B" button on your PICO controller to switch the presentation form of the picture
4. Then confirm your description in the `data_record` pane. Confirm the security alerts in `deploy_stream` pane. When everything is ready, you can press "`]`" in `deploy_stream` pane to start the robot to stand on the ground. 
5. **Stand in calibration pose** — Upright, feet together, arms in down. Then press "`enter`" to start teleoperation.

### recording controls
| Input | Action |
|---|---|
| **Left Grip + A** | **Toggle** recording — starts a new episode, or stops and saves the current one |
| **Left Grip + B** | **Discard** the current episode without saving |

### Emergency stop methods
**Keyboard (deployment terminal):**
- Press **`O`** in the `deploy_stream` pane for immediate stop

**PICO controllers:**
- Press **A + B + X + Y** simultaneously

Both methods immediately halt the policy and exit control mode.