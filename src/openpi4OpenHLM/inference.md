# Robot Inference

## Test in SIM

before testing on the real robot, you can first test the inference pipeline in simulation to ensure everything is working.

### Policy Server

Start a policy server for remote inference:

```bash
# Use a trained checkpoint
cd src/openpi4OpenHLM
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
--env SONICG1 \
--num-steps 10 \
policy:checkpoint \
--policy.config=openhlm_example \
--policy.dir=path/to/trained/checkpoint 
```

**On the PC/client side:**

```bash
### terminal 1: launch virtual robot in MuJoCo Simulator
cd src/GR00T-WholeBodyControl4OpenHLM
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

```bash
### terminal 2: start the low-level control server
cd src/GR00T-WholeBodyControl4OpenHLM/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq sim

```

In Terminal 2 (deploy.sh), press "`]`" to start the policy.\
Then press "`enter`" to streaming mode.\
In Terminal 1, press 9 to drop the robot to the ground.

```bash
### terminal 3: start the OpenPI deployment client
cd src/GR00T-WholeBodyControl4OpenHLM
source .venv_teleop/bin/activate
python scripts/openpi-eval/main.py \
--control_hz 30 \
--max_steps 10000 \
--save_video \
--instruction "example" \
--exp_name openhlm_example
```
After robot raises into the initial pose, press "`s`" to start the inference and control loop.\
Press "Ctrl + C" in terminal 3 to stop the high-level policy and recover to the initial pose.\
Press "O" in terminal 2 to stop the low-level control server.

## Inference in REAL

### Policy Server

Start a policy server for remote inference:

```bash
# Use a trained checkpoint
cd src/openpi4OpenHLM
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
--env SONICG1 \
--num-steps 10 \
policy:checkpoint \
--policy.config=openhlm_example \
--policy.dir=path/to/trained/checkpoint 
```

### Robot Inference

The deployment client connects to the OpenPI policy server via websocket for action inference and controls the G1 robot via the GR00T WBC framework.

**setup the hardware on g1.**

```bash
### g1 side, ssh into the g1's onboard computer and run the hardware setup script to initialize the cameras, grippers, and VR system
cd path/to/hardware4OpenHLM
uv run hardware_setup

### remember to fill in the sudo password in the gripper pane
```

**On the PC/client side:**

```bash
### terminal 1: start the low-level control server
cd src/GR00T-WholeBodyControl4OpenHLM/scripts
bash deploy_stream.sh
```

Confirm the security alerts in `deploy_stream` pane.\
When everything is ready, you can press "`]`" in `deploy_stream` pane to start the robot to stand on the ground.\
Then press "`enter`" to streaming mode.

```bash
### terminal 2: start the OpenPI deployment client
cd src/GR00T-WholeBodyControl4OpenHLM/scripts
python openpi-eval/main.py \
--control_hz 30 \
--max_steps 10000 \
--save_video \
--instruction "example" \
--exp_name openhlm_example
```

After robot raises into the initial pose, press "`s`" to start the inference and control loop.\
Press "Ctrl + C" in terminal 2 to stop the high-level policy and recover to the initial pose.\
Press "O" in terminal 1 to stop the low-level control server.