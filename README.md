# Dodo Sim-to-Sim Deployment

## Overview
This project deploys a walking policy trained in **Isaac Lab** to **Genesis** through **ROS 2** for sim-to-sim validation on the Dodo biped robot.

The current pipeline is:

**Isaac Lab policy (TorchScript) → ROS 2 → Genesis**

The goal is to verify whether the trained walking policy can be transferred to another simulator while preserving basic locomotion behavior.

---

## Main Files

- `src/genesis_bridge.py`  
  Runs the Genesis simulation, loads the robot model, publishes robot states, and executes policy actions.

- `src/policy_node.py`  
  Loads the exported TorchScript actor, reconstructs the policy observation, and publishes actions through ROS 2.

- `requirements.txt`  
  Contains the Conda/Python environment dependencies.

---

## Environment Setup
Create and activate the environment:

```bash
conda create -n dodo_sim2sim python=3.10 -y
conda activate dodo_sim2sim
pip install -r requirements.txt
```

Then source ROS 2:

```bash
source /opt/ros/humble/setup.bash
```

If you use a local ROS workspace, also source:

```bash
source install/setup.bash
```

---

## Before Running
Please check the following paths in the code:

- the URDF path in `src/genesis_bridge.py`
- the TorchScript policy path in `src/policy_node.py`

Make sure they match your local setup.

---

## Run
Open two terminals.

### Terminal 1: start Genesis bridge
```bash
conda activate dodo_sim2sim
source /opt/ros/humble/setup.bash
python src/genesis_bridge.py
```

### Terminal 2: start policy node
```bash
conda activate dodo_sim2sim
source /opt/ros/humble/setup.bash
python src/policy_node.py
```

### Optional: send a command
You can send a velocity command through ROS 2:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

---

## Current Status
The current system is already runnable.

At this stage, it can:
- load the Dodo robot in Genesis,
- connect the trained policy,
- publish robot states through ROS 2,
- execute policy actions in Genesis,
- and produce preliminary walking behavior.

However, the walking behavior is **not yet fully stable**.

---

## Next Steps
The next stage will focus on:
- improving frame alignment between Isaac Lab and Genesis,
- validating the policy under fixed command settings,
- improving transfer robustness,
- and preparing for future sim-to-real deployment.
