# Sim-to-Sim Deployment for Dodo in Genesis

## Overview

This project deploys a walking policy trained in **Isaac Lab** to the **Genesis** simulator through **ROS 2**, forming a complete sim-to-sim validation pipeline for the Dodo biped robot.

The deployment pipeline is:

**Isaac Lab policy (TorchScript) → ROS 2 → Genesis simulator**

The main goal is to verify whether a walking policy trained in Isaac Lab can be transferred to another simulation environment while preserving control semantics, observation consistency, and basic locomotion behavior.

At the current stage, the system is already able to:

* load the Dodo robot in Genesis,
* publish robot states through ROS 2,
* run the exported TorchScript actor online,
* convert policy outputs into joint position targets,
* execute the policy in Genesis with implicit PD control,
* and produce preliminary walking behavior.

The current status can be summarized as **runnable but not yet fully stable**.

---

## Repository Structure

```text
src/
├── genesis_bridge.py   # Genesis-side simulation bridge
└── policy_node.py      # ROS2 policy inference node

requirements.txt        # Conda/pip environment dependencies
```

---

## System Architecture

The sim-to-sim system consists of two main ROS 2 nodes.

### 1. `genesis_bridge.py`

This node runs on the Genesis side and is responsible for:

* loading the robot from URDF,
* building the Genesis simulation scene,
* configuring rigid-body dynamics and contact properties,
* applying joint PD gains and force limits,
* aligning joint ordering between training and Genesis,
* publishing robot states,
* receiving policy actions,
* and executing joint position targets in Genesis.

### 2. `policy_node.py`

This node runs the trained actor network and is responsible for:

* subscribing to robot state topics,
* reconstructing the policy observation vector,
* running TorchScript inference,
* publishing the predicted action back to Genesis,
* and optionally handling commanded velocity input from `/cmd_vel`.

---

## Control Semantics

A key requirement of this project is that the deployment-side control semantics must remain consistent with those used during training.

In Isaac Lab, the walking policy was trained with:

* **joint position actions**, not torque actions,
* action range in **[-1, 1]**, and
* an **implicit PD controller** as the low-level actuator interface.

Therefore, in Genesis the action is interpreted as a **joint position offset**:

```math
q_des = q_default + scale * action
```

where:

* `q_default` is the default standing joint pose,
* `scale` is the action scaling factor,
* and `action` is the policy output after tanh clipping.

This is implemented in `genesis_bridge.py`, and the resulting target positions are sent to Genesis through `control_dofs_position(...)`.

This alignment is essential. Even if the model loads correctly, the policy will not behave as expected if the deployed action no longer has the same meaning as in training.

---

## Joint Order Alignment

The policy was trained using the following **fixed joint order**:

```text
[left_joint_1, left_joint_2, left_joint_3, left_joint_4,
 right_joint_1, right_joint_2, right_joint_3, right_joint_4]
```

However, Genesis may internally store the local DOF indices in a different order. Therefore, `genesis_bridge.py` explicitly defines a mapping from the **training joint order** to the **Genesis local DOF indices**:

```python
self.joint_dofs_idx_local = np.array([7, 9, 11, 13, 6, 8, 10, 12], dtype=np.int32)
```

This mapping ensures that:

* published joint states follow the same semantic ordering as in training,
* action targets are sent to the correct physical joints,
* and the observation-action loop remains consistent.

If the URDF or Genesis model changes, this mapping must be re-verified.

---

## Observation Construction

The deployed policy reconstructs the observation vector from ROS topics.

The observation currently used in `policy_node.py` is:

* base height `(1)`
* base linear velocity `(3)`
* base angular velocity `(3)`
* projected gravity `(3)`
* velocity command `(3)`
* joint positions `(8)`
* joint velocities `(8)`
* previous actions `(8)`

Total observation dimension:

```text
1 + 3 + 3 + 3 + 3 + 8 + 8 + 8 = 37
```

This matches the inferred actor input dimension in the script.

### Important Note

The comments in earlier design discussions distinguish between **policy observations** and **critic observations**. During deployment, only the **policy-side observation** should be reconstructed. The critic is only used during training and is not needed at inference time.

---

## Quaternion and Frame Handling

One of the main difficulties in sim-to-sim transfer is frame consistency.

To address this, `genesis_bridge.py` and `policy_node.py` both include logic for:

* quaternion normalization,
* automatic quaternion convention selection,
* body/world rotation conversion,
* projected gravity reconstruction,
* and body-frame velocity interpretation.

The bridge automatically tests candidate quaternion conventions and selects the one whose projected gravity is closest to `[0, 0, -1]` under upright standing.

This is necessary because simulator APIs may differ in:

* quaternion storage order (`xyzw` vs `wxyz`),
* whether the quaternion maps body-to-world or world-to-body,
* and body-frame vs world-frame velocity definitions.

If locomotion appears unstable or rotated incorrectly, frame alignment should be checked first.

---

## ROS 2 Topics

### Published by `genesis_bridge.py`

#### `/dodo/joint_states`

Type: `sensor_msgs/msg/JointState`

Publishes:

* joint names in training order,
* joint positions,
* joint velocities.

#### `/dodo/base_state`

Type: `std_msgs/msg/Float32MultiArray`

Format:

```text
[z, v_body_x, v_body_y, v_body_z, w_body_x, w_body_y, w_body_z]
```

Publishes:

* base height,
* body-frame linear velocity,
* body-frame angular velocity.

#### `/dodo/imu`

Type: `sensor_msgs/msg/Imu`

Publishes:

* orientation quaternion,
* body-frame angular velocity.

### Subscribed by `genesis_bridge.py`

#### `/dodo/action`

Type: `std_msgs/msg/Float32MultiArray`

Receives the policy output action and converts it to joint position targets.

---

### Published by `policy_node.py`

#### `/dodo/action`

Type: `std_msgs/msg/Float32MultiArray`

Publishes the actor output after tanh clipping.

### Subscribed by `policy_node.py`

#### `/dodo/joint_states`

Type: `sensor_msgs/msg/JointState`

Used to reconstruct joint positions and velocities.

#### `/dodo/base_state`

Type: `std_msgs/msg/Float32MultiArray`

Used to obtain base height, linear velocity, and angular velocity.

#### `/dodo/imu`

Type: `sensor_msgs/msg/Imu`

Used to reconstruct projected gravity.

#### `/cmd_vel`

Type: `geometry_msgs/msg/Twist`

Optional external command input. If `/cmd_vel` is not updated recently, the script falls back to internal command sampling.

---

## Simulation and Control Parameters

The main parameters currently used in `genesis_bridge.py` are:

### Genesis simulation

* simulation timestep: `dt = 1 / 120.0`
* substeps: `4`
* gravity: `(0.0, 0.0, -9.81)`
* solver: Newton constraint solver
* iterations: `50`
* tolerance: `1e-4`

### Contact / rigid body material

* ground friction: `1.0`
* robot friction: `1.0`
* restitution: `0.0`

### Joint control

* `kp = 42.0`
* `kv = 2.5`
* force limit: `[-6.0, 6.0]`
* action scale: `0.5`

### Default standing pose

```text
[0.0, -0.3, 0.90, -0.65,
 0.0, -0.3, 0.90, -0.65]
```

### Action smoothing and safety logic

* initial ramp-in duration: `1.0 s`
* target position low-pass filter: `qdes_lpf = 0.2`
* action timeout fallback: if no action is received for `0.25 s`, return toward default pose

These mechanisms are important because they help avoid immediate instability when the policy first connects or when the action stream is interrupted.

---

## Command Handling

`policy_node.py` supports two ways of generating commands.

### 1. External velocity command through `/cmd_vel`

If a recent `/cmd_vel` message is available, the node directly uses:

* linear x velocity,
* linear y velocity,
* angular z velocity.

### 2. Internal command sampling

If no recent `/cmd_vel` input is received, the node samples commands internally according to the configured ranges.

Current values in the script are:

* `resample_T = 10.0`
* `rel_standing = 0.02`
* `lin_x_range = (0.5, 0.5)`
* `lin_y_range = (0.0, 0.0)`
* `ang_z_range = (-1.0, 1.0)`

These values should be checked against the exact training configuration if strict alignment is required.

---

## Environment Setup

This project uses a Conda environment defined by:

```text
requirements.txt
```

### Create environment

A typical setup workflow is:

```bash
conda create -n dodo_sim2sim python=3.10 -y
conda activate dodo_sim2sim
pip install -r requirements.txt
```

If your `requirements.txt` is Conda-style rather than pip-style, use:

```bash
conda create -n dodo_sim2sim --file requirements.txt -y
conda activate dodo_sim2sim
```

### Additional requirements

You also need:

* **ROS 2** installed and sourced,
* **Genesis** correctly installed,
* GPU support if running with `gs.gpu`,
* and a valid URDF file for the Dodo robot.

Before running, make sure your ROS 2 environment is sourced, for example:

```bash
source /opt/ros/humble/setup.bash
```

If you are using a workspace overlay, also source:

```bash
source install/setup.bash
```

---

## Paths to Update

Before running the project, check the hard-coded paths in the scripts.

### In `genesis_bridge.py`

Update:

* `urdf_path`

Example:

```python
urdf_path = "/path/to/dodobot_v3_simple.urdf"
```

### In `policy_node.py`

Update:

* `self.model_path`

Example:

```python
self.model_path = "/path/to/policy_actor_ts.pt"
```

If these paths are incorrect, the system will fail to load the robot or actor.

---

## Running the Pipeline

Open two terminals.

### Terminal 1: start the Genesis bridge

```bash
conda activate dodo_sim2sim
source /opt/ros/humble/setup.bash
python src/genesis_bridge.py
```

This launches Genesis, loads the robot, initializes the controller, and starts publishing robot states.

### Terminal 2: start the policy node

```bash
conda activate dodo_sim2sim
source /opt/ros/humble/setup.bash
python src/policy_node.py
```

This loads the TorchScript policy, subscribes to the robot state topics, reconstructs the observation, and publishes actions.

### Optional: send velocity commands

You can publish commands manually through ROS 2:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

---

## Expected Runtime Behavior

When the system is running correctly:

* the Genesis window should show the robot standing in the default pose,
* `/dodo/joint_states`, `/dodo/base_state`, and `/dodo/imu` should be published continuously,
* the policy node should report the inferred observation dimension,
* the bridge should print the selected quaternion convention,
* actions should begin ramping in when the policy connects,
* and the robot should start showing policy-driven motion.

At the current stage, stable standing and preliminary walking are expected, but fully robust walking is not yet guaranteed.

---

## Common Issues and Debugging

### 1. Joint order mismatch

Symptoms:

* legs move incorrectly,
* robot falls immediately,
* action semantics look swapped.

Check:

* URDF movable joint order,
* `joint_names_expected`,
* `joint_dofs_idx_local`,
* and the published `JointState.name` ordering.

### 2. Observation dimension mismatch

Symptoms:

* `Obs dim mismatch` error in `policy_node.py`.

Check:

* the actor input dimension,
* the observation terms concatenated in deployment,
* whether the deployed observation matches the training observation exactly.

### 3. Quaternion / frame inconsistency

Symptoms:

* projected gravity is incorrect,
* robot orientation appears flipped,
* velocity signs are wrong,
* turning behavior is unstable.

Check:

* quaternion convention (`xyzw` vs `wxyz`),
* body-to-world vs world-to-body interpretation,
* body-frame conversion for linear and angular velocities.

### 4. Policy outputs saturate

Symptoms:

* many actions close to `±1`,
* unstable or jerky motion.

Check:

* action scale,
* observation normalization consistency,
* command range alignment,
* whether the actor was exported from the correct checkpoint.

### 5. Robot returns to default pose repeatedly

Symptoms:

* policy seems connected, but robot keeps relaxing toward standing pose.

Check:

* whether `/dodo/action` is being published continuously,
* whether the action timeout of `0.25 s` is being triggered,
* whether the policy node is actually receiving valid state messages.

---

## Current Status

The current implementation already supports a complete sim-to-sim loop:

* simulator bridge,
* state publishing,
* policy inference,
* and action execution.

The robot can stand stably in Genesis, and basic walking behavior has been observed. However, the transferred locomotion is still not fully stable, and frame alignment remains one of the main issues to refine.

In short, the system is **functional but still under validation and tuning**.

---

## Future Work

Planned next steps include:

* fully aligning body-frame and velocity-frame definitions between Isaac Lab and Genesis,
* verifying deployment under fixed command distributions,
* introducing domain randomization,
* refining actuator and contact parameter matching,
* and preparing the interface for future sim-to-real deployment.

---

## Notes

* This README documents the current code behavior based on `src/genesis_bridge.py` and `src/policy_node.py`.
* If the training configuration changes, the deployment-side observation, command generation, or action semantics may also need to be updated.
* The most important principle in sim-to-sim transfer is **strict consistency** between training and deployment.
