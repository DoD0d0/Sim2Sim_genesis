#!/usr/bin/env python3
import time
import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

import torch


def quat_xyzw_to_rotmat(q):
    """q=[x,y,z,w], R maps v_body -> v_world"""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)],
    ], dtype=np.float32)
    return R






class PolicyNode(Node):
    def __init__(self):
        super().__init__("dodo_policy_node")

        # ---- model ----
        self.model_path = "/home/rczh/workspace/isaac_dodo/trained_models/command_velocity/2026-01-10_21-15-43/policy_actor_ts.pt"
        self.actor = torch.jit.load(self.model_path, map_location="cpu")
        self.actor.eval()
        self.get_logger().info(f"Loaded actor: {self.model_path}")

        self.obs_dim = self._infer_obs_dim()
        self.get_logger().info(f"Inferred obs_dim = {self.obs_dim} (expect 37)")

        # ---- training-aligned command cfg ----
        self.resample_T = 10.0
        self.rel_standing = 0.02
        self.lin_x_range = (0.5, 0.5)
        self.lin_y_range = (0.0, 0.0)
        self.ang_z_range = (-1.0, 1.0)

        self.cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # [vx, vy, wz]
        self._last_cmdvel_time = time.time()
        self._next_resample_time = time.time()

        # ---- expected joint order (MUST match training) ----
        self.joint_names_expected = [
            "left_joint_1","left_joint_2","left_joint_3","left_joint_4",
            "right_joint_1","right_joint_2","right_joint_3","right_joint_4",
        ]

        # ---- cached states ----
        self.last_action = np.zeros(8, dtype=np.float32)

        self.imu_quat = None       # xyzw
        self.base_z = None         # float
        self.base_lin_vel = None   # body frame (3,)
        self.base_ang_vel = None   # body frame (3,)  (we'll prefer from /dodo/base_state, fallback to /imu ang vel)
        self.base_height_target = 0.5      # 训练 init_state.pos.z
        self.base_height_offset = None     # 启动后自动估计


        # joint cache
        self.q = None
        self.qd = None

        # ---- decimation alignment ----
        self.decimation = 4   # Isaac cfg
        self._js_counter = 0

        # ---- ROS I/O ----
        self.pub = self.create_publisher(Float32MultiArray, "/dodo/action", 10)
        self.sub_js = self.create_subscription(JointState, "/dodo/joint_states", self.on_js, 10)
        self.sub_imu = self.create_subscription(Imu, "/dodo/imu", self.on_imu, 50)
        self.sub_base = self.create_subscription(Float32MultiArray, "/dodo/base_state", self.on_base_state, 50)
        self.sub_cmdvel = self.create_subscription(Twist, "/cmd_vel", self.on_cmdvel, 10)

        self.step_count = 0

    def _infer_obs_dim(self):
        for d in [37, 36, 35, 40, 48, 32, 24]:
            try:
                x = torch.zeros(1, d, dtype=torch.float32)
                with torch.no_grad():
                    y = self.actor(x)
                if hasattr(y, "shape") and y.shape[-1] == 8:
                    return d
            except Exception:
                pass
        return None

    def on_cmdvel(self, msg: Twist):
        self.cmd[0] = float(msg.linear.x)
        self.cmd[1] = float(msg.linear.y)
        self.cmd[2] = float(msg.angular.z)
        self._last_cmdvel_time = time.time()

    def _maybe_resample_command(self):
        # 如果外部 /cmd_vel 最近 1s 有输入，就不采样
        if time.time() - self._last_cmdvel_time < 1.0:
            return

        now = time.time()
        if now < self._next_resample_time:
            return
        self._next_resample_time = now + self.resample_T

        if np.random.rand() < self.rel_standing:
            self.cmd[:] = 0.0
            return

        vx = np.random.uniform(*self.lin_x_range)
        vy = np.random.uniform(*self.lin_y_range)
        wz = np.random.uniform(*self.ang_z_range)
        self.cmd[:] = np.array([vx, vy, wz], dtype=np.float32)

    def on_imu(self, msg: Imu):
        q = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n > 1e-6:
            q = q / n
        if np.isfinite(q).all():
            self.imu_quat = q

    def projected_gravity_from_quat_auto(self, q_raw_xyzw: np.ndarray) -> np.ndarray:
        """
        输出对齐 IsaacLab projected_gravity_b：直立时约 [0,0,-1]
        自动尝试：xyzw vs wxyz，以及 g_body = R^T*g_world vs R*g_world
        选 g_z 最接近 -1 的那种。
        """
        q = np.asarray(q_raw_xyzw, dtype=np.float32).reshape(4)
        if not np.isfinite(q).all():
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        q = q / n

        # 备选1：按 ROS 常见 xyzw 使用
        q_xyzw = q

        # 备选2：假设其实是 wxyz，把它转成 xyzw
        # 如果 q_raw 其实是 [w,x,y,z]，那么 xyzw = [x,y,z,w] = [q1,q2,q3,q0]
        q_wxyz_assumed = q
        q2_xyzw = np.array([q_wxyz_assumed[1], q_wxyz_assumed[2], q_wxyz_assumed[3], q_wxyz_assumed[0]], dtype=np.float32)

        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        cands = []
        for name, qq in [("xyzw", q_xyzw), ("wxyz?", q2_xyzw)]:
            R = quat_xyzw_to_rotmat(qq)  # v_body -> v_world
            g1 = (R.T @ g_world)         # world->body
            g2 = (R @ g_world)           # (如果 quat 是 world->body，这个才是对的)
            for subname, gg in [("RT", g1), ("R", g2)]:
                ng = float(np.linalg.norm(gg))
                if ng > 1e-6:
                    gg = gg / ng
                cands.append((f"{name}:{subname}", gg.astype(np.float32)))

        # 选 g_z 最接近 -1 的
        best_name, best_g = min(cands, key=lambda kv: abs(float(kv[1][2]) - (-1.0)))

        # 可选：如果你想看选了哪一种，首次打印一次
        if not hasattr(self, "_pg_choice_printed"):
            self._pg_choice_printed = True
            self.get_logger().info(f"[proj_g auto] chose {best_name}, sample g={best_g.tolist()}")

        return best_g

    def on_base_state(self, msg: Float32MultiArray):
        # expected: [z, v_body_x,y,z, w_body_x,y,z]
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if data.shape[0] < 7:
            return
        self.base_z = float(data[0])
        self.base_lin_vel = data[1:4].astype(np.float32)
        self.base_ang_vel = data[4:7].astype(np.float32)

    def _reorder_joints(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        try:
            idx = [name_to_i[n] for n in self.joint_names_expected]
        except KeyError:
            return None
        q = np.array([msg.position[i] for i in idx], dtype=np.float32)
        qd = np.array([msg.velocity[i] for i in idx], dtype=np.float32)
        return q, qd

    def on_js(self, msg: JointState):
        reordered = self._reorder_joints(msg)
        if reordered is None:
            self.get_logger().warning("joint_states names mismatch. Ignored.")
            return
        self.q, self.qd = reordered

        # 生成/更新 command（与你训练的 UniformVelocityCommandCfg 对齐）
        self._maybe_resample_command()

        # decimation：每 4 帧才推理一次
        self._js_counter += 1
        if (self._js_counter % self.decimation) != 0:
            return
        
        if self.base_height_offset is None:
            # 第一次看到 base_z 时，认为此时是站姿附近，估计 offset
            # 让当前 base_z + offset ~= 0.5
            self.base_height_offset = self.base_height_target - float(self.base_z)
            self.get_logger().info(f"[base_height] offset set to {self.base_height_offset:.3f} (z_raw={self.base_z:.3f} -> z_used=0.5)")


        # 必须有 base_state + imu 才能拼齐 policy obs
        if self.base_z is None or self.base_lin_vel is None:
            return
        if self.base_ang_vel is None:
            return
        if self.imu_quat is None:
            return

        # ---- build obs (NO noise by default; you can add later if needed) ----
        z_used = float(self.base_z) + float(self.base_height_offset)
        # base_height = np.array([z_used], dtype=np.float32)            # (1,)
        # base_lin_vel = self.base_lin_vel.astype(np.float32)                 # (3,)
        base_ang_vel = self.base_ang_vel.astype(np.float32)                 # (3,)
        proj_g = self.projected_gravity_from_quat_auto(self.imu_quat)       # (3,)
        vel_cmd = self.cmd.astype(np.float32)                               # (3,)
        joint_pos = self.q.astype(np.float32)                               # (8,)
        joint_vel = self.qd.astype(np.float32)                              # (8,)
        last_a = self.last_action.astype(np.float32)                        # (8,)

        obs = np.concatenate(
            [base_height, base_lin_vel, base_ang_vel, proj_g, vel_cmd, joint_pos, joint_vel, last_a],
            axis=0
        ).astype(np.float32)

        if self.obs_dim is not None and obs.shape[0] != self.obs_dim:
            self.get_logger().error(f"Obs dim mismatch: built={obs.shape[0]} model={self.obs_dim}")
            return

        if not np.isfinite(obs).all():
            act = np.zeros(8, dtype=np.float32)
            raw = None
        else:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                raw_t = self.actor(obs_t).squeeze(0)   # (8,)
                act_t = torch.tanh(raw_t)              # 输出限幅到 [-1,1]
                act = act_t.cpu().numpy().astype(np.float32)
                raw = raw_t.cpu().numpy().astype(np.float32)

        # update + publish
        self.last_action = act
        out = Float32MultiArray()
        out.data = act.tolist()
        self.pub.publish(out)

        # ---- debug (每60次 policy step 打印一次) ----
        self.step_count += 1
        if self.step_count % 30 == 0:
            sat = float(np.mean(np.abs(act) > 0.95)) * 100.0
            self.get_logger().info(
                f"cmd(vx,vy,wz)={np.round(self.cmd,3).tolist()} | "
                f"base_z={float(self.base_z):.3f} | "
                f"v_b={np.round(base_lin_vel,3).tolist()} w_b={np.round(base_ang_vel,3).tolist()} | "
                f"proj_g={np.round(proj_g,3).tolist()} | "
                f"act_sat(|a|>0.95)={sat:.1f}% act={np.round(act,3).tolist()}"
            )
            if raw is not None:
                self.get_logger().info(f"raw range: min={float(raw.min()):.3f} max={float(raw.max()):.3f}")

def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
