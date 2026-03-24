#!/usr/bin/env python3
import time
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState, Imu

import genesis as gs
import xml.etree.ElementTree as ET
from typing import List


def urdf_get_movable_joint_names(urdf_path: str) -> List[str]:
    """Return joint names whose type != fixed, in URDF order."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    names = []
    for j in root.findall("joint"):
        jtype = j.get("type", "")
        if jtype != "fixed":
            n = j.get("name")
            if n is not None:
                names.append(n)
    return names


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


def rotate_world_to_body(quat_xyzw, v_world):
    """v_body = R^T * v_world, assuming quat is body->world."""
    R = quat_xyzw_to_rotmat(quat_xyzw)
    return (R.T @ np.asarray(v_world, dtype=np.float32).reshape(3)).astype(np.float32)

def normalize_quat(q):
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n > 1e-6:
        q = q / n
    return q

def wxyz_to_xyzw(q_wxyz):
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)

def projected_gravity_from_quat_xyzw(quat_xyzw, use_RT=True):
    # gravity direction in body frame; upright should be about [0,0,-1]
    R = quat_xyzw_to_rotmat(quat_xyzw)  # v_body -> v_world
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    g_body = (R.T @ g_world) if use_RT else (R @ g_world)
    ng = float(np.linalg.norm(g_body))
    if ng > 1e-6:
        g_body = g_body / ng
    return g_body.astype(np.float32)

def choose_quat_convention(q_raw_4):
    """
    返回：
      quat_xyzw   : 用于 ROS 发布的 xyzw
      use_RT      : True 表示 world->body 用 R^T；False 表示用 R
      tag         : 选中的方案名
    选择标准：让 projected_gravity.z 最接近 -1（站立时）
    """
    q_raw = normalize_quat(q_raw_4)

    # 两种分量解释：raw 当 xyzw，raw 当 wxyz
    q_xyzw_a = q_raw
    q_xyzw_b = wxyz_to_xyzw(q_raw)

    cands = []
    for name, q_xyzw in [("raw_is_xyzw", q_xyzw_a), ("raw_is_wxyz", q_xyzw_b)]:
        for use_RT in [True, False]:
            g = projected_gravity_from_quat_xyzw(q_xyzw, use_RT=use_RT)
            score = abs(float(g[2]) - (-1.0))  # 越接近 -1 越好
            cands.append((score, name, use_RT, q_xyzw, g))

    cands.sort(key=lambda x: x[0])
    best = cands[0]
    _, name, use_RT, q_xyzw, g = best
    return q_xyzw.astype(np.float32), bool(use_RT), f"{name}:{'RT' if use_RT else 'R'}", g



class GenesisBridge(Node):
    def __init__(
        self,
        urdf_path: str,
        show_viewer: bool = True,
        viewer_res=(1280, 960),
        viewer_max_fps: int = 60,
        enable_interaction: bool = False,
        run_in_thread: bool = False,
    ):
        super().__init__("genesis_bridge")

        # ---------------- ROS2 ----------------
        self.pub_js = self.create_publisher(JointState, "/dodo/joint_states", 10)
        self.pub_base = self.create_publisher(Float32MultiArray, "/dodo/base_state", 10)  # [z, v_body(3), w_body(3)]
        self.pub_imu = self.create_publisher(Imu, "/dodo/imu", 10)

        self.sub_action = self.create_subscription(Float32MultiArray, "/dodo/action", self.on_action, 10)

        # ---------------- Genesis ----------------
        gs.init(backend=gs.gpu)  # 没 GPU 改 gs.cpu

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                res=viewer_res,
                max_FPS=viewer_max_fps,
                enable_interaction=enable_interaction,
                run_in_thread=run_in_thread,
            ),
            sim_options=gs.options.SimOptions(
                dt=1 / 120.0,
                substeps=4,
                gravity=(0.0, 0.0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                iterations=50,
                tolerance=1e-4,
                constraint_timeconst=0.01,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                show_link_frame=False,
            ),
        )

        ground_mat = gs.materials.Rigid(friction=1.0, coup_restitution=0.0)
        robot_mat = gs.materials.Rigid(friction=1.0, coup_restitution=0.0)
        self.scene.add_entity(gs.morphs.Plane(), material=ground_mat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=(0.0, 0.0, 0.55),
                default_armature=0.01,
            ),
            material=robot_mat,
        )

        self.scene.build(n_envs=1)

        # ---------------- DOF / joint order ----------------
        self.n_dofs = int(self.robot.n_dofs)
        self.get_logger().info(f"Robot DOFs = {self.n_dofs}")

        # 1) 从 URDF 读 movable joint（URDF 原始顺序）
        urdf_joints = urdf_get_movable_joint_names(urdf_path)
        self.get_logger().info(f"URDF movable joints = {len(urdf_joints)}")
        for i, n in enumerate(urdf_joints):
            self.get_logger().info(f"URDF_JOINT[{i}] = {n}")
        
        q0 = self._get_dofs_pos_1d()

        # 2) 你的模型：floating base 6 + joints 8 = 14（如果不同就提示）
        self.n_joints = 8
        self.n_base_dofs = self.n_dofs - self.n_joints
        self.get_logger().info(f"Inferred base DOFs = {self.n_base_dofs}, joint DOFs = {self.n_joints}")

        if not (self.n_base_dofs == 6 and len(urdf_joints) == 8 and self.n_dofs == 14):
            self.get_logger().warning(
                f"Assumption mismatch: base_dofs={self.n_base_dofs}, urdf_joints={len(urdf_joints)}, total_dofs={self.n_dofs}"
            )

        # 3) ✅ 训练/策略顺序（必须固定）：left1..4, right1..4
        self.joint_names_expected = [
            "left_joint_1","left_joint_2","left_joint_3","left_joint_4",
            "right_joint_1","right_joint_2","right_joint_3","right_joint_4",
        ]

        # 4) ✅ joint_states 发布的 name 就用训练顺序（避免误导）
        self.joint_names = self.joint_names_expected

        # 5) ✅ Genesis local DOF index：要和训练顺序对齐
        # 你验证 mapping（URDF: r1 r2 r3 r4 l1 l2 l3 l4 -> Genesis local）
        # r1->6 r2->8 r3->10 r4->12 l1->7 l2->9 l3->11 l4->13
        # 所以训练顺序 l1 l2 l3 l4 r1 r2 r3 r4 -> [7,9,11,13, 6,8,10,12]
        self.joint_dofs_idx_local = np.array([7, 9, 11, 13, 6, 8, 10, 12], dtype=np.int32)
        self.get_logger().info(
            f"Using joint_dofs_idx_local (train order -> Genesis): {self.joint_dofs_idx_local.tolist()}"
        )

        # ---------------- Actuator gains & limits (match training actuators) ----------------
        # stiffness=42 damping=2.5 armature=0.01 effort_limit_sim=6
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = self.device
        kp = torch.full((self.n_joints,), 42.0, device=device, dtype=torch.float32)
        kv = torch.full((self.n_joints,), 2.5, device=device, dtype=torch.float32)
        fmin = torch.full((self.n_joints,), -6.0, device=device, dtype=torch.float32)
        fmax = torch.full((self.n_joints,),  6.0, device=device, dtype=torch.float32)

        if hasattr(self.robot, "set_dofs_kp"):
            self.robot.set_dofs_kp(kp, dofs_idx_local=self.joint_dofs_idx_local)
        if hasattr(self.robot, "set_dofs_kv"):
            self.robot.set_dofs_kv(kv, dofs_idx_local=self.joint_dofs_idx_local)
        if hasattr(self.robot, "set_dofs_force_range"):
            self.robot.set_dofs_force_range(fmin, fmax, dofs_idx_local=self.joint_dofs_idx_local)

        self.get_logger().info("Applied kp/kv/force_range on JOINT DOFs (position control).")

        # ---------------- Default pose (q_default) ----------------
        # 你贴的 init_state:
        # joint_1: 0.0, joint_2: -0.3, joint_3: 0.90, joint_4: -0.65 for both legs
        self.q_default = np.array(
            [0.0, -0.3, 0.90, -0.65,
             0.0, -0.3, 0.90, -0.65],
            dtype=np.float32
        )

        # 用 default pose 初始化机器人
        q0 = self._get_dofs_pos_1d()
        q0[self.joint_dofs_idx_local] = self.q_default
        self.robot.set_dofs_position(q0[None, :])

        try:
            self.robot.set_dofs_velocity(np.zeros((1, self.n_dofs), dtype=np.float32))
        except Exception:
            pass

        # ---------------- Policy action -> joint position target ----------------
        self.action_scale = 0.5  # ActionsCfg.joint_pos.scale
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # 平滑/渐入
        self.alpha_ramp_sec = 1.0
        self._policy_ready = False
        self._policy_start_time = None
        self._last_action_time = 0.0

        self.qdes_lpf = 0.2
        self.last_qdes = self.q_default.copy()

        self._quat_xyzw_fixed = None
        self._quat_use_RT = True
        self._quat_tag = None


        # settle 1s
        for _ in range(120):
            self.robot.control_dofs_position(torch.tensor(self.q_default, device=device, dtype=torch.float32),
                                            dofs_idx_local=self.joint_dofs_idx_local)
            self.scene.step()

        self.get_logger().info("Genesis bridge started (POSITION action sim2sim).")

    # ---------------- helpers ----------------
    def _get_dofs_pos_1d(self) -> np.ndarray:
        """Get full dofs position as 1D numpy array (len=self.n_dofs)."""
        q = self.robot.get_dofs_position()
        if hasattr(q, "ndim") and q.ndim == 2:
            q = q[0]
        return self._to_numpy_1d(q, np.float32)

    def _to_numpy_1d(self, x, dtype=np.float32):
        """torch / numpy -> 1D numpy"""
        if hasattr(x, "detach"):
            x = x.detach()
            if hasattr(x, "cpu"):
                x = x.cpu()
            x = x.numpy()
        return np.array(x, dtype=dtype).reshape(-1)

    def _get_q_qd_full(self):
        """Return (q, qd) for the full DOFs (len = self.n_dofs)."""
        q = self.robot.get_dofs_position()
        qd = self.robot.get_dofs_velocity()

        # Genesis 有时返回 (1, n_dofs)
        if hasattr(q, "ndim") and q.ndim == 2:
            q = q[0]
        if hasattr(qd, "ndim") and qd.ndim == 2:
            qd = qd[0]

        q = self._to_numpy_1d(q, np.float32)
        qd = self._to_numpy_1d(qd, np.float32)
        return q, qd

    def _get_joint_state_train_order(self):
        """
        Return (qj, qdj) in TRAIN order:
        [left1,left2,left3,left4, right1,right2,right3,right4]
        using self.joint_dofs_idx_local which is already in train order.
        """
        q, qd = self._get_q_qd_full()
        qj = q[self.joint_dofs_idx_local].astype(np.float32)
        qdj = qd[self.joint_dofs_idx_local].astype(np.float32)
        return qj, qdj



    def _get_root_state_best_effort(self):
        """
        尝试从 Genesis 取 root pose/vel.
        不同版本 API 可能不同，这里做 best-effort：
        返回：pos_world(3), quat_xyzw(4), lin_vel_world(3), ang_vel_world(3)
        """
        # ---- position ----
        pos = None
        quat = None
        v = None
        w = None

        # 常见可能的接口：get_pos/get_quat/get_vel/get_ang
        if hasattr(self.robot, "get_pos"):
            pos = self.robot.get_pos()
        if hasattr(self.robot, "get_quat"):
            quat = self.robot.get_quat()
        if hasattr(self.robot, "get_vel"):
            v = self.robot.get_vel()
        if hasattr(self.robot, "get_ang"):
            w = self.robot.get_ang()

        # fallback：有些版本是 get_position/get_orientation/get_velocity/get_angular_velocity
        if pos is None and hasattr(self.robot, "get_position"):
            pos = self.robot.get_position()
        if quat is None and hasattr(self.robot, "get_orientation"):
            quat = self.robot.get_orientation()
        if v is None and hasattr(self.robot, "get_velocity"):
            v = self.robot.get_velocity()
        if w is None and hasattr(self.robot, "get_angular_velocity"):
            w = self.robot.get_angular_velocity()

        if pos is None or quat is None or v is None or w is None:
            raise RuntimeError(
                "Cannot read root state from Genesis robot. "
                "Please check Genesis API for root pose/vel getters."
            )

        pos = self._to_numpy_1d(pos, np.float32)[:3]
        quat = self._to_numpy_1d(quat, np.float32)[:4]
        v = self._to_numpy_1d(v, np.float32)[:3]
        w = self._to_numpy_1d(w, np.float32)[:3]

        # normalize quat
        n = float(np.linalg.norm(quat))
        if n > 1e-6:
            quat = quat / n
        return pos, quat, v, w

    # ---------------- ROS callbacks ----------------
    def on_action(self, msg: Float32MultiArray):
        a = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.n_joints:
            self.get_logger().warning(f"/dodo/action len {a.shape[0]} != {self.n_joints}, ignored")
            return

        # action should already be [-1,1], but clip anyway
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        self.last_action = a
        self._last_action_time = time.time()

        if not self._policy_ready:
            self._policy_ready = True
            self._policy_start_time = time.time()
            self.get_logger().info("Policy connected. Start ramping in (position targets)...")

    # ---------------- main step ----------------
    def step(self):
        # ---- publish joint_states ----
        qj, qdj = self._get_joint_state_train_order()
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names_expected
        js.position = [float(x) for x in qj]
        js.velocity = [float(x) for x in qdj]
        self.pub_js.publish(js)

        if not hasattr(self, "_printed_joint_order"):
            self._printed_joint_order = True
            self.get_logger().info(f"Publishing joint_states in order: {self.joint_names}")

        # ---- publish base_state + imu ----
        try:
            pos_w, quat_raw, v_w, w_w = self._get_root_state_best_effort()

            # 第一次进来，自动判别 Genesis quat 的约定
            if self._quat_tag is None:
                q_xyzw, use_RT, tag, g0 = choose_quat_convention(quat_raw)
                self._quat_tag = tag
                self._quat_use_RT = use_RT
                self._quat_xyzw_fixed = q_xyzw
                self.get_logger().info(f"[quat auto] chose {tag}, g_init={g0.tolist()}, quat_xyzw={np.round(q_xyzw,4).tolist()}")

            # 每次都把 raw 按固定约定转成 xyzw
            if self._quat_tag.startswith("raw_is_wxyz"):
                quat_xyzw = wxyz_to_xyzw(quat_raw)
            else:
                quat_xyzw = np.asarray(quat_raw, dtype=np.float32).reshape(4)
            quat_xyzw = normalize_quat(quat_xyzw)

            # world->body 旋转：由 use_RT 决定
            R = quat_xyzw_to_rotmat(quat_xyzw)  # v_body -> v_world
            if self._quat_use_RT:
                v_b = (R.T @ np.asarray(v_w, np.float32).reshape(3)).astype(np.float32)
                w_b = (R.T @ np.asarray(w_w, np.float32).reshape(3)).astype(np.float32)
            else:
                v_b = (R @ np.asarray(v_w, np.float32).reshape(3)).astype(np.float32)
                w_b = (R @ np.asarray(w_w, np.float32).reshape(3)).astype(np.float32)

            # （可选强自检）proj_g 也用同一套约定算出来，站立时应该接近 [0,0,-1]
            proj_g_dbg = projected_gravity_from_quat_xyzw(quat_xyzw, use_RT=self._quat_use_RT)


            base_msg = Float32MultiArray()
            base_msg.data = [float(pos_w[2])] + [float(x) for x in v_b] + [float(x) for x in w_b]
            self.pub_base.publish(base_msg)

            imu = Imu()
            imu.header.stamp = js.header.stamp
            imu.orientation.x = float(quat_xyzw[0])
            imu.orientation.y = float(quat_xyzw[1])
            imu.orientation.z = float(quat_xyzw[2])
            imu.orientation.w = float(quat_xyzw[3])
            imu.angular_velocity.x = float(w_b[0])
            imu.angular_velocity.y = float(w_b[1])
            imu.angular_velocity.z = float(w_b[2])
            self.pub_imu.publish(imu)
        except Exception:
            # 如果你的 Genesis 版本暂时拿不到 root state，先不 publish base/imu
            pass

        # ---- compute desired joint positions (IsaacLab JointPositionActionCfg) ----
        # q_des = q_default + scale * action   (use_default_offset=True)
        q_des_policy = self.q_default + self.action_scale * self.last_action

        # ramp-in
        if self._policy_ready and self._policy_start_time is not None:
            t = time.time() - self._policy_start_time
            alpha = float(np.clip(t / self.alpha_ramp_sec, 0.0, 1.0))
        else:
            alpha = 0.0

        # if action stream lost, go back to default pose
        if (not self._policy_ready) or (time.time() - self._last_action_time > 0.25):
            alpha = 0.0

        q_des = (1.0 - alpha) * self.q_default + alpha * q_des_policy

        # low-pass on target positions (helps a lot)
        self.last_qdes = (1.0 - self.qdes_lpf) * self.last_qdes + self.qdes_lpf * q_des

        # apply position targets (implicit PD inside Genesis)
        qdes_t = torch.tensor(self.last_qdes, device=self.device, dtype=torch.float32)
        self.robot.control_dofs_position(qdes_t, dofs_idx_local=self.joint_dofs_idx_local)

        self.scene.step()


def main():
    rclpy.init()
    urdf_path = "/home/rczh/workspace/dodo/dodobot_v3/urdf/dodobot_v3_simple.urdf"

    node = GenesisBridge(
        urdf_path,
        show_viewer=True,
        viewer_res=(1280, 960),
        viewer_max_fps=60,
        enable_interaction=False,
        run_in_thread=False,
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.step()
            time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
