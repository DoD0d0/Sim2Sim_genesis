#!/usr/bin/env python3
# verify_joint_order_clean.py
import argparse
import math
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

import genesis as gs


# ------------------ URDF parsing ------------------

def parse_urdf_movable_joints(urdf_path: str) -> List[Dict]:
    """
    Return movable joints in URDF order:
    [
      {"name":..., "type":..., "parent":..., "child":...},
      ...
    ]
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = []
    for j in root.findall("joint"):
        jtype = j.get("type", "")
        if jtype == "fixed":
            continue
        name = j.get("name")
        parent = j.find("parent").get("link") if j.find("parent") is not None else None
        child = j.find("child").get("link") if j.find("child") is not None else None
        if name and parent and child:
            joints.append({"name": name, "type": jtype, "parent": parent, "child": child})
    return joints


# ------------------ Helpers: Genesis tensor -> numpy ------------------

def to_numpy_1d(x, dtype=np.float32) -> np.ndarray:
    """torch(cuda/cpu)/numpy/list -> (N,) numpy on CPU"""
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        x = x.numpy()
    return np.array(x, dtype=dtype).reshape(-1)


# ------------------ Quaternion math ------------------

def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0, 0, 0, 1], dtype=np.float64)
    return q / n

def quat_conj(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ], dtype=np.float64)

def quat_inv(q):
    q = quat_normalize(q)
    return quat_conj(q)

def quat_angle(q):
    """Return rotation angle (rad) represented by quaternion q (x,y,z,w)"""
    q = quat_normalize(q)
    w = float(np.clip(abs(q[3]), 0.0, 1.0))
    return 2.0 * math.acos(w)

def relative_rot(q_parent, q_child):
    """q_rel = inv(q_parent) * q_child"""
    return quat_mul(quat_inv(q_parent), q_child)

def angle_between_rel_rots(qrel0, qrel1):
    """angle of dq = inv(qrel0)*qrel1"""
    dq = quat_mul(quat_inv(qrel0), qrel1)
    return quat_angle(dq)


# ------------------ Genesis link pose access ------------------

def get_link_handle(robot, name: str):
    """
    Genesis v0.3.x 有时 fixed joint 会把 body_link 折叠成 base_link.
    所以如果找不到 body_link，就尝试 base_link。
    """
    try:
        return robot.get_link(name=name)
    except Exception:
        if name == "body_link":
            return robot.get_link(name="base_link")
        raise

def link_quat_xyzw(robot, link_name: str) -> np.ndarray:
    link = get_link_handle(robot, link_name)

    # 尝试常见 API
    if hasattr(link, "get_quat"):
        q = link.get_quat()
    elif hasattr(link, "quat"):
        q = link.quat
    elif hasattr(link, "get_pose"):
        pose = link.get_pose()
        # pose 可能是 (pos, quat)
        q = pose[1]
    else:
        raise RuntimeError(f"Cannot read quaternion from link: {link_name}")

    q = to_numpy_1d(q, np.float64)
    # 保证是 (4,)
    if q.shape[0] != 4:
        q = q.reshape(-1)[:4]
    return quat_normalize(q)

def reset_to_q0(robot, scene, q0_full: np.ndarray, settle_steps: int):
    robot.set_dofs_position(q0_full[None, :].astype(np.float32))
    try:
        robot.set_dofs_velocity(np.zeros((1, q0_full.shape[0]), dtype=np.float32))
    except Exception:
        pass
    for _ in range(settle_steps):
        scene.step()


# ------------------ Main probing ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--delta", type=float, default=0.05, help="joint perturbation in rad")
    ap.add_argument("--settle", type=int, default=60, help="steps to settle after initial pose")
    ap.add_argument("--probe_settle", type=int, default=10, help="steps to settle after each perturb/reset")
    ap.add_argument("--dt", type=float, default=1/240.0)
    ap.add_argument("--z", type=float, default=0.7)
    args = ap.parse_args()

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend)

    # ✅ A) 悬空 + gravity=0 + 不加 Plane，避免接触/摩擦造成串扰
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=1,
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=30,
            tolerance=1e-4,
            constraint_timeconst=0.01,
        ),
    )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=args.urdf,
            pos=(0.0, 0.0, args.z),
            default_armature=0.01,
        ),
        material=gs.materials.Rigid(friction=1.0, coup_restitution=0.0),
    )

    scene.build(n_envs=1)

    # URDF joints
    joints = parse_urdf_movable_joints(args.urdf)
    n_joints = len(joints)
    n_dofs = int(robot.n_dofs)
    n_base = n_dofs - n_joints

    print(f"[INFO] n_dofs={n_dofs}, movable_joints={n_joints}, inferred_base={n_base}")
    for i, j in enumerate(joints):
        print(f"  J[{i}] {j['name']:<14} parent={j['parent']:<12} child={j['child']:<12}")

    # ✅ B) probe 前固定到同一个初始姿态（关节全 0），并 settle
    q0 = robot.get_dofs_position()
    if hasattr(q0, "ndim") and q0.ndim == 2:
        q0 = q0[0]
    q0 = to_numpy_1d(q0, np.float32)

    # 只把关节 DOF 置 0（base 不动）
    q0[n_base:] = 0.0
    reset_to_q0(robot, scene, q0, args.settle)

    # 要 probe 的 DOF（默认 joint DOFs 在最后）
    dof_list = list(range(n_base, n_dofs))

    # 记录矩阵：rows = dof_list, cols = URDF joint order
    mat = np.zeros((len(dof_list), n_joints), dtype=np.float64)
    best = []

    print("\n[INFO] Probing DOFs by parent-child RELATIVE rotation change (悬空+gravity0)...")

    for ri, dof in enumerate(dof_list):
        # 每次 probe 前都回到同一个 q0（非常关键）
        reset_to_q0(robot, scene, q0, args.probe_settle)

        # baseline: relative rot for each joint
        qrel0 = []
        for j in joints:
            qp = link_quat_xyzw(robot, j["parent"])
            qc = link_quat_xyzw(robot, j["child"])
            qrel0.append(relative_rot(qp, qc))

        # perturb this DOF
        q1 = q0.copy()
        q1[dof] += float(args.delta)
        robot.set_dofs_position(q1[None, :].astype(np.float32))
        for _ in range(args.probe_settle):
            scene.step()

        # measure
        qrel1 = []
        for j in joints:
            qp = link_quat_xyzw(robot, j["parent"])
            qc = link_quat_xyzw(robot, j["child"])
            qrel1.append(relative_rot(qp, qc))

        # fill row
        for ci in range(n_joints):
            mat[ri, ci] = angle_between_rel_rots(qrel0[ci], qrel1[ci])

        ci_best = int(np.argmax(mat[ri]))
        best_name = joints[ci_best]["name"]
        best_angle = mat[ri, ci_best]
        best.append((dof, best_name, best_angle))
        print(f"  dof_idx_local={dof:2d} -> best_joint={best_name:<14} angle={best_angle:.6f} rad")

    # print matrix
    print("\n[RESULT] angle-matrix rows=dof_idx_local cols=URDF joint order:")
    header = "          " + " ".join([j["name"][:6].ljust(6) for j in joints])
    print(header)
    for ri, dof in enumerate(dof_list):
        row = "dof%02d: " % dof + " ".join([f"{mat[ri,ci]:.3f}".rjust(6) for ci in range(n_joints)])
        print(row)

    print("\n[NOTE] 每行应该只有一个明显最大值 => DOF->joint 映射清晰。")
    print("[NOTE] 如果仍有多列接近，把 --delta 再降到 0.03。")


if __name__ == "__main__":
    main()
