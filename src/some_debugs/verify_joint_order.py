#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import xml.etree.ElementTree as ET
import genesis as gs


def list_robot_link_names(robot):
    # try multiple APIs for compatibility
    for attr in ["link_names", "links_name", "links_names", "names"]:
        if hasattr(robot, attr):
            try:
                return list(getattr(robot, attr))
            except Exception:
                pass
    if hasattr(robot, "get_links"):
        try:
            links = robot.get_links()
            if isinstance(links, dict):
                return list(links.keys())
            out = []
            for l in links:
                if hasattr(l, "name"):
                    out.append(l.name)
                else:
                    out.append(str(l))
            return out
        except Exception:
            pass
    return []


def resolve_link_name(robot, urdf_name: str):
    names = list_robot_link_names(robot)
    if not names:
        return urdf_name  # fallback, let get_link raise

    # 1) exact
    if urdf_name in names:
        return urdf_name

    # 2) suffix match (most common: "xxx::body_link")
    suf = [n for n in names if n.endswith(urdf_name)]
    if len(suf) == 1:
        return suf[0]
    if len(suf) > 1:
        # choose shortest (closest)
        return sorted(suf, key=len)[0]

    # 3) contains
    con = [n for n in names if urdf_name in n]
    if len(con) == 1:
        return con[0]
    if len(con) > 1:
        return sorted(con, key=len)[0]

    # 4) try common base link alternatives
    if urdf_name == "body_link":
        for cand in ["base_link", "base", "torso", "pelvis", "trunk"]:
            if cand in names:
                return cand
            suf2 = [n for n in names if n.endswith(cand)]
            if suf2:
                return sorted(suf2, key=len)[0]

    return urdf_name

def parse_urdf_movable_joints(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            continue
        name = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        joints.append({"name": name, "parent": parent, "child": child})
    return joints

def link_quat_xyzw(robot, link_name: str) -> np.ndarray:
    """Return quat (x,y,z,w) for env0 as numpy float64."""
    real = resolve_link_name(robot, link_name)
    link = robot.get_link(name=real)

    # try common APIs
    q = None
    if hasattr(link, "get_quat"):
        q = link.get_quat()
    elif hasattr(link, "get_quaternion"):
        q = link.get_quaternion()
    elif hasattr(link, "get_rot"):  # if returns 3x3, convert to quat (fallback)
        R = link.get_rot()
        if hasattr(R, "ndim") and R.ndim == 3:
            R = R[0]
        if hasattr(R, "detach"):
            R = R.detach().cpu().numpy()
        R = np.array(R, dtype=np.float64).reshape(3,3)
        # rotmat -> quat (xyzw)
        w = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
        x = np.sqrt(max(0.0, 1.0 + R[0,0] - R[1,1] - R[2,2])) / 2.0 * np.sign(R[2,1] - R[1,2] + 1e-12)
        y = np.sqrt(max(0.0, 1.0 - R[0,0] + R[1,1] - R[2,2])) / 2.0 * np.sign(R[0,2] - R[2,0] + 1e-12)
        z = np.sqrt(max(0.0, 1.0 - R[0,0] - R[1,1] + R[2,2])) / 2.0 * np.sign(R[1,0] - R[0,1] + 1e-12)
        return np.array([x,y,z,w], dtype=np.float64)

    if hasattr(q, "ndim") and q.ndim == 2:
        q = q[0]
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    q = np.array(q, dtype=np.float64).reshape(4)
    return q

def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

def quat_mul(a, b):
    ax,ay,az,aw = a
    bx,by,bz,bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ], dtype=np.float64)

def quat_rel(parent_q, child_q):
    # q_rel = q_parent^{-1} * q_child
    return quat_mul(quat_conj(parent_q), child_q)

def quat_angle(q_rel):
    # angle in [0, pi]
    q = q_rel / (np.linalg.norm(q_rel) + 1e-12)
    w = float(np.clip(abs(q[3]), -1.0, 1.0))
    return 2.0 * np.arccos(w)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--backend", default="gpu", choices=["gpu","cpu"])
    ap.add_argument("--dt", type=float, default=1/120.0)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--z", type=float, default=0.7)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--settle", type=int, default=120)
    args = ap.parse_args()

    backend = gs.gpu if args.backend=="gpu" else gs.cpu
    gs.init(backend=backend)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=args.dt, substeps=args.substeps, gravity=(0,0,-9.81)),
        rigid_options=gs.options.RigidOptions(constraint_solver=gs.constraint_solver.Newton, iterations=50),
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0, coup_restitution=0.0))
    robot = scene.add_entity(gs.morphs.URDF(file=args.urdf, pos=(0,0,args.z), default_armature=0.01))
    scene.build(n_envs=1)

    print("=== URDF link name -> Genesis link name ===")
    for j in joints:
        for ln in [j["parent"], j["child"]]:
            print(f"{ln:12s} -> {resolve_link_name(robot, ln)}")


    joints = parse_urdf_movable_joints(args.urdf)
    n_joints = len(joints)
    n_dofs = int(robot.n_dofs)
    n_base = n_dofs - n_joints

    print(f"[INFO] n_dofs={n_dofs}, movable_joints={n_joints}, inferred_base={n_base}")
    for i,j in enumerate(joints):
        print(f"  J[{i}] {j['name']:14s} parent={j['parent']:12s} child={j['child']:12s}")

    for _ in range(args.settle):
        scene.step()

    # baseline relative quats for each joint (parent->child)
    base_rel = []
    for j in joints:
        qp = link_quat_xyzw(robot, j["parent"])
        qc = link_quat_xyzw(robot, j["child"])
        base_rel.append(quat_rel(qp, qc))

    q = robot.get_dofs_position()
    if hasattr(q, "ndim") and q.ndim == 2:
        q = q[0]
    if hasattr(q, "detach"):
        q = q.detach()
    q0 = q.clone()

    dofs = list(range(n_base, n_dofs))
    score = np.zeros((len(dofs), n_joints), dtype=np.float64)

    print("\n[INFO] Probing DOFs; measuring change of EACH joint's parent-child RELATIVE rotation...")
    for di, dof_idx in enumerate(dofs):
        robot.set_dofs_position(q0[None, :])
        for _ in range(5):
            scene.step()

        q1 = q0.clone()
        q1[dof_idx] += float(args.delta)
        robot.set_dofs_position(q1[None, :])
        for _ in range(10):
            scene.step()

        for ji, j in enumerate(joints):
            qp = link_quat_xyzw(robot, j["parent"])
            qc = link_quat_xyzw(robot, j["child"])
            rel = quat_rel(qp, qc)
            # relative rotation difference magnitude
            dq = quat_mul(quat_conj(base_rel[ji]), rel)
            score[di, ji] = quat_angle(dq)

        best = int(np.argmax(score[di]))
        print(f"  dof_idx_local={dof_idx:2d} -> best_joint={joints[best]['name']:14s}  angle={score[di,best]:.6f} rad")

    print("\n[RESULT] angle-matrix rows=dof_idx_local cols=URDF joint order:")
    header = "          " + " ".join([f"{j['name'][:6]:>6s}" for j in joints])
    print(header)
    for di, dof_idx in enumerate(dofs):
        row = f"dof{dof_idx:02d}: " + " ".join([f"{score[di,ji]:6.3f}" for ji in range(n_joints)])
        print(row)

    print("\n[NOTE] Ideally each row has a single dominant column => mapping is clear.")
    print("[NOTE] If two columns dominate, reduce --delta to 0.05 and rerun.")

if __name__ == "__main__":
    main()
