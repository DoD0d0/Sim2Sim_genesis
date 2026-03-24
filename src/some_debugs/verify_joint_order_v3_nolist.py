#!/usr/bin/env python3
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import genesis as gs

def parse_urdf(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    robot_name = root.get("name", "robot")

    joints = []
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            continue
        name = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        joints.append({"name": name, "parent": parent, "child": child})
    return robot_name, joints

def try_get_link(robot, name: str):
    try:
        return robot.get_link(name=name)
    except Exception:
        return None

def resolve_link(robot, urdf_link: str, robot_name: str):
    """
    Resolve URDF link name to something Genesis accepts, WITHOUT needing a link list API.
    Tries common namespaces/prefixes and special-case fixed-joint reduction.
    """
    # direct
    cands = [urdf_link]

    # common namespacing patterns
    cands += [
        f"{robot_name}::{urdf_link}",
        f"{robot_name}:{urdf_link}",
        f"{robot_name}/{urdf_link}",
        f"/{robot_name}/{urdf_link}",
    ]

    # special-case: body_link often merged into base_link via fixed joint reduction
    if urdf_link == "body_link":
        cands += ["base_link",
                  f"{robot_name}::base_link",
                  f"{robot_name}/base_link",
                  f"/{robot_name}/base_link"]

    # also try: if urdf_link is base_link, sometimes becomes body_link (rare)
    if urdf_link == "base_link":
        cands += ["body_link",
                  f"{robot_name}::body_link",
                  f"{robot_name}/body_link",
                  f"/{robot_name}/body_link"]

    for c in cands:
        lk = try_get_link(robot, c)
        if lk is not None:
            return c, lk

    # last resort: return None
    return None, None

def to_np4(x):
    # x may be torch tensor on gpu/cpu or numpy/list
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        x = x.numpy()
    x = np.array(x, dtype=np.float64).reshape(-1)
    return x

def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ], dtype=np.float64)

def quat_rel(parent_q, child_q):
    return quat_mul(quat_conj(parent_q), child_q)

def quat_angle(q):
    q = q / (np.linalg.norm(q) + 1e-12)
    w = float(np.clip(abs(q[3]), -1.0, 1.0))
    return 2.0 * np.arccos(w)

def link_quat_xyzw(robot, link_name: str, robot_name: str) -> np.ndarray:
    real_name, link = resolve_link(robot, link_name, robot_name)
    if link is None:
        raise RuntimeError(f"Cannot resolve link '{link_name}'. Tried common prefixes; still not found.")

    # Genesis link quat API may vary
    if hasattr(link, "get_quat"):
        q = link.get_quat()
    elif hasattr(link, "get_quaternion"):
        q = link.get_quaternion()
    else:
        raise RuntimeError(f"Resolved '{link_name}' -> '{real_name}', but link has no get_quat/get_quaternion")

    if hasattr(q, "ndim") and q.ndim == 2:
        q = q[0]
    return to_np4(q)  # (4,) xyzw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--backend", default="gpu", choices=["gpu","cpu"])
    ap.add_argument("--dt", type=float, default=1/120.0)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--z", type=float, default=0.7)
    ap.add_argument("--delta", type=float, default=0.04)
    ap.add_argument("--settle", type=int, default=120)
    args = ap.parse_args()

    robot_name, joints = parse_urdf(args.urdf)

    backend = gs.gpu if args.backend=="gpu" else gs.cpu
    gs.init(backend=backend)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=args.dt, substeps=args.substeps, gravity=(0,0,0)),
        rigid_options=gs.options.RigidOptions(constraint_solver=gs.constraint_solver.Newton, iterations=50),
    )
    # scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0, coup_restitution=0.0))
    robot = scene.add_entity(gs.morphs.URDF(file=args.urdf, pos=(0,0,args.z), default_armature=0.01))
    scene.build(n_envs=1)

    n_joints = len(joints)
    n_dofs = int(robot.n_dofs)
    n_base = n_dofs - n_joints

    print(f"[INFO] urdf_robot_name='{robot_name}'")
    print(f"[INFO] n_dofs={n_dofs}, movable_joints={n_joints}, inferred_base={n_base}")
    for i,j in enumerate(joints):
        print(f"  J[{i}] {j['name']:14s} parent={j['parent']:12s} child={j['child']:12s}")

    # show how links resolve (very useful)
    print("\n=== URDF link -> resolved Genesis link (trial) ===")
    for j in joints:
        for ln in [j["parent"], j["child"]]:
            rn, lk = resolve_link(robot, ln, robot_name)
            print(f"{ln:12s} -> {rn}")

    for _ in range(args.settle):
        scene.step()

    # baseline parent-child relative quat
    base_rel = []
    for j in joints:
        qp = link_quat_xyzw(robot, j["parent"], robot_name)
        qc = link_quat_xyzw(robot, j["child"], robot_name)
        base_rel.append(quat_rel(qp, qc))

    q = robot.get_dofs_position()
    if hasattr(q, "ndim") and q.ndim == 2:
        q = q[0]
    if hasattr(q, "detach"):
        q0 = q.detach().clone()
    else:
        q0 = np.array(q, dtype=np.float32)

    dofs = list(range(n_base, n_dofs))
    score = np.zeros((len(dofs), n_joints), dtype=np.float64)

    print("\n[INFO] Probing DOFs by parent-child RELATIVE rotation change...")
    for di, dof_idx in enumerate(dofs):
        robot.set_dofs_position(q0[None, :])
        for _ in range(5):
            scene.step()

        q1 = q0.clone() if hasattr(q0, "clone") else q0.copy()
        q1[dof_idx] += float(args.delta)
        robot.set_dofs_position(q1[None, :])
        for _ in range(10):
            scene.step()

        for ji, j in enumerate(joints):
            qp = link_quat_xyzw(robot, j["parent"], robot_name)
            qc = link_quat_xyzw(robot, j["child"], robot_name)
            rel = quat_rel(qp, qc)
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

    print("\n[NOTE] Each row should have a single dominant column => clear DOF->joint mapping.")
    print("[NOTE] If ambiguous, reduce --delta to 0.05 and rerun.")

if __name__ == "__main__":
    main()
