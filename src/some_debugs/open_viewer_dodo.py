#!/usr/bin/env python3
import argparse
import numpy as np
import genesis as gs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--urdf", default="/home/rczh/workspace/dodo/dodobot_v3/urdf/dodobot_v3_simple.urdf")
    ap.add_argument("--res", default="1280x960")
    ap.add_argument("--max_fps", type=int, default=60)
    ap.add_argument("--dt", type=float, default=1/120.0)
    args = ap.parse_args()

    w, h = map(int, args.res.lower().split("x"))
    backend = gs.gpu if args.backend == "gpu" else gs.cpu

    gs.init(backend=backend)

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            res=(w, h),
            max_FPS=args.max_fps,
            enable_interaction=False, 
            run_in_thread=False,
        ),
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=1,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            iterations=20,
            tolerance=1e-4,
            constraint_timeconst=0.01,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            show_link_frame=False,
        ),
    )

    ground_mat = gs.materials.Rigid(friction=1.0, coup_restitution=0.0)
    scene.add_entity(gs.morphs.Plane(), material=ground_mat)

    robot_mat = gs.materials.Rigid(friction=1.0, coup_restitution=0.0)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=args.urdf,
            pos=(0.0, 0.0, 0.60),
            default_armature=0.01,
        ),
        material=robot_mat,
    )

    scene.build(n_envs=1)

    # 可选：给一个站立初始姿态（先用0，后面你可以按关节映射填）
    n = int(robot.n_dofs)
    q0 = np.zeros(n, dtype=np.float32)
    robot.set_dofs_position(q0)

    print("[OK] DODO loaded. Viewer should show robot on plane. Ctrl+C to exit.")
    try:
        while True:
            scene.step()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
