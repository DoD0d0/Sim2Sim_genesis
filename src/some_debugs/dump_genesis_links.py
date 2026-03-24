#!/usr/bin/env python3
import genesis as gs

URDF = "/home/rczh/workspace/dodo/dodobot_v3/urdf/dodobot_v3_simple.urdf"

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=False,
    sim_options=gs.options.SimOptions(dt=1/120.0, substeps=4, gravity=(0,0,-9.81)),
)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(file=URDF, pos=(0,0,0.7)))
scene.build(n_envs=1)

# ✅ 尝试多种方式拿 link names（不同版本 Genesis 字段名可能不同）
names = None
for attr in ["link_names", "links_name", "links_names", "names"]:
    if hasattr(robot, attr):
        try:
            names = list(getattr(robot, attr))
            break
        except Exception:
            pass

# 最保险：尝试 get_links()
if names is None and hasattr(robot, "get_links"):
    try:
        links = robot.get_links()
        # links 可能是 list[Link]，也可能是 dict
        if isinstance(links, dict):
            names = list(links.keys())
        else:
            names = []
            for l in links:
                if hasattr(l, "name"):
                    names.append(l.name)
                else:
                    names.append(str(l))
    except Exception:
        pass

print("=== Genesis link names ===")
if names is None:
    print("Could not find link list API on this Genesis version.")
else:
    for i, n in enumerate(names):
        print(f"[{i:02d}] {n}")
