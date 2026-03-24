#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import math

URDF = "/home/rczh/workspace/dodo/dodobot_v3/urdf/dodobot_v3_simple.urdf"

def f3(s, default=0.0):
    if s is None: return (default, default, default)
    parts = s.strip().split()
    if len(parts) != 3: return (default, default, default)
    return tuple(float(x) for x in parts)

tree = ET.parse(URDF)
root = tree.getroot()

mins = [math.inf, math.inf, math.inf]
maxs = [-math.inf, -math.inf, -math.inf]

mesh_scales = []
bad_masses = []
coll_origins = []

for link in root.findall("link"):
    name = link.get("name","?")
    # inertial mass
    inert = link.find("inertial")
    if inert is not None:
        mass = inert.find("mass")
        if mass is not None and mass.get("value") is not None:
            m = float(mass.get("value"))
            if m <= 0 or m < 1e-6:
                bad_masses.append((name, m))

    for col in link.findall("collision"):
        org = col.find("origin")
        xyz = (0.0, 0.0, 0.0)
        if org is not None:
            xyz = f3(org.get("xyz"))
        coll_origins.append((name, xyz))

        geom = col.find("geometry")
        if geom is not None:
            mesh = geom.find("mesh")
            if mesh is not None:
                sc = mesh.get("scale")
                if sc is not None:
                    mesh_scales.append((name, sc, mesh.get("filename")))

        # update rough bounds using origin only (not true size, but catches huge offsets)
        for i in range(3):
            mins[i] = min(mins[i], xyz[i])
            maxs[i] = max(maxs[i], xyz[i])

print("== collision origin xyz range (rough) ==")
print("min:", mins, "max:", maxs)
print("\n== suspicious collision origins (|xyz|>2m) ==")
for name, xyz in coll_origins:
    if any(abs(v) > 2.0 for v in xyz):
        print(name, xyz)

print("\n== mesh scales used in collision ==")
for it in mesh_scales:
    print(it)

print("\n== suspicious masses (<=0 or extremely small) ==")
for it in bad_masses:
    print(it)
