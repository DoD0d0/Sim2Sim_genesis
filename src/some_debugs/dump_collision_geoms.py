#!/usr/bin/env python3
import xml.etree.ElementTree as ET

URDF = "/home/rczh/workspace/dodo/dodobot_v3/urdf/dodobot_v3_simple.urdf"

def f3(s, default=0.0):
    if s is None:
        return (default, default, default)
    p = s.strip().split()
    if len(p) != 3:
        return (default, default, default)
    return tuple(float(x) for x in p)

tree = ET.parse(URDF)
root = tree.getroot()

def get_link_name(col_elem):
    # parent is link
    return col_elem.getparent().get("name","?")  # not available in std ET

# std xml.etree doesn't have getparent; so we iterate links
rows = []
for link in root.findall("link"):
    lname = link.get("name","?")
    for col in link.findall("collision"):
        org = col.find("origin")
        xyz = (0.0,0.0,0.0)
        if org is not None:
            xyz = f3(org.get("xyz"))
        geom = col.find("geometry")
        if geom is None:
            rows.append((lname, xyz, "UNKNOWN", ""))
            continue
        if geom.find("box") is not None:
            size = geom.find("box").get("size","")
            rows.append((lname, xyz, "box", size))
        elif geom.find("cylinder") is not None:
            c = geom.find("cylinder")
            rows.append((lname, xyz, "cylinder", f"radius={c.get('radius')} length={c.get('length')}"))
        elif geom.find("sphere") is not None:
            s = geom.find("sphere")
            rows.append((lname, xyz, "sphere", f"radius={s.get('radius')}"))
        elif geom.find("mesh") is not None:
            m = geom.find("mesh")
            rows.append((lname, xyz, "mesh", f"file={m.get('filename')} scale={m.get('scale')}"))
        else:
            rows.append((lname, xyz, "OTHER", ""))

print("Found", len(rows), "collision geometries")
# 打印所有 box/cyl/sphere（最容易离谱）
for lname, xyz, typ, detail in rows:
    if typ in ("box","cylinder","sphere"):
        print(f"[{typ}] link={lname} origin={xyz} {detail}")

print("\nMesh collisions:")
for lname, xyz, typ, detail in rows:
    if typ == "mesh":
        print(f"[mesh] link={lname} origin={xyz} {detail}")
