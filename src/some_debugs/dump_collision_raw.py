#!/usr/bin/env python3
import xml.etree.ElementTree as ET

URDF = "/home/rczh/workspace/dodo/dodobot_v3/urdf/dodobot_v3_simple.urdf"

tree = ET.parse(URDF)
root = tree.getroot()

def get_xyz(col):
    org = col.find("origin")
    if org is None:
        return None
    s = org.get("xyz")
    if s is None:
        return None
    return s.strip()

def first_child_tag(elem):
    if elem is None:
        return None
    for c in list(elem):
        return c.tag
    return None

count = 0
for link in root.findall("link"):
    lname = link.get("name","?")
    for col in link.findall("collision"):
        count += 1
        xyz = get_xyz(col)
        geom = col.find("geometry")
        gtag = first_child_tag(geom)
        # dump details
        print(f"\n=== collision #{count} link={lname} origin_xyz={xyz} geom_tag={gtag} ===")
        if geom is None:
            print("NO <geometry>!")
            continue
        # print all children and attributes
        for child in list(geom):
            print("child:", child.tag, "attrs:", child.attrib)
            # if mesh, print filename/scale
            if child.tag.endswith("mesh"):
                print("  mesh filename:", child.get("filename"), "scale:", child.get("scale"))
            if child.tag.endswith("box"):
                print("  box size:", child.get("size"))
            if child.tag.endswith("cylinder"):
                print("  cylinder radius:", child.get("radius"), "length:", child.get("length"))
            if child.tag.endswith("sphere"):
                print("  sphere radius:", child.get("radius"))

print("\nTotal collisions:", count)
