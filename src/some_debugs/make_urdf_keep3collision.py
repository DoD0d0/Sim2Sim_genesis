#!/usr/bin/env python3
import xml.etree.ElementTree as ET

SRC = "/home/rczh/workspace/dodo/dodobot_v3/urdf/dodobot_v3_simple.urdf"
DST = "/home/rczh/workspace/dodo/dodobot_v3/urdf/dodobot_v3_keep3collision.urdf"

KEEP_LINKS = {"body_link", "left_link_4", "right_link_4"}

tree = ET.parse(SRC)
root = tree.getroot()

removed = 0
kept = 0

for link in root.findall("link"):
    lname = link.get("name","?")
    cols = list(link.findall("collision"))
    for col in cols:
        if lname in KEEP_LINKS:
            kept += 1
            continue
        link.remove(col)
        removed += 1

tree.write(DST)
print("Wrote:", DST)
print("Kept collisions:", kept, "Removed collisions:", removed)
