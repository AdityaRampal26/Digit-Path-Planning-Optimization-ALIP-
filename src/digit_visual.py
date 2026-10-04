"""Give the Digit model a proper colour scheme.

The MJCF paints every visual mesh with the `visual` default class,
`rgba="0.8 0.8 0.8 1"`, and STL files carry no colour of their own -- so the whole
robot renders as one uniform grey mass and you cannot tell a shin from an arm.

This assigns colours per component by mesh name, following Agility's livery:
teal torso (#1e5d88), light grey frame and limb shells (#d5d5d5), and matte dark
joints and motor housings.

`tint_sides=True` additionally warms the left leg and cools the right, which makes
gait footage far easier to follow -- useful for a portfolio video, less faithful
to the hardware.

Used by the playback scripts; can also be run directly to write a coloured model:

    python src/digit_visual.py            # -> models/digit/xml/digit_colored.xml
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
XML_DIR = ROOT / "models" / "digit" / "xml"

# Agility Digit's actual livery.
#   torso / primary body   teal    #1e5d88  = (30, 93, 136)/255
#   frame and limb shells  light   #d5d5d5  = (213, 213, 213)/255
#   joints and motors      matte dark, low reflectance
#
# Matched against the mesh name, longest key first so specific beats generic.
TEAL = "0.118 0.365 0.533 1"      # #1e5d88
SHELL = "0.835 0.835 0.835 1"     # #d5d5d5
DARK = "0.14 0.15 0.16 1"         # joints / motor housings
DARKER = "0.10 0.11 0.12 1"       # feet
LINKAGE = "0.45 0.46 0.48 1"      # rods: mid grey metallic, still readable

PALETTE = {
    # teal body
    "torso":            TEAL,
    "waist-cap":        TEAL,
    "shoulder-cap":     TEAL,
    # dark joint and motor housings
    "hip-yaw-housing":  DARK,
    "hip-pitch-housing": DARK,
    "hip-pitch":        DARK,
    "knee":             DARK,
    # light frame and limb shells
    "shin":             SHELL,
    "tarsus":           SHELL,
    "arm-L1":           SHELL,
    "arm-L2":           SHELL,
    "arm-L3":           SHELL,
    "arm-L4":           SHELL,
    # closed-chain linkage: distinct enough to watch the four-bars articulate
    "achilles-rod":     LINKAGE,
    "heel-spring":      LINKAGE,
    "toe-A-rod":        LINKAGE,
    "toe-B-rod":        LINKAGE,
    # feet
    "toe-output":       DARK,
    "toe-pitch":        DARK,
    "toe-roll":         DARKER,
}

LEFT_TINT = (1.06, 0.98, 0.88)
RIGHT_TINT = (0.88, 0.98, 1.08)

# Match a whole self-closing <geom .../> tag. Attribute order varies and some tags
# carry pos/quat/type before the mesh name -- the waist and shoulder caps do, and a
# narrower pattern silently leaves them the default grey.
GEOM_RE = re.compile(r"<geom\b[^>]*?/>", re.DOTALL)
MESH_RE = re.compile(r'mesh="([A-Za-z0-9\-]+)"')


def _base_name(mesh: str) -> str:
    """Right-side meshes are the left ones with a trailing '1'."""
    return mesh[:-1] if mesh.endswith("1") and not mesh.endswith("L1") else mesh


def _lookup(mesh: str) -> str:
    name = _base_name(mesh)
    for key in sorted(PALETTE, key=len, reverse=True):
        if key in name:
            return PALETTE[key]
    return "0.55 0.56 0.58 1"


def _apply_tint(rgba: str, tint) -> str:
    r, g, b, a = (float(v) for v in rgba.split())
    r, g, b = (min(1.0, c * t) for c, t in zip((r, g, b), tint))
    return f"{r:.3f} {g:.3f} {b:.3f} {a:.3f}"


def colorize(text: str, tint_sides: bool = False) -> str:
    """Insert an explicit rgba on every visual mesh geom."""

    count = 0

    def repl(match):
        nonlocal count
        tag = match.group(0)
        if 'class="visual"' not in tag or "rgba=" in tag:
            return tag
        m = MESH_RE.search(tag)
        if not m:
            return tag
        mesh = m.group(1)
        rgba = _lookup(mesh)
        if tint_sides:
            is_right = mesh.endswith("1") and not mesh.endswith("L1")
            rgba = _apply_tint(rgba, RIGHT_TINT if is_right else LEFT_TINT)
        count += 1
        return tag[:-2].rstrip() + f' rgba="{rgba}" />'

    out = GEOM_RE.sub(repl, text)
    if count == 0:
        raise RuntimeError("no visual mesh geoms matched; model layout changed")
    return out


def write_colored(src=None, dst=None, tint_sides: bool = False) -> pathlib.Path:
    src = pathlib.Path(src or XML_DIR / "digit_patched.xml")
    dst = pathlib.Path(dst or XML_DIR / "digit_colored.xml")
    dst.write_text(colorize(src.read_text(), tint_sides))
    return dst


if __name__ == "__main__":
    import sys

    import mujoco

    tint = "--tint" in sys.argv
    path = write_colored(tint_sides=tint)
    model = mujoco.MjModel.from_xml_path(str(path))
    n = sum(1 for g in range(model.ngeom) if model.geom_group[g] == 1)
    print(f"wrote {path}  ({n} visual geoms coloured, tint_sides={tint})")
