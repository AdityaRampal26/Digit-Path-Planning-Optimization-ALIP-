"""Render a solid, shaded still of Digit without OpenGL.

MuJoCo's own renderer needs a GL context, which isn't available on every machine
(no Mesa, no root to install it). But the model already contains everything a
renderer needs: mesh vertices, per-vertex normals, geom poses and colours. So this
projects the vertices itself, depth-sorts them, applies Lambert shading, and splats
them with matplotlib.

717k triangles is too many for matplotlib's polygon path, but 366k *vertices*
splat fine and read as solid geometry at slide resolution.

Usage:
    python sim/render_still.py --qpos digit_stageC_up22_qpos.npy --slope-deg 22 \
        --frame 0.6 --out out/still_up22.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "sim"))

import digit_params as dp  # noqa: E402

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def gather_vertices(model, data):
    """World-space vertices, normals and colours for every visual mesh geom."""
    P, N, C = [], [], []
    for g in range(model.ngeom):
        if model.geom_group[g] != 1:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[adr:adr + num].reshape(-1, 3)
        n = model.mesh_normal[adr:adr + num].reshape(-1, 3)

        # NOTE: do NOT apply mesh_pos / mesh_quat here. MuJoCo has already folded the
        # mesh frame into geom_xpos / geom_xmat -- mesh_pos is literally the geom's
        # own offset within its body. Applying it again displaces every part by its
        # own offset and the robot renders as an exploded diagram.
        Rg = np.array(data.geom_xmat[g]).reshape(3, 3)
        P.append(v @ Rg.T + np.array(data.geom_xpos[g]))
        N.append(n @ Rg.T)
        C.append(np.tile(model.geom_rgba[g][:3], (num, 1)))
    return np.vstack(P), np.vstack(N), np.vstack(C)


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.vstack([s, u, -f])          # world -> camera


def render(qpos_file, slope_deg, frame_frac, out_path, width=2000, height=1250,
           azim=118.0, elev=12.0, dist=3.4, point_size=1.6, dpi=170):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from play_digit_ramp import make_ramp_xml, rotate_trajectory

    xml = make_ramp_xml(slope_deg)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    qpos = rotate_trajectory(np.load(OUT / qpos_file), slope_deg)
    i = int(np.clip(frame_frac, 0, 1) * (len(qpos) - 1))
    data.qpos[:] = qpos[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    P, N, C = gather_vertices(model, data)

    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    target = np.array(data.xpos[torso]) + np.array([0.0, 0.0, -0.15])
    a, e = np.radians(azim), np.radians(elev)
    eye = target + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    R = look_at(eye, target)

    # Lambert shading with a soft fill, so unlit faces don't go black.
    light = np.array([-0.35, -0.75, 0.85]); light /= np.linalg.norm(light)
    nn = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    lam = np.clip(nn @ light, 0.0, 1.0)
    shade = (0.42 + 0.58 * lam)[:, None]
    rgb = np.clip(C * shade, 0, 1)

    cam = (P - eye) @ R.T
    z = -cam[:, 2]
    keep = z > 0.05
    cam, rgb, z = cam[keep], rgb[keep], z[keep]

    f = 1.5
    sx = f * cam[:, 0] / z
    sy = f * cam[:, 1] / z

    order = np.argsort(-z)               # far to near: painter's algorithm
    sx, sy, rgb, z = sx[order], sy[order], rgb[order], z[order]

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#f2f4f6")

    # Ground: project the tilted plane as a large quad.
    gamma = np.radians(-slope_deg)
    Rg = np.array([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0],
                   [-np.sin(gamma), 0, np.cos(gamma)]])
    L = 14.0
    corners = np.array([[-L, -L, 0], [L, -L, 0], [L, L, 0], [-L, L, 0]]) @ Rg.T
    gc = (corners - eye) @ R.T
    gz = -gc[:, 2]
    if np.all(gz > 0.05):
        gx, gy = f * gc[:, 0] / gz, f * gc[:, 1] / gz
        ax.add_patch(Polygon(np.column_stack([gx, gy]), closed=True,
                             facecolor="#d9dee3", edgecolor="none", zorder=0))

    ax.scatter(sx, sy, c=rgb, s=point_size, marker="s", linewidths=0, zorder=2)

    pad = 0.10
    ax.set_xlim(sx.min() - pad, sx.max() + pad)
    ax.set_ylim(sy.min() - pad, sy.max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    out_path = pathlib.Path(out_path)
    fig.savefig(out_path, dpi=dpi, facecolor="#f2f4f6")
    plt.close(fig)
    print(f"wrote {out_path}  ({len(sx)} points, frame {i}/{len(qpos)-1})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", required=True)
    ap.add_argument("--slope-deg", type=float, default=0.0)
    ap.add_argument("--frame", type=float, default=0.6, help="0..1 through the run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--azim", type=float, default=118.0)
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--dist", type=float, default=3.4)
    args = ap.parse_args()
    render(args.qpos, args.slope_deg, args.frame, args.out,
           azim=args.azim, elev=args.elev, dist=args.dist)
