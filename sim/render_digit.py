"""Render a Digit trajectory as a skeleton animation, without needing OpenGL.

Same idea as sim/render_skeleton.py but for Digit's kinematic tree, and it draws
the arms as well as the legs. Foot outlines come from the calibrated sole corners,
so the box foot is drawn as an actual footprint rather than Cassie's line blade.

Outputs (prefix set by --name):
    out/<name>_skeleton.gif
    out/<name>_overlay.png
    out/<name>_footprints.png

Run:  python sim/render_digit.py [--qpos digit_walk_qpos.npy] [--name digit_walk]
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import digit_params as dp  # noqa: E402

OUT = ROOT / "out"

LEG_CHAIN = ("hip_roll", "hip_yaw", "hip_pitch", "knee", "tarsus", "toe_pitch", "toe_roll")
ARM_CHAIN = ("shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow")
CORNERS = ("corner_tl", "corner_tr", "corner_hr", "corner_hl")
STRIDE = 3
FPS = 100 // STRIDE


def slope_view(frames, slope_deg):
    """Rotate everything so a tilted-gravity run displays as an actual hill.

    The physics runs with a flat floor and rotated gravity, which is exactly
    equivalent to a tilted floor -- but it *looks* flat. Rotating the drawn points
    by -gamma about y puts the picture back in the gravity-vertical frame, so
    uphill reads as uphill.
    """
    if not slope_deg:
        return frames
    g = np.radians(-slope_deg)
    R = np.array([[np.cos(g), 0.0, np.sin(g)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(g), 0.0, np.cos(g)]])
    out = []
    for f in frames:
        nf = {}
        for k, v in f.items():
            v = np.asarray(v)
            nf[k] = (v @ R.T) if v.ndim == 2 else (R @ v)
        out.append(nf)
    return out


def collect(qpos):
    model, data = dp.load()
    feet = dp.FootGeometry(model)

    # Calibrate from the nominal standing pose so heel/toe are labelled correctly.
    data.qpos[:] = dp.nominal_pose(model, data, feet)
    mujoco.mj_forward(model, data)
    feet.calibrate(data)

    bid = {"torso": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")}
    for side in dp.SIDES:
        for b in LEG_CHAIN + ARM_CHAIN:
            nm = f"{side}_{b}"
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if i < 0:  # a few arm bodies use a different naming order
                i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{b}_{side}")
            bid[nm] = i

    frames = []
    for k in range(0, len(qpos), STRIDE):
        data.qpos[:] = qpos[k]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        f = {"com": np.array(data.subtree_com[0]), "torso": np.array(data.xpos[bid["torso"]])}
        for nm, i in bid.items():
            if i >= 0:
                f[nm] = np.array(data.xpos[i])
        for side in dp.SIDES:
            f[f"{side}_sole"] = np.array([feet.world(data, side, c) for c in CORNERS])
            f[f"{side}_toe"] = feet.world(data, side, "toe")
            f[f"{side}_heel"] = feet.world(data, side, "heel")
        frames.append(f)
    return frames


def chain_pts(f, side, chain, root="torso"):
    pts = [f[root]]
    for b in chain:
        key = f"{side}_{b}"
        if key in f:
            pts.append(f[key])
    return np.array(pts)


def draw(ax, f, ix, iy, arms=True):
    for side, c in (("left", "C1"), ("right", "C2")):
        leg = chain_pts(f, side, LEG_CHAIN)
        ax.plot(leg[:, ix], leg[:, iy], "-o", color=c, ms=3.0, lw=2.0)
        sole = f[f"{side}_sole"]
        closed = np.vstack([sole, sole[:1]])
        ax.plot(closed[:, ix], closed[:, iy], "-", color=c, lw=2.5)
        if arms:
            arm = chain_pts(f, side, ARM_CHAIN)
            if len(arm) > 1:
                ax.plot(arm[:, ix], arm[:, iy], "-", color=c, lw=1.2, alpha=0.65)
    ax.plot([f["com"][ix]], [f["com"][iy]], "x", color="0.1", ms=7, mew=2)
    ax.plot(
        [f["left_hip_roll"][ix], f["right_hip_roll"][ix]],
        [f["left_hip_roll"][iy], f["right_hip_roll"][iy]],
        "-", color="0.25", lw=5, solid_capstyle="round",
    )
    # Torso as a stub from the hip bar upward.
    mid = 0.5 * (f["left_hip_roll"] + f["right_hip_roll"])
    ax.plot([mid[ix], f["torso"][ix]], [mid[iy], f["torso"][iy]], "-", color="0.25", lw=5)


def overlay(frames, name, n=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.linspace(int(len(frames) * 0.45), len(frames) - 1, n).astype(int)
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.axhline(0.0, color="0.7", lw=1.2)
    for j, i in enumerate(idx):
        a = 0.25 + 0.75 * j / (n - 1)
        f = frames[i]
        for side, c in (("left", "C1"), ("right", "C2")):
            leg = chain_pts(f, side, LEG_CHAIN)
            ax.plot(leg[:, 0], leg[:, 2], "-", color=c, lw=1.8, alpha=a)
            sole = f[f"{side}_sole"]
            ax.plot(np.vstack([sole, sole[:1]])[:, 0], np.vstack([sole, sole[:1]])[:, 2],
                    "-", color=c, lw=2.2, alpha=a)
        mid = 0.5 * (f["left_hip_roll"] + f["right_hip_roll"])
        ax.plot([mid[0], f["torso"][0]], [mid[2], f["torso"][2]], "-", color="0.25", lw=4, alpha=a)
        ax.plot([f["com"][0]], [f["com"][2]], "x", color="0.1", ms=6, mew=1.6, alpha=a)

    com = np.array([f["com"] for f in frames])
    ax.plot(com[:, 0], com[:, 2], "-", color="0.1", lw=0.9, alpha=0.55, label="CoM path")
    ax.set_aspect("equal")
    # Auto-scale: on a slope the whole gait translates vertically, so a fixed
    # window clips the robot straight off the figure.
    lo = min(com[:, 2].min(), min(f[f"{s}_sole"][:, 2].min() for f in frames
                                  for s in dp.SIDES))
    ax.set_ylim(lo - 0.15, com[:, 2].max() + 0.30)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Digit ALIP gait, world frame, fading = earlier (orange=left, green=right)")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_overlay.png", dpi=125)
    print(f"wrote {OUT / f'{name}_overlay.png'}")


def footprints(frames, name):
    """Top-down view: the box feet make real footprints, unlike Cassie's blade."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 3.0))
    seen = {"left": [], "right": []}
    for f in frames:
        for side, c in (("left", "C1"), ("right", "C2")):
            sole = f[f"{side}_sole"]
            if sole[:, 2].max() < 0.005:  # foot planted
                x = sole[:, 0].mean()
                if not seen[side] or abs(x - seen[side][-1]) > 0.03:
                    seen[side].append(x)
                    closed = np.vstack([sole, sole[:1]])
                    ax.fill(closed[:, 0], closed[:, 1], color=c, alpha=0.35)
                    ax.plot(closed[:, 0], closed[:, 1], "-", color=c, lw=1.0)

    com = np.array([f["com"] for f in frames])
    ax.plot(com[:, 0], com[:, 1], "-", color="0.1", lw=1.2, label="CoM path")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Digit footprints, top-down (real support polygons)")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_footprints.png", dpi=125)
    print(f"wrote {OUT / f'{name}_footprints.png'}")


def animate(frames, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (ax_s, ax_f) = plt.subplots(1, 2, figsize=(12, 4.8), width_ratios=[2, 1])
    for ax in (ax_s, ax_f):
        ax.set_aspect("equal")
        ax.set_ylim(-0.1, 1.2)
    ax_f.set_xlim(-0.5, 0.5)
    ax_s.set_ylabel("z [m]")
    ax_s.set_xlabel("x [m]")
    ax_f.set_xlabel("y [m]")
    ax_f.set_title("front view")

    def update(i):
        f = frames[i]
        for ax in (ax_s, ax_f):
            ax.clear()
            ax.set_aspect("equal")
            ax.set_ylim(-0.1, 1.2)
            ax.axhline(0.0, color="0.75", lw=1.0)
        draw(ax_s, f, 0, 2)
        draw(ax_f, f, 1, 2)
        ax_s.set_xlim(f["torso"][0] - 1.0, f["torso"][0] + 0.8)
        ax_f.set_xlim(-0.5, 0.5)
        ax_s.set_title(f"Digit ALIP walk   t = {i * STRIDE / 100:.2f} s")
        ax_f.set_title("front view")
        ax_s.set_xlabel("x [m]")
        ax_s.set_ylabel("z [m]")
        ax_f.set_xlabel("y [m]")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)
    path = OUT / f"{name}_skeleton.gif"
    anim.save(path, writer=PillowWriter(fps=FPS))
    print(f"wrote {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", default="digit_walk_qpos.npy")
    ap.add_argument("--name", default="digit_walk")
    ap.add_argument("--only", default="all", choices=("all", "gif", "still"))
    ap.add_argument("--slope-deg", type=float, default=0.0,
                    help="rotate the view so a tilted-gravity run reads as a hill")
    args = ap.parse_args()

    qpos = np.load(OUT / args.qpos)
    print(f"loaded {qpos.shape} from {args.qpos}")
    frames = slope_view(collect(qpos), args.slope_deg)
    print(f"rendering {len(frames)} frames")
    OUT.mkdir(exist_ok=True)
    if args.only in ("all", "still"):
        overlay(frames, args.name)
        footprints(frames, args.name)
    if args.only in ("all", "gif"):
        animate(frames, args.name)
