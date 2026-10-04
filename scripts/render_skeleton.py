"""Render the solved walking trajectory as a skeleton animation (no OpenGL needed).

MuJoCo's own renderer needs a GL context, which is not always available (headless
boxes, containers without the Mesa libraries). This draws the same motion from
body and site positions with matplotlib, which is enough to eyeball whether the
gait is sane. For the real thing, use scripts/play_walk.py under mjpython.

Outputs:
    out/walk_skeleton.gif        animation
    out/walk_filmstrip.png       key frames side by side

Run:  python scripts/render_skeleton.py
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cassie_params  # noqa: E402

OUT = ROOT / "out"

# Chain of body origins that reads as a leg, plus the foot blade from its sites.
LEG_BODIES = ("hip-roll", "hip-pitch", "knee", "tarsus", "foot")
STRIDE = 3  # render every Nth frame


def collect(qpos):
    """Forward kinematics for every frame, gathering the points we want to draw."""
    model, data = cassie_params.load()

    bid = {
        f"{side}-{b}": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}-{b}")
        for side in ("left", "right")
        for b in LEG_BODIES
    }
    bid["pelvis"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cassie-pelvis")
    sid = {
        f"{side}-{s}": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}-{s}")
        for side in ("left", "right")
        for s in ("toe", "heel")
    }

    frames = []
    for i in range(0, len(qpos), STRIDE):
        data.qpos[:] = qpos[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        f = {"com": np.array(data.subtree_com[0])}
        for name, b in bid.items():
            f[name] = np.array(data.xpos[b])
        for name, s in sid.items():
            f[name] = np.array(data.site_xpos[s])
        frames.append(f)
    return frames


def leg_chain(f, side):
    pts = [f["pelvis"]] + [f[f"{side}-{b}"] for b in LEG_BODIES]
    return np.array(pts)


def foot_blade(f, side):
    return np.array([f[f"{side}-heel"], f[f"{side}-toe"]])


def draw(ax, f, ix, iy, colors=("C1", "C2")):
    """Draw one frame projected onto axes (ix, iy) of world coordinates."""
    artists = []
    for side, c in zip(("left", "right"), colors):
        chain = leg_chain(f, side)
        (ln,) = ax.plot(chain[:, ix], chain[:, iy], "-o", color=c, ms=3.5, lw=2.0)
        blade = foot_blade(f, side)
        (bl,) = ax.plot(blade[:, ix], blade[:, iy], "-", color=c, lw=4.0, solid_capstyle="round")
        artists += [ln, bl]
    # Pelvis as a short bar, and the CoM.
    p = f["pelvis"]
    (cm,) = ax.plot([f["com"][ix]], [f["com"][iy]], "x", color="0.1", ms=7, mew=2)
    (pv,) = ax.plot(
        [f["left-hip-roll"][ix], f["right-hip-roll"][ix]],
        [f["left-hip-roll"][iy], f["right-hip-roll"][iy]],
        "-",
        color="0.25",
        lw=5,
        solid_capstyle="round",
    )
    artists += [cm, pv]
    return artists


def filmstrip(frames, n=6, name="walk"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sample one full step cycle from the cruise phase.
    start = int(len(frames) * 0.62)
    idx = np.linspace(start, min(start + len(frames) // 7, len(frames) - 1), n).astype(int)

    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 4.2), sharey=True)
    for ax, i in zip(axes, idx):
        f = frames[i]
        ax.axhline(0.0, color="0.75", lw=1.0)
        draw(ax, f, 0, 2)
        ax.set_xlim(f["pelvis"][0] - 0.6, f["pelvis"][0] + 0.6)
        ax.set_ylim(-0.1, 1.25)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_title(f"t={i * STRIDE / 100:.2f}s", fontsize=9)
    axes[0].set_ylabel("z [m]")
    fig.suptitle(f"{name}: side view -- orange=left, green=right", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_filmstrip.png", dpi=120)
    print(f"wrote {OUT / f'{name}_filmstrip.png'}")


def overlay(frames, n=9, name="walk"):
    """World-frame overlay: successive poses side by side, so translation shows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = int(len(frames) * 0.45)
    idx = np.linspace(start, len(frames) - 1, n).astype(int)

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axhline(0.0, color="0.7", lw=1.2)
    for j, i in enumerate(idx):
        a = 0.25 + 0.75 * j / (n - 1)
        f = frames[i]
        for side, c in (("left", "C1"), ("right", "C2")):
            chain = leg_chain(f, side)
            ax.plot(chain[:, 0], chain[:, 2], "-", color=c, lw=1.8, alpha=a)
            blade = foot_blade(f, side)
            ax.plot(blade[:, 0], blade[:, 2], "-", color=c, lw=3.5, alpha=a,
                    solid_capstyle="round")
        ax.plot(
            [f["left-hip-roll"][0], f["right-hip-roll"][0]],
            [f["left-hip-roll"][2], f["right-hip-roll"][2]],
            "-", color="0.25", lw=4, alpha=a, solid_capstyle="round",
        )
        ax.plot([f["com"][0]], [f["com"][2]], "x", color="0.1", ms=6, mew=1.6, alpha=a)

    com = np.array([f["com"] for f in frames])
    ax.plot(com[:, 0], com[:, 2], "-", color="0.1", lw=0.8, alpha=0.5, label="CoM path")
    ax.set_aspect("equal")
    ax.set_ylim(-0.08, 1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{name}: world coordinates, fading = earlier (orange=left, green=right)")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_overlay.png", dpi=125)
    print(f"wrote {OUT / f'{name}_overlay.png'}")


def animate(frames, name="walk"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (ax_side, ax_front) = plt.subplots(1, 2, figsize=(12, 4.6), width_ratios=[2, 1])

    for ax in (ax_side, ax_front):
        ax.set_aspect("equal")
        ax.axhline(0.0, color="0.75", lw=1.0)
        ax.set_ylim(-0.1, 1.25)
    ax_side.set_title("side view (walking direction ->)")
    ax_front.set_title("front view")
    ax_side.set_xlabel("x [m]")
    ax_front.set_xlabel("y [m]")
    ax_side.set_ylabel("z [m]")
    ax_front.set_xlim(-0.45, 0.45)

    # Footstep marks accumulate on the side view so the stride is visible.
    step_marks = []
    prev_contact = {"left": None, "right": None}

    def update(i):
        f = frames[i]
        for ax in (ax_side, ax_front):
            for art in list(ax.lines):
                if getattr(art, "_keep", False):
                    continue
                art.remove()
            ax.axhline(0.0, color="0.75", lw=1.0)

        draw(ax_side, f, 0, 2)
        draw(ax_front, f, 1, 2)

        # Mark a footprint whenever a foot is on the ground.
        for side, c in (("left", "C1"), ("right", "C2")):
            z = f[f"{side}-toe"][2]
            if z < 0.005:
                x = f[f"{side}-foot"][0]
                if prev_contact[side] is None or abs(x - prev_contact[side]) > 0.02:
                    prev_contact[side] = x
                    step_marks.append((x, c))

        for x, c in step_marks:
            (m,) = ax_side.plot([x], [0.0], "v", color=c, ms=5, alpha=0.45)
            m._keep = False

        ax_side.set_xlim(f["pelvis"][0] - 1.0, f["pelvis"][0] + 0.8)
        ax_side.set_title(f"side view   t = {i * STRIDE / 100:.2f} s")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=STRIDE * 10, blit=False)
    path = OUT / f"{name}_skeleton.gif"
    anim.save(path, writer=PillowWriter(fps=int(100 / STRIDE)))
    print(f"wrote {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", default="walk_qpos.npy", help="file in out/")
    ap.add_argument("--name", default="walk", help="output filename prefix")
    ap.add_argument("--only", default="all", choices=("all", "gif", "still"))
    args = ap.parse_args()

    NAME = args.name
    qpos = np.load(OUT / args.qpos)
    print(f"loaded {qpos.shape[0]} frames from {args.qpos}")
    frames = collect(qpos)
    print(f"rendering {len(frames)} frames")
    OUT.mkdir(exist_ok=True)
    if args.only in ("all", "still"):
        filmstrip(frames, name=NAME)
        overlay(frames, name=NAME)
    if args.only in ("all", "gif"):
        animate(frames, name=NAME)
