"""2D side-view animations of Digit for the README.

    python scripts/make_gifs.py            # everything (~3 min)
    python scripts/make_gifs.py terrains   # just the three-terrain GIF
    python scripts/make_gifs.py compare    # just ALIP vs no ALIP

Writes images/anim_*_light.gif and images/anim_*_dark.gif (GitHub shows the one
that matches the viewer's theme). Edit THEMES to change colours.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import alip  # noqa: E402
import digit_params as dp  # noqa: E402
from osc import OSC, OSCConfig  # noqa: E402
from walk_controller import WalkCommand, WalkController  # noqa: E402

IMG = ROOT / "images"
FPS = 25
FRAME_DT = 1.0 / FPS

THEMES = {
    "light": dict(bg="#FFFFFF", ground="#D0D7DE", ground_line="#8C959F", text="#1F2328",
                  sub="#57606A", left="#D9822B", right="#2B7BB9", torso="#57606A",
                  com="#1F2328", good="#1A7F37", bad="#CF222E"),
    "dark": dict(bg="#0D1117", ground="#21262D", ground_line="#6E7681", text="#E6EDF3",
                 sub="#8B949E", left="#F0883E", right="#58A6FF", torso="#8B949E",
                 com="#E6EDF3", good="#3FB950", bad="#F85149"),
}

CHAIN = ("hip_roll", "hip_pitch", "knee", "tarsus", "toe_pitch")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _blind_footstep(ctl):
    """'No ALIP': the foot always lands one fixed stride ahead, whatever the body does."""
    original = ctl.plan_footstep

    def plan(data):
        _, st = original(data)
        s = ctl.state
        sx = ctl.cmd.v_forward * ctl.p.T
        sy = -ctl.p.W if s.stance_is_left else ctl.p.W
        t = s.stance_point + np.array([sx, sy, 0.0])
        t[2] = ctl.ground_z
        return t, st

    ctl.plan_footstep = plan


def simulate(seconds, speed, slope_deg=0.0, blind=False, ramp_after=0.5, ramp_time=1.0):
    """Closed-loop walk; returns poses sampled at FPS. Keeps running after a fall."""
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q_nom = dp.nominal_pose(model, data, feet)
    slope = np.radians(slope_deg)
    g = 9.806
    model.opt.gravity[:] = [-g * np.sin(slope), 0.0, -g * np.cos(slope)]
    data.qpos[:] = q_nom
    data.qpos[2] -= 0.001
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    feet.calibrate(data)

    meas = dp.measure(model, data, feet)
    p = alip.AlipParams(m=meas["total_mass"], H=meas["com_height_above_feet"], T=0.35,
                        W=meas["foot_separation_y"], g=g, slope=slope)
    if slope:  # start leaning into the hill, as digit_walk.py does
        from digit_ik import DigitIK
        ik = DigitIK()
        ik.data.qpos[:] = data.qpos
        mujoco.mj_forward(ik.model, ik.data)
        ik.feet.calibrate(ik.data)
        tg = {s: ik.feet.contact_point(ik.data, s).copy() for s in dp.SIDES}
        base = np.array(data.qpos[0:3])
        base[0] += p.x_equilibrium
        r = ik.solve(np.array(data.qpos), tg, base_pos=base, base_quat=(1, 0, 0, 0),
                     flat_feet=True, yaw_target={"left": 0.0, "right": 0.0},
                     max_iter=200, tol=1e-6)
        if r.converged:
            data.qpos[:] = r.qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

    cfg = OSCConfig()
    cfg.w_stance_ankle = 20.0
    osc = OSC(model, feet, config=cfg)
    cmd = WalkCommand(v_forward=0.0, swing_height=0.08)
    ctl = WalkController(model, feet, osc, p, cmd)
    ctl.reset(data, stance_is_left=True)
    if blind:
        _blind_footstep(ctl)

    dt = model.opt.timestep
    every = max(1, int(round(FRAME_DT / dt)))
    tau = np.zeros(osc.act.n)
    fell_at = None
    frames = []
    bid = {f"{s}_{b}": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_{b}")
           for s in dp.SIDES for b in CHAIN}
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    for i in range(int(seconds / dt)):
        t = i * dt
        cmd.v_forward = (0.0 if t < ramp_after else
                         speed * min(1.0, (t - ramp_after) / ramp_time))
        if i % 2 == 0:
            try:
                tn, info = ctl.update(data, t, 2 * dt)
                if tn is not None and info["info"]["ok"]:
                    tau = tn
            except Exception:
                tau = np.zeros(osc.act.n)
        if fell_at is not None:
            tau = np.zeros(osc.act.n)   # after a fall, go limp like a real e-stop
        osc.act.apply_torque(data, tau)

        if i % every == 0:
            f = dict(t=t, com=np.array(data.subtree_com[0]),
                     torso=np.array(data.xpos[torso]),
                     torso_up=np.array(data.xmat[torso]).reshape(3, 3)[:, 2])
            for s in dp.SIDES:
                f[s] = np.array([data.xpos[bid[f"{s}_{b}"]] for b in CHAIN])
                f[f"{s}_heel"] = feet.world(data, s, "heel")
                f[f"{s}_toe"] = feet.world(data, s, "toe")
            frames.append(f)
        mujoco.mj_step(model, data)
        if fell_at is None and data.qpos[2] < q_nom[2] - 0.25:
            fell_at = t
    return dict(frames=frames, fell_at=fell_at, slope=slope_deg, steps=ctl.state.step_index)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _rot(p, deg):
    """Slope frame -> screen frame (gravity straight down)."""
    a = np.radians(deg)
    p = np.asarray(p)
    return np.stack([p[..., 0] * np.cos(a) - p[..., 2] * np.sin(a),
                     p[..., 0] * np.sin(a) + p[..., 2] * np.cos(a)], -1)


class Panel:
    """One robot in one axes; update(frame) redraws it."""

    def __init__(self, ax, run, title, T, half_width=1.0):
        self.ax, self.run, self.T, self.hw = ax, run, T, half_width
        self.deg = run["slope"]
        ax.set_facecolor(T["bg"])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=T["text"], fontsize=13, fontweight="bold", pad=6)

        fr = run["frames"]
        gz = min(min(f["left_heel"][2], f["right_heel"][2]) for f in fr[:10])
        xs = np.array([fr[0]["com"][0] - 5, fr[-1]["com"][0] + 5])
        self.ground = _rot(np.stack([xs, 0 * xs, gz + 0 * xs], -1), self.deg)
        gx = self.ground[:, 0]
        gy = self.ground[:, 1]
        ax.fill_between(gx, gy, gy - 3, color=T["ground"], lw=0, zorder=0)
        ax.plot(gx, gy, color=T["ground_line"], lw=1.6, zorder=1)
        # tick marks on the ground every 0.5 m so you can see it moving
        for x in np.arange(np.floor(xs[0]), xs[1], 0.5):
            a = _rot(np.array([x, 0, gz]), self.deg)
            b = _rot(np.array([x, 0, gz - 0.06]), self.deg)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=T["ground_line"], lw=1, zorder=1)

        self.trail, = ax.plot([], [], color=T["com"], lw=0.8, alpha=0.35, zorder=2)
        self.legs = {}
        for s in ("right", "left"):   # right leg drawn behind
            leg, = ax.plot([], [], "-", color=T[s], lw=3.2, solid_capstyle="round",
                           solid_joinstyle="round", zorder=3 if s == "right" else 5)
            foot, = ax.plot([], [], "-", color=T[s], lw=6, solid_capstyle="round",
                            zorder=3 if s == "right" else 5)
            self.legs[s] = (leg, foot)
        self.body, = ax.plot([], [], "-", color=T["torso"], lw=9, solid_capstyle="round",
                             zorder=4)
        self.head, = ax.plot([], [], "o", color=T["torso"], ms=13, zorder=4)
        self.dot, = ax.plot([], [], "o", color=T["com"], ms=6, zorder=6)
        self.label = ax.text(0.03, 0.95, "", transform=ax.transAxes, ha="left", va="top",
                             fontsize=12, fontweight="bold", zorder=7)
        self.clock = ax.text(0.97, 0.95, "", transform=ax.transAxes, ha="right", va="top",
                             fontsize=10.5, color=T["sub"], zorder=7, family="monospace")
        # ground height on screen as a function of screen x, for the camera
        k = (gy[1] - gy[0]) / (gx[1] - gx[0])
        self.ground_y = lambda x: gy[0] + k * (x - gx[0])
        self.cam = None

    def update(self, f, note=None, note_color=None):
        d = self.deg
        for s in ("left", "right"):
            pts = _rot(np.vstack([f["torso"], f[s]]), d)
            self.legs[s][0].set_data(pts[:, 0], pts[:, 1])
            ft = _rot(np.vstack([f[f"{s}_heel"], f[f"{s}_toe"]]), d)
            self.legs[s][1].set_data(ft[:, 0], ft[:, 1])
        top = f["torso"] + 0.42 * f["torso_up"]
        bd = _rot(np.vstack([f["torso"], top]), d)
        self.body.set_data(bd[:, 0], bd[:, 1])
        hd = _rot(f["torso"] + 0.56 * f["torso_up"], d)
        self.head.set_data([hd[0]], [hd[1]])
        c = _rot(f["com"], d)
        self.dot.set_data([c[0]], [c[1]])
        self._trail.append(c)
        tr = np.array(self._trail[-60:])
        self.trail.set_data(tr[:, 0], tr[:, 1])
        # camera slides along the ground with the robot; it never drops when she falls
        self.cam = c[0] if self.cam is None else 0.85 * self.cam + 0.15 * c[0]
        cx = self.cam
        cy = self.ground_y(cx)
        self.ax.set_xlim(cx - self.hw, cx + self.hw)
        self.ax.set_ylim(cy - 0.35, cy + 1.6)
        self.clock.set_text(f"t = {f['t']:.1f} s")
        self.label.set_text(note or "")
        if note_color:
            self.label.set_color(note_color)

    def reset(self):
        self._trail = []
        self.cam = None


def _save(fig, update, n, name):
    from matplotlib.animation import FuncAnimation, PillowWriter

    IMG.mkdir(exist_ok=True)
    path = IMG / f"{name}.gif"
    anim = FuncAnimation(fig, update, frames=n, blit=False)
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=fig.dpi,
              savefig_kwargs=dict(facecolor=fig.get_facecolor()))
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1e6:.1f} MB)")


def gif_terrains(runs, theme):
    import matplotlib.pyplot as plt

    T = THEMES[theme]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=96)
    fig.patch.set_facecolor(T["bg"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.04)
    titles = ["Flat ground", "22° uphill", "22° downhill"]
    panels = [Panel(ax, r, t, T) for ax, r, t in zip(axes, runs, titles)]
    for p in panels:
        p.reset()
    start = int(1.5 * FPS)
    n = min(len(r["frames"]) for r in runs) - start

    def update(i):
        for p in panels:
            p.update(p.run["frames"][start + i])

    _save(fig, update, n, f"anim_terrains_{theme}")
    plt.close(fig)


def gif_compare(with_alip, without, theme):
    import matplotlib.pyplot as plt

    T = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), dpi=100)
    fig.patch.set_facecolor(T["bg"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.04)
    pa = Panel(axes[0], with_alip, "With ALIP: decides where to step", T)
    pb = Panel(axes[1], without, "Without ALIP: fixed steps", T)
    for p in (pa, pb):
        p.reset()
    n = min(len(with_alip["frames"]), len(without["frames"]))
    fell = without["fell_at"]

    def update(i):
        pa.update(with_alip["frames"][i], "walking", T["good"])
        f = without["frames"][i]
        down = fell is not None and f["t"] >= fell
        pb.update(f, "fell over" if down else "", T["bad"])

    _save(fig, update, n, f"anim_alip_vs_none_{theme}")
    plt.close(fig)


def main(which):
    import matplotlib
    matplotlib.use("Agg")

    if "terrains" in which:
        print("simulating terrains ...", flush=True)
        runs = [simulate(7.5, 0.4, s) for s in (0.0, 22.0, -22.0)]
        for th in THEMES:
            gif_terrains(runs, th)
    if "compare" in which:
        print("simulating ALIP vs no ALIP ...", flush=True)
        a = simulate(4.0, 0.4, 0.0, blind=False)
        b = simulate(4.0, 0.4, 0.0, blind=True)
        print(f"  no ALIP fell at t={b['fell_at']}")
        for th in THEMES:
            gif_compare(a, b, th)


if __name__ == "__main__":
    main(sys.argv[1:] or ["terrains", "compare"])
