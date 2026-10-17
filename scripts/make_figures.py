"""Every README figure, from one place.

Two steps, so the slow part only runs once:

    python scripts/make_figures.py --collect     # run the sims (~10 min), cache to out/figdata.pkl
    python scripts/make_figures.py               # redraw all figures from the cache (seconds)
    python scripts/make_figures.py --only envelope telemetry

Figures land in images/. To restyle everything, edit the STYLE block below;
to change one figure, edit its fig_* function.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CACHE = ROOT / "out" / "figdata.pkl"
IMG = ROOT / "images"

# Grades swept for the capability envelope (deg, + = uphill).
GRADES = [-30, -25, -22, -18, -15, -10, -5, 0, 5, 10, 15, 18, 22, 25]
TORQUE_GRADES = [0, 22, -22, 25]
V_CMD = 0.4  # m/s, same as sweep_envelope.py

# ---------------------------------------------------------------------------
# STYLE — change these and every figure follows.
# ---------------------------------------------------------------------------
STYLE = dict(
    left="#D9822B",       # left leg / left foot
    right="#2B7BB9",      # right leg / right foot
    com="#222222",        # centre of mass
    good="#3A8D5C",       # pass / target band
    bad="#C0392B",        # fail / limit
    muted="#9AA0A6",      # reference lines, ground
    ink="#1F2328",        # text (GitHub's text colour)
    bg="#FFFFFF",
    font="DejaVu Sans",
    dpi=200,
)


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": STYLE["font"],
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.edgecolor": "#D0D7DE",
        "axes.labelcolor": STYLE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#EAEEF2",
        "grid.linewidth": 0.8,
        "xtick.color": "#57606A",
        "ytick.color": "#57606A",
        "text.color": STYLE["ink"],
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.facecolor": STYLE["bg"],
        "axes.facecolor": STYLE["bg"],
        "savefig.facecolor": STYLE["bg"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
    return plt


def save(fig, name):
    IMG.mkdir(exist_ok=True)
    path = IMG / f"{name}.png"
    fig.savefig(path, dpi=STYLE["dpi"])
    print(f"wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Data collection (slow; cached)
# ---------------------------------------------------------------------------

def _frames(qpos, stride=1):
    """Digit leg/torso points for each frame, for side-view stick figures."""
    import mujoco
    import digit_params as dp

    model, data = dp.load()
    feet = dp.FootGeometry(model)
    chain = ("hip_roll", "hip_pitch", "knee", "tarsus", "toe_pitch")
    bid = {f"{s}_{b}": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_{b}")
           for s in ("left", "right") for b in chain}
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    frames = []
    for q in qpos[::stride]:
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        f = {"com": np.array(data.subtree_com[0]), "torso": np.array(data.xpos[torso])}
        for s in ("left", "right"):
            f[s] = np.array([data.xpos[bid[f"{s}_{b}"]] for b in chain])
            f[f"{s}_heel"] = feet.world(data, s, "heel")
            f[f"{s}_toe"] = feet.world(data, s, "toe")
        frames.append(f)
    return frames


def _walk(seconds, speed, slope):
    import digit_walk

    out = digit_walk.run(seconds=seconds, speed=speed, slope_deg=slope, verbose=False)
    log = out["log"]
    keys = ("t", "vx", "cmd_v", "com_z", "tilt", "sat", "Ly", "stance")
    arr = {k: np.array([d[k] for d in log]) for k in keys}
    arr["com"] = np.array([d["com"] for d in log])
    return dict(log=arr, qpos=out["qpos"], touchdowns=out["touchdowns"],
                steps=out["steps"], fell_at=out["fell_at"], failures=out["failures"],
                params=dict(T=out["params"].T, H=out["params"].H, m=out["params"].m),
                slope=slope, speed=speed)


def _cassie_pd():
    from physics_rollout import rollout

    qpos, diag = rollout(seconds=2.5, controller="pd", verbose=False)
    return dict(qpos=qpos, t=np.array([d["t"] for d in diag]),
                com_z=np.array([d["com_z"] for d in diag]),
                joint_err=np.degrees(np.array([d["joint_err"] for d in diag])))


def collect():
    """Run every sim once. Saves after each stage and skips stages already cached,
    so an interrupted run picks up where it stopped (delete the cache to redo all)."""
    import sweep_envelope as se

    data = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    CACHE.parent.mkdir(exist_ok=True)

    def stage(key, fn):
        if key in data:
            return
        print(f"{key} ...", flush=True)
        data[key] = fn()
        CACHE.write_bytes(pickle.dumps(data))

    def walk(seconds, speed, slope):
        d = _walk(seconds, speed, slope)
        d["frames"] = _frames(d["qpos"], stride=2)
        return d

    stage("flat", lambda: walk(20.0, 0.5, 0.0))
    stage("up22", lambda: walk(8.0, V_CMD, 22.0))
    stage("down22", lambda: walk(8.0, V_CMD, -22.0))
    stage("pd", _cassie_pd)
    data.setdefault("envelope", {})
    data.setdefault("torque", {})
    for g in GRADES:
        if g not in data["envelope"]:
            r = se.evaluate(g)
            print(f"  {g:+4d} deg  {r['verdict']}  v={r['speed']:.3f}  fell={r['fell']}",
                  flush=True)
            data["envelope"][g] = r
            CACHE.write_bytes(pickle.dumps(data))
    for g in TORQUE_GRADES:
        if g not in data["torque"]:
            data["torque"][g] = se.peak_torque(g)
            print(f"  torque profile {g:+d} deg", flush=True)
            CACHE.write_bytes(pickle.dumps(data))
    print(f"cached -> {CACHE.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _outcome(r):
    """pass / walks (but misses a strict check) / fell, plus a short reason."""
    if r["verdict"] == "PASS":
        return "pass", ""
    if r["fell"]:
        return "fell", f"falls after {r['steps']} steps"
    why = []
    if not r["checks"]["tracking"]:
        why.append("slows to %.2f m/s" % r["speed"] if r["speed"] < V_CMD
                   else "speeds up to %.2f m/s" % r["speed"])
    if not r["checks"]["com_height"]:
        why.append("CoM bobs %.0f cm" % (100 * r["com_ptp"]))
    if not r["checks"]["upright"]:
        why.append("tilts %.0f°" % r["tilt"])
    if not r["checks"]["solver"]:
        n = r["solver_failures"]
        why.append(f"{n} QP solve miss" + ("es" if n > 1 else ""))
    return "walks", ", ".join(why)


# Where each failure label sits relative to its point, in points (dx, dy).
LABEL_OFFSET = {-30: (14, 0), -25: (0, -32), -22: (14, -14), 22: (0, -32), 25: (-14, 14)}


def fig_envelope(D, plt):
    """Slope vs measured speed: where it works, where it breaks, and why."""
    runs = sorted(D["envelope"].values(), key=lambda r: r["grade"])
    g = np.array([r["grade"] for r in runs])
    v = np.array([r["speed"] if np.isfinite(r["speed"]) else 0.0 for r in runs])
    kind = [_outcome(r)[0] for r in runs]
    ok = np.array([k == "pass" for k in kind])

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axvspan(g[ok].min() - 1.5, g[ok].max() + 1.5, color=STYLE["good"], alpha=0.06, lw=0)
    ax.axhspan(V_CMD * 0.9, V_CMD * 1.1, color=STYLE["good"], alpha=0.15, lw=0)
    ax.axhline(V_CMD, color=STYLE["good"], lw=1, ls="--")
    ax.text(-4.5, V_CMD * 1.1 + 0.012, f"commanded {V_CMD:.1f} m/s ±10%",
            color=STYLE["good"], fontsize=9.5, va="bottom", ha="center")
    ax.plot(g, v, "-", color=STYLE["muted"], lw=1, zorder=1)

    mk = dict(pass_=("o", STYLE["good"], 55, "passes every check"),
              walks=("D", STYLE["left"], 50, "keeps walking, misses a check"),
              fell=("X", STYLE["bad"], 95, "falls"))
    for k, (m, c, sz, lab) in mk.items():
        sel = np.array([x == k.rstrip("_") for x in kind])
        if sel.any():
            ax.scatter(g[sel], v[sel], s=sz, marker=m, color=c, zorder=3, label=lab)
    for r, k in zip(runs, kind):
        if k == "pass":
            continue
        y = r["speed"] if np.isfinite(r["speed"]) else 0.0
        why = _outcome(r)[1]
        dx, dy = LABEL_OFFSET.get(int(r["grade"]), (0, 14))
        ax.annotate(f"{r['grade']:+.0f}°\n" + why.replace(", ", "\n"), (r["grade"], y),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 10 else ("right" if dx < -10 else "center"),
                    va="center", fontsize=8.6,
                    color=STYLE["bad"] if k == "fell" else "#9A5B13", linespacing=1.15)

    ax.set_title(f"Clean pass from {g[ok].min():+.0f}° to {g[ok].max():+.0f}°, "
                 f"still walking at ±22°, falls past that")
    ax.set_xlabel("slope (°)        ← downhill   ·   uphill →")
    ax.set_ylabel("measured speed (m/s)")
    ax.set_xticks(g)
    ax.set_ylim(-0.05, max(0.75, np.nanmax(v) * 1.35))
    ax.set_xlim(g.min() - 3, g.max() + 3)
    ax.legend(loc="upper right")
    fig.text(0.01, -0.03, "Each point is one 8 s closed-loop run. Strict pass = no fall, "
             "speed within ±10%, torso tilt < 10°, CoM height swing < 5 cm, zero QP failures.",
             fontsize=9, color="#57606A")
    save(fig, "fig_envelope")


def fig_telemetry(D, plt):
    """20 s flat walk: does it do what it was told, and how steady is it?"""
    d = D["flat"]
    L = d["log"]
    t = L["t"]
    td = d["touchdowns"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.4, 1, 1], hspace=0.28))
    ax = axes[0]
    ax.plot(t, L["vx"], color=STYLE["right"], lw=0.8, alpha=0.35, label="instantaneous")
    if len(td) > 2:
        tt = np.array([x["t"] for x in td])
        xx = np.array([x["landed"][0] for x in td])
        sv = np.diff(xx) / np.diff(tt)
        ax.step(tt[1:], sv, where="pre", color=STYLE["right"], lw=2, label="per step")
    ax.plot(t, L["cmd_v"], color=STYLE["ink"], lw=1.4, ls="--", label="commanded")
    ax.set_ylabel("speed (m/s)")
    ax.legend(ncol=3, loc="lower right")
    cruise = L["cmd_v"] >= L["cmd_v"].max() - 1e-6
    half = np.where(cruise)[0]
    half = half[len(half) // 2:]
    mv = (L["com"][half[-1], 0] - L["com"][half[0], 0]) / (t[half[-1]] - t[half[0]])
    ax.set_title(f"Flat ground, {d['steps']} steps: commanded {L['cmd_v'].max():.2f} m/s, "
                 f"measured {mv:.3f} m/s")

    ax = axes[1]
    z = (L["com_z"] - np.median(L["com_z"][cruise])) * 1000
    ax.plot(t, z, color=STYLE["com"], lw=1.1)
    ax.axhspan(-4, 4, color=STYLE["good"], alpha=0.12, lw=0)
    ax.set_ylabel("CoM height\nchange (mm)")

    ax = axes[2]
    ax.plot(t, L["tilt"], color=STYLE["left"], lw=1.1)
    ax.set_ylabel("torso tilt (°)")
    ax.set_xlabel("time (s)")
    ax.set_xlim(t[0], t[-1])
    for a in axes:
        a.axvline(t[np.argmax(L["cmd_v"] > 0)], color=STYLE["muted"], lw=0.8, ls=":")
    axes[0].text(t[np.argmax(L["cmd_v"] > 0)] + 0.15, 0.02, "starts walking",
                 color="#57606A", fontsize=9, transform=axes[0].get_xaxis_transform())
    fig.align_ylabels(axes)
    save(fig, "fig_telemetry")


JOINTS = [("hip_roll", "hip roll"), ("hip_yaw", "hip yaw"), ("hip_pitch", "hip pitch"),
          ("knee", "knee"), ("toe_A", "ankle A"), ("toe_B", "ankle B")]


def fig_torque(D, plt):
    """Peak torque per joint as % of the motor limit — which joint runs out first."""
    tq = D["torque"]
    grades = [g for g in TORQUE_GRADES if g in tq]
    cols = {0: STYLE["muted"], 22: STYLE["left"], -22: STYLE["right"], 25: STYLE["bad"]}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    n = len(grades)
    w = 0.8 / n
    x = np.arange(len(JOINTS))
    for i, gr in enumerate(grades):
        vals = [100 * max(tq[gr].get(f"{s}_{j}_joint", 0) for s in ("left", "right"))
                for j, _ in JOINTS]
        lab = "flat" if gr == 0 else f"{gr:+d}°" + (" (falls)" if gr == 25 else "")
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=cols.get(gr, STYLE["ink"]),
               label=lab)
    ax.axhline(100, color=STYLE["bad"], lw=1.3, ls="--")
    ax.text(len(JOINTS) - 0.5, 101, "motor limit", color=STYLE["bad"], ha="right",
            va="bottom", fontsize=9.5)
    ax.set_xticks(x, [lab for _, lab in JOINTS])
    ax.set_ylabel("peak torque (% of limit)")
    ax.set_ylim(0, 115)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    worst = {}
    for gr in grades:
        j, lab = max(JOINTS, key=lambda jl: max(tq[gr].get(f"{s}_{jl[0]}_joint", 0)
                                                 for s in ("left", "right")))
        worst[gr] = lab
    ax.set_title("Peak torque per joint, worst of both legs: at +22° the knees and hips "
                 "are already maxed out", pad=34)
    save(fig, "fig_torque")
    return worst


def _rot(p, deg):
    """World (= slope) frame -> display frame with gravity pointing straight down."""
    a = np.radians(deg)
    x, z = p[..., 0], p[..., 2]
    return np.stack([x * np.cos(a) - z * np.sin(a), x * np.sin(a) + z * np.cos(a)], -1)


def _stick(ax, f, deg, alpha, lw=2.0):
    for s in ("left", "right"):
        c = STYLE[s]
        leg = _rot(np.vstack([f["torso"], f[s]]), deg)
        ax.plot(leg[:, 0], leg[:, 1], "-", color=c, lw=lw, alpha=alpha,
                solid_capstyle="round", solid_joinstyle="round")
        foot = _rot(np.vstack([f[f"{s}_heel"], f[f"{s}_toe"]]), deg)
        ax.plot(foot[:, 0], foot[:, 1], "-", color=c, lw=lw * 2, alpha=alpha,
                solid_capstyle="round")
    c = _rot(f["com"], deg)
    ax.plot(c[0], c[1], "o", color=STYLE["com"], ms=4, alpha=alpha)


def fig_slopes(D, plt):
    """Side view on the real incline: flat, 22° up, 22° down."""
    jobs = [("flat", 0.0, "flat ground"), ("up22", 22.0, "22° uphill"),
            ("down22", -22.0, "22° downhill")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (key, deg, title) in zip(axes, jobs):
        d = D[key]
        fr = d["frames"]
        tsec = np.arange(len(fr)) * 0.02
        # ~2.4 s of steady walking, 8 poses
        t0 = max(tsec[-1] - 3.0, 0)
        idx = np.linspace(np.searchsorted(tsec, t0), len(fr) - 1, 8).astype(int)
        for k, i in enumerate(idx):
            _stick(ax, fr[i], deg, 0.25 + 0.75 * k / (len(idx) - 1))
        com = _rot(np.array([f["com"] for f in fr[idx[0]:]]), deg)
        ax.plot(com[:, 0], com[:, 1], color=STYLE["com"], lw=0.8, alpha=0.6)
        gz = min(min(f["left_heel"][2], f["right_heel"][2]) for f in fr[idx[0]:])
        xs = np.array([fr[idx[0]]["com"][0] - 3, fr[idx[-1]]["com"][0] + 3])
        ground = _rot(np.stack([xs, 0 * xs, gz + 0 * xs], -1), deg)
        ax.fill_between(ground[:, 0], ground[:, 1], ground[:, 1].min() - 0.6,
                        color="#EAEEF2", lw=0)
        ax.plot(ground[:, 0], ground[:, 1], color=STYLE["muted"], lw=1.5)
        ax.set_aspect("equal")
        cx = 0.5 * (com[:, 0].min() + com[:, 0].max())
        cy = 0.5 * (com[:, 1].min() + com[:, 1].max()) - 0.35
        ax.set_xlim(cx - 1.3, cx + 1.3)
        ax.set_ylim(cy - 0.95, cy + 0.85)
        ax.set_title(title, loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.legend(handles=[plt.Line2D([], [], color=STYLE["left"], lw=3, label="left leg"),
                        plt.Line2D([], [], color=STYLE["right"], lw=3, label="right leg"),
                        plt.Line2D([], [], color=STYLE["com"], marker="o", lw=0.8,
                                   label="center of mass")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Same controller on three terrains (faded = earlier)", x=0.02, ha="left",
                 fontweight="bold", fontsize=12.5)
    save(fig, "fig_slopes")


def fig_footsteps(D, plt):
    """Where each foot landed, and how the step length settles."""
    d = D["flat"]
    td = d["touchdowns"]
    fr = d["frames"]
    tsec = np.arange(len(fr)) * 0.02
    com = d["log"]["com"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6),
                                 gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.45))
    xmax = 3.2
    for x in td:
        if x["landed"][0] > xmax + 0.2:
            break
        i = min(np.searchsorted(tsec, x["t"] + 0.05), len(fr) - 1)
        side = "left" if x["stance"] == "right" else "right"   # the foot that just landed
        h, tq = fr[i][f"{side}_heel"], fr[i][f"{side}_toe"]
        a1.plot([h[0], tq[0]], [h[1], tq[1]], color=STYLE[side], lw=7,
                solid_capstyle="butt", alpha=0.85)
    m = com[:, 0] <= xmax
    a1.plot(com[m, 0], com[m, 1], color=STYLE["com"], lw=1.2, label="center of mass")
    a1.set_aspect("equal")
    a1.set_xlim(com[0, 0] - 0.2, xmax)
    a1.set_ylim(-0.46, 0.4)
    a1.set_xlabel("forward (m)")
    a1.set_ylabel("sideways (m)")
    a1.set_title("Top-down: each bar is a foot placed by the planner (first 3 m)")
    a1.legend(handles=[plt.Line2D([], [], color=STYLE["left"], lw=6, label="left foot"),
                       plt.Line2D([], [], color=STYLE["right"], lw=6, label="right foot"),
                       plt.Line2D([], [], color=STYLE["com"], lw=1.2, label="center of mass")],
              ncol=3, loc="upper right", bbox_to_anchor=(1, 1.02))

    xs = np.array([x["landed"][0] for x in td])
    ys = np.array([x["landed"][1] for x in td])
    tt = np.array([x["t"] for x in td])
    n = np.arange(1, len(xs))
    a2.plot(n, np.diff(xs) * 100, "o-", ms=3.5, color=STYLE["right"], lw=1.2,
            label="step length")
    a2.plot(n, np.abs(np.diff(ys)) * 100, "s-", ms=3.2, color=STYLE["left"], lw=1.2,
            label="step width")
    a2.set_xlabel("step number")
    a2.set_ylabel("cm")
    a2.set_xlim(0, len(xs))
    a2.legend(ncol=2, loc="lower right")
    dur = np.diff(tt)
    a2.set_title(f"Step length settles in ~8 steps and holds "
                 f"(step time {dur[len(dur)//2:].mean():.2f} s)")
    save(fig, "fig_footsteps")


def fig_baseline(D, plt):
    """Why a planner at all: joint PD tracks its joints and still falls."""
    pd = D["pd"]
    dg = D["flat"]["log"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    m = dg["t"] <= pd["t"][-1]
    ax.plot(dg["t"][m], dg["com_z"][m] / dg["com_z"][0] * 100, color=STYLE["good"], lw=2,
            label="Digit, ALIP + whole-body QP")
    ax.plot(pd["t"], pd["com_z"] / pd["com_z"][0] * 100, color=STYLE["bad"], lw=2,
            label="Cassie, joint PD only")
    ax.set_ylabel("CoM height (% of start)")
    ax.set_xlabel("time (s)")
    ax.set_ylim(0, 115)
    ax2 = ax.twinx()
    ax2.plot(pd["t"], pd["joint_err"], color=STYLE["bad"], lw=1, ls=":", alpha=0.8)
    ax.legend(handles=ax.get_lines() + [plt.Line2D([], [], color=STYLE["bad"], lw=1, ls=":",
                                                   label="Cassie PD joint error (right axis)")],
              loc="center left", bbox_to_anchor=(0, 0.45))
    ax2.set_ylabel("PD worst joint error (°)", color=STYLE["bad"])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, max(10, pd["joint_err"].max() * 1.3))
    ax.set_title(f"Joint PD holds every joint within {pd['joint_err'].max():.1f}° "
                 f"and the robot still falls")
    save(fig, "fig_baseline")


FIGS = dict(envelope=fig_envelope, telemetry=fig_telemetry, torque=fig_torque,
            slopes=fig_slopes, footsteps=fig_footsteps, baseline=fig_baseline)


def draw(only=None):
    D = pickle.loads(CACHE.read_bytes())
    plt = setup_matplotlib()
    for name, fn in FIGS.items():
        if only and name not in only:
            continue
        r = fn(D, plt)
        if r:
            print(f"  {name}: {r}")
        plt.close("all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="run the sims and cache results")
    ap.add_argument("--only", nargs="*", default=None, choices=list(FIGS))
    a = ap.parse_args()
    if a.collect:
        collect()
    draw(a.only)
