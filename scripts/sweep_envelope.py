"""Capability-envelope sweep: terrain grade vs pass/fail and performance.

Produces the two artefacts a test audience actually wants:
  1. capability envelope  — grade vs outcome, with the pass/fail boundary marked
  2. actuator utilisation — peak torque as a fraction of limit, per joint, per grade

Results append to out/envelope.json so the sweep can be run in chunks.

    python scripts/sweep_envelope.py --grades -30 -25 -22 -15
    python scripts/sweep_envelope.py --plot
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import digit_params as dp  # noqa: E402
import digit_walk  # noqa: E402

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
STORE = OUT / "envelope.json"

# Pre-declared acceptance criteria — fixed before the sweep is run.
CRIT = dict(tilt_max_deg=10.0, speed_err_frac=0.10, com_height_tol=0.05)
V_CMD = 0.4


def load_store():
    return json.loads(STORE.read_text()) if STORE.exists() else {}


def evaluate(deg, seconds=8.0):
    """Run one grade and score it against the pre-declared criteria."""
    out = digit_walk.run(seconds=seconds, control_hz=500, speed=V_CMD, ramp_after=0.8,
                         ramp_time=1.2, slope_deg=deg, verbose=False)
    log = out["log"]
    xs = np.array([d["com"][0] for d in log])
    tilt = float(max(d["tilt"] for d in log))
    late = [d for d in log if d["t"] > seconds * 0.6]
    v = ((late[-1]["com"][0] - late[0]["com"][0]) / (late[-1]["t"] - late[0]["t"])
         if len(late) > 2 else float("nan"))
    com_z = np.array([d["com_z"] for d in log])
    fell = out["fell_at"] is not None

    # bool() casts are deliberate: numpy bools are not JSON serialisable.
    checks = {
        "upright": bool((not fell) and tilt < CRIT["tilt_max_deg"]),
        "tracking": bool((not fell) and abs(v - V_CMD) <= CRIT["speed_err_frac"] * V_CMD),
        "com_height": bool((not fell) and float(np.ptp(com_z)) < CRIT["com_height_tol"]),
        "solver": bool(out["failures"] == 0),
    }
    return dict(grade=float(deg), steps=int(out["steps"]), distance=float(xs[-1] - xs[0]),
                speed=float(v), tilt=tilt, com_ptp=float(np.ptp(com_z)),
                solver_failures=int(out["failures"]), fell=bool(fell),
                fell_at=out["fell_at"], checks=checks,
                verdict="PASS" if all(checks.values()) else "FAIL")


def peak_torque(deg, seconds=6.0):
    """Peak |tau| / limit per actuated joint, for one grade."""
    import alip
    from osc import OSC, OSCConfig
    from walk_controller import WalkCommand, WalkController

    model, data = dp.load()
    feet = dp.FootGeometry(model)
    qn = dp.nominal_pose(model, data, feet)
    data.qpos[:] = qn
    data.qpos[2] -= 0.001
    data.qvel[:] = 0.0
    g, sl = 9.806, np.radians(deg)
    model.opt.gravity[:] = [-g * np.sin(sl), 0.0, -g * np.cos(sl)]
    mujoco.mj_forward(model, data)
    feet.calibrate(data)

    meas = dp.measure(model, data, feet)
    p = alip.AlipParams(m=meas["total_mass"], H=meas["com_height_above_feet"], T=0.35,
                        W=meas["foot_separation_y"], g=g, slope=sl)
    cfg = OSCConfig()
    cfg.w_stance_ankle = 20.0
    osc = OSC(model, feet, config=cfg)
    cmd = WalkCommand(0.0, swing_height=0.07)
    ctl = WalkController(model, feet, osc, p, cmd)

    if sl:
        from digit_ik import DigitIK
        ik = DigitIK()
        ik.data.qpos[:] = data.qpos
        mujoco.mj_forward(ik.model, ik.data)
        ik.feet.calibrate(ik.data)
        tg = {s: ik.feet.contact_point(ik.data, s).copy() for s in dp.SIDES}
        base = np.array(data.qpos[0:3]); base[0] += p.x_equilibrium
        r = ik.solve(np.array(data.qpos), tg, base_pos=base, base_quat=(1, 0, 0, 0),
                     flat_feet=True, yaw_target={"left": 0.0, "right": 0.0},
                     max_iter=200, tol=1e-6)
        if r.converged:
            data.qpos[:] = r.qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
    ctl.reset(data, True)

    dt = model.opt.timestep
    tau = np.zeros(osc.act.n)
    peak = np.zeros(osc.act.n)
    for i in range(int(seconds / dt)):
        t = i * dt
        cmd.v_forward = 0.0 if t < 0.8 else min(V_CMD, V_CMD * (t - 0.8) / 1.2)
        if i % 2 == 0:
            tn, info = ctl.update(data, t, 2 * dt)
            if tn is not None and info["info"]["ok"]:
                tau = tn
            peak = np.maximum(peak, np.abs(tau) / osc.act.torque_limit)
        osc.act.apply_torque(data, tau)
        mujoco.mj_step(model, data)
        if data.qpos[2] < qn[2] - 0.25:
            break
    return {n: float(v) for n, v in zip(osc.act.names, peak)}


# ---------------------------------------------------------------------------


def plot(store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted((v for k, v in store.items() if k.startswith("run_")),
                  key=lambda r: r["grade"])
    g = np.array([r["grade"] for r in runs])
    sp = np.array([r["speed"] for r in runs])
    ok = np.array([r["verdict"] == "PASS" for r in runs])

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.axhspan(V_CMD * 0.9, V_CMD * 1.1, color="#4a8a4a", alpha=0.13,
               label="acceptance band, ±10% of command")
    ax.axhline(V_CMD, color="#4a8a4a", lw=1.0, ls="--")
    ax.plot(g[ok], sp[ok], "o", ms=8, color="#1E5D88", label="PASS", zorder=3)
    bad = ~ok
    if bad.any():
        ax.plot(g[bad], np.nan_to_num(sp[bad], nan=0.0), "X", ms=11,
                color="#B3261E", label="FAIL", zorder=3)
    lo = g[ok].min() if ok.any() else 0
    hi = g[ok].max() if ok.any() else 0
    for b, lab in ((lo, f"{lo:+.0f}°"), (hi, f"{hi:+.0f}°")):
        ax.axvline(b, color="#B3261E", lw=1.2, ls=":", alpha=0.8)
    ax.text(0.5 * (lo + hi), V_CMD * 1.62, f"verified envelope  {lo:+.0f}° to {hi:+.0f}°",
            ha="center", fontsize=10, color="#1E5D88", weight="bold")
    ax.set_xlabel("terrain grade [deg]   (negative = downhill)")
    ax.set_ylabel("measured speed [m/s]")
    ax.set_title("Capability envelope — commanded 0.40 m/s, pre-declared pass criteria")
    ax.set_ylim(0, max(0.75, np.nanmax(sp) * 1.15))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "envelope_capability.png", dpi=150)
    print(f"wrote {OUT/'envelope_capability.png'}")

    tq = {k[3:]: v for k, v in store.items() if k.startswith("tq_")}
    if tq:
        grades = sorted(tq, key=float)
        joints = [j for j in tq[grades[0]] if "shoulder" not in j and "elbow" not in j]
        short = [j.replace("_joint", "").replace("left_", "L ").replace("right_", "R ")
                 for j in joints]
        fig2, ax2 = plt.subplots(figsize=(9.2, 3.6))
        w = 0.8 / len(grades)
        cols = ["#9EC5DE", "#1E5D88", "#D98324"]
        for i, gr in enumerate(grades):
            vals = [100 * tq[gr][j] for j in joints]
            ax2.bar(np.arange(len(joints)) + i * w, vals, w,
                    label=f"{float(gr):+.0f}°", color=cols[i % len(cols)])
        ax2.axhline(100, color="#B3261E", lw=1.4, ls="--", label="actuator limit")
        ax2.set_xticks(np.arange(len(joints)) + 0.4 - w / 2)
        ax2.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
        ax2.set_ylabel("peak torque, % of limit")
        ax2.set_title("Actuator utilisation — ankle (toe A/B) is the binding constraint")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(fontsize=8, ncol=4)
        fig2.tight_layout()
        fig2.savefig(OUT / "envelope_torque.png", dpi=150)
        print(f"wrote {OUT/'envelope_torque.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", type=float, nargs="*", default=[])
    ap.add_argument("--torque", type=float, nargs="*", default=[])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    store = load_store()
    for deg in args.grades:
        r = evaluate(deg)
        store[f"run_{deg:+06.1f}"] = r
        print(f"{deg:+6.1f}°  {r['verdict']:>4}  steps={r['steps']:>3} "
              f"v={r['speed']:.3f}  tilt={r['tilt']:.2f}°  fell={r['fell']}")
        STORE.write_text(json.dumps(store, indent=1))
    for deg in args.torque:
        store[f"tq_{deg:+06.1f}"] = peak_torque(deg)
        print(f"torque profile recorded for {deg:+.1f}°")
        STORE.write_text(json.dumps(store, indent=1))
    if args.plot:
        plot(store)
