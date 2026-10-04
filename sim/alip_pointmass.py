"""Stage 1: closed-loop ALIP walking on a point mass.

No robot, no contacts, no springs -- just the reduced-order model driven by the
footstep planner in src/alip.py. The point is to confirm the planner produces a
sane, stable gait (and to see what the reference trajectories *should* look like)
before any of it goes near Cassie's 32 degrees of freedom.

Commanded gait: walk in place for 6 steps, ramp to 0.5 m/s over 4 steps, hold.

Run:
    source .venv/bin/activate
    python sim/alip_pointmass.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import alip  # noqa: E402
from alip import FRONTAL, SAGITTAL, AlipParams  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

P = AlipParams()
DT = 1e-3
N_STEPS = 24
HOLD_STEPS = 6      # steps walked in place before the ramp
RAMP_STEPS = 4      # steps spent ramping up to cruise
V_CRUISE = 0.5      # target step-averaged forward speed [m/s]


def commanded_speed(step_index: int) -> float:
    """Step-averaged forward speed command as a function of step number."""
    if step_index < HOLD_STEPS:
        return 0.0
    if step_index < HOLD_STEPS + RAMP_STEPS:
        frac = (step_index - HOLD_STEPS + 1) / RAMP_STEPS
        return V_CRUISE * frac
    return V_CRUISE


def simulate():
    # Start on the in-place periodic orbit with the left foot in stance.
    stance_is_left = True
    x, Ly = alip.periodic_orbit_sagittal(0.0, P)
    y, Lx = alip.periodic_orbit_frontal(stance_is_left, P)

    # World-frame position of the stance foot. CoM world = stance + (x, y).
    stance_world = np.array([-x, -y])  # so the CoM starts at the origin

    log = {k: [] for k in ("t", "x", "y", "Lx", "Ly", "Lx_des", "Ly_des", "vx", "vy")}
    footsteps = []  # (world_x, world_y, is_left)
    t = 0.0

    for k in range(N_STEPS):
        v_avg = commanded_speed(k)
        vx_des = alip.average_to_terminal_velocity(v_avg, P) if v_avg else 0.0
        Ly_des = alip.desired_L_sagittal(vx_des, P)
        Lx_des = alip.desired_L_frontal(0.0, stance_is_left, P)

        footsteps.append((stance_world[0], stance_world[1], stance_is_left))

        n_ticks = int(round(P.T / DT))
        for i in range(n_ticks):
            log["t"].append(t)
            log["x"].append(stance_world[0] + x)
            log["y"].append(stance_world[1] + y)
            log["Lx"].append(Lx)
            log["Ly"].append(Ly)
            log["Lx_des"].append(Lx_des)
            log["Ly_des"].append(Ly_des)
            log["vx"].append(Ly / (P.m * P.H))
            log["vy"].append(-Lx / (P.m * P.H))

            x, Ly = alip.flow(x, Ly, DT, SAGITTAL, P)
            y, Lx = alip.flow(y, Lx, DT, FRONTAL, P)
            t += DT

        # Foot strike: plan where the swing foot lands, then swap stance.
        # Both planners target the END OF THE NEXT step, so the sagittal command
        # is next step's speed and the lateral target flips stance side.
        v_next = commanded_speed(k + 1)
        vx_next = alip.average_to_terminal_velocity(v_next, P) if v_next else 0.0
        step_x = alip.plan_sagittal_step(x, Ly, 0.0, vx_next, P)
        step_y = alip.plan_lateral_step(y, Lx, 0.0, 0.0, stance_is_left, P)
        x, Ly = alip.reset_map(x, Ly, step_x)
        y, Lx = alip.reset_map(y, Lx, step_y)
        stance_world = stance_world + np.array([step_x, step_y])
        stance_is_left = not stance_is_left

    return {k: np.asarray(v) for k, v in log.items()}, footsteps


def report(log, footsteps):
    """Print a few numbers that should hold if the planner is behaving."""
    print(f"omega = {P.omega:.4f} rad/s   (H = {P.H} m)")
    print(f"step duration T = {P.T} s, step width W = {P.W} m")

    # Cruise-phase average speed, measured from the footstep sequence.
    cruise = footsteps[HOLD_STEPS + RAMP_STEPS :]
    if len(cruise) >= 3:
        dx = cruise[-1][0] - cruise[0][0]
        v_meas = dx / ((len(cruise) - 1) * P.T)
        print(f"commanded cruise speed = {V_CRUISE:.3f} m/s")
        print(f"measured  cruise speed = {v_meas:.3f} m/s")

    # Lateral drift should be zero: the feet straddle y = const.
    y_left = [f[1] for f in footsteps if f[2]]
    y_right = [f[1] for f in footsteps if not f[2]]
    print(f"left  foot y: mean {np.mean(y_left):+.4f}, spread {np.ptp(y_left):.2e}")
    print(f"right foot y: mean {np.mean(y_right):+.4f}, spread {np.ptp(y_right):.2e}")
    print(f"realised step width = {abs(np.mean(y_left) - np.mean(y_right)):.4f} m")

    print(f"CoM lateral sway range = {np.ptp(log['y']):.4f} m")
    print(f"peak |vx| = {np.max(np.abs(log['vx'])):.3f} m/s")


def plot(log, footsteps):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plots. pip install matplotlib")
        return

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(log["t"], log["x"], label="CoM x")
    for fx, _, _ in footsteps:
        ax[0].axhline(fx, lw=0.3, color="0.8")
    ax[0].set_ylabel("forward x [m]")
    ax[0].legend(loc="upper left")
    ax[0].set_title("ALIP point-mass gait: in place, then ramp to 0.5 m/s")

    ax[1].plot(log["t"], log["y"], label="CoM y")
    y_left = np.mean([f[1] for f in footsteps if f[2]])
    y_right = np.mean([f[1] for f in footsteps if not f[2]])
    ax[1].axhline(y_left, ls="--", lw=0.8, color="C1", label="left foot")
    ax[1].axhline(y_right, ls="--", lw=0.8, color="C2", label="right foot")
    ax[1].set_ylabel("lateral y [m]")
    ax[1].legend(loc="upper left")

    ax[2].plot(log["t"], log["Ly"], label="$L_y$")
    ax[2].plot(log["t"], log["Ly_des"], ls="--", label="$L_y$ target")
    ax[2].set_ylabel("sagittal $L$ [kg m$^2$/s]")
    ax[2].legend(loc="upper left")

    ax[3].plot(log["t"], log["Lx"], label="$L_x$")
    ax[3].plot(log["t"], log["Lx_des"], ls="--", label="$L_x$ target")
    ax[3].set_ylabel("frontal $L$ [kg m$^2$/s]")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "alip_pointmass_traj.png", dpi=130)

    # Top-down view: CoM path weaving between the footprints.
    fig2, ax2 = plt.subplots(figsize=(11, 3.2))
    ax2.plot(log["x"], log["y"], lw=1.0, color="0.3", label="CoM path")
    for fx, fy, is_left in footsteps:
        ax2.plot(
            fx,
            fy,
            marker="s" if is_left else "o",
            color="C1" if is_left else "C2",
            ms=7,
        )
    ax2.plot([], [], "s", color="C1", label="left footstep")
    ax2.plot([], [], "o", color="C2", label="right footstep")
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper left", ncol=3)
    ax2.set_title("Footstep plan, top-down")
    fig2.tight_layout()
    fig2.savefig(OUT / "alip_pointmass_footsteps.png", dpi=130)

    print(f"\nPlots written to {OUT}/")


if __name__ == "__main__":
    log, footsteps = simulate()
    report(log, footsteps)
    plot(log, footsteps)
