"""Stage B, step 1: can the OSC balance Digit on one leg?

Before adding a moving swing foot and stance switching, verify the single-support
QP in isolation. This is the tightest possible test of the two things walking needs
that standing did not:

  * only ONE foot is constrained -- the swing foot must NOT get a no-slip
    constraint, or the QP demands that a foot in mid-air not accelerate, which is
    instantly infeasible;
  * the swing foot is held by a task instead, so the leg does not dangle.

The CoM target moves over the stance foot, which is a ~13 cm lateral shift from the
nominal stance. The starting pose is built by homotopy through the IK rather than
guessed.

Run:  python sim/digit_one_leg.py [--seconds 3]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import digit_params as dp  # noqa: E402
import digit_state as ds  # noqa: E402
from digit_ik import DigitIK  # noqa: E402
from osc import OSC  # noqa: E402

OUT = ROOT / "out"
SINK = 0.001
LOG_HZ = 100.0


def build_pose(stance="left", lift=0.06, n=20, verbose=False):
    """Pose balanced over one foot with the other raised, via IK homotopy."""
    ik = DigitIK()
    model = ik.model
    q = dp.nominal_pose(model, ik.data, ik.feet)
    ik.data.qpos[:] = q
    mujoco.mj_forward(model, ik.data)
    ik.feet.calibrate(ik.data)

    swing = "right" if stance == "left" else "left"
    p_stance = ik.feet.contact_point(ik.data, stance).copy()
    p_swing0 = ik.feet.contact_point(ik.data, swing).copy()
    base0 = q[0:3].copy()

    # Shift the base laterally over the stance foot, and raise the swing foot.
    base_goal = base0.copy()
    base_goal[1] = p_stance[1]
    swing_goal = p_swing0 + np.array([0.0, 0.0, lift])

    for k in range(1, n + 1):
        a = k / n
        res = ik.solve(
            q,
            {stance: p_stance, swing: (1 - a) * p_swing0 + a * swing_goal},
            base_pos=(1 - a) * base0 + a * base_goal,
            base_quat=(1, 0, 0, 0),
            flat_feet=True,
            yaw_target={"left": 0.0, "right": 0.0},
            max_iter=120,
            tol=1e-6,
        )
        q = res.qpos
        if verbose:
            print(f"  homotopy {k:>3}/{n}  task {res.task_error*1e3:7.4f} mm  "
                  f"closure {res.closure_error*1e3:7.4f} mm  {res.converged}")
    return ik, q, p_stance, swing_goal


def run(seconds=3.0, control_hz=500.0, stance="left", lift=0.06, verbose=True):
    ik, q0, p_stance, swing_goal = build_pose(stance, lift, verbose=verbose)
    model, data = dp.load()
    feet = dp.FootGeometry(model)

    data.qpos[:] = q0
    data.qpos[2] -= SINK
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    feet.calibrate(data)

    osc = OSC(model, feet)
    det = ds.DigitContactDetector(model)
    est = ds.DigitAlipEstimator(model, feet)
    swing = "right" if stance == "left" else "left"

    # Hold the CoM over the stance contact point, at the height it starts at.
    com_target = np.array([p_stance[0], p_stance[1], float(data.subtree_com[0][2])])
    posture_target = q0[osc.act.qposadr].copy()

    dt = model.opt.timestep
    n_steps = int(seconds / dt)
    ctrl_every = max(1, int(round(1.0 / (control_hz * dt))))
    log_every = max(1, int(round(1.0 / (LOG_HZ * dt))))
    weight = ds.total_mass(model) * 9.806

    tau = np.zeros(osc.act.n)
    log, qpos_log = [], []
    failures = 0
    fell_at = None

    if verbose:
        print(f"\nsingle support on {stance}, swing {swing} raised {lift*100:.0f} cm")
        print(f"CoM target ({com_target[0]:+.3f}, {com_target[1]:+.3f}, {com_target[2]:.3f})")

    for step in range(n_steps):
        t = step * dt
        if step % ctrl_every == 0:
            tau_new, info = osc.solve(
                data,
                stance_sides=(stance,),           # ONE foot constrained
                com_target=com_target,
                torso_quat_target=(1, 0, 0, 0),
                posture_target=posture_target,
                swing_side=swing,                 # the other held by a task
                swing_target=swing_goal,
            )
            if tau_new is None or not info["ok"]:
                failures += 1
            else:
                tau = tau_new
        osc.act.apply_torque(data, tau)

        if step % log_every == 0:
            cs = det.state(data)
            st = est.estimate(data, stance_is_left=(stance == "left"))
            log.append(dict(
                t=t,
                com_err=float(np.linalg.norm(np.array(data.subtree_com[0]) - com_target)),
                swing_err=float(np.linalg.norm(feet.contact_point(data, swing) - swing_goal)),
                swing_z=float(feet.contact_point(data, swing)[2]),
                fz_stance=cs.force_left if stance == "left" else cs.force_right,
                fz_swing=cs.force_right if stance == "left" else cs.force_left,
                tilt=float(np.degrees(np.arccos(np.clip(
                    np.array(data.xmat[osc.torso_id]).reshape(3, 3)[2, 2], -1, 1)))),
                sat=float(np.max(np.abs(tau) / osc.act.torque_limit)),
                Lx=st.Lx, Ly=st.Ly,
            ))
            qpos_log.append(data.qpos.copy())

        mujoco.mj_step(model, data)
        if fell_at is None and data.qpos[2] < q0[2] - 0.25:
            fell_at = t

    st = dict(log=log, qpos=np.array(qpos_log), failures=failures,
              fell_at=fell_at, weight=weight)

    if verbose:
        print(f"\nQP failures: {failures}")
        print(f"{'t':>6} {'comErr':>8} {'swingErr':>9} {'swingZ':>7} {'FzStance':>9} "
              f"{'FzSwing':>8} {'tilt':>6} {'sat':>5}")
        for d in log[:: max(1, len(log) // 12)]:
            print(f"{d['t']:6.2f} {d['com_err']*1e3:8.2f} {d['swing_err']*1e3:9.2f} "
                  f"{d['swing_z']:7.4f} {d['fz_stance']/weight:9.2f} "
                  f"{d['fz_swing']/weight:8.2f} {d['tilt']:6.2f} {d['sat']:5.2f}")
        ce = np.array([d["com_err"] for d in log])
        se = np.array([d["swing_err"] for d in log])
        tl = np.array([d["tilt"] for d in log])
        print(f"\nsteady CoM error   : {ce[len(ce)//2:].mean()*1e3:.2f} mm")
        print(f"steady swing error : {se[len(se)//2:].mean()*1e3:.2f} mm")
        print(f"max torso tilt     : {tl.max():.2f} deg")
        print(f"stance Fz / weight : {np.mean([d['fz_stance'] for d in log])/weight:.3f}")
        print(f"swing  Fz / weight : {np.mean([d['fz_swing'] for d in log])/weight:.3f}"
              "   (should be ~0)")
        if fell_at is not None:
            print(f"\nVERDICT: FELL at t={fell_at:.2f} s")
        else:
            ok = tl.max() < 10 and ce[-1] < 0.05
            print(f"\nVERDICT: {'BALANCED' if ok else 'UPRIGHT BUT POOR'} on one leg "
                  f"for {seconds:.0f} s")
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--control-hz", type=float, default=500.0)
    ap.add_argument("--stance", default="left", choices=("left", "right"))
    ap.add_argument("--lift", type=float, default=0.06)
    args = ap.parse_args()
    st = run(args.seconds, args.control_hz, args.stance, args.lift)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "digit_one_leg_qpos.npy", st["qpos"])
    print(f"\nwrote {OUT / 'digit_one_leg_qpos.npy'}  shape {st['qpos'].shape}")
