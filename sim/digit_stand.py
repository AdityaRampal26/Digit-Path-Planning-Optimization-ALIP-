"""Stage A gate: can the OSC hold Digit standing under real physics?

This is the milestone that decides whether anything downstream is worth building.
Joint PD provably cannot do it (see tests/test_cassie_state.py --
`test_joint_pd_alone_cannot_balance`), because nothing regulates the CoM over the
support polygon. The OSC should, and with a box foot it has centre-of-pressure
authority to do it without moving a foot.

Note on initialisation: the nominal pose puts the soles exactly at z = 0, where
MuJoCo reports **zero contacts** -- tangent geoms generate no force. The robot is
therefore started a fraction of a millimetre low so contacts exist on the first
tick; the closed loop then settles to whatever penetration balances body weight.

Run:  python sim/digit_stand.py [--seconds 30] [--push 150]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import digit_params as dp  # noqa: E402
import digit_state as ds  # noqa: E402
from osc import OSC, OSCConfig  # noqa: E402

OUT = ROOT / "out"
LOG_HZ = 100.0
SINK = 0.001   # start 1 mm low so contacts are live on tick 0


def run(seconds=5.0, control_hz=1000.0, push=None, push_time=2.0, push_duration=0.1,
        verbose=True, config: OSCConfig | None = None):
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q_nom = dp.nominal_pose(model, data, feet)

    data.qpos[:] = q_nom
    data.qpos[2] -= SINK
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    feet.calibrate(data)

    osc = OSC(model, feet, config=config)
    det = ds.DigitContactDetector(model)
    est = ds.DigitAlipEstimator(model, feet)

    # Hold the CoM above the midpoint of the two feet, at the nominal height.
    mid = 0.5 * (feet.contact_point(data, "left") + feet.contact_point(data, "right"))
    com_target = np.array([mid[0], mid[1], q_nom[2] - SINK - 0.008 + 0.0])
    com_target[2] = float(data.subtree_com[0][2])
    posture_target = q_nom[osc.act.qposadr].copy()

    dt = model.opt.timestep
    n_steps = int(seconds / dt)
    ctrl_every = max(1, int(round(1.0 / (control_hz * dt))))
    log_every = max(1, int(round(1.0 / (LOG_HZ * dt))))
    weight = ds.total_mass(model) * 9.806

    log = []
    qpos_log = []
    tau = np.zeros(osc.act.n)
    solve_times = []
    failures = 0
    fell_at = None

    if verbose:
        print(f"Digit standing: {seconds}s, sim dt={dt}, control {control_hz:.0f} Hz")
        print(f"weight {weight:.1f} N, CoM target z={com_target[2]:.4f}")
        if push:
            print(f"push: {push} N in +x for {push_duration*1000:.0f} ms at t={push_time}s")

    for step in range(n_steps):
        t = step * dt

        if step % ctrl_every == 0:
            stance = [s for s in dp.SIDES if det.normal_force(data, s) > 5.0] or list(dp.SIDES)
            t0 = time.perf_counter()
            tau_new, info = osc.solve(
                data,
                stance_sides=stance,
                com_target=com_target,
                torso_quat_target=(1, 0, 0, 0),
                posture_target=posture_target,
            )
            solve_times.append(time.perf_counter() - t0)
            if tau_new is None or not info["ok"]:
                failures += 1
            else:
                tau = tau_new

        osc.act.apply_torque(data, tau)

        # External disturbance.
        data.xfrc_applied[:] = 0.0
        if push and push_time <= t < push_time + push_duration:
            data.xfrc_applied[osc.torso_id, 0] = push

        if step % log_every == 0:
            cs = det.state(data)
            st = est.estimate(data, stance_is_left=True)
            cop_l = det.center_of_pressure(data, "left")
            cop_r = det.center_of_pressure(data, "right")
            log.append(dict(
                t=t,
                com=np.array(data.subtree_com[0]),
                com_err=float(np.linalg.norm(np.array(data.subtree_com[0]) - com_target)),
                base_z=float(data.qpos[2]),
                fz=cs.force_left + cs.force_right,
                tilt=float(np.degrees(np.arccos(np.clip(
                    (np.array(data.xmat[osc.torso_id]).reshape(3, 3))[2, 2], -1, 1)))),
                tau_max=float(np.max(np.abs(tau))),
                sat=float(np.max(np.abs(tau) / osc.act.torque_limit)),
                cop_l=cop_l, cop_r=cop_r,
                Lx=st.Lx, Ly=st.Ly,
                double=cs.double_support,
            ))
            qpos_log.append(data.qpos.copy())

        mujoco.mj_step(model, data)

        if fell_at is None and data.qpos[2] < q_nom[2] - 0.25:
            fell_at = t

    qpos_log = np.array(qpos_log)
    st = dict(log=log, qpos=qpos_log, failures=failures,
              solve_ms=1e3 * np.mean(solve_times), fell_at=fell_at,
              com_target=com_target, weight=weight)

    if verbose:
        _report(st, osc, seconds)
    return st


def _report(st, osc, seconds):
    log = st["log"]
    print(f"\nQP: {st['failures']} failed solves, mean solve {st['solve_ms']:.2f} ms "
          f"({1e3/st['solve_ms']:.0f} Hz capability)")
    print(f"{'t':>6} {'comErr':>8} {'baseZ':>7} {'tilt':>6} {'Fz/W':>6} {'tauMax':>7} {'sat':>5}")
    for d in log[:: max(1, len(log) // 12)]:
        print(f"{d['t']:6.2f} {d['com_err']*1000:8.2f} {d['base_z']:7.4f} "
              f"{d['tilt']:6.2f} {d['fz']/st['weight']:6.2f} {d['tau_max']:7.1f} "
              f"{d['sat']:5.2f}")

    final = log[-1]
    com_err = np.array([d["com_err"] for d in log])
    tilt = np.array([d["tilt"] for d in log])
    print()
    print(f"final CoM error   : {final['com_err']*1000:.2f} mm")
    print(f"max CoM error     : {com_err.max()*1000:.2f} mm")
    print(f"steady CoM error  : {com_err[len(com_err)//2:].mean()*1000:.2f} mm (2nd half mean)")
    print(f"max torso tilt    : {tilt.max():.2f} deg")
    print(f"max torque satur. : {max(d['sat'] for d in log)*100:.1f}% of limit")
    print(f"mean Fz / weight  : {np.mean([d['fz'] for d in log])/st['weight']:.3f}")
    if st["fell_at"] is not None:
        print(f"\nVERDICT: FELL at t={st['fell_at']:.2f} s")
    else:
        upright = tilt.max() < 10.0 and com_err.max() < 0.10
        print(f"\nVERDICT: {'STANDING' if upright else 'UPRIGHT BUT POOR'} "
              f"for the full {seconds:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--control-hz", type=float, default=1000.0)
    ap.add_argument("--push", type=float, default=None)
    ap.add_argument("--out", default="digit_stand_qpos.npy")
    args = ap.parse_args()

    st = run(args.seconds, args.control_hz, push=args.push)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / args.out, st["qpos"])
    print(f"\nwrote {OUT / args.out}  shape {st['qpos'].shape}")
