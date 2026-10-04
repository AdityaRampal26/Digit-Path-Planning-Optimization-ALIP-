"""Run Cassie forward under real physics and record the result.

Right now the only controller that exists is the joint-PD baseline, which does
not balance -- so this records the fall. It is here so the failure mode is
visible and so the same harness can replay the OSC once that exists.

Run:  python sim/physics_rollout.py [--seconds 2.5] [--controller pd]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cassie_params  # noqa: E402
import cassie_state as cs  # noqa: E402

OUT = ROOT / "out"
LOG_HZ = 100.0


def rollout(seconds=2.5, controller="pd", kp=1500.0, kd=50.0, verbose=True):
    model, data = cassie_params.load()
    data.qpos[:] = cassie_params.QPOS_INIT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    act = cassie_params.Actuation(model)
    det = cs.ContactDetector(model)
    est = cs.AlipEstimator(model)
    q_target = cassie_params.QPOS_INIT[act.qposadr]

    dt = model.opt.timestep
    n_steps = int(seconds / dt)
    log_every = max(1, int(1.0 / (LOG_HZ * dt)))

    qpos_log, diag = [], []

    for step in range(n_steps):
        if controller == "pd":
            tau = act.joint_pd(data, q_target, kp, kd)
        elif controller == "none":
            tau = np.zeros(model.nu)
        else:
            raise ValueError(f"unknown controller {controller!r}")
        act.apply_torque(data, tau)

        if step % log_every == 0:
            st = est.estimate(data, stance_is_left=True)
            qpos_log.append(data.qpos.copy())
            diag.append(
                dict(
                    t=step * dt,
                    pelvis_z=float(data.qpos[2]),
                    com_z=float(st.com[2]),
                    vx=st.vx,
                    Lx=st.Lx,
                    Ly=st.Ly,
                    Fz=sum(det.normal_force(data, s) for s in ("left", "right")),
                    joint_err=float(np.max(np.abs(data.qpos[act.qposadr] - q_target))),
                )
            )
        mujoco.mj_step(model, data)

    qpos_log = np.array(qpos_log)

    if verbose:
        weight = cs.total_mass(model) * 9.81
        print(f"controller={controller}  {seconds}s at dt={dt}  ({len(qpos_log)} logged frames)")
        print(f"{'t':>6} {'pelvis z':>9} {'com z':>7} {'vx':>7} {'Fz/W':>6} {'joint err':>10}")
        for d_ in diag[:: max(1, len(diag) // 12)]:
            print(
                f"{d_['t']:6.2f} {d_['pelvis_z']:9.3f} {d_['com_z']:7.3f} "
                f"{d_['vx']:7.3f} {d_['Fz'] / weight:6.2f} "
                f"{np.degrees(d_['joint_err']):9.2f} deg"
            )
        final = diag[-1]
        print()
        print(f"final pelvis z = {final['pelvis_z']:.3f} m "
              f"(started {diag[0]['pelvis_z']:.3f})")
        print(f"max joint tracking error = "
              f"{np.degrees(max(d_['joint_err'] for d_ in diag)):.2f} deg")
        verdict = "STANDING" if final["pelvis_z"] > 0.85 else "FELL"
        print(f"verdict: {verdict}")

    return qpos_log, diag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--controller", default="pd", choices=("pd", "none"))
    ap.add_argument("--out", default="pd_fall_qpos.npy")
    args = ap.parse_args()

    qpos, diag = rollout(args.seconds, args.controller)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / args.out, qpos)
    print(f"\nwrote {OUT / args.out}  shape {qpos.shape}")
