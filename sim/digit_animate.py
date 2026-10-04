"""Solve the ALIP walking reference through Digit's IK and save the qpos trajectory.

The point of this file is how little of it there is. `src/alip.py` and `src/gait.py`
are reused **byte-for-byte** from the Cassie work -- the planner does not know or
care which robot it is driving, it only takes (m, H, T, W, g). Everything Digit-
specific lives in digit_params.py and digit_ik.py.

Kinematic stage: the torso is prescribed along the ALIP CoM path and the legs are
solved to put the feet where the planner asked. No contact forces, no torques.

Outputs out/digit_walk_qpos.npy.

Run:  python sim/digit_animate.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import digit_params as dp  # noqa: E402
from digit_ik import DigitIK  # noqa: E402
from gait import GaitReference, GaitSchedule  # noqa: E402

OUT = ROOT / "out"
DT = 1.0 / 100.0


def build(dt: float = DT, schedule: GaitSchedule | None = None, verbose: bool = True):
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q_nom = dp.nominal_pose(model, data, feet)

    data.qpos[:] = q_nom
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    meas = dp.measure(model, data, feet)

    import alip

    params = alip.AlipParams(
        m=meas["total_mass"], H=meas["com_height_above_feet"],
        T=0.35, W=meas["foot_separation_y"],
    )
    schedule = schedule or GaitSchedule(swing_height=0.10)

    ref = GaitReference(
        params, schedule,
        pelvis_to_com=meas["base_to_com"],   # torso origin -> CoM, measured
        half_width=meas["foot_separation_y"] / 2,
    ).generate(dt)
    n = len(ref["t"])

    if verbose:
        print(f"Digit ALIP params: m={params.m:.3f} kg  H={params.H:.4f} m  "
              f"W={params.W:.4f} m  omega={params.omega:.4f}")
        print(f"reference: {n} ticks, {ref['t'][-1] + dt:.2f} s, "
              f"{len(ref['footsteps'])} steps")

    ik = DigitIK()
    # Fix heel/toe orientation from the standing pose before asking for foot yaw.
    ik.data.qpos[:] = q_nom
    mujoco.mj_forward(ik.model, ik.data)
    signs = ik.feet.calibrate(ik.data)
    if verbose:
        print(f"foot long-axis signs (heel->toe calibration): {signs}")

    q = q_nom.copy()
    qpos = np.zeros((n, model.nq))
    task = np.zeros(n)
    closure = np.zeros(n)
    tilt = np.zeros(n)
    iters = np.zeros(n, dtype=int)
    reach = np.zeros((n, 2))

    hip_id = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hip_roll")
              for s in dp.SIDES}

    t0 = time.time()
    for i in range(n):
        res = ik.solve(
            q,
            {"left": ref["left"][i], "right": ref["right"][i]},
            base_pos=ref["pelvis"][i],
            base_quat=(1.0, 0.0, 0.0, 0.0),
            flat_feet=True,
            yaw_target={"left": 0.0, "right": 0.0},
            max_iter=60,
            tol=1e-5,
        )
        qpos[i] = res.qpos
        task[i], closure[i], tilt[i] = res.task_error, res.closure_error, res.tilt_error
        iters[i] = res.iterations
        for k, s in enumerate(dp.SIDES):
            reach[i, k] = np.linalg.norm(
                ik.feet.contact_point(ik.data, s) - np.array(ik.data.xpos[hip_id[s]])
            )
        q = res.qpos
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    elapsed = time.time() - t0

    if verbose:
        print(f"\nsolved {n} frames in {elapsed:.1f} s ({n/elapsed:.0f} frames/s)")
        print(f"foot position error : mean {task.mean()*1e3:.5f} mm, max {task.max()*1e3:.5f} mm")
        print(f"loop closure error  : mean {closure.mean()*1e3:.5f} mm, max {closure.max()*1e3:.5f} mm")
        print(f"sole tilt           : max {np.degrees(tilt.max()):.5f} deg")
        print(f"IK iterations       : mean {iters.mean():.1f}, max {iters.max()}")
        print(f"leg extension       : max {reach.max():.4f} m of {dp.MAX_LEG_REACH:.4f} "
              f"({100*reach.max()/dp.MAX_LEG_REACH:.1f}%)")
        if reach.max() / dp.MAX_LEG_REACH > 0.97:
            print("  WARNING: leg is near full extension; gait is at the workspace edge")

        act = dp.Actuation(model, include_arms=False)
        print("\nleg joint travel over the gait:")
        for name, adr, jid in zip(act.names, act.qposadr, act.joint_ids):
            lo, hi = model.jnt_range[jid]
            s = qpos[:, adr]
            margin = min((s.min() - lo), (hi - s.max())) / (hi - lo)
            flag = "  <-- near limit" if margin < 0.02 else ""
            print(f"  {name:<28} {np.degrees(s.min()):>8.2f} .. {np.degrees(s.max()):>8.2f} deg"
                  f"   (limits {np.degrees(lo):>7.1f} .. {np.degrees(hi):>7.1f}){flag}")

    return ref, qpos, dict(task=task, closure=closure, tilt=tilt, iters=iters, reach=reach)


if __name__ == "__main__":
    ref, qpos, stats = build()
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "digit_walk_qpos.npy", qpos)
    np.savez(
        OUT / "digit_walk_reference.npz",
        t=ref["t"], pelvis=ref["pelvis"], left=ref["left"], right=ref["right"],
        com=ref["com"], stance_is_left=ref["stance_is_left"], dt=DT,
    )
    print(f"\nwrote {OUT / 'digit_walk_qpos.npy'}  shape {qpos.shape}")
