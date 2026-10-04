"""Solve the whole walking reference through IK and save the qpos trajectory.

This is the kinematic stage: the pelvis is prescribed along the ALIP CoM path and
the legs are solved to put the feet where the planner asked. Cassie will look like
she is walking, but nothing is pushing her -- there are no contact forces and no
torques, so she cannot fall. That is the point: it isolates the planner, the
trajectory generator and the IK, so any bug found later in the physics stage is
genuinely a *controller* bug.

Outputs out/walk_qpos.npy, which scripts/play_walk.py and scripts/render_skeleton.py read.

Run:  python scripts/alip_animate.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cassie_params  # noqa: E402
from cassie_ik import CassieIK  # noqa: E402
from gait import GaitReference, GaitSchedule  # noqa: E402

OUT = ROOT / "out"
DT = 1.0 / 100.0


def build(dt: float = DT, schedule: GaitSchedule | None = None, verbose: bool = True):
    params = cassie_params.alip_params()
    ref = GaitReference(params, schedule or GaitSchedule()).generate(dt)
    n = len(ref["t"])

    ik = CassieIK()
    q = cassie_params.QPOS_INIT.copy()

    qpos = np.zeros((n, ik.model.nq))
    task_err = np.zeros(n)
    closure_err = np.zeros(n)
    iters = np.zeros(n, dtype=int)

    t0 = time.time()
    for i in range(n):
        res = ik.solve(
            q,
            {"left": ref["left"][i], "right": ref["right"][i]},
            pelvis_pos=ref["pelvis"][i],
            pelvis_quat=(1.0, 0.0, 0.0, 0.0),
            max_iter=60,
            tol=1e-5,
        )
        qpos[i] = res.qpos
        task_err[i] = res.task_error
        closure_err[i] = res.closure_error
        iters[i] = res.iterations
        q = res.qpos  # warm start the next frame

        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} frames", flush=True)

    elapsed = time.time() - t0

    if verbose:
        print(f"\nsolved {n} frames in {elapsed:.1f} s ({n / elapsed:.0f} frames/s)")
        print(f"foot position error: mean {task_err.mean() * 1000:.4f} mm, "
              f"max {task_err.max() * 1000:.4f} mm")
        print(f"loop closure error:  mean {closure_err.mean() * 1000:.4f} mm, "
              f"max {closure_err.max() * 1000:.4f} mm")
        print(f"IK iterations: mean {iters.mean():.1f}, max {iters.max()}")

        bad = int(np.sum(task_err > 1e-3))
        if bad:
            print(f"WARNING: {bad} frames with >1 mm foot error (likely out of reach)")

        # Joint travel, to sanity check nothing is pinned against a limit.
        model = ik.model
        print("\njoint travel over the gait:")
        for i_u in range(model.nu):
            jid = model.actuator_trnid[i_u, 0]
            import mujoco

            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            adr = model.jnt_qposadr[jid]
            lo, hi = model.jnt_range[jid]
            series = qpos[:, adr]
            frac_lo = (series.min() - lo) / (hi - lo)
            frac_hi = (hi - series.max()) / (hi - lo)
            flag = "  <-- near limit" if min(frac_lo, frac_hi) < 0.02 else ""
            print(
                f"  {name:<17} {np.degrees(series.min()):>7.1f} .. "
                f"{np.degrees(series.max()):>7.1f} deg"
                f"   (limits {np.degrees(lo):>6.1f} .. {np.degrees(hi):>6.1f}){flag}"
            )

    return ref, qpos, dict(task_err=task_err, closure_err=closure_err, iters=iters)


if __name__ == "__main__":
    ref, qpos, stats = build()
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "walk_qpos.npy", qpos)
    np.savez(
        OUT / "walk_reference.npz",
        t=ref["t"],
        pelvis=ref["pelvis"],
        left=ref["left"],
        right=ref["right"],
        com=ref["com"],
        stance_is_left=ref["stance_is_left"],
        dt=DT,
    )
    print(f"\nwrote {OUT / 'walk_qpos.npy'}  shape {qpos.shape}")
