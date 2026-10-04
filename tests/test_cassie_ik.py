"""Verification of the Cassie leg IK.

The key checks are independence-based: after solving, we re-evaluate the pose on
a *separate* clean model (contacts enabled, fresh MjData) and confirm the feet
really are where we asked. That catches the classic IK failure where the solver's
own bookkeeping is self-consistent but wrong.

Run:  python tests/test_cassie_ik.py
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import cassie_params  # noqa: E402
from cassie_ik import SIDES, CassieIK  # noqa: E402

TOL_M = 1e-5  # 10 microns
Q0 = cassie_params.QPOS_INIT.copy()


def _fresh_model():
    """A clean model, deliberately not the solver's private copy."""
    model, data = cassie_params.load()
    return model, data


def _foot_centre_on(model, data, qpos, side):
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    toe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}-toe")
    heel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}-heel")
    return 0.5 * (np.array(data.site_xpos[toe]) + np.array(data.site_xpos[heel]))


def _nominal_feet(ik):
    ik.data.qpos[:] = Q0
    mujoco.mj_forward(ik.model, ik.data)
    return ik.foot_centre("left").copy(), ik.foot_centre("right").copy()


# ---------------------------------------------------------------------------


def test_identity_target_is_a_fixed_point():
    """Asking for the pose it already has must not move the robot."""
    ik = CassieIK()
    left, right = _nominal_feet(ik)
    res = ik.solve(Q0, {"left": left, "right": right})
    assert res.converged
    assert res.task_error < TOL_M
    # Legs may shift slightly because IK also repairs the 5.6mm closure error,
    # but the feet must stay put.
    for side, want in (("left", left), ("right", right)):
        got = _foot_centre_on(*_fresh_model(), res.qpos, side)
        assert np.max(np.abs(got - want)) < 1e-4, (side, got, want)


def test_solution_verified_on_independent_model():
    """The whole point: re-check the solved qpos with a fresh model instance."""
    ik = CassieIK()
    left, right = _nominal_feet(ik)
    model, data = _fresh_model()

    for dx, dy, dz in [
        (0.10, 0.00, 0.06),
        (-0.08, 0.00, 0.04),
        (0.00, 0.04, 0.08),
        (0.15, -0.03, 0.05),
    ]:
        target = left + np.array([dx, dy, dz])
        res = ik.solve(Q0, {"left": target, "right": right})
        assert res.converged, (dx, dy, dz, res.task_error)
        got = _foot_centre_on(model, data, res.qpos, "left")
        assert np.max(np.abs(got - target)) < 1e-4, (dx, dy, dz, got, target)


def test_loop_closure_is_enforced_not_ignored():
    """Solved poses must satisfy the connect constraints far better than qpos_init."""
    ik = CassieIK()
    left, right = _nominal_feet(ik)

    nominal_violation = ik.errors({})[1]
    assert nominal_violation > 1e-3, "expected qpos_init to be off the manifold"

    res = ik.solve(Q0, {"left": left + [0.12, 0, 0.07], "right": right})
    assert res.closure_error < 1e-6, res.closure_error
    assert res.closure_error < nominal_violation / 1000


def test_floating_base_is_held_fixed():
    ik = CassieIK()
    left, right = _nominal_feet(ik)
    res = ik.solve(Q0, {"left": left + [0.1, 0, 0.05], "right": right})
    assert np.allclose(res.qpos[0:7], Q0[0:7], atol=1e-12)


def test_pelvis_override_is_applied():
    ik = CassieIK()
    left, right = _nominal_feet(ik)
    new_pos = np.array([0.3, -0.05, 0.95])
    res = ik.solve(Q0, {"left": left, "right": right}, pelvis_pos=new_pos)
    assert np.allclose(res.qpos[0:3], new_pos, atol=1e-12)
    # Feet stay in world coordinates even though the pelvis moved.
    got = _foot_centre_on(*_fresh_model(), res.qpos, "left")
    assert np.max(np.abs(got - left)) < 1e-4


def test_joint_limits_respected():
    ik = CassieIK()
    left, right = _nominal_feet(ik)
    # Deliberately reach far enough to press against the limits.
    res = ik.solve(Q0, {"left": left + [0.45, 0, 0.02], "right": right})
    model = ik.model
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        adr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        q = res.qpos[adr]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert lo - 1e-9 <= q <= hi + 1e-9, f"{name}: {q} outside [{lo}, {hi}]"


def test_flat_sole_maintained():
    """The toe/heel level objective should keep the sole roughly horizontal."""
    ik = CassieIK()
    left, right = _nominal_feet(ik)
    res = ik.solve(Q0, {"left": left + [0.10, 0, 0.08], "right": right})
    model, data = _fresh_model()
    data.qpos[:] = res.qpos
    mujoco.mj_forward(model, data)
    toe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left-toe")
    heel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left-heel")
    dz = abs(data.site_xpos[toe][2] - data.site_xpos[heel][2])
    assert dz < 0.02, f"sole tilted by {dz * 1000:.1f} mm across the foot"


def test_workspace_sweep_converges():
    """A swing-foot-sized workspace should solve everywhere, quickly."""
    ik = CassieIK()
    left, right = _nominal_feet(ik)
    failures, worst_iters = [], 0
    for dx in np.linspace(-0.25, 0.30, 8):
        for dz in np.linspace(0.0, 0.15, 4):
            target = left + np.array([dx, 0.0, dz])
            res = ik.solve(Q0, {"left": target, "right": right})
            worst_iters = max(worst_iters, res.iterations)
            if not res.converged:
                failures.append((round(dx, 3), round(dz, 3), res.task_error))
    assert not failures, f"{len(failures)}/32 failed: {failures[:5]}"
    assert worst_iters < 40, worst_iters


def test_warm_start_is_cheaper_than_cold_start():
    """Seeding from the previous solution should cut iterations, as in a gait loop."""
    ik = CassieIK()
    left, right = _nominal_feet(ik)
    path = [left + np.array([dx, 0.0, 0.05]) for dx in np.linspace(0.0, 0.2, 10)]

    cold = sum(ik.solve(Q0, {"left": t, "right": right}).iterations for t in path)

    q = Q0.copy()
    warm = 0
    for t in path:
        res = ik.solve(q, {"left": t, "right": right})
        warm += res.iterations
        q = res.qpos
    assert warm <= cold, (warm, cold)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
