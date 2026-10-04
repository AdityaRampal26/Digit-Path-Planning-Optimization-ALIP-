"""Verify the dynamics layer against MuJoCo itself before any controller uses it.

If these do not pass, nothing built on top can work, and the symptom would look
like bad controller tuning rather than a wrong model. So they are deliberately
strict and check against MuJoCo's own numbers rather than against our derivation.

Run:  python tests/test_dynamics.py
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import digit_params as dp  # noqa: E402
from dynamics import CORNERS, DynamicsModel  # noqa: E402


def _state(qvel_scale=0.0, seed=0, settle=0):
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    data.qpos[:] = dp.nominal_pose(model, data, feet)
    data.qvel[:] = 0.0
    if qvel_scale:
        rng = np.random.default_rng(seed)
        data.qvel[:] = rng.normal(0, qvel_scale, model.nv)
    for _ in range(settle):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data, feet


# ---------------------------------------------------------------------------
# Equation of motion
# ---------------------------------------------------------------------------


def test_equation_of_motion_reproduced():
    """M qacc + h == B u + qfrc_constraint, using MuJoCo's own constraint forces."""
    for scale, seed in [(0.0, 0), (0.3, 1), (1.0, 2)]:
        model, data, feet = _state(scale, seed)
        dyn = DynamicsModel(model, feet)
        act = dyn.act
        rng = np.random.default_rng(seed + 50)
        tau = rng.normal(0, 20.0, act.n)
        act.apply_torque(data, tau)
        mujoco.mj_forward(model, data)

        res = dyn.residual_against_mujoco(data)
        assert np.max(np.abs(res)) < 1e-8, (scale, np.max(np.abs(res)))


def test_passive_forces_are_not_ignored():
    """This model has damping up to 2.0; dropping qfrc_passive breaks the EoM."""
    model, data, feet = _state(0.8, seed=7)
    dyn = DynamicsModel(model, feet)
    assert np.max(np.abs(np.asarray(data.qfrc_passive))) > 1e-6, "expected passive forces"

    good = np.max(np.abs(dyn.residual_against_mujoco(data)))
    # Same check but naively using qfrc_bias alone.
    M = dyn.mass_matrix(data)
    bad_lhs = M @ np.asarray(data.qacc) + np.asarray(data.qfrc_bias)
    bad = np.max(np.abs(bad_lhs - (np.asarray(data.qfrc_actuator)
                                   + np.asarray(data.qfrc_constraint)
                                   + np.asarray(data.qfrc_applied))))
    assert good < 1e-8
    assert bad > 1e-4, "omitting qfrc_passive should visibly break the identity"


def test_actuation_matrix_matches_qfrc_actuator():
    model, data, feet = _state()
    dyn = DynamicsModel(model, feet)
    rng = np.random.default_rng(3)
    tau = rng.normal(0, 15.0, dyn.act.n)
    applied = dyn.act.apply_torque(data, tau)
    mujoco.mj_forward(model, data)
    assert np.allclose(dyn.B @ applied, np.asarray(data.qfrc_actuator), atol=1e-9)


def test_mass_matrix_symmetric_positive_definite():
    model, data, feet = _state(0.5, seed=11)
    M = DynamicsModel(model, feet).mass_matrix(data)
    assert np.allclose(M, M.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(M)) > 0.0


# ---------------------------------------------------------------------------
# Jacobians and drift
# ---------------------------------------------------------------------------


def test_contact_jacobian_shape_and_finite_difference():
    """Jc must equal d(point)/dq, checked by finite difference."""
    model, data, feet = _state()
    dyn = DynamicsModel(model, feet)
    sides = ("left",)
    Jc = dyn.contact_jacobian(data, sides)
    assert Jc.shape == (3 * len(CORNERS), model.nv)

    rng = np.random.default_rng(5)
    dq = rng.normal(0, 1.0, model.nv)
    h = 1e-7
    p0 = dyn.contact_points(data, sides).reshape(-1)

    d2 = mujoco.MjData(model)
    d2.qpos[:] = data.qpos
    d2.qvel[:] = dq
    mujoco.mj_integratePos(model, d2.qpos, d2.qvel, h)
    mujoco.mj_forward(model, d2)
    p1 = dyn.contact_points(d2, sides).reshape(-1)

    assert np.allclose((p1 - p0) / h, Jc @ dq, rtol=1e-4, atol=1e-6)


def test_eq_jacobian_rows_are_equality_only():
    model, data, feet = _state()
    dyn = DynamicsModel(model, feet)
    Jeq = dyn.eq_jacobian(data)
    assert Jeq.shape == (3 * model.neq, model.nv) == (18, model.nv)


def test_drift_term_matches_second_derivative():
    """Jdot*qdot must equal the acceleration of the point at zero qacc."""
    model, data, feet = _state(0.7, seed=13)
    dyn = DynamicsModel(model, feet)
    sides = ("left",)

    drift = dyn.contact_drift(data, sides)

    # Independent estimate: central difference of point position along a
    # constant-velocity path gives d2p/dt2 = Jdot qdot when qacc = 0.
    h = 1e-4
    qvel = np.asarray(data.qvel)
    pts = []
    for s in (-1, 0, 1):
        d2 = mujoco.MjData(model)
        d2.qpos[:] = data.qpos
        d2.qvel[:] = qvel
        if s:
            mujoco.mj_integratePos(model, d2.qpos, d2.qvel, s * h)
        mujoco.mj_forward(model, d2)
        pts.append(dyn.contact_points(d2, sides).reshape(-1))
    second = (pts[2] - 2 * pts[1] + pts[0]) / h**2

    assert np.allclose(drift, second, rtol=2e-3, atol=1e-3), (
        np.max(np.abs(drift - second)))


def test_drift_is_zero_at_rest():
    model, data, feet = _state(0.0)
    dyn = DynamicsModel(model, feet)
    assert np.allclose(dyn.contact_drift(data, ("left", "right")), 0.0)
    assert np.allclose(dyn.eq_drift(data), 0.0)


# ---------------------------------------------------------------------------
# The end-to-end check: our model must predict MuJoCo's acceleration
# ---------------------------------------------------------------------------


def test_predicted_qacc_matches_mujoco_with_its_own_forces():
    """Feed MuJoCo's constraint forces back in and recover MuJoCo's qacc.

    This is the real gate on the whole layer: it exercises M, h, B, and the
    Jacobian transposes together. Constraint forces are taken from MuJoCo (rather
    than solved for) so that a failure here can only be our dynamics.
    """
    model, data, feet = _state(0.4, seed=17, settle=200)
    dyn = DynamicsModel(model, feet)
    rng = np.random.default_rng(23)
    tau = rng.normal(0, 10.0, dyn.act.n)
    applied = dyn.act.apply_torque(data, tau)
    mujoco.mj_forward(model, data)

    M = dyn.mass_matrix(data)
    rhs = dyn.B @ applied - dyn.bias(data) + np.asarray(data.qfrc_constraint)
    qacc_pred = np.linalg.solve(M, rhs)
    assert np.allclose(qacc_pred, np.asarray(data.qacc), rtol=1e-6, atol=1e-7), (
        np.max(np.abs(qacc_pred - np.asarray(data.qacc))))


def test_constraint_forces_decompose_onto_our_jacobians():
    """qfrc_constraint must lie in the row space of our stacked Jacobians.

    If it does not, the QP is missing a constraint that MuJoCo is enforcing, and
    the controller would be fighting an invisible force.
    """
    model, data, feet = _state(0.0, settle=300)
    dyn = DynamicsModel(model, feet)
    sides = ("left", "right")
    Jc = dyn.contact_jacobian(data, sides)
    Jeq = dyn.eq_jacobian(data)
    A = np.vstack([Jc, Jeq]).T                 # nv x (nc + neq)
    f = np.asarray(data.qfrc_constraint)

    sol, *_ = np.linalg.lstsq(A, f, rcond=None)
    residual = np.max(np.abs(A @ sol - f))
    scale = max(1.0, np.max(np.abs(f)))
    assert residual / scale < 5e-2, (residual, scale)


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
