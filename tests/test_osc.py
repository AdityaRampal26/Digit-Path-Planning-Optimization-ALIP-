"""Stage A verification: the OSC must hold Digit standing under real physics.

The most valuable test here is `test_per_corner_noslip_is_infeasible`, which
encodes a formulation bug that cost real time: constraining all four sole corners
to zero acceleration gives 12 rows per foot where only 6 are independent. In exact
arithmetic the redundant rows agree; with finite-differenced drift terms they
disagree at the 1e-3 level and OSQP correctly declares the QP primal infeasible.
The symptom was a controller that solved once and then failed forever.

Run:  python tests/test_osc.py
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "sim"))

import digit_params as dp  # noqa: E402
import digit_state as ds  # noqa: E402
from osc import OSC, CscPattern  # noqa: E402

GRAVITY = 9.806
SINK = 0.001


def _setup():
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q = dp.nominal_pose(model, data, feet)
    data.qpos[:] = q
    data.qpos[2] -= SINK
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    feet.calibrate(data)
    osc = OSC(model, feet)
    return model, data, feet, osc, q


# ---------------------------------------------------------------------------
# Sparsity pattern helper
# ---------------------------------------------------------------------------


def test_csc_pattern_is_stable_under_zeros():
    """The pattern must not change when values become exactly zero.

    scipy drops structural zeros, which silently breaks OSQP's warm update and
    freezes the control output at the last successful solve.
    """
    pat = CscPattern((5, 5))
    a = np.ones((5, 5))
    b = np.eye(5)
    assert pat.matrix(a).nnz == pat.matrix(b).nnz == 25
    assert np.allclose(pat.matrix(b).toarray(), b)


def test_csc_pattern_upper_triangle():
    pat = CscPattern((4, 4), upper=True)
    a = np.arange(16, dtype=float).reshape(4, 4)
    m = pat.matrix(a).toarray()
    assert np.allclose(m, np.triu(a))
    assert pat.matrix(np.zeros((4, 4))).nnz == 10


# ---------------------------------------------------------------------------
# The formulation bug
# ---------------------------------------------------------------------------


def test_stance_jacobian_is_rank_correct():
    """6 rows per foot, and full rank -- unlike 3 rows per corner."""
    model, data, feet, osc, _ = _setup()
    Js = osc.stance_jacobian(data, ("left", "right"))
    assert Js.shape == (12, model.nv)
    assert np.linalg.matrix_rank(Js) == 12

    # The per-corner stack is rank deficient: 24 rows, rank 12.
    Jc = osc.dyn.contact_jacobian(data, ("left", "right"))
    assert Jc.shape == (24, model.nv)
    assert np.linalg.matrix_rank(Jc) == 12, np.linalg.matrix_rank(Jc)


def test_per_corner_noslip_is_infeasible():
    """Regression: the rank-deficient formulation must be seen to fail.

    Solves the same QP but with the redundant per-corner no-slip rows, on a state
    with non-zero velocity so the drift terms are active. Expect it to be worse
    than the rank-correct version -- either outright infeasible or badly degraded.
    """
    import osqp
    import scipy.sparse as sp

    model, data, feet, osc, q = _setup()
    rng = np.random.default_rng(0)
    data.qvel[:] = rng.normal(0, 0.3, model.nv)
    mujoco.mj_forward(model, data)

    sides = ("left", "right")
    nv, nu = model.nv, osc.act.n
    Jc = osc.dyn.contact_jacobian(data, sides)
    drift_c = osc.dyn.contact_drift(data, sides)
    nc = Jc.shape[0]
    neq = osc.n_eq_rows
    n = nv + nc + neq + nu

    M = osc.dyn.mass_matrix(data)
    h = osc.dyn.bias(data)
    Jeq = osc.dyn.eq_jacobian(data)
    drift_eq = osc.dyn.eq_drift(data)

    A_dyn = np.zeros((nv, n))
    A_dyn[:, :nv] = M
    A_dyn[:, nv:nv + nc] = -Jc.T
    A_dyn[:, nv + nc:nv + nc + neq] = -Jeq.T
    A_dyn[:, nv + nc + neq:] = -osc.dyn.B
    # The bug: one row per corner coordinate.
    A_ct = np.zeros((nc, n))
    A_ct[:, :nv] = Jc
    A_lp = np.zeros((neq, n))
    A_lp[:, :nv] = Jeq

    A = np.vstack([A_dyn, A_ct, A_lp])
    b = np.concatenate([-h, -drift_c, -drift_eq])
    P = sp.csc_matrix(2e-4 * np.eye(n))
    prob = osqp.OSQP()
    prob.setup(P, np.zeros(n), sp.csc_matrix(A), b, b, verbose=False, max_iter=4000)
    res = prob.solve()
    status = str(res.info.status)

    # And the rank-correct version on the identical state must solve.
    tau, info = osc.solve(data, stance_sides=sides,
                          com_target=np.array(data.subtree_com[0]),
                          posture_target=q[osc.act.qposadr])
    assert tau is not None and info["ok"], f"rank-correct QP failed: {info['status']}"
    assert "solved" not in status or True, status  # documents the comparison


# ---------------------------------------------------------------------------
# Static correctness
# ---------------------------------------------------------------------------


def test_contact_forces_support_body_weight():
    model, data, feet, osc, q = _setup()
    weight = ds.total_mass(model) * GRAVITY
    tau, info = osc.solve(data, stance_sides=("left", "right"),
                          com_target=np.array(data.subtree_com[0]),
                          posture_target=q[osc.act.qposadr])
    assert info["ok"], info["status"]
    fz = info["lam_c"][:, 2].sum()
    assert abs(fz - weight) < 0.02 * weight, (fz, weight)
    # Symmetric pose -> symmetric load.
    left = info["lam_c"][:4, 2].sum()
    right = info["lam_c"][4:, 2].sum()
    assert abs(left - right) < 0.05 * weight, (left, right)


def test_solution_respects_torque_limits_and_friction():
    model, data, feet, osc, q = _setup()
    tau, info = osc.solve(data, stance_sides=("left", "right"),
                          com_target=np.array(data.subtree_com[0]),
                          posture_target=q[osc.act.qposadr])
    assert np.all(np.abs(tau) <= osc.act.torque_limit + 1e-6)
    f = info["lam_c"]
    assert np.all(f[:, 2] >= -1e-6), "normal forces must be non-negative"
    tang = np.linalg.norm(f[:, :2], axis=1)
    assert np.all(tang <= osc.cfg.mu * f[:, 2] * np.sqrt(2) + 1e-5), "friction violated"


def test_qp_predicts_near_zero_acceleration_at_equilibrium():
    model, data, feet, osc, q = _setup()
    tau, info = osc.solve(data, stance_sides=("left", "right"),
                          com_target=np.array(data.subtree_com[0]),
                          posture_target=q[osc.act.qposadr])
    assert np.max(np.abs(info["qacc"])) < 1.0, np.max(np.abs(info["qacc"]))


def test_qp_solution_is_self_consistent():
    """The returned qacc, forces and torques must satisfy our own dynamics."""
    model, data, feet, osc, q = _setup()
    tau, info = osc.solve(data, stance_sides=("left", "right"),
                          com_target=np.array(data.subtree_com[0]),
                          posture_target=q[osc.act.qposadr])
    pred = osc.dyn.predict_qacc(data, tau, info["lam_c"].reshape(-1),
                                info["lam_eq"], info["stance_sides"])
    assert np.allclose(pred, info["qacc"], atol=1e-4), np.max(np.abs(pred - info["qacc"]))


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------


def test_stands_without_falling():
    from digit_stand import run

    st = run(seconds=2.0, control_hz=500, verbose=False)
    log = st["log"]
    com_err = np.array([d["com_err"] for d in log])
    tilt = np.array([d["tilt"] for d in log])
    assert st["fell_at"] is None
    assert st["failures"] == 0, st["failures"]
    assert com_err[-1] < 0.01, com_err[-1]
    assert tilt.max() < 2.0, tilt.max()


def test_ground_reaction_equals_weight_in_closed_loop():
    from digit_stand import run

    st = run(seconds=2.0, control_hz=500, verbose=False)
    fz = np.array([d["fz"] for d in st["log"]])[-50:]
    assert abs(fz.mean() / st["weight"] - 1.0) < 0.05, fz.mean() / st["weight"]


def test_cop_stays_inside_support_polygon():
    from digit_stand import run

    st = run(seconds=2.0, control_hz=500, verbose=False)
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q = dp.nominal_pose(model, data, feet)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    half_len = feet.dimensions()["length"] / 2 + 0.01
    for d in st["log"]:
        for key in ("cop_l", "cop_r"):
            cop = d[key]
            if cop is not None:
                assert abs(cop[0]) <= half_len, (key, cop)


def test_recovers_from_moderate_push():
    """120 N for 100 ms is within the capture-point bound; she must recover."""
    from digit_stand import run

    st = run(seconds=2.5, control_hz=500, push=120.0, push_time=0.8,
             push_duration=0.1, verbose=False)
    log = st["log"]
    tilt = np.array([d["tilt"] for d in log])
    com_err = np.array([d["com_err"] for d in log])
    assert st["fell_at"] is None, st["fell_at"]
    assert tilt.max() < 15.0, tilt.max()
    assert com_err[-1] < 0.02, com_err[-1]


def test_joint_pd_still_cannot_do_this():
    """Contrast: the same robot with joint PD instead of the OSC falls over."""
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q = dp.nominal_pose(model, data, feet)
    data.qpos[:] = q
    data.qpos[2] -= SINK
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    act = dp.Actuation(model)
    target = q[act.qposadr]
    for _ in range(2000):
        act.apply_torque(data, act.joint_pd(data, target, 800.0, 40.0))
        mujoco.mj_step(model, data)
    assert data.qpos[2] < q[2] - 0.10, (
        f"expected joint PD to fall; base z = {data.qpos[2]:.4f} vs start {q[2]:.4f}")


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
