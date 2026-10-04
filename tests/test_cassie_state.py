"""Verification of the ALIP state estimator against independent oracles.

The important test is `test_angmom_matches_bruteforce`: MuJoCo's `subtree_angmom`
and a hand-rolled sum over all 26 bodies must agree. They share no code, so a
wrong frame, a missing rotational inertia, or a bad transfer term shows up
immediately.

Run:  python tests/test_cassie_state.py
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import cassie_params  # noqa: E402
import cassie_state as cs  # noqa: E402

RNG = np.random.default_rng(0)


def _model_with_state(qvel_scale=0.0, seed=0):
    model, data = cassie_params.load()
    data.qpos[:] = cassie_params.QPOS_INIT
    if qvel_scale:
        rng = np.random.default_rng(seed)
        data.qvel[:] = rng.normal(0, qvel_scale, model.nv)
    mujoco.mj_forward(model, data)
    return model, data


# ---------------------------------------------------------------------------


def test_total_mass_matches_model():
    model, _ = cassie_params.load()
    assert abs(cs.total_mass(model) - float(np.sum(model.body_mass))) < 1e-12
    assert 30.0 < cs.total_mass(model) < 36.0


def test_angmom_matches_bruteforce():
    """subtree_angmom-based transfer vs an explicit sum over bodies."""
    for scale in (0.0, 0.2, 1.0, 3.0):
        model, data = _model_with_state(scale, seed=int(scale * 10))
        for point in (
            np.zeros(3),
            np.array([0.3, -0.2, 0.0]),
            cs.com_position(data),
        ):
            fast = cs.angular_momentum_about(model, data, point)
            slow = cs.angular_momentum_bruteforce(model, data, point)
            assert np.allclose(fast, slow, rtol=1e-8, atol=1e-9), (
                scale, point, fast, slow
            )


def test_com_velocity_matches_finite_difference():
    """subtree_linvel must equal d(com)/dt."""
    model, data = _model_with_state(0.5, seed=3)
    v_analytic = cs.com_velocity(model, data)

    h = 1e-7
    com0 = cs.com_position(data).copy()
    mujoco.mj_integratePos(model, data.qpos, data.qvel, h)
    mujoco.mj_forward(model, data)
    v_fd = (cs.com_position(data) - com0) / h

    assert np.allclose(v_analytic, v_fd, rtol=1e-4, atol=1e-6), (v_analytic, v_fd)


def test_angmom_transfer_identity():
    """L about p must equal L about com plus the transfer term."""
    model, data = _model_with_state(0.8, seed=5)
    com = cs.com_position(data)
    p = np.array([0.11, -0.07, 0.013])
    L_com = cs.angular_momentum_bruteforce(model, data, com)
    L_p = cs.angular_momentum_bruteforce(model, data, p)
    transfer = np.cross(com - p, cs.total_mass(model) * cs.com_velocity(model, data))
    assert np.allclose(L_p, L_com + transfer, rtol=1e-8, atol=1e-9)


def test_angmom_zero_at_rest():
    model, data = _model_with_state(0.0)
    L = cs.angular_momentum_about(model, data, np.zeros(3))
    assert np.max(np.abs(L)) < 1e-12


def test_pure_translation_gives_expected_momentum():
    """Rigid horizontal translation: L about a point below must be m*H*v."""
    model, data = cassie_params.load()
    data.qpos[:] = cassie_params.QPOS_INIT
    data.qvel[:] = 0.0
    data.qvel[0] = 0.7  # pelvis slide in x, whole body moves together
    mujoco.mj_forward(model, data)

    est = cs.AlipEstimator(model)
    st = est.estimate(data, stance_is_left=True)

    # With no relative joint motion the reduced model is exact.
    want = cs.total_mass(model) * st.H * 0.7
    assert abs(st.Ly - want) < 1e-6 * max(1.0, abs(want)), (st.Ly, want)
    dsag, dfrontal = st.alip_consistency(cs.total_mass(model))
    assert abs(dsag) < 1e-6 and abs(dfrontal) < 1e-6


def test_lateral_translation_sign_convention():
    """Lx = -m*H*vy is the convention src/alip.py assumes."""
    model, data = cassie_params.load()
    data.qpos[:] = cassie_params.QPOS_INIT
    data.qvel[:] = 0.0
    data.qvel[1] = 0.4  # +y is left
    mujoco.mj_forward(model, data)

    est = cs.AlipEstimator(model)
    st = est.estimate(data, stance_is_left=True)
    want = -cs.total_mass(model) * st.H * 0.4
    assert abs(st.Lx - want) < 1e-6 * max(1.0, abs(want)), (st.Lx, want)
    assert st.Lx < 0, "moving left must give negative Lx under this convention"


def test_estimator_geometry_at_nominal_pose():
    model, data = _model_with_state(0.0)
    est = cs.AlipEstimator(model)
    st = est.estimate(data, stance_is_left=True)

    # Contact point should be under the left foot.
    assert st.contact[1] > 0.1, st.contact
    # CoM height above contact should match what cassie_params measured.
    m = cassie_params.measure(model, data)
    assert abs(st.H - m["com_height_above_feet"]) < 1e-6

    st_r = est.estimate(data, stance_is_left=False)
    assert st_r.contact[1] < -0.1
    # Symmetric pose: y offsets should mirror.
    assert abs(st.y + st_r.y) < 1e-3


def test_swing_leg_breaks_the_reduced_model():
    """A moving swing leg must show up as m*H*v being wrong -- that is the point."""
    model, data = cassie_params.load()
    data.qpos[:] = cassie_params.QPOS_INIT
    data.qvel[:] = 0.0
    # Swing the right hip pitch fast, pelvis still.
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right-hip-pitch")
    data.qvel[model.jnt_dofadr[jid]] = 6.0
    mujoco.mj_forward(model, data)

    est = cs.AlipEstimator(model)
    st = est.estimate(data, stance_is_left=True)
    dsag, _ = st.alip_consistency(cs.total_mass(model))
    assert abs(dsag) > 0.05, (
        "a fast swing leg should make the reduced-model approximation visibly "
        f"wrong, got {dsag}"
    )


def test_contact_detector_finds_foot_geoms():
    model, _ = cassie_params.load()
    det = cs.ContactDetector(model)
    for side in ("left", "right"):
        assert det.foot_geoms[side], f"no geoms found for {side} foot"


def _pd_stand(model, data, seconds, kp=1500.0, kd=50.0):
    """Hold the nominal joint angles with PD. Returns the recorded history."""
    act = cassie_params.Actuation(model)
    q_target = cassie_params.QPOS_INIT[act.qposadr]
    det = cs.ContactDetector(model)
    hist = []
    for _ in range(int(seconds / model.opt.timestep)):
        act.apply_torque(data, act.joint_pd(data, q_target, kp, kd))
        mujoco.mj_step(model, data)
        hist.append(
            (
                data.qpos[2],
                sum(det.normal_force(data, s) for s in ("left", "right")),
            )
        )
    return np.array(hist)


def test_contact_detector_measures_body_weight_while_upright():
    """While she is still standing, ground reaction must balance her weight."""
    model, data = cassie_params.load()
    data.qpos[:] = cassie_params.QPOS_INIT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    hist = _pd_stand(model, data, 0.6)
    weight = cs.total_mass(model) * 9.81

    # The series springs make the force ring at a few Hz, so average it.
    settled = hist[int(0.3 / model.opt.timestep) :, 1]
    mean_force = float(settled.mean())
    assert 0.6 * weight < mean_force < 1.5 * weight, (
        f"mean ground reaction {mean_force:.1f} N vs weight {weight:.1f} N"
    )


def test_joint_pd_alone_cannot_balance():
    """Regression test encoding *why* a balance controller is required.

    Joint PD tracks the leg configuration essentially perfectly -- and she still
    falls over, because nothing regulates the CoM above the support polygon. If a
    future change makes this pass, the model or gravity has changed, not the
    controller.
    """
    model, data = cassie_params.load()
    data.qpos[:] = cassie_params.QPOS_INIT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    act = cassie_params.Actuation(model)
    q_target = cassie_params.QPOS_INIT[act.qposadr]
    hist = _pd_stand(model, data, 2.5)

    # Legs held their shape ...
    joint_err = np.max(np.abs(data.qpos[act.qposadr] - q_target))
    # ... yet the pelvis dropped a long way.
    assert hist[-1, 0] < 0.75, f"expected a fall, pelvis ended at {hist[-1, 0]:.3f} m"
    assert joint_err < 0.5, (
        f"joints should still be near target during the fall, max error {joint_err:.3f} rad"
    )


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
