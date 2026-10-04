"""Verification of the Digit port: geometry, IK, estimator, contact.

The tests that earn their keep here are the ones encoding bugs actually hit during
the port:

  * `test_foot_geom_rotation_is_respected` -- the collision box carries
    euler='-1.047' (60 deg about x). Ignoring `geom_quat` leaves the sole tilted by
    ~50 deg while the position task still reports success.
  * `test_heel_toe_calibration_is_mirrored` -- the geom eulers are mirrored between
    sides, so the geom's +y axis points in OPPOSITE world directions on the two
    feet. Without calibration a yaw target of 0 demands a 180 deg flip and saturates
    the hip yaw joint.
  * `test_efc_rows_must_be_filtered` -- Digit has active joint limits at ordinary
    poses, so `efc` mixes limit rows with equality rows.

Run:  python tests/test_digit.py
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import alip  # noqa: E402
import digit_params as dp  # noqa: E402
import digit_state as ds  # noqa: E402
from digit_ik import DigitIK  # noqa: E402

TOL_M = 1e-5


def _standing():
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q = dp.nominal_pose(model, data, feet)
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    feet.calibrate(data)
    return model, data, feet, q


# ---------------------------------------------------------------------------
# Model audit
# ---------------------------------------------------------------------------


def test_model_loads_and_basic_counts():
    model, _ = dp.load()
    assert model.nq == 57 and model.nv == 50
    assert model.neq == 6
    assert 45.0 < np.sum(model.body_mass) < 55.0


def test_model_has_no_springs():
    """Digit's MJCF has no series compliance, unlike Cassie's 1500/1250 springs."""
    model, _ = dp.load()
    assert not np.any(model.jnt_stiffness > 0)


def test_model_declares_no_torque_limits():
    """Documents why digit_params imposes TORQUE_LIMIT itself."""
    model, _ = dp.load()
    assert not np.any(model.actuator_ctrllimited)
    assert not np.any(model.actuator_forcelimited)
    assert np.allclose(model.actuator_gear[:, 0], 1.0), "ctrl should be torque directly"


def test_fictitious_actuators_are_excluded():
    """tarsus / toe_pitch / toe_roll are dependent coordinates on hardware."""
    model, _ = dp.load()
    all_act = dp.Actuation(model, real_only=False)
    real = dp.Actuation(model)
    legs = dp.Actuation(model, include_arms=False)
    assert all_act.n == 26
    assert real.n == 20, real.n              # 12 leg + 8 arm
    assert legs.n == 12, legs.n              # 6 per leg
    for nm in real.names:
        assert not any(f in nm for f in dp.FICTITIOUS_ACTUATORS), nm


def test_torque_limits_are_applied_and_clipped():
    model, data = dp.load()
    act = dp.Actuation(model, include_arms=False)
    applied = act.apply_torque(data, np.full(act.n, 1e6))
    assert np.allclose(applied, act.torque_limit)
    assert np.all(np.abs(data.ctrl) <= act.torque_limit.max() + 1e-9)
    # Untouched actuators must stay at zero.
    mask = np.ones(model.nu, bool)
    mask[act.act_ids] = False
    assert np.allclose(data.ctrl[mask], 0.0)


# ---------------------------------------------------------------------------
# Foot geometry -- the bugs
# ---------------------------------------------------------------------------


def test_foot_geom_rotation_is_respected():
    """The sole normal must come from geom_quat, not from body +z."""
    model, _ = dp.load()
    feet = dp.FootGeometry(model)
    for side in dp.SIDES:
        n_body = feet.normal_body[side]
        # 60 deg about x => normal is (0, +-0.866, 0.5) in the body frame.
        assert abs(abs(n_body[1]) - np.sqrt(3) / 2) < 1e-3, n_body
        assert abs(n_body[2] - 0.5) < 1e-3, n_body
        assert not np.allclose(n_body, [0, 0, 1]), "body +z would be the naive bug"


def test_foot_dimensions():
    model, _ = dp.load()
    feet = dp.FootGeometry(model)
    d = feet.dimensions()
    assert abs(d["length"] - 0.235) < 1e-6
    assert abs(d["width"] - 0.08) < 1e-6


def test_sole_is_flat_and_level_when_standing():
    _, data, feet, _ = _standing()
    for side in dp.SIDES:
        assert np.allclose(feet.sole_normal(data, side), [0, 0, 1], atol=1e-4)
        assert np.ptp(feet.corner_heights(data, side)) < 1e-6


def test_heel_toe_calibration_is_mirrored():
    """The raw geom y axis points opposite ways on the two feet."""
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q = dp.nominal_pose(model, data, feet)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    raw = {}
    for side in dp.SIDES:
        R = np.array(data.xmat[feet.body_id[side]]).reshape(3, 3)
        raw[side] = (R @ feet.long_body[side])[0]
    assert raw["left"] * raw["right"] < 0, "expected mirrored long axes"

    signs = feet.calibrate(data)
    assert signs["left"] * signs["right"] < 0
    # After calibration both point forward, i.e. toe is ahead of heel.
    for side in dp.SIDES:
        assert feet.long_direction(data, side)[0] > 0.9
        toe = feet.world(data, side, "toe")
        heel = feet.world(data, side, "heel")
        assert toe[0] > heel[0], side


def test_efc_rows_must_be_filtered():
    """qpos0 has limit rows mixed in with equality rows."""
    model, data = dp.load()
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    rows = dp.equality_rows(model, data)
    assert len(rows) == 3 * model.neq == 18
    assert data.nefc > len(rows), "expected extra limit/contact rows to filter out"
    types = np.asarray(data.efc_type[: data.nefc])[rows]
    assert np.all(types == mujoco.mjtConstraint.mjCNSTR_EQUALITY)


def test_loop_closure_satisfied_at_qpos0():
    """Unlike Cassie's qpos_init (5.6 mm off), Digit's qpos0 is on the manifold."""
    model, data = dp.load()
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    rows = dp.equality_rows(model, data)
    assert np.max(np.abs(np.asarray(data.efc_pos)[rows])) < 1e-9


# ---------------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------------


def test_nominal_pose_is_a_valid_stance():
    model, data, feet, q = _standing()
    m = dp.measure(model, data, feet)
    assert abs(m["com_height_above_feet"] - 0.80) < 1e-3
    assert abs(m["foot_separation_y"] - 0.26) < 1e-3
    for side, sgn in (("left", +1), ("right", -1)):
        p = feet.contact_point(data, side)
        assert abs(p[2]) < 1e-6, p
        assert np.sign(p[1]) == sgn
    rows = dp.equality_rows(model, data)
    assert np.max(np.abs(np.asarray(data.efc_pos)[rows])) < 1e-7


def test_nominal_pose_leaves_workspace_margin():
    """A near-straight leg has no room for a swing and is badly conditioned."""
    model, data, feet, _ = _standing()
    for side in dp.SIDES:
        hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hip_roll")
        reach = np.linalg.norm(feet.contact_point(data, side) - np.array(data.xpos[hip]))
        assert reach / dp.MAX_LEG_REACH < 0.95, f"{side} leg {reach:.4f} m is too straight"


def test_ik_reaches_targets_verified_on_fresh_model():
    _, _, _, q0 = _standing()
    ik = DigitIK()
    ik.data.qpos[:] = q0
    mujoco.mj_forward(ik.model, ik.data)
    ik.feet.calibrate(ik.data)
    left0 = ik.feet.contact_point(ik.data, "left").copy()
    right0 = ik.feet.contact_point(ik.data, "right").copy()

    model, data = dp.load()
    feet = dp.FootGeometry(model)

    for dx, dy, dz in [(0.10, 0, 0.06), (-0.08, 0, 0.05), (0.15, 0.03, 0.08)]:
        target = left0 + np.array([dx, dy, dz])
        res = ik.solve(q0, {"left": target, "right": right0}, base_pos=q0[0:3],
                       base_quat=(1, 0, 0, 0), flat_feet=True)
        assert res.converged, (dx, dy, dz, res.task_error)
        assert res.closure_error < 1e-6
        # Re-verify on an independent model instance.
        data.qpos[:] = res.qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        got = feet.contact_point(data, "left")
        assert np.max(np.abs(got - target)) < 1e-4, (got, target)


def test_ik_keeps_feet_flat_while_swinging():
    _, _, _, q0 = _standing()
    ik = DigitIK()
    ik.data.qpos[:] = q0
    mujoco.mj_forward(ik.model, ik.data)
    ik.feet.calibrate(ik.data)
    left0 = ik.feet.contact_point(ik.data, "left").copy()
    right0 = ik.feet.contact_point(ik.data, "right").copy()

    res = ik.solve(q0, {"left": left0 + [0.12, 0, 0.10], "right": right0},
                   base_pos=q0[0:3], base_quat=(1, 0, 0, 0), flat_feet=True,
                   yaw_target={"left": 0.0, "right": 0.0})
    assert res.converged
    assert np.degrees(res.tilt_error) < 0.5, np.degrees(res.tilt_error)


def test_yaw_target_without_calibration_raises():
    """Refuse to silently demand a 180 degree flip."""
    ik = DigitIK()
    q0 = ik.model.qpos0.copy()
    try:
        ik.solve(q0, {}, flat_feet=True, yaw_target={"left": 0.0}, max_iter=1)
    except RuntimeError as exc:
        assert "calibrat" in str(exc).lower()
    else:
        raise AssertionError("expected a RuntimeError about calibration")


def test_base_is_held_fixed_by_ik():
    _, _, _, q0 = _standing()
    ik = DigitIK()
    res = ik.solve(q0, {}, base_pos=(0.2, -0.05, 0.79), base_quat=(1, 0, 0, 0))
    assert np.allclose(res.qpos[0:3], (0.2, -0.05, 0.79), atol=1e-12)
    assert np.allclose(res.qpos[3:7], (1, 0, 0, 0), atol=1e-12)


def test_arms_are_frozen_when_requested():
    _, _, _, q0 = _standing()
    ik = DigitIK(freeze_arms=True)
    arm_adr = [ik.model.jnt_qposadr[j] for j in range(ik.model.njnt)
               if any(a in (mujoco.mj_id2name(ik.model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
                      for a in dp.ARM_MOTORS)]
    ik.data.qpos[:] = q0
    mujoco.mj_forward(ik.model, ik.data)
    ik.feet.calibrate(ik.data)
    left0 = ik.feet.contact_point(ik.data, "left").copy()
    res = ik.solve(q0, {"left": left0 + [0.1, 0, 0.05]}, base_pos=q0[0:3])
    assert np.allclose(res.qpos[arm_adr], q0[arm_adr], atol=1e-12)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


def test_angmom_matches_bruteforce_on_digit():
    model, data = dp.load()
    for scale in (0.0, 0.4, 1.5):
        data.qpos[:] = model.qpos0
        rng = np.random.default_rng(int(scale * 100))
        data.qvel[:] = rng.normal(0, scale, model.nv) if scale else 0.0
        mujoco.mj_forward(model, data)
        for point in (np.zeros(3), np.array([0.2, -0.1, 0.0])):
            fast = ds.angular_momentum_about(model, data, point)
            slow = ds.angular_momentum_bruteforce(model, data, point)
            assert np.allclose(fast, slow, rtol=1e-8, atol=1e-9), (scale, fast, slow)


def test_estimator_sign_conventions_match_alip():
    """Ly = +mH*vx and Lx = -mH*vy, matching src/alip.py."""
    model, data, feet, q = _standing()
    est = ds.DigitAlipEstimator(model, feet)
    mass = ds.total_mass(model)

    for axis, sgn, label in ((0, +1.0, "Ly"), (1, -1.0, "Lx")):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        data.qvel[axis] = 0.6
        mujoco.mj_forward(model, data)
        st = est.estimate(data, stance_is_left=True)
        want = sgn * mass * st.H * 0.6
        got = st.Ly if axis == 0 else st.Lx
        assert abs(got - want) < 1e-5 * max(1.0, abs(want)), (label, got, want)


def test_estimator_height_matches_alip_params():
    model, data, feet, q = _standing()
    est = ds.DigitAlipEstimator(model, feet)
    st = est.estimate(data, stance_is_left=True)
    p = alip.AlipParams(m=ds.total_mass(model), H=st.H, W=0.26)
    assert abs(st.H - 0.80) < 1e-3
    assert 3.4 < p.omega < 3.6, p.omega


def test_contact_detector_and_cop_under_load():
    model, data, feet, q = _standing()
    det = ds.DigitContactDetector(model)
    data.qpos[:] = q
    data.qvel[:] = 0.0
    # Settle briefly so contacts form; no controller, so only check early frames.
    for _ in range(40):
        mujoco.mj_step(model, data)
    cs = det.state(data)
    assert cs.force_left + cs.force_right > 0.0
    for side in dp.SIDES:
        cop = det.center_of_pressure(data, side)
        if cop is not None:
            # CoP must lie within the foot's own footprint.
            corners = np.array([feet.world(data, side, c)[:2] for c in
                                ("corner_tl", "corner_tr", "corner_hl", "corner_hr")])
            lo, hi = corners.min(axis=0) - 0.02, corners.max(axis=0) + 0.02
            assert np.all(cop >= lo) and np.all(cop <= hi), (side, cop, lo, hi)


# ---------------------------------------------------------------------------
# The planner is shared, unchanged
# ---------------------------------------------------------------------------


def test_alip_planner_is_robot_agnostic():
    """The same src/alip.py serves both robots; only the parameters differ."""
    p_digit = dp.alip_params()
    assert abs(p_digit.m - 50.067) < 0.01
    assert abs(p_digit.H - 0.80) < 1e-3
    assert 3.4 < p_digit.omega < 3.6

    # Deadbeat property must hold for Digit's parameters too.
    x, Ly = alip.periodic_orbit_sagittal(0.4, p_digit)
    x_T, L_T = alip.flow(x, Ly, p_digit.T, alip.SAGITTAL, p_digit)
    L_des = alip.desired_L_sagittal(0.4, p_digit)
    step = alip.foot_placement(x_T, L_T, L_des, alip.SAGITTAL, p_digit)
    xp, Lp = alip.reset_map(x_T, L_T, step)
    _, L_next = alip.flow(xp, Lp, p_digit.T, alip.SAGITTAL, p_digit)
    assert abs(L_next - L_des) < 1e-9

    # Lateral periodic orbit closes for Digit's width too.
    y, Lx = alip.periodic_orbit_frontal(True, p_digit)
    stance = True
    for _ in range(2):
        y_T, L_T = alip.flow(y, Lx, p_digit.T, alip.FRONTAL, p_digit)
        s = alip.plan_lateral_step(y, Lx, p_digit.T, 0.0, stance, p_digit)
        assert abs(abs(s) - p_digit.W) < 1e-9
        y, Lx = alip.reset_map(y_T, L_T, s)
        stance = not stance
    y0, Lx0 = alip.periodic_orbit_frontal(True, p_digit)
    assert abs(y - y0) < 1e-12 and abs(Lx - Lx0) < 1e-12


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
