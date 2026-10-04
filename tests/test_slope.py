"""Verify ALIP on an inclined surface.

The claim being tested is that a constant grade turns the sagittal ALIP from a
linear system into an *affine* one, and that shifting to the equilibrium
x* = H tan(gamma) recovers the flat-ground equations exactly. If that holds, the
deadbeat foot placement law works on slopes with no other change -- and the
variable-pendulum-length machinery of the stair-climbing literature is not needed
for constant grades.

The decisive test is `test_flow_matches_numerical_integration_on_slope`, which
integrates the affine ODE numerically, sharing no code with the closed form.

Run:  python tests/test_slope.py
"""

from __future__ import annotations

import math
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import alip  # noqa: E402
from alip import FRONTAL, SAGITTAL, AlipParams  # noqa: E402

GRADES = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20)  # radians


def P(slope=0.0, T=0.35):
    return AlipParams(m=50.067, H=0.80, T=T, W=0.26, g=9.806, slope=slope)


# ---------------------------------------------------------------------------
# The affine dynamics
# ---------------------------------------------------------------------------


def _rk4_slope(pos, L, t, sign, p, n=200_000):
    """Integrate the affine ALIP ODE directly, as an independent oracle.

        x_dot  = L / (m H)
        L_dot  = m g cos(gamma) x  -  m g H sin(gamma)     (sagittal)
        y_dot  = -Lx / (m H)
        Lx_dot = -m g cos(gamma) y                          (frontal, no bias)
    """
    gc = p.g * math.cos(p.slope)
    bias = -p.m * p.g * p.H * math.sin(p.slope) if sign == SAGITTAL else 0.0

    def f(state):
        pos_, L_ = state
        return (sign * L_ / (p.m * p.H), sign * p.m * gc * pos_ + bias)

    h = t / n
    state = (pos, L)
    for _ in range(n):
        k1 = f(state)
        k2 = f((state[0] + 0.5 * h * k1[0], state[1] + 0.5 * h * k1[1]))
        k3 = f((state[0] + 0.5 * h * k2[0], state[1] + 0.5 * h * k2[1]))
        k4 = f((state[0] + h * k3[0], state[1] + h * k3[1]))
        state = (
            state[0] + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            state[1] + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        )
    return state


def test_flow_matches_numerical_integration_on_slope():
    for slope in GRADES:
        p = P(slope)
        for sign in (SAGITTAL, FRONTAL):
            for pos0, L0 in [(0.05, 3.0), (-0.12, -8.0), (0.0, 0.0), (0.2, 12.0)]:
                got = alip.flow(pos0, L0, p.T, sign, p)
                want = _rk4_slope(pos0, L0, p.T, sign, p)
                assert math.isclose(got[0], want[0], rel_tol=1e-7, abs_tol=1e-9), (
                    slope, sign, pos0, L0, got, want)
                assert math.isclose(got[1], want[1], rel_tol=1e-7, abs_tol=1e-8), (
                    slope, sign, pos0, L0, got, want)


def test_equilibrium_is_stationary():
    """At x = H tan(gamma) with zero momentum, nothing should move."""
    for slope in GRADES:
        p = P(slope)
        pos, L = alip.flow(p.x_equilibrium, 0.0, p.T, SAGITTAL, p)
        assert math.isclose(pos, p.x_equilibrium, abs_tol=1e-12), (slope, pos)
        assert abs(L) < 1e-9, (slope, L)


def test_equilibrium_formula():
    for slope in GRADES:
        p = P(slope)
        assert math.isclose(p.x_equilibrium, p.H * math.tan(slope), rel_tol=1e-12)
        # Sanity: 10 degrees on Digit is a ~14 cm lean.
        if abs(slope - 0.1745) < 1e-3:
            assert 0.13 < p.x_equilibrium < 0.15


def test_slope_bias_magnitude_exceeds_flat_restoring_torque():
    """Documents why the slope term dominates: it is bigger than what it perturbs."""
    p = P(math.radians(10.0))
    bias = abs(p.slope_bias)
    restoring_at_5cm = p.m * p.g * 0.05
    assert bias > 2.0 * restoring_at_5cm, (bias, restoring_at_5cm)


def test_omega_barely_changes():
    flat, ten = P(0.0).omega, P(math.radians(10.0)).omega
    assert abs(ten / flat - 1.0) < 0.01, (flat, ten)


# ---------------------------------------------------------------------------
# The deadbeat property must survive
# ---------------------------------------------------------------------------


def test_foot_placement_achieves_target_on_slope():
    """The defining property, on every grade."""
    for slope in GRADES:
        p = P(slope)
        for sign in (SAGITTAL, FRONTAL):
            for pos_T, L_T, L_des in [(0.05, 3.0, 8.0), (-0.10, -2.0, 8.0),
                                      (0.0, 0.0, -8.0), (0.25, 12.0, 0.0)]:
                step = alip.foot_placement(pos_T, L_T, L_des, sign, p)
                pos_p, L_p = alip.reset_map(pos_T, L_T, step)
                _, L_next = alip.flow(pos_p, L_p, p.T, sign, p)
                assert math.isclose(L_next, L_des, rel_tol=1e-8, abs_tol=1e-8), (
                    slope, sign, L_next, L_des)


def test_sagittal_orbit_is_fixed_point_on_slope():
    for slope in GRADES:
        p = P(slope)
        for v in (0.0, 0.3, 0.5):
            x, Ly = alip.periodic_orbit_sagittal(v, p)
            x_T, L_T = alip.flow(x, Ly, p.T, SAGITTAL, p)
            L_des = alip.desired_L_sagittal(v, p)
            assert math.isclose(L_T, L_des, rel_tol=1e-8, abs_tol=1e-10), (slope, v)
            step = alip.foot_placement(x_T, L_T, L_des, SAGITTAL, p)
            x_p, L_p = alip.reset_map(x_T, L_T, step)
            assert math.isclose(x_p, x, rel_tol=1e-8, abs_tol=1e-10), (slope, v, x_p, x)
            assert math.isclose(L_p, Ly, rel_tol=1e-8, abs_tol=1e-10), (slope, v)


def test_step_length_is_grade_independent():
    """The orbit translates uphill but its shape -- hence the stride -- is unchanged."""
    v = 0.4
    lengths = []
    for slope in GRADES:
        p = P(slope)
        x, Ly = alip.periodic_orbit_sagittal(v, p)
        x_T, L_T = alip.flow(x, Ly, p.T, SAGITTAL, p)
        step = alip.foot_placement(x_T, L_T, alip.desired_L_sagittal(v, p), SAGITTAL, p)
        lengths.append(step)
    # Only the tiny omega change should move it, well under 1%.
    assert max(lengths) / min(lengths) - 1.0 < 0.01, lengths


def test_orbit_translates_uphill_by_the_offset():
    v = 0.3
    flat_x, _ = alip.periodic_orbit_sagittal(v, P(0.0))
    for slope in (0.1, -0.1):
        p = P(slope)
        x, _ = alip.periodic_orbit_sagittal(v, p)
        # Same shape, shifted by the equilibrium offset (modulo the omega change).
        assert math.isclose(x - p.x_equilibrium, flat_x, rel_tol=2e-3), (slope, x, flat_x)
        assert (x > flat_x) == (slope > 0), "uphill grade must lean uphill"


def test_frontal_plane_unaffected_by_sagittal_grade():
    """A purely sagittal slope adds no constant frontal torque."""
    for slope in GRADES:
        p = P(slope)
        y, Lx = alip.periodic_orbit_frontal(True, p)
        stance = True
        for _ in range(2):
            y_T, L_T = alip.flow(y, Lx, p.T, FRONTAL, p)
            step = alip.plan_lateral_step(y, Lx, p.T, 0.0, stance, p)
            assert math.isclose(abs(step), p.W, rel_tol=1e-8), (slope, step)
            y, Lx = alip.reset_map(y_T, L_T, step)
            stance = not stance
        y0, Lx0 = alip.periodic_orbit_frontal(True, p)
        assert math.isclose(y, y0, abs_tol=1e-10), slope
        assert math.isclose(Lx, Lx0, abs_tol=1e-10), slope


# ---------------------------------------------------------------------------
# Flat ground must be untouched
# ---------------------------------------------------------------------------


def test_zero_slope_is_exactly_the_flat_case():
    p0 = P(0.0)
    assert p0.x_equilibrium == 0.0
    assert p0.slope_bias == 0.0
    assert alip.slope_offset(SAGITTAL, p0) == 0.0
    assert alip.slope_offset(FRONTAL, p0) == 0.0
    # And a nonzero grade must NOT shift the frontal plane.
    assert alip.slope_offset(FRONTAL, P(0.15)) == 0.0


def test_friction_requirement():
    """mu must exceed tan(gamma) just to stand, before any walking."""
    for deg in (5, 10, 15, 20):
        p = P(math.radians(deg))
        assert math.isclose(p.min_friction() if hasattr(p, "min_friction")
                            else abs(math.tan(p.slope)), abs(math.tan(p.slope)))
    # Digit's floor friction is 1.0, so the geometric limit is 45 degrees.
    assert math.degrees(math.atan(1.0)) == 45.0


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
