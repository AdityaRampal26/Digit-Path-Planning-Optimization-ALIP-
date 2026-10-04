"""Verification of the ALIP model against independent oracles.

Each test checks the closed-form math against something that does not share its
derivation: numerical integration, or a fixed-point condition.

Run:  python -m pytest tests/test_alip.py -v      (or just: python tests/test_alip.py)
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import alip  # noqa: E402
from alip import FRONTAL, SAGITTAL, AlipParams  # noqa: E402

P = AlipParams()


# ---------------------------------------------------------------------------
# 1. The closed-form flow must match brute-force numerical integration.
# ---------------------------------------------------------------------------


def _rk4_flow(pos, L, t, sign, p, n=200_000):
    """Integrate the ALIP ODE with RK4. Deliberately naive, as an oracle."""

    def f(state):
        pos_, L_ = state
        return (sign * L_ / (p.m * p.H), sign * p.m * p.g * pos_)

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


def test_flow_matches_numerical_integration():
    for sign in (SAGITTAL, FRONTAL):
        for pos0, L0 in [(0.05, 3.0), (-0.12, -8.0), (0.0, 5.5), (0.2, 0.0)]:
            got = alip.flow(pos0, L0, P.T, sign, P)
            want = _rk4_flow(pos0, L0, P.T, sign, P)
            assert math.isclose(got[0], want[0], rel_tol=1e-9, abs_tol=1e-11), (sign, pos0, L0)
            assert math.isclose(got[1], want[1], rel_tol=1e-9, abs_tol=1e-11), (sign, pos0, L0)


def test_L_definition_consistent_with_velocity():
    """Ly = +mH*vx and Lx = -mH*vy must hold along the flow, not just at t=0."""
    pos0, L0, dt = 0.03, 4.0, 1e-6
    for sign, vel_sign in ((SAGITTAL, +1.0), (FRONTAL, -1.0)):
        a = alip.flow(pos0, L0, 0.1, sign, P)
        b = alip.flow(pos0, L0, 0.1 + dt, sign, P)
        vel_fd = (b[0] - a[0]) / dt
        vel_from_L = vel_sign * a[1] / (P.m * P.H)
        assert math.isclose(vel_fd, vel_from_L, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 2. The foot placement law must land L exactly on target one step later.
# ---------------------------------------------------------------------------


def test_foot_placement_achieves_target_L():
    """This is the defining property of the planner, so test it directly."""
    for sign in (SAGITTAL, FRONTAL):
        for pos_T, L_T, L_des in [
            (0.05, 3.0, 8.0),
            (-0.10, -2.0, 8.0),
            (0.0, 0.0, -8.0),
            (0.15, 12.0, 0.0),
        ]:
            step = alip.foot_placement(pos_T, L_T, L_des, sign, P)
            pos_plus, L_plus = alip.reset_map(pos_T, L_T, step)
            _, L_next = alip.flow(pos_plus, L_plus, P.T, sign, P)
            assert math.isclose(L_next, L_des, rel_tol=1e-9, abs_tol=1e-9)


def test_plan_step_matches_foot_placement_at_touchdown():
    """With zero time remaining, one-step-ahead planning is plain foot placement."""
    got = alip.plan_step(0.04, 5.0, 0.0, 9.0, SAGITTAL, P)
    want = alip.foot_placement(0.04, 5.0, 9.0, SAGITTAL, P)
    assert math.isclose(got, want, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 3. The analytic periodic orbits must actually be periodic.
# ---------------------------------------------------------------------------


def test_frontal_orbit_is_period_2_fixed_point():
    """Two steps from the frontal orbit must return to the starting state."""
    y, Lx = alip.periodic_orbit_frontal(stance_is_left=True, p=P)
    y0, Lx0, stance_is_left = y, Lx, True

    for _ in range(2):
        y_T, L_T = alip.flow(y, Lx, P.T, FRONTAL, P)
        # On the orbit the flow already delivers this step's desired momentum.
        assert math.isclose(L_T, alip.desired_L_frontal(0.0, stance_is_left, P), rel_tol=1e-9)

        # Planning uses the NEXT step's target, which has the opposite sign.
        step = alip.plan_lateral_step(y, Lx, P.T, 0.0, stance_is_left, P)
        # The footstep must be exactly one step width, toward the swing side.
        assert math.isclose(abs(step), P.W, rel_tol=1e-9), f"step={step}, W={P.W}"
        assert step < 0 if stance_is_left else step > 0

        y, Lx = alip.reset_map(y_T, L_T, step)
        stance_is_left = not stance_is_left

    assert math.isclose(y, y0, abs_tol=1e-12)
    assert math.isclose(Lx, Lx0, abs_tol=1e-12)


def test_frontal_orbit_starts_half_a_width_from_stance_foot():
    """Left stance => CoM begins W/2 to the right of the left foot."""
    y0, _ = alip.periodic_orbit_frontal(stance_is_left=True, p=P)
    assert math.isclose(y0, -0.5 * P.W, rel_tol=1e-12)


def test_sagittal_orbit_is_fixed_point_and_step_length_is_consistent():
    for v in (0.0, 0.25, 0.5, 1.0):
        x, Ly = alip.periodic_orbit_sagittal(v, P)
        x_T, L_T = alip.flow(x, Ly, P.T, SAGITTAL, P)
        L_des = alip.desired_L_sagittal(v, P)
        assert math.isclose(L_T, L_des, rel_tol=1e-9, abs_tol=1e-12)

        step = alip.foot_placement(x_T, L_T, L_des, SAGITTAL, P)
        x_plus, L_plus = alip.reset_map(x_T, L_T, step)
        assert math.isclose(x_plus, x, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(L_plus, Ly, rel_tol=1e-9, abs_tol=1e-12)

        # Step length closed form, derived independently of foot_placement.
        want = 2.0 * v * math.tanh(0.5 * P.omega * P.T) / P.omega
        assert math.isclose(step, want, rel_tol=1e-9, abs_tol=1e-12)


def test_average_velocity_conversion_round_trips():
    """Commanding the converted terminal velocity must yield the requested average."""
    for v_avg in (0.1, 0.5, 1.2):
        v_term = alip.average_to_terminal_velocity(v_avg, P)
        x, Ly = alip.periodic_orbit_sagittal(v_term, P)
        x_T, L_T = alip.flow(x, Ly, P.T, SAGITTAL, P)
        step = alip.foot_placement(x_T, L_T, alip.desired_L_sagittal(v_term, P), SAGITTAL, P)
        assert math.isclose(step / P.T, v_avg, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 4. Closed-loop convergence: the planner must reject a perturbation.
# ---------------------------------------------------------------------------


def test_planner_recovers_from_perturbation():
    """Off-orbit initial condition must converge back to the target momentum."""
    x, Ly = alip.periodic_orbit_sagittal(0.0, P)
    x += 0.08  # shove the CoM 8 cm forward
    L_des = alip.desired_L_sagittal(0.0, P)

    for _ in range(6):
        x_T, L_T = alip.flow(x, Ly, P.T, SAGITTAL, P)
        step = alip.foot_placement(x_T, L_T, L_des, SAGITTAL, P)
        x, Ly = alip.reset_map(x_T, L_T, step)

    # Deadbeat in one step by construction, so after six it should be exact.
    _, L_end = alip.flow(x, Ly, P.T, SAGITTAL, P)
    assert math.isclose(L_end, L_des, abs_tol=1e-9)


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
