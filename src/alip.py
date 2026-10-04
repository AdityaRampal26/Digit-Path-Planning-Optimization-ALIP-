"""Angular-momentum Linear Inverted Pendulum (ALIP) model and footstep planner.

Reference: Gong & Grizzle, "One-Step Ahead Prediction of Angular Momentum about
the Contact Point for Control of Bipedal Locomotion", ICRA 2021.

Conventions
-----------
World frame: x forward, y left, z up.

`pos` is the horizontal CoM position **relative to the current stance contact
point**. `L` is the angular momentum of the robot **about that contact point**.
The CoM is assumed to stay at constant height H, and centroidal inertia is
neglected, so with r = (x, y, H) and v = (vx, vy, 0):

    L = r x (m v)  =>  Lx = -m H vy
                       Ly = +m H vx

Gravity torque about the contact point, tau = r x (0, 0, -m g), gives

    Lx_dot = -m g y
    Ly_dot = +m g x

so each plane is a 2-state linear system:

    sagittal (x, Ly):  d/dt [x ; Ly] = [[0, +1/(mH)], [+mg, 0]] [x ; Ly]
    frontal  (y, Lx):  d/dt [y ; Lx] = [[0, -1/(mH)], [-mg, 0]] [y ; Lx]

They differ only in the sign of the off-diagonal terms, so both are handled by
one implementation with `sign = +1` (sagittal) or `sign = -1` (frontal).

Why angular momentum instead of CoM velocity: L about the contact point is
continuous through the foot-strike impact (it depends only on CoM velocity and
height, both of which the idealised impact leaves unchanged), and it is far less
noisy to estimate on hardware than CoM acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SAGITTAL = +1
FRONTAL = -1


@dataclass(frozen=True)
class AlipParams:
    """Physical and gait parameters.

    Defaults for m, H and W are measured from cassie.xml at the nominal standing
    pose -- see src/cassie_params.py, which is the authority for these numbers.
    T is a gait choice, not a measurement.

    Attributes
    ----------
    m : total robot mass [kg]
    H : nominal constant CoM height above the contact point [m]
    T : single-support (step) duration [s]
    W : nominal lateral step width, i.e. foot-to-foot spacing [m]
    g : gravitational acceleration [m/s^2]
    """

    m: float = 33.3120
    H: float = 0.8577
    T: float = 0.35
    W: float = 0.2699
    g: float = 9.81
    slope: float = 0.0     # ground inclination [rad]; +ve = uphill in +x

    @property
    def omega(self) -> float:
        """Natural frequency, sqrt(g_perp / H) with g_perp = g cos(slope).

        Only the component of gravity normal to the surface drives the pendulum.
        The effect is tiny -- at 10 degrees cos(gamma) = 0.985, so omega moves 0.8%
        -- which is exactly why the *other* slope term (see `slope_bias`) is the one
        that matters.
        """
        return math.sqrt(self.g * math.cos(self.slope) / self.H)

    @property
    def mHw(self) -> float:
        """The recurring product m*H*omega."""
        return self.m * self.H * self.omega

    @property
    def slope_bias(self) -> float:
        """Constant gravity torque about the contact point on a slope, -m g H sin(gamma).

        This is the whole story of walking on an incline. On flat ground the only
        gravity torque is m g x, which vanishes when the CoM is over the foot. On a
        slope there is an additional CONSTANT term that does not vanish, pulling the
        robot downhill at all times.

        Scale check for Digit at 10 degrees: 50 * 9.81 * 0.8 * 0.174 = 68 N m, against
        a flat-ground restoring torque of only ~25 N m at a 5 cm CoM offset. The
        perturbation is larger than the thing it perturbs.
        """
        return -self.m * self.g * self.H * math.sin(self.slope)

    @property
    def g_normal(self) -> float:
        """Gravity component along the surface normal, g cos(gamma)."""
        return self.g * math.cos(self.slope)

    @property
    def grade_percent(self) -> float:
        """Slope as a road-sign style percentage grade, 100 tan(gamma)."""
        return 100.0 * math.tan(self.slope)

    def min_friction(self) -> float:
        """Friction coefficient needed merely to stand on this grade, tan(gamma)."""
        return abs(math.tan(self.slope))

    @property
    def x_equilibrium(self) -> float:
        """CoM offset, along the slope, at which the gravity torque cancels: H tan(gamma).

        The robot must stand leaning *uphill* of its own foot. At 10 degrees with
        H = 0.8 m that is 14 cm -- comparable to the length of the foot itself.
        """
        return self.H * math.tan(self.slope)


# ---------------------------------------------------------------------------
# Slope handling
# ---------------------------------------------------------------------------


def slope_offset(sign: int, p: AlipParams) -> float:
    """Equilibrium shift for the given plane on an inclined surface.

    Only the sagittal plane sees the constant downhill torque, so only it is
    shifted. Returns zero on flat ground, which is why every flat-ground result
    and test carries over untouched.
    """
    return p.x_equilibrium if sign == SAGITTAL else 0.0


# ---------------------------------------------------------------------------
# Continuous flow
# ---------------------------------------------------------------------------


def flow(pos: float, L: float, t: float, sign: int, p: AlipParams) -> tuple[float, float]:
    """Exact solution of the ALIP dynamics after time `t`.

    Parameters
    ----------
    pos : CoM position relative to the contact point (x if sagittal, y if frontal)
    L   : angular momentum about the contact point (Ly if sagittal, Lx if frontal)
    t   : elapsed time [s]
    sign: SAGITTAL (+1) or FRONTAL (-1)

    Returns
    -------
    (pos, L) at time `t`.

    On a slope the sagittal system is affine, not linear -- there is a constant
    downhill gravity torque. It is handled by solving about the shifted
    equilibrium `p.x_equilibrium = H tan(gamma)`, which turns the affine system
    back into the flat-ground linear one exactly. The frontal plane is untouched:
    a purely sagittal grade produces no constant frontal torque.
    """
    off = slope_offset(sign, p)
    w = p.omega
    c, s = math.cosh(w * t), math.sinh(w * t)
    zeta = pos - off
    pos_t = c * zeta + sign * (s / p.mHw) * L
    L_t = sign * p.mHw * s * zeta + c * L
    return pos_t + off, L_t


# ---------------------------------------------------------------------------
# Desired end-of-step angular momentum
# ---------------------------------------------------------------------------


def desired_L_sagittal(vx_des: float, p: AlipParams) -> float:
    """Ly at the end of a step that yields forward velocity `vx_des`.

    Note `vx_des` is the CoM velocity *at the instant of foot strike*, not the
    step-averaged velocity. Use `average_to_terminal_velocity` if you want to
    command an average speed instead.
    """
    return p.m * p.H * vx_des


def desired_L_frontal(vy_des: float, stance_is_left: bool, p: AlipParams) -> float:
    """Lx at the end of the step whose stance foot is given by `stance_is_left`.

    NOTE the sign alternates with stance side, so this is easy to get wrong when
    feeding `foot_placement` -- which targets the *next* step. Prefer
    `plan_lateral_step`, which handles the alternation for you.

    The alternating term is what makes the robot sway side to side instead of
    toppling laterally. Derivation: imposing Lx(T) = -Lx(0) and the mirror
    condition y+(0) = -y(0) on the frontal dynamics gives, for left stance,
    y(0) = -W/2 and Lx(T) = +(1/2) m H omega W tanh(omega T / 2).
    """
    sway = 0.5 * p.mHw * p.W * math.tanh(0.5 * p.omega * p.T)
    return -p.m * p.H * vy_des + (sway if stance_is_left else -sway)


def average_to_terminal_velocity(v_avg: float, p: AlipParams) -> float:
    """Convert a desired step-averaged speed to the terminal velocity to command.

    On the periodic sagittal orbit the step length is
    ``2 * v_terminal * tanh(omega T / 2) / omega``, so the average speed is
    ``v_terminal * 2 tanh(omega T / 2) / (omega T)`` -- about 10% below the
    terminal velocity for typical Cassie gait parameters.
    """
    u = p.omega * p.T
    return v_avg * u / (2.0 * math.tanh(0.5 * u))


# ---------------------------------------------------------------------------
# Footstep planner
# ---------------------------------------------------------------------------


def foot_placement(pos_T: float, L_T: float, L_des: float, sign: int, p: AlipParams) -> float:
    """Where to put the swing foot, relative to the current stance foot.

    Given the predicted end-of-step state ``(pos_T, L_T)``, returns the footstep
    displacement ``step`` such that the angular momentum at the end of the
    *next* step equals ``L_des``.

    ``L_des`` is therefore the **next** step's target. In the frontal plane that
    target alternates with stance side, so pass the target for the opposite foot
    -- or use `plan_lateral_step`, which does it for you.

    Derivation: at foot strike ``pos+(0) = pos_T - step`` and ``L+(0) = L_T``
    (L is impact-invariant). Substituting into `flow` over one step and solving
    the resulting linear equation for ``step`` gives the expression below.

    The footstep is a *displacement*, so it is identical in shifted and unshifted
    coordinates; only the measured position needs the slope offset removed. The
    deadbeat property therefore survives on an incline unchanged.
    """
    w, T = p.omega, p.T
    c, s = math.cosh(w * T), math.sinh(w * T)
    zeta_T = pos_T - slope_offset(sign, p)
    return zeta_T + sign * (c * L_T - L_des) / (p.mHw * s)


def plan_step(
    pos: float,
    L: float,
    t_remaining: float,
    L_des: float,
    sign: int,
    p: AlipParams,
) -> float:
    """One-step-ahead footstep plan, callable at any point during a step.

    Predicts the state at foot strike `t_remaining` seconds from now, then
    applies `foot_placement`. Call this every control tick so the swing foot
    target keeps updating as the state estimate improves -- that continuous
    replanning is what gives ALIP its push-recovery behaviour.
    """
    pos_T, L_T = flow(pos, L, t_remaining, sign, p)
    return foot_placement(pos_T, L_T, L_des, sign, p)


def plan_lateral_step(
    y: float,
    Lx: float,
    t_remaining: float,
    vy_des: float,
    stance_is_left: bool,
    p: AlipParams,
) -> float:
    """Lateral footstep plan. `stance_is_left` is the **current** stance foot.

    Wraps `plan_step` and takes care of the fact that the frontal momentum target
    flips sign each step, so the target belongs to the opposite foot. On the
    periodic orbit this returns exactly -W for left stance and +W for right.
    """
    L_des = desired_L_frontal(vy_des, not stance_is_left, p)
    return plan_step(y, Lx, t_remaining, L_des, FRONTAL, p)


def plan_sagittal_step(
    x: float,
    Ly: float,
    t_remaining: float,
    vx_des_next: float,
    p: AlipParams,
) -> float:
    """Sagittal footstep plan. `vx_des_next` is the target for the next step.

    Provided for symmetry with `plan_lateral_step`. No alternation here, so the
    only subtlety is that the command should be the one you want realised at the
    *end of the next step*, which matters while a velocity ramp is in progress.
    """
    return plan_step(x, Ly, t_remaining, desired_L_sagittal(vx_des_next, p), SAGITTAL, p)


def reset_map(pos_T: float, L_T: float, step: float) -> tuple[float, float]:
    """State immediately after foot strike, expressed about the new contact point.

    The CoM position is measured from the new foot, so it shifts by `step`.
    Angular momentum is unchanged: the idealised impact alters neither the CoM
    velocity nor its height.
    """
    return pos_T - step, L_T


# ---------------------------------------------------------------------------
# Analytic periodic orbits (useful as initial conditions and as test oracles)
# ---------------------------------------------------------------------------


def periodic_orbit_sagittal(vx_des: float, p: AlipParams) -> tuple[float, float]:
    """Start-of-step (x, Ly) on the periodic sagittal orbit.

    On a slope the whole orbit translates uphill by `p.x_equilibrium`: the CoM
    rides ahead of the stance foot by H tan(gamma) throughout. The *shape* of the
    orbit, and hence the step length and momentum, are unchanged.
    """
    w = p.omega
    x0 = -vx_des * math.tanh(0.5 * w * p.T) / w
    return x0 + p.x_equilibrium, p.m * p.H * vx_des


def periodic_orbit_frontal(stance_is_left: bool, p: AlipParams) -> tuple[float, float]:
    """Start-of-step (y, Lx) on the periodic frontal orbit, zero lateral drift.

    For left stance the CoM starts W/2 to the *right* of the stance foot and
    sways rightward toward where the right foot will land.
    """
    y0 = -0.5 * p.W if stance_is_left else +0.5 * p.W
    L0 = p.mHw * y0 * math.tanh(0.5 * p.omega * p.T)
    return y0, L0
