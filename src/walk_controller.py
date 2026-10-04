"""Closed-loop ALIP walking controller for Digit.

Stage B. This is where the reduced-order planner finally drives the real robot:

    estimator  ->  ALIP footstep planner  ->  swing trajectory  ->  OSC  ->  torques

Every control tick, the true angular momentum about the stance contact point is
measured from the full 38-body model, the planner solves for the footstep that puts
that momentum on target one step later, and the swing foot is steered toward it.
No part of the reference is precomputed.

Three things that differ from the standing controller
----------------------------------------------------
1. **CoM height only.** The horizontal CoM is deliberately *not* regulated. It is
   the pendulum state that foot placement stabilises; commanding it would defeat
   the entire method.
2. **One stance foot.** The swing foot gets no no-slip constraint -- only a task.
3. **Stance handoff.** At touchdown the contact set changes, which changes the QP's
   shape, and the ALIP state must be re-referenced to the new contact point.

Continuous replanning is what makes this a controller rather than playback: the
footstep target is recomputed from the measured state every tick, so a disturbance
is absorbed by moving the foot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

import alip
import digit_params as dp
import digit_state as ds
from alip import FRONTAL, SAGITTAL, AlipParams
from gait import smoothstep, swing_height_profile


@dataclass
class WalkCommand:
    """What the operator asks for."""

    v_forward: float = 0.0        # step-averaged forward speed [m/s]
    v_lateral: float = 0.0
    swing_height: float = 0.08    # peak clearance [m]

    def terminal_vx(self, p: AlipParams) -> float:
        return alip.average_to_terminal_velocity(self.v_forward, p) if self.v_forward else 0.0


@dataclass
class WalkState:
    stance_is_left: bool = True
    phase_time: float = 0.0       # time elapsed in the current step [s]
    step_index: int = 0
    swing_start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    swing_target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    stance_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    touchdowns: list = field(default_factory=list)

    @property
    def stance(self):
        return "left" if self.stance_is_left else "right"

    @property
    def swing(self):
        return "right" if self.stance_is_left else "left"


class WalkController:
    def __init__(self, model, feet, osc, params: AlipParams, command: WalkCommand | None = None,
                 ground_z: float = 0.0, min_swing_time: float = 0.6,
                 freeze_phase: float = 0.75):
        self.model = model
        self.feet = feet
        self.osc = osc
        self.p = params
        self.cmd = command or WalkCommand()
        self.ground_z = ground_z
        self.est = ds.DigitAlipEstimator(model, feet)
        self.det = ds.DigitContactDetector(model)
        self.state = WalkState()
        # Fraction of the step that must elapse before touchdown is accepted, so a
        # grazing contact early in swing does not end the step.
        self.min_swing_frac = min_swing_time
        # Stop replanning this late into swing and hold the latched target.
        #
        # Continuous replanning is what gives ALIP its disturbance rejection, but if
        # it continues to touchdown the foot is chasing a target that is still
        # moving, and lands past it. Measured on Digit: a systematic 19 mm forward
        # overshoot, which pushes the contact point ahead of plan, *decelerates* the
        # CoM, and cost ~23% of commanded speed. Freezing the target late in swing
        # lets the foot converge to a fixed point while keeping replanning for the
        # part of the step where it actually matters.
        self.freeze_phase = freeze_phase
        self._frozen_target = None
        self.posture_target = None
        self.com_height = None
        self.last_info = {}

        # Outer integral loop on step-averaged speed.
        #
        # The deadbeat law is exact for the reduced model, but the real robot has a
        # steady-state bias: swing-foot placement overshoots by ~14 mm (worth
        # m*H*omega*sinh(omega T) = 220 kg m^2/s per metre of error), the stance
        # ankle is suppressed but not zero, and the swing leg carries momentum the
        # point-mass model ignores. Measured effect: 0.385 m/s achieved against
        # 0.500 commanded. Rather than pretend the model is exact, close the gap
        # with integral action on the measured stride.
        self.ki_speed = 0.6
        self.v_bias = 0.0
        self.v_bias_limit = 0.5
        self._last_stance_x = None
        self.speed_history = []

        # Outer proportional loop on lateral position.
        #
        # The frontal planner holds step WIDTH accurately (measured: 0.5 mm) but
        # nothing pins the gait's lateral CENTRE, so small asymmetries integrate into
        # a slow sideways drift (~0.14 m over 8 s). A position-hold on CoM y, fed in
        # as a lateral velocity command, removes it. Position rather than velocity
        # feedback because the requirement is zero net displacement.
        self.kp_lateral = 0.8
        self.v_lateral_limit = 0.25
        self.y_hold = None

        # Contact-driven step transitions (Stage C).
        # Force on the swing foot that counts as landed. This must fire on FIRST
        # contact, not on a loaded foot. Measured: at the nominal touchdown instant
        # the foot has just reached ground level carrying ~0 N, and MuJoCo's soft
        # contacts need real penetration to build tens of newtons -- so a 60 N
        # threshold silently dragged every step to 1.6x T and the gait fell over.
        # 5 N and 15 N behave identically; 15 N is used for noise margin.
        self.touchdown_force = 15.0
        self.max_overtime = 0.60       # keep reaching for up to (1+this)*T
        self.reach_rate = 0.25         # [m/s] descent past nominal touchdown
        # Both feet stay constrained for a moment after landing. The reduced model
        # assumes an instantaneous foot swap; the real robot spends time in double
        # support, and that is where the impact is absorbed. Releasing the old
        # stance foot the instant the new one lands throws away that authority.
        self.double_support_time = 0.05
        self.ds_remaining = 0.0
        self.prev_stance = None
        self.overtime_history = []

    # -- initialisation -----------------------------------------------------

    def reset(self, data, stance_is_left=True, posture_target=None):
        mujoco.mj_forward(self.model, data)
        s = self.state
        s.stance_is_left = stance_is_left
        s.phase_time = 0.0
        s.step_index = 0
        s.touchdowns = []
        s.stance_point = self.feet.contact_point(data, s.stance).copy()
        s.swing_start = self.feet.contact_point(data, s.swing).copy()
        s.swing_target = s.swing_start.copy()
        self.posture_target = (data.qpos[self.osc.act.qposadr].copy()
                               if posture_target is None else posture_target)
        mujoco.mj_subtreeVel(self.model, data)
        self.com_height = float(data.subtree_com[0][2]) - s.stance_point[2]

    # -- planning -----------------------------------------------------------

    def plan_footstep(self, data):
        """One-step-ahead ALIP footstep, in world coordinates.

        Uses the *measured* angular momentum about the current contact point, so
        the target moves as the state evolves. That continuous replanning is the
        disturbance rejection.
        """
        s = self.state
        st = self.est.estimate(data, s.stance_is_left)
        t_rem = max(self.p.T - s.phase_time, 1e-3)

        v_cmd = self.cmd.v_forward + self.v_bias
        v_term = alip.average_to_terminal_velocity(v_cmd, self.p) if v_cmd else 0.0
        step_x = alip.plan_sagittal_step(st.x, st.Ly, t_rem, v_term, self.p)

        # Lateral position hold layered on the commanded lateral velocity.
        if self.y_hold is None:
            self.y_hold = float(st.com[1])
        vy = self.cmd.v_lateral + float(np.clip(
            -self.kp_lateral * (st.com[1] - self.y_hold),
            -self.v_lateral_limit, self.v_lateral_limit))
        step_y = alip.plan_lateral_step(st.y, st.Lx, t_rem, vy, s.stance_is_left, self.p)

        # Keep the step inside something physically sane; the raw law can ask for a
        # very long step while the state is far off the orbit.
        step_x = float(np.clip(step_x, -0.45, 0.45))
        lo, hi = (-0.55, -0.10) if s.stance_is_left else (0.10, 0.55)
        step_y = float(np.clip(step_y, lo, hi))

        target = s.stance_point + np.array([step_x, step_y, 0.0])
        target[2] = self.ground_z

        # Latch the target for the final part of swing; see `freeze_phase`.
        phase = s.phase_time / self.p.T
        if phase >= self.freeze_phase and self._frozen_target is not None:
            target = self._frozen_target
        else:
            self._frozen_target = target.copy()
        return target, st

    def swing_reference(self, target):
        """Swing foot position and velocity at the current phase.

        Past the nominal step duration the foot keeps descending at `reach_rate`
        instead of hovering at the expected ground height. Without this, waiting for
        contact is pointless: the reference would sit exactly at the height the
        ground was *assumed* to be, and if the real ground is lower the foot never
        arrives. Reaching is what turns the timeout into genuine contact detection.
        """
        s = self.state
        T = self.p.T
        if s.phase_time > T:
            over = s.phase_time - T
            pos = target.copy()
            pos[2] = self.ground_z - self.reach_rate * over
            return pos, np.array([0.0, 0.0, -self.reach_rate])

        phase = np.clip(s.phase_time / T, 0.0, 1.0)

        horiz = smoothstep(phase)
        pos = s.swing_start + (target - s.swing_start) * horiz
        pos[2] = self.ground_z + swing_height_profile(phase, self.cmd.swing_height)

        # Analytic derivatives of the same profiles, so the task has a velocity
        # reference instead of implicitly asking for zero.
        dhoriz = 6.0 * phase * (1.0 - phase) / T
        vel = (target - s.swing_start) * dhoriz
        vel[2] = (self.cmd.swing_height * np.pi / T
                  * np.sin(2.0 * np.pi * phase) / 1.0)
        return pos, vel

    # -- step transition ----------------------------------------------------

    def _touchdown(self, data):
        """Has the swing foot actually landed?

        The previous rule was "switch at T, or earlier on contact", which can only
        ever fire EARLY. Late touchdown was structurally undetectable: the timer
        always won. Going downhill, or onto ground lower than expected, that means
        declaring stance on a foot still in the air -- the QP then pins an airborne
        foot with a no-slip constraint and plans about a contact point that does not
        exist.

        Now contact decides, in both directions. The timer only survives as a
        timeout, so a missed contact cannot hang the gait forever.
        """
        s = self.state
        if s.phase_time < self.min_swing_frac * self.p.T:
            return False
        if self.det.normal_force(data, s.swing) > self.touchdown_force:
            return True
        return s.phase_time >= self.p.T * (1.0 + self.max_overtime)

    def _update_speed_bias(self, data, stride, elapsed):
        """Integral action on the measured stride, to remove the steady-state bias."""
        if elapsed <= 0.0 or self.cmd.v_forward == 0.0:
            return
        measured = stride / elapsed
        self.speed_history.append(measured)
        err = self.cmd.v_forward - measured
        self.v_bias = float(np.clip(self.v_bias + self.ki_speed * err,
                                    -self.v_bias_limit, self.v_bias_limit))

    def _switch_stance(self, data, t):
        s = self.state
        landed = self.feet.contact_point(data, s.swing).copy()
        if self._last_stance_x is not None and s.touchdowns:
            self._update_speed_bias(data, landed[0] - self._last_stance_x,
                                    t - s.touchdowns[-1]["t"])
        self._last_stance_x = float(landed[0])
        s.touchdowns.append(dict(t=t, step=s.step_index, stance=s.stance,
                                 landed=landed, overtime=s.phase_time - self.p.T))
        self.overtime_history.append(s.phase_time - self.p.T)
        # Hold both feet briefly; see `double_support_time`.
        self.prev_stance = s.stance
        self.ds_remaining = self.double_support_time
        s.stance_is_left = not s.stance_is_left
        s.step_index += 1
        s.phase_time = 0.0
        s.stance_point = self.feet.contact_point(data, s.stance).copy()
        s.swing_start = self.feet.contact_point(data, s.swing).copy()
        self._frozen_target = None

    # -- main tick ----------------------------------------------------------

    def update(self, data, t, dt):
        s = self.state
        target, alip_state = self.plan_footstep(data)
        s.swing_target = target
        pos, vel = self.swing_reference(target)

        com_target = np.array([0.0, 0.0, s.stance_point[2] + self.com_height])

        # Double support for a moment after landing: both feet carry load, and the
        # QP gets a wider support polygon to absorb the impact with.
        stance_sides = (s.stance,)
        in_double = False
        if self.ds_remaining > 0.0 and self.prev_stance is not None:
            if self.det.normal_force(data, self.prev_stance) > 20.0:
                stance_sides = (s.stance, self.prev_stance)
                in_double = True
            self.ds_remaining -= dt

        tau, info = self.osc.solve(
            data,
            stance_sides=stance_sides,
            com_target=com_target,
            com_axes=(2,),                 # HEIGHT ONLY -- see module docstring
            torso_quat_target=(1, 0, 0, 0),
            posture_target=self.posture_target,
            # The trailing foot is still loaded during double support, so it must
            # not also be chased as a swing target.
            swing_side=None if in_double else s.swing,
            swing_target=None if in_double else pos,
            swing_vel_target=None if in_double else vel,
        )

        self.last_info = dict(
            info=info, alip=alip_state, swing_ref=pos, swing_target=target,
            phase=s.phase_time / self.p.T, stance=s.stance, step=s.step_index,
            double_support=in_double, stance_sides=stance_sides,
        )

        s.phase_time += dt
        if self._touchdown(data):
            self._switch_stance(data, t)

        return tau, self.last_info
