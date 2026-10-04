"""Steps 4-5: turn the ALIP plan into a full-body reference trajectory.

Given the ALIP footstep planner, this produces, at every control tick:

  * pelvis position (the CoM path, offset to the pelvis frame)
  * left and right foot positions in world coordinates
  * which foot is in stance

That is exactly the input the IK needs, and later the same reference feeds the
torque controller -- so this module is shared between the kinematic animation and
the physics stage.

Swing foot shaping
------------------
Horizontal:  smoothstep, 3s^2 - 2s^3, so the foot leaves and arrives with zero
             horizontal velocity.
Vertical:    h * sin^2(pi s), which is zero at both ends *with zero derivative*
             at both ends. A plain sin(pi s) bump peaks at the same height but
             slams down at -h*pi on touchdown, which is exactly the impact the
             ALIP reduced model assumes away.

A note on replanning: real ALIP controllers recompute the footstep every tick,
which is where the push recovery comes from. Here the plant *is* the ALIP model,
and `flow` is exact, so the predicted end-of-step state is identical no matter
when it is computed -- replanning each tick provably returns the same target.
The target is therefore computed once per step, and `plan_lateral_step` /
`plan_sagittal_step` are called with the true time remaining so the code path is
the one the physics stage will use.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import alip  # noqa: E402
from alip import FRONTAL, SAGITTAL, AlipParams  # noqa: E402


@dataclass
class GaitSchedule:
    """What gait to walk, and how the speed command evolves."""

    n_steps: int = 14
    hold_steps: int = 3          # steps in place before moving
    ramp_steps: int = 3          # steps spent ramping to cruise
    v_cruise: float = 0.5        # step-averaged forward speed [m/s]
    swing_height: float = 0.12   # peak swing foot clearance [m]
    first_stance_is_left: bool = True

    def speed(self, k: int) -> float:
        """Step-averaged forward speed command for step index `k`."""
        if k < self.hold_steps:
            return 0.0
        if k < self.hold_steps + self.ramp_steps:
            return self.v_cruise * (k - self.hold_steps + 1) / self.ramp_steps
        return self.v_cruise


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def swing_height_profile(s, height):
    """sin^2 bump: zero value and zero slope at both lift-off and touchdown."""
    return height * np.sin(np.pi * np.clip(s, 0.0, 1.0)) ** 2


class GaitReference:
    """Generates the walking reference from the ALIP planner."""

    def __init__(
        self,
        params: AlipParams | None = None,
        schedule: GaitSchedule | None = None,
        *,
        pelvis_to_com=(-0.01754, 0.00012, -0.12853),
        ground_z: float = 0.0,
        half_width: float | None = None,
    ):
        self.p = params or AlipParams()
        self.s = schedule or GaitSchedule()
        self.pelvis_to_com = np.asarray(pelvis_to_com, dtype=float)
        self.ground_z = ground_z
        self.half_width = self.p.W / 2 if half_width is None else half_width

    def _terminal_speed(self, k: int) -> float:
        v = self.s.speed(k)
        return alip.average_to_terminal_velocity(v, self.p) if v else 0.0

    def generate(self, dt: float = 1.0 / 200.0):
        p, s = self.p, self.s
        stance_is_left = s.first_stance_is_left

        # Start on the in-place periodic orbit.
        x, Ly = alip.periodic_orbit_sagittal(0.0, p)
        y, Lx = alip.periodic_orbit_frontal(stance_is_left, p)

        # Feet start straddling the origin at the nominal width.
        feet = {
            "left": np.array([0.0, +self.half_width, self.ground_z]),
            "right": np.array([0.0, -self.half_width, self.ground_z]),
        }
        # Place the stance foot so the CoM starts above the origin.
        stance = "left" if stance_is_left else "right"
        feet[stance][0] = -x

        out = {k: [] for k in ("t", "pelvis", "left", "right", "com", "stance_is_left", "phase")}
        footsteps = []
        t = 0.0
        n_ticks = int(round(p.T / dt))

        for k in range(s.n_steps):
            stance = "left" if stance_is_left else "right"
            swing = "right" if stance_is_left else "left"
            stance_pos = feet[stance].copy()
            swing_start = feet[swing].copy()
            footsteps.append((stance_pos.copy(), stance_is_left))

            # One-step-ahead footstep target, in world coordinates.
            vx_next = self._terminal_speed(k + 1)
            step_x = alip.plan_sagittal_step(x, Ly, p.T, vx_next, p)
            step_y = alip.plan_lateral_step(y, Lx, p.T, 0.0, stance_is_left, p)
            swing_target = stance_pos + np.array([step_x, step_y, 0.0])

            for i in range(n_ticks):
                phase = i / n_ticks

                com = np.array(
                    [
                        stance_pos[0] + x,
                        stance_pos[1] + y,
                        self.ground_z + p.H,
                    ]
                )

                swing_pos = swing_start + (swing_target - swing_start) * smoothstep(phase)
                swing_pos[2] = self.ground_z + swing_height_profile(phase, s.swing_height)

                out["t"].append(t)
                out["com"].append(com)
                out["pelvis"].append(com - self.pelvis_to_com)
                out[stance].append(stance_pos.copy())
                out[swing].append(swing_pos)
                out["stance_is_left"].append(stance_is_left)
                out["phase"].append(phase)

                x, Ly = alip.flow(x, Ly, dt, SAGITTAL, p)
                y, Lx = alip.flow(y, Lx, dt, FRONTAL, p)
                t += dt

            # Foot strike.
            feet[swing] = swing_target
            x, Ly = alip.reset_map(x, Ly, step_x)
            y, Lx = alip.reset_map(y, Lx, step_y)
            stance_is_left = not stance_is_left

        ref = {k: np.asarray(v) for k, v in out.items()}
        ref["footsteps"] = footsteps
        return ref


if __name__ == "__main__":
    import cassie_params

    g = GaitReference(cassie_params.alip_params())
    ref = g.generate()
    n = len(ref["t"])
    print(f"{n} ticks over {ref['t'][-1] + 1 / 200:.2f} s, {len(ref['footsteps'])} steps")
    print(f"pelvis z: {ref['pelvis'][:, 2].min():.4f} .. {ref['pelvis'][:, 2].max():.4f} m")
    print(f"CoM y sway: {np.ptp(ref['com'][:, 1]) * 100:.2f} cm")
    print(f"swing clearance: {max(ref['left'][:, 2].max(), ref['right'][:, 2].max()):.4f} m")
    print(f"final pelvis x: {ref['pelvis'][-1, 0]:.4f} m")

    # Stance foot must never move while it is the stance foot.
    for side in ("left", "right"):
        is_stance = ref["stance_is_left"] if side == "left" else ~ref["stance_is_left"]
        pos = ref[side][is_stance]
        jumps = np.abs(np.diff(pos[:, :2], axis=0)).max(axis=1)
        # Jumps at stance-phase boundaries are expected; check within-phase only.
        breaks = np.where(np.diff(np.flatnonzero(is_stance)) > 1)[0]
        mask = np.ones(len(jumps), bool)
        mask[breaks] = False
        print(f"{side} stance slip: {jumps[mask].max() * 1e6:.3f} um")
