"""ALIP state estimation and contact detection for Digit.

The momentum mathematics is robot-agnostic, so the generic helpers are imported
from `cassie_state` rather than duplicated -- `angular_momentum_about`,
`angular_momentum_bruteforce`, `com_position`, `com_velocity` and
`update_subtree_velocities` work on any MuJoCo model. Only two things are
robot-specific and live here:

  * the stance contact point, which for Digit comes from the sole box rather than
    from named sites (the MJCF has no foot sites at all);
  * contact detection, which matches on the `<side>_toe_roll` body.

Reminder carried over from the Cassie work: `subtree_angmom` and `subtree_linvel`
are only populated by `mj_subtreeVel`, never by `mj_forward` alone. Read them
straight after `mj_forward` and you silently get zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

import digit_params as dp
from cassie_state import (  # noqa: F401  (re-exported deliberately)
    AlipState,
    angular_momentum_about,
    angular_momentum_bruteforce,
    com_position,
    com_velocity,
    total_mass,
    update_subtree_velocities,
)


class DigitAlipEstimator:
    """Turns Digit's simulator state into the planner's (x, y, Lx, Ly)."""

    def __init__(self, model, feet: dp.FootGeometry | None = None):
        self.model = model
        self.feet = feet or dp.FootGeometry(model)
        self.mass = total_mass(model)

    def contact_point(self, data, stance_is_left: bool) -> np.ndarray:
        side = "left" if stance_is_left else "right"
        return self.feet.contact_point(data, side)

    def estimate(self, data, stance_is_left: bool) -> AlipState:
        update_subtree_velocities(self.model, data)
        com = com_position(data)
        v_com = np.array(data.subtree_linvel[0])
        contact = self.contact_point(data, stance_is_left)
        L = angular_momentum_about(self.model, data, contact)
        rel = com - contact
        return AlipState(
            x=float(rel[0]), y=float(rel[1]),
            Lx=float(L[0]), Ly=float(L[1]),
            com=com, v_com=v_com, contact=contact,
            H=float(rel[2]), L_about_contact=L,
        )


@dataclass
class ContactState:
    left: bool
    right: bool
    force_left: float
    force_right: float

    @property
    def double_support(self) -> bool:
        return self.left and self.right

    @property
    def flight(self) -> bool:
        return not self.left and not self.right


class DigitContactDetector:
    """Per-foot normal force from MuJoCo's contact list.

    Digit's foot is a box with a real support polygon, so unlike Cassie's blade the
    centre of pressure is a meaningful quantity and is reported here -- it is the
    extra stabilisation channel Digit has and Cassie does not.
    """

    def __init__(self, model, force_threshold: float = 30.0):
        self.model = model
        self.threshold = force_threshold
        self.foot_geoms = {}
        for side in dp.SIDES:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, dp.FOOT_BODY[side])
            ids = {g for g in range(model.ngeom) if model.geom_bodyid[g] == body}
            if not ids:
                raise RuntimeError(f"no geoms on {dp.FOOT_BODY[side]}")
            self.foot_geoms[side] = ids

    def _contacts(self, data, side):
        wrench = np.zeros(6)
        for c in range(data.ncon):
            con = data.contact[c]
            if con.geom1 in self.foot_geoms[side] or con.geom2 in self.foot_geoms[side]:
                mujoco.mj_contactForce(self.model, data, c, wrench)
                frame = np.array(con.frame).reshape(3, 3)
                f_world = frame.T @ wrench[:3]
                yield np.array(con.pos), f_world

    def normal_force(self, data, side: str) -> float:
        return float(sum(abs(f[2]) for _, f in self._contacts(data, side)))

    def center_of_pressure(self, data, side: str):
        """CoP under one foot, or None if unloaded.

        Only meaningful because the foot has area. This is what lets Digit shift
        its centre of pressure to regulate the CoM without moving a foot.
        """
        num = np.zeros(2)
        den = 0.0
        for pos, f in self._contacts(data, side):
            w = abs(f[2])
            num += w * pos[:2]
            den += w
        return None if den < 1e-9 else num / den

    def state(self, data) -> ContactState:
        fl = self.normal_force(data, "left")
        fr = self.normal_force(data, "right")
        return ContactState(fl > self.threshold, fr > self.threshold, fl, fr)
