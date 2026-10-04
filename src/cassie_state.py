"""Step 3: ALIP state estimation from the full-order simulator state.

The planner in src/alip.py needs three things every tick:

    x, y  -- CoM position relative to the stance contact point
    Ly    -- angular momentum about that point, sagittal component
    Lx    -- angular momentum about that point, frontal component

The reduced model *defines* L as m*H*v, but the real robot's angular momentum
also includes every link's own rotational inertia and its motion relative to the
CoM. Feeding the reduced-model approximation back into the planner would be
circular: the whole reason ALIP works better than a plain LIP is that L is the
honest quantity. So this computes the true angular momentum of the full 26-body
mechanism.

The transfer identity used throughout:

    L_about_p = L_about_com + (com - p) x (M_total * v_com)

MuJoCo gives us `subtree_angmom[0]` (angular momentum of the whole tree about the
tree's CoM) for free, so the estimator is cheap. `angular_momentum_bruteforce`
recomputes it by summing over bodies, and the tests require the two to agree --
that is what catches a wrong frame or a missing inertia term.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class AlipState:
    """ALIP state, plus the full-order quantities it was derived from."""

    x: float           # CoM position relative to contact point, forward [m]
    y: float           # CoM position relative to contact point, lateral [m]
    Lx: float          # angular momentum about contact point, frontal [kg m^2/s]
    Ly: float          # angular momentum about contact point, sagittal [kg m^2/s]
    com: np.ndarray    # CoM in world coordinates [m]
    v_com: np.ndarray  # CoM velocity in world coordinates [m/s]
    contact: np.ndarray  # stance contact point in world coordinates [m]
    H: float           # CoM height above the contact point [m]
    L_about_contact: np.ndarray  # full 3-vector, for diagnostics

    @property
    def vx(self) -> float:
        return float(self.v_com[0])

    @property
    def vy(self) -> float:
        return float(self.v_com[1])

    def alip_consistency(self, mass: float) -> tuple[float, float]:
        """How far the true L is from the reduced model's m*H*v assumption.

        Returns (sagittal, frontal) discrepancies in kg m^2/s. These are never
        zero on a real mechanism -- the swing leg alone carries momentum -- and
        watching them is a good way to see when the reduced model is being asked
        to do something it cannot represent.
        """
        return (
            self.Ly - mass * self.H * self.vx,
            self.Lx + mass * self.H * self.vy,
        )


def total_mass(model) -> float:
    return float(np.sum(model.body_mass))


def com_position(data) -> np.ndarray:
    """Whole-robot CoM. subtree_com[0] is the root subtree, i.e. everything."""
    return np.array(data.subtree_com[0])


def update_subtree_velocities(model, data) -> None:
    """Populate `subtree_linvel` and `subtree_angmom`.

    These are NOT filled in by mj_forward -- MuJoCo only computes them in
    mj_subtreeVel, which the standard pipeline invokes solely when a subtree
    sensor demands it. Reading them straight after mj_forward silently returns
    zeros, which is exactly the bug the estimator tests caught.
    """
    mujoco.mj_subtreeVel(model, data)


def com_velocity(model, data) -> np.ndarray:
    """Whole-robot CoM velocity. Calls mj_subtreeVel first; see the note there."""
    update_subtree_velocities(model, data)
    return np.array(data.subtree_linvel[0])


def angular_momentum_about(model, data, point) -> np.ndarray:
    """Angular momentum of the whole robot about an arbitrary world point."""
    update_subtree_velocities(model, data)
    com = com_position(data)
    v_com = np.array(data.subtree_linvel[0])
    L_com = np.array(data.subtree_angmom[0])
    return L_com + np.cross(com - np.asarray(point), total_mass(model) * v_com)


def angular_momentum_bruteforce(model, data, point) -> np.ndarray:
    """Independent computation, summing every body's contribution.

    Deliberately does not use subtree_angmom. Used by the tests as an oracle.
    """
    p = np.asarray(point, dtype=float)
    L = np.zeros(3)
    vel = np.zeros(6)
    for i in range(1, model.nbody):  # body 0 is the world
        m_i = model.body_mass[i]
        if m_i == 0:
            continue
        # mjOBJ_BODY reports at the body's inertial frame origin (its CoM),
        # with flg_local=0 giving world-aligned axes.
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, i, vel, 0)
        omega, v = vel[0:3], vel[3:6]

        r = np.array(data.xipos[i])           # body CoM in world
        R = np.array(data.ximat[i]).reshape(3, 3)  # inertial frame orientation
        I_world = R @ np.diag(model.body_inertia[i]) @ R.T

        L += m_i * np.cross(r - p, v) + I_world @ omega
    return L


class AlipEstimator:
    """Turns simulator state into the planner's (x, y, Lx, Ly)."""

    def __init__(self, model):
        self.model = model
        self.mass = total_mass(model)
        self.site = {
            f"{side}-{part}": mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"{side}-{part}"
            )
            for side in ("left", "right")
            for part in ("toe", "heel")
        }

    def contact_point(self, data, stance_is_left: bool) -> np.ndarray:
        """Stance contact point: midpoint of the toe and heel sites."""
        side = "left" if stance_is_left else "right"
        toe = np.array(data.site_xpos[self.site[f"{side}-toe"]])
        heel = np.array(data.site_xpos[self.site[f"{side}-heel"]])
        return 0.5 * (toe + heel)

    def estimate(self, data, stance_is_left: bool) -> AlipState:
        update_subtree_velocities(self.model, data)
        com = com_position(data)
        v_com = np.array(data.subtree_linvel[0])
        contact = self.contact_point(data, stance_is_left)
        L = angular_momentum_about(self.model, data, contact)
        rel = com - contact
        return AlipState(
            x=float(rel[0]),
            y=float(rel[1]),
            Lx=float(L[0]),
            Ly=float(L[1]),
            com=com,
            v_com=v_com,
            contact=contact,
            H=float(rel[2]),
            L_about_contact=L,
        )


# ---------------------------------------------------------------------------
# Contact detection (step 5)
# ---------------------------------------------------------------------------


class ContactDetector:
    """Which feet are actually loaded, from MuJoCo's contact list.

    Timer-based step transitions are the usual first implementation and the usual
    first thing to break: real touchdown is early on a bump and late on a drop.
    """

    def __init__(self, model, force_threshold: float = 20.0):
        self.model = model
        self.threshold = force_threshold
        # The robot's geoms are all unnamed in cassie.xml, so match on the owning
        # body instead. Anything rigidly part of the foot counts as foot contact.
        self.foot_geoms = {}
        for side in ("left", "right"):
            ids = set()
            for g in range(model.ngeom):
                body = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]
                )
                if body == f"{side}-foot":
                    ids.add(g)
            if not ids:
                raise RuntimeError(f"no geoms found on body {side}-foot")
            self.foot_geoms[side] = ids

    def normal_force(self, data, side: str) -> float:
        """Total vertical contact force on one foot [N]."""
        total = 0.0
        wrench = np.zeros(6)
        for c in range(data.ncon):
            con = data.contact[c]
            if con.geom1 in self.foot_geoms[side] or con.geom2 in self.foot_geoms[side]:
                mujoco.mj_contactForce(self.model, data, c, wrench)
                # Rotate the contact-frame normal into world and take z.
                frame = np.array(con.frame).reshape(3, 3)
                total += abs(float((frame.T @ wrench[:3])[2]))
        return total

    def in_contact(self, data, side: str) -> bool:
        return self.normal_force(data, side) > self.threshold

    def state(self, data) -> dict:
        return {side: self.in_contact(data, side) for side in ("left", "right")}
