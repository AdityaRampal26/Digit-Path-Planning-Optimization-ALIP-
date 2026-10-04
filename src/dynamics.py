"""Constrained rigid-body dynamics terms, extracted from MuJoCo for the OSC.

The controller needs to predict what the robot will do, which means reproducing
MuJoCo's own equation of motion:

    M qacc + qfrc_bias = qfrc_actuator + qfrc_passive + qfrc_constraint

Rearranged into the form the QP uses, with the constraint forces split into the
parts we model explicitly:

    M qacc + h = B u + Jc^T lambda_c + Jeq^T lambda_eq
    h := qfrc_bias - qfrc_passive

`qfrc_passive` matters here: this model has joint damping up to 2.0 and non-zero
armature. Dropping it makes the predicted acceleration wrong in a way that looks
like a mysteriously badly tuned controller.

Two traps this module exists to handle
--------------------------------------
1. **Loop-closure forces.** Digit has six `connect` constraints (18 rows). Omit
   `Jeq^T lambda_eq` from the dynamics and every computed torque is wrong, while
   still looking plausible. They are solved for as unknowns, not ignored.

2. **The drift terms `Jdot qdot`.** A contact point is held still by
   `Jc qacc + Jdot_c qdot = 0`, so the drift term is not optional. It is obtained
   here by finite-differencing `J(q) qdot` along the current velocity, which is
   slower than an analytic route but straightforward to verify -- and it *is*
   verified, against MuJoCo's own accelerations in tests/test_dynamics.py.
"""

from __future__ import annotations

import mujoco
import numpy as np

import digit_params as dp

CORNERS = ("corner_tl", "corner_tr", "corner_hr", "corner_hl")


class DynamicsModel:
    """Assembles M, h, B and the constraint Jacobians with their drift terms."""

    def __init__(self, model, feet: dp.FootGeometry | None = None,
                 actuation: dp.Actuation | None = None, fd_step: float = 1e-6):
        self.model = model
        self.feet = feet or dp.FootGeometry(model)
        self.act = actuation or dp.Actuation(model)
        self.nv = model.nv
        self.nu = self.act.n
        self.fd_step = fd_step
        self._scratch = mujoco.MjData(model)

    # -- inertia and bias ---------------------------------------------------

    def mass_matrix(self, data):
        """Dense mass matrix.

        MuJoCo changed this API: older versions take the sparse array as the source
        and the dense matrix as the second argument, `mj_fullM(m, dst, d.qM)`; newer
        ones take the MjData and put the destination last, `mj_fullM(m, d, dst)`.
        Both are supported so this works across versions.
        """
        M = np.zeros((self.nv, self.nv))
        if hasattr(data, "qM"):
            mujoco.mj_fullM(self.model, M, data.qM)   # mujoco < 3.11
        else:
            mujoco.mj_fullM(self.model, data, M)      # mujoco >= 3.11
        return M

    def bias(self, data):
        """h = qfrc_bias - qfrc_passive: gravity, Coriolis, damping, springs."""
        return np.asarray(data.qfrc_bias) - np.asarray(data.qfrc_passive)

    @property
    def B(self):
        return self.act.B

    # -- contact ------------------------------------------------------------

    def contact_jacobian(self, data, sides):
        """Stacked translational Jacobian of every sole corner in contact.

        Four points per foot rather than one. That is what makes the centre of
        pressure a decision variable: with a single point the QP cannot express
        where under the foot the load sits, and CoP regulation is the whole
        advantage Digit's box foot has over Cassie's blade.
        """
        rows = []
        for side in sides:
            for c in CORNERS:
                rows.append(self.feet.jacobian(self.model, data, side, c))
        return np.vstack(rows) if rows else np.zeros((0, self.nv))

    def contact_points(self, data, sides):
        return np.array([self.feet.world(data, side, c)
                         for side in sides for c in CORNERS])

    # -- loop closure -------------------------------------------------------

    def eq_jacobian(self, data):
        """Jacobian of the six `connect` constraints (18 rows).

        Rows are filtered by constraint type: Digit has active joint limits at
        ordinary poses, and a stray limit row here would silently corrupt the
        constraint block of the QP.
        """
        rows = dp.equality_rows(self.model, data)
        if len(rows) == 0:
            return np.zeros((0, self.nv))
        return np.asarray(data.efc_J).reshape(data.nefc, self.nv)[rows].copy()

    # -- drift terms --------------------------------------------------------

    def _drift(self, data, jac_fn):
        """Jdot*qdot, by finite-differencing J(q)*qdot along the current velocity.

        d/dt (J qdot) = Jdot qdot when qdot is held fixed, so stepping q by
        h*qdot and re-evaluating gives the drift directly.
        """
        s = self._scratch
        qvel = np.asarray(data.qvel)
        if not np.any(qvel):
            return np.zeros(jac_fn(data).shape[0])

        j0 = jac_fn(data) @ qvel

        s.qpos[:] = data.qpos
        s.qvel[:] = data.qvel
        mujoco.mj_integratePos(self.model, s.qpos, s.qvel, self.fd_step)
        mujoco.mj_forward(self.model, s)
        j1 = jac_fn(s) @ qvel
        return (j1 - j0) / self.fd_step

    def contact_drift(self, data, sides):
        return self._drift(data, lambda d: self.contact_jacobian(d, sides))

    def eq_drift(self, data):
        return self._drift(data, self.eq_jacobian)

    # -- verification helper ------------------------------------------------

    def predict_qacc(self, data, tau, lam_c, lam_eq, sides):
        """qacc implied by our model, given torques and constraint forces."""
        M = self.mass_matrix(data)
        h = self.bias(data)
        Jc = self.contact_jacobian(data, sides)
        Jeq = self.eq_jacobian(data)
        rhs = self.B @ tau - h
        if Jc.shape[0]:
            rhs = rhs + Jc.T @ lam_c
        if Jeq.shape[0]:
            rhs = rhs + Jeq.T @ lam_eq
        return np.linalg.solve(M, rhs)

    def residual_against_mujoco(self, data):
        """How well our M and h reproduce MuJoCo's own equation of motion.

        Uses MuJoCo's `qfrc_constraint` directly, so this isolates the inertia and
        bias terms from any modelling of the constraint forces. Should be ~0.
        """
        M = self.mass_matrix(data)
        lhs = M @ np.asarray(data.qacc) + self.bias(data)
        rhs = np.asarray(data.qfrc_actuator) + np.asarray(data.qfrc_constraint) \
            + np.asarray(data.qfrc_applied)
        return lhs - rhs
