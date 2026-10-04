"""Leg inverse kinematics for Digit, with closed-loop constraints enforced.

Structurally the same Levenberg-Marquardt solve as src/cassie_ik.py, generalised
in three ways that Digit forces on us:

1. **efc rows are filtered by constraint type.** For Cassie we could disable
   contacts and assert every remaining row was a loop closure. Digit has joint
   limits active at ordinary poses, so `efc` mixes limit rows in with equality
   rows. Feeding those into the closure residual would corrupt the solve.

2. **Contact points come from `mj_jac` at explicit points, not `mj_jacSite`.**
   The MJCF has no foot sites, so sole points are derived from the collision box
   (see digit_params.FootGeometry) and addressed positionally.

3. **Full 3-DOF foot orientation.** Cassie's blade foot only needed toe/heel
   levelling, since it cannot resist roll anyway. Digit's foot is a box with a
   real support polygon and actuated ankle roll, so the sole normal is driven to
   vertical in both directions and foot yaw is regulated.

There is no spring-pinning term: this model has zero joint stiffness everywhere,
so the failure mode where the IK cheats by deflecting a spring cannot occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

import digit_params as dp
from digit_params import NUM_BASE_DOF, SIDES, FootGeometry, equality_rows


@dataclass
class IKResult:
    qpos: np.ndarray
    iterations: int
    task_error: float      # max sole-point position error [m]
    closure_error: float   # max loop-closure violation [m]
    tilt_error: float      # max sole tilt [rad]
    converged: bool
    history: list = field(default_factory=list)


class DigitIK:
    def __init__(
        self,
        model_path=None,
        *,
        w_task: float = 1.0,
        w_tilt: float = 0.3,
        w_yaw: float = 0.1,
        w_closure: float = 2.0,
        damping: float = 1e-6,
        regularisation: float = 1e-7,
        freeze_arms: bool = True,
    ):
        self.model, self.data = dp.load(model_path)
        self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        # Contacts would fight the IK while feet are being moved to targets.
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.feet = FootGeometry(self.model)

        self.w_task = w_task
        self.w_tilt = w_tilt
        self.w_yaw = w_yaw
        self.w_closure = w_closure
        self.damping = damping
        self.regularisation = regularisation
        self.nv = self.model.nv

        # Solver may move everything except the floating base, and optionally
        # except the arms (which do nothing for foot placement and otherwise just
        # drift around in the nullspace).
        free = list(range(NUM_BASE_DOF, self.nv))
        if freeze_arms:
            arm_dofs = set()
            for j in range(self.model.njnt):
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if any(a in nm for a in dp.ARM_MOTORS):
                    adr = self.model.jnt_dofadr[j]
                    # int() keys: newer mujoco/numpy return np.int32 here, which
                    # does not hash-match the mjtJoint enum members.
                    ndof = {int(mujoco.mjtJoint.mjJNT_HINGE): 1,
                            int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
                            int(mujoco.mjtJoint.mjJNT_BALL): 3,
                            int(mujoco.mjtJoint.mjJNT_FREE): 6}[int(self.model.jnt_type[j])]
                    arm_dofs.update(range(adr, adr + ndof))
            free = [d for d in free if d not in arm_dofs]
        self.free = np.array(free)

        self.limited = [
            (self.model.jnt_qposadr[j], *self.model.jnt_range[j])
            for j in range(self.model.njnt)
            if self.model.jnt_limited[j] and self.model.jnt_type[j] in
            (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ]

    # -- residual blocks ----------------------------------------------------

    def _closure(self):
        rows = equality_rows(self.model, self.data)
        if len(rows) == 0:
            return np.zeros(0), np.zeros((0, self.nv))
        r = np.asarray(self.data.efc_pos)[rows].copy()
        J = np.asarray(self.data.efc_J).reshape(self.data.nefc, self.nv)[rows].copy()
        return r, J

    def _foot_rotation_jac(self, side):
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, self.feet.body_id[side])
        return jacr

    def _residual(self, targets, flat_feet, yaw_target):
        rows_r, rows_J = [], []

        for side in SIDES:
            target = targets.get(side)
            if target is not None:
                # Drive the sole *contact point* (toe/heel midpoint) to target.
                J_toe = self.feet.jacobian(self.model, self.data, side, "toe")
                J_heel = self.feet.jacobian(self.model, self.data, side, "heel")
                err = self.feet.contact_point(self.data, side) - np.asarray(target, float)
                rows_r.append(self.w_task * err)
                rows_J.append(self.w_task * 0.5 * (J_toe + J_heel))

            if flat_feet:
                # Sole normal to vertical. The x and y components of the world
                # normal must vanish; that is two constraints (pitch and roll),
                # and their derivative w.r.t. an angular velocity w is
                # d(n)/dt = w x n, so the Jacobian rows come from -[n]_x Jr.
                n = self.feet.sole_normal(self.data, side)
                Jr = self._foot_rotation_jac(side)
                skew = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
                Jn = -skew @ Jr                     # d n / d(theta)
                rows_r.append(self.w_tilt * n[:2])
                rows_J.append(self.w_tilt * Jn[:2, :])

                # Foot yaw, so the feet stay pointing along the walking direction.
                # Only meaningful once FootGeometry.calibrate has fixed the heel/toe
                # sign, so refuse to guess rather than silently demand a 180 flip.
                if yaw_target and not self.feet._calibrated:
                    raise RuntimeError(
                        "yaw_target given but FootGeometry is not calibrated; call "
                        "feet.calibrate(data) at an upright pose first"
                    )
                long_axis = self.feet.long_direction(self.data, side)
                yaw = np.arctan2(long_axis[1], long_axis[0])
                want = yaw_target.get(side, yaw) if yaw_target else yaw
                dyaw = np.arctan2(np.sin(yaw - want), np.cos(yaw - want))
                rows_r.append(self.w_yaw * np.array([dyaw]))
                rows_J.append(self.w_yaw * Jr[2:3, :])

        r_eq, J_eq = self._closure()
        if len(r_eq):
            rows_r.append(self.w_closure * r_eq)
            rows_J.append(self.w_closure * J_eq)

        return np.concatenate(rows_r), np.vstack(rows_J)

    def _clamp(self):
        for adr, lo, hi in self.limited:
            self.data.qpos[adr] = min(max(self.data.qpos[adr], lo), hi)

    # -- solve --------------------------------------------------------------

    def solve(
        self,
        qpos_init,
        targets: dict,
        *,
        base_pos=None,
        base_quat=None,
        flat_feet: bool = True,
        yaw_target: dict | None = None,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> IKResult:
        self.data.qpos[:] = np.asarray(qpos_init, dtype=float)
        if base_pos is not None:
            self.data.qpos[dp.QPOS_BASE_POS] = base_pos
        if base_quat is not None:
            self.data.qpos[dp.QPOS_BASE_QUAT] = base_quat
        self.data.qvel[:] = 0.0
        q_ref = self.data.qpos.copy()

        lam = self.damping
        mujoco.mj_forward(self.model, self.data)
        cost = float(np.sum(self._residual(targets, flat_feet, yaw_target)[0] ** 2))

        history, converged, it = [], False, 0
        for it in range(1, max_iter + 1):
            r, J = self._residual(targets, flat_feet, yaw_target)
            Jf = J[:, self.free]

            dq_ref = np.zeros(self.nv)
            mujoco.mj_differentiatePos(self.model, dq_ref, 1.0, self.data.qpos, q_ref)

            H = Jf.T @ Jf + (lam + self.regularisation) * np.eye(self.free.size)
            g = Jf.T @ r - self.regularisation * dq_ref[self.free]
            dq_free = -np.linalg.solve(H, g)

            dq = np.zeros(self.nv)
            dq[self.free] = dq_free

            backup = self.data.qpos.copy()
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            self._clamp()
            mujoco.mj_forward(self.model, self.data)
            new_cost = float(np.sum(self._residual(targets, flat_feet, yaw_target)[0] ** 2))

            if new_cost < cost:
                cost = new_cost
                lam = max(lam * 0.7, 1e-12)
            else:
                self.data.qpos[:] = backup
                mujoco.mj_forward(self.model, self.data)
                lam *= 4.0
                if lam > 1e6:
                    break
                continue

            errs = self.errors(targets)
            history.append(errs)
            if max(errs[0], errs[1]) < tol:
                converged = True
                break

        mujoco.mj_forward(self.model, self.data)
        task, closure, tilt = self.errors(targets)
        return IKResult(
            qpos=self.data.qpos.copy(),
            iterations=it,
            task_error=task,
            closure_error=closure,
            tilt_error=tilt,
            converged=converged,
            history=history,
        )

    def errors(self, targets):
        mujoco.mj_forward(self.model, self.data)
        task = 0.0
        for side in SIDES:
            t = targets.get(side)
            if t is not None:
                task = max(task, float(np.max(np.abs(
                    self.feet.contact_point(self.data, side) - np.asarray(t)))))
        r_eq, _ = self._closure()
        closure = float(np.max(np.abs(r_eq))) if len(r_eq) else 0.0
        tilt = max(
            float(np.arccos(np.clip(self.feet.sole_normal(self.data, s)[2], -1, 1)))
            for s in SIDES
        )
        return task, closure, tilt


if __name__ == "__main__":
    ik = DigitIK()
    print(f"free DOFs available to IK: {ik.free.size} of {ik.nv}")
    print(f"foot: {ik.feet.dimensions()}")

    q = ik.model.qpos0.copy()
    ik.data.qpos[:] = q
    mujoco.mj_forward(ik.model, ik.data)
    print(f"qpos0 feet: left {ik.feet.contact_point(ik.data,'left')}  "
          f"right {ik.feet.contact_point(ik.data,'right')}")
    print()
    print("solving nominal standing pose by homotopy from qpos0 ...")
    q = dp.solve_nominal(ik, verbose=True)
    np.save(dp._NOMINAL_CACHE, q)
    ik.data.qpos[:] = q
    mujoco.mj_forward(ik.model, ik.data)
    print()
    print(f"left foot  {ik.feet.contact_point(ik.data,'left')}")
    print(f"right foot {ik.feet.contact_point(ik.data,'right')}")
    print(f"CoM        {np.array(ik.data.subtree_com[0])}")
    print(f"sole tilt  {np.degrees(ik.errors({})[2]):.4f} deg")
    print(f"corner height spread {np.ptp(ik.feet.corner_heights(ik.data,'left'))*1000:.5f} mm")
    hb = mujoco.mj_name2id(ik.model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_roll")
    reach = np.linalg.norm(ik.feet.contact_point(ik.data, "left") - np.array(ik.data.xpos[hb]))
    print(f"leg extension {reach:.4f} m of {dp.MAX_LEG_REACH:.4f} max "
          f"({100*reach/dp.MAX_LEG_REACH:.1f}%)")
    print()
    print("leg joint angles (deg):")
    for j in range(ik.model.njnt):
        nm = mujoco.mj_id2name(ik.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.startswith("left") and ik.model.jnt_type[j] == 3 and "shoulder" not in nm and "elbow" not in nm:
            print(f"  {nm:<28} {np.degrees(q[ik.model.jnt_qposadr[j]]):>8.2f}")
