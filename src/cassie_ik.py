"""Step 2: leg inverse kinematics for Cassie, respecting the closed loops.

Cassie's legs are not serial chains: the achilles rod and the plantar rod each
close a loop, enforced in cassie.xml by `connect` equality constraints. Solving
IK on the actuated joints alone and letting the rods dangle produces a pose that
looks broken and is dynamically meaningless.

Instead this solves a damped Gauss-Newton problem over *all* leg degrees of
freedom, stacking two kinds of residual:

  * task    -- foot centre position, and toe/heel at equal height (flat sole)
  * closure -- MuJoCo's own equality-constraint violation, read from `efc_pos`
               with the exact Jacobian from `efc_J`

Driving both to zero simultaneously gives a foot placed where we asked with the
linkage still physically assembled. The floating base is held fixed: the
animation stage prescribes the pelvis and asks only "where do the legs go".
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent

NUM_BASE_DOF = 6  # 3 slide + 1 ball = 6 velocity DOFs
SIDES = ("left", "right")


@dataclass
class IKResult:
    qpos: np.ndarray
    iterations: int
    task_error: float       # max foot-centre position error [m]
    closure_error: float    # max loop-closure violation [m]
    converged: bool
    history: list = field(default_factory=list)


class CassieIK:
    """Damped least-squares IK over the leg DOFs, with loop closure enforced.

    Contacts are disabled on this private model copy: during IK the feet are
    being moved to wherever we ask, and contact forces would fight that.
    """

    def __init__(
        self,
        model_path: str | pathlib.Path | None = None,
        *,
        w_task: float = 1.0,
        w_level: float = 0.05,
        w_closure: float = 2.0,
        w_spring: float = 2.0,
        damping: float = 1e-6,
        regularisation: float = 1e-7,
    ):
        if model_path is None:
            model_path = ROOT / "models" / "cassie-mujoco-sim" / "model" / "cassie.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(self.model)

        self.w_task = w_task
        self.w_level = w_level
        self.w_closure = w_closure
        self.w_spring = w_spring
        self.damping = damping
        self.regularisation = regularisation

        # Compliant joints (shin and heel springs). A purely kinematic solve will
        # happily deflect these to reach a target, because bending a spring costs
        # nothing without dynamics -- but on the robot it costs a large force. Pin
        # them to their rest position so the actuated joints have to do the work.
        self.spring_dofs = [
            (self.model.jnt_qposadr[j], self.model.jnt_dofadr[j], float(self.model.qpos_spring[self.model.jnt_qposadr[j]]))
            for j in range(self.model.njnt)
            if self.model.jnt_stiffness[j] > 0
        ]

        self.nv = self.model.nv
        self.site = {
            f"{side}-{part}": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}-{part}"
            )
            for side in SIDES
            for part in ("toe", "heel")
        }
        if any(v < 0 for v in self.site.values()):
            raise RuntimeError("expected left/right toe and heel sites in the model")

        # Which velocity DOFs the solver may move: everything but the base.
        self.free = np.arange(NUM_BASE_DOF, self.nv)

        # Position limits, for clamping after each Gauss-Newton update.
        self.limited = [
            (self.model.jnt_qposadr[j], *self.model.jnt_range[j])
            for j in range(self.model.njnt)
            if self.model.jnt_limited[j]
        ]

    # -- geometry helpers ---------------------------------------------------

    def _site_jac(self, site_id):
        jacp = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, site_id)
        return jacp

    def foot_centre(self, side):
        toe = self.data.site_xpos[self.site[f"{side}-toe"]]
        heel = self.data.site_xpos[self.site[f"{side}-heel"]]
        return 0.5 * (np.asarray(toe) + np.asarray(heel))

    def _closure(self):
        """Equality-constraint violation and its Jacobian.

        With contacts disabled, every row of `efc` belongs to one of the four
        `connect` constraints, so no filtering is needed -- but assert it, since
        a silent extra row would corrupt the solve.
        """
        n = self.data.nefc
        expected = 3 * self.model.neq
        if n != expected:
            raise RuntimeError(f"expected {expected} equality rows, got {n}")
        r = np.asarray(self.data.efc_pos[:n]).copy()
        J = np.asarray(self.data.efc_J).reshape(n, self.nv).copy()
        return r, J

    # -- the solve ----------------------------------------------------------

    def _residual(self, targets):
        """Stack weighted residuals and Jacobians. Residual is what we drive to 0."""
        rows_r, rows_J = [], []

        for side in SIDES:
            target = targets.get(side)
            toe_id, heel_id = self.site[f"{side}-toe"], self.site[f"{side}-heel"]
            J_toe, J_heel = self._site_jac(toe_id), self._site_jac(heel_id)

            if target is not None:
                err = self.foot_centre(side) - np.asarray(target, dtype=float)
                rows_r.append(self.w_task * err)
                rows_J.append(self.w_task * 0.5 * (J_toe + J_heel))

            # Flat sole: toe and heel at the same height.
            dz = self.data.site_xpos[toe_id][2] - self.data.site_xpos[heel_id][2]
            rows_r.append(self.w_level * np.array([dz]))
            rows_J.append(self.w_level * (J_toe - J_heel)[2:3, :])

        r_eq, J_eq = self._closure()
        rows_r.append(self.w_closure * r_eq)
        rows_J.append(self.w_closure * J_eq)

        # Keep the compliant joints at rest.
        if self.spring_dofs:
            r_spring = np.array(
                [self.data.qpos[adr] - rest for adr, _, rest in self.spring_dofs]
            )
            J_spring = np.zeros((len(self.spring_dofs), self.nv))
            for row, (_, dof, _) in enumerate(self.spring_dofs):
                J_spring[row, dof] = 1.0
            rows_r.append(self.w_spring * r_spring)
            rows_J.append(self.w_spring * J_spring)

        return np.concatenate(rows_r), np.vstack(rows_J)

    def _clamp(self):
        for adr, lo, hi in self.limited:
            self.data.qpos[adr] = min(max(self.data.qpos[adr], lo), hi)

    def solve(
        self,
        qpos_init,
        targets: dict,
        *,
        pelvis_pos=None,
        pelvis_quat=None,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> IKResult:
        """Place the feet at `targets` with the pelvis held where it is.

        Parameters
        ----------
        qpos_init : starting configuration, also the regularisation reference
        targets   : {"left": (3,), "right": (3,)} desired foot-centre positions in
                    world coordinates. Omit a side to leave that leg free.
        pelvis_pos, pelvis_quat : optionally override the base pose before solving
        """
        self.data.qpos[:] = np.asarray(qpos_init, dtype=float)
        if pelvis_pos is not None:
            self.data.qpos[0:3] = pelvis_pos
        if pelvis_quat is not None:
            self.data.qpos[3:7] = pelvis_quat
        self.data.qvel[:] = 0.0
        q_ref = self.data.qpos.copy()

        history = []
        converged = False
        it = 0
        lam = self.damping
        mujoco.mj_forward(self.model, self.data)
        cost = float(np.sum(self._residual(targets)[0] ** 2))

        for it in range(1, max_iter + 1):
            r, J = self._residual(targets)
            Jf = J[:, self.free]

            # Pull weakly toward the reference pose to resolve the nullspace.
            dq_ref = np.zeros(self.nv)
            mujoco.mj_differentiatePos(self.model, dq_ref, 1.0, self.data.qpos, q_ref)

            H = Jf.T @ Jf + (lam + self.regularisation) * np.eye(self.free.size)
            g = Jf.T @ r - self.regularisation * dq_ref[self.free]
            dq_free = -np.linalg.solve(H, g)

            dq = np.zeros(self.nv)
            dq[self.free] = dq_free

            # Levenberg-Marquardt: accept the step only if the cost improves,
            # otherwise retreat toward gradient descent and try again.
            q_backup = self.data.qpos.copy()
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            self._clamp()
            mujoco.mj_forward(self.model, self.data)
            new_cost = float(np.sum(self._residual(targets)[0] ** 2))

            if new_cost < cost:
                cost = new_cost
                lam = max(lam * 0.7, 1e-12)
            else:
                self.data.qpos[:] = q_backup
                mujoco.mj_forward(self.model, self.data)
                lam *= 4.0
                if lam > 1e6:
                    break
                continue

            task_err, closure_err = self.errors(targets)
            history.append((task_err, closure_err))
            if max(task_err, closure_err) < tol:
                converged = True
                break

        mujoco.mj_forward(self.model, self.data)
        task_err, closure_err = self.errors(targets)
        return IKResult(
            qpos=self.data.qpos.copy(),
            iterations=it,
            task_error=task_err,
            closure_error=closure_err,
            converged=converged,
            history=history,
        )

    def errors(self, targets):
        """Recompute errors from scratch, in metres, unweighted."""
        mujoco.mj_forward(self.model, self.data)
        task = 0.0
        for side in SIDES:
            if targets.get(side) is not None:
                task = max(
                    task,
                    float(np.max(np.abs(self.foot_centre(side) - np.asarray(targets[side])))),
                )
        closure = float(np.max(np.abs(self._closure()[0])))
        return task, closure

    def spring_deflection(self):
        """Max |deflection| of the compliant joints, in radians. Should be ~0."""
        if not self.spring_dofs:
            return 0.0
        return float(
            max(abs(self.data.qpos[adr] - rest) for adr, _, rest in self.spring_dofs)
        )


if __name__ == "__main__":
    import cassie_params

    ik = CassieIK()
    q0 = cassie_params.QPOS_INIT.copy()

    ik.data.qpos[:] = q0
    mujoco.mj_forward(ik.model, ik.data)
    left0, right0 = ik.foot_centre("left"), ik.foot_centre("right")
    print(f"nominal left foot  {np.array2string(left0, precision=4)}")
    print(f"nominal right foot {np.array2string(right0, precision=4)}")
    print(f"nominal closure violation {ik.errors({})[1] * 1000:.3f} mm")
    print()

    # Swing the left foot forward and up, as a mid-swing pose would.
    for dx, dz in [(0.0, 0.0), (0.05, 0.05), (0.15, 0.10), (-0.10, 0.06)]:
        res = ik.solve(q0, {"left": left0 + [dx, 0, dz], "right": right0})
        print(
            f"target dx={dx:+.2f} dz={dz:+.2f} -> "
            f"iters {res.iterations:>3}  task err {res.task_error * 1000:7.4f} mm  "
            f"closure {res.closure_error * 1000:7.4f} mm  "
            f"{'converged' if res.converged else 'NOT CONVERGED'}"
        )
