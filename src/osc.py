"""Operational-space controller: QP-based inverse dynamics for Digit.

Stage A of the plan. Solves, every control tick, for joint accelerations, contact
forces, loop-closure forces and motor torques simultaneously, subject to the true
constrained dynamics -- then sends only the torques to the robot.

Decision variable
-----------------
    z = [ qacc (nv) | lambda_c (3 per contact point) | lambda_eq (18) | tau (nu) ]

Hard constraints (equalities)
-----------------------------
    dynamics    M qacc + h = B tau + Jc^T lambda_c + Jeq^T lambda_eq
    no-slip     Jc qacc + Jdot_c qdot = 0        (stance points do not accelerate)
    loop closure Jeq qacc + Jdot_eq qdot = 0     (the six `connect` constraints)

Hard constraints (inequalities)
-------------------------------
    friction    |fx| <= mu fz, |fy| <= mu fz     (linearised pyramid)
    unilateral  fz >= 0                          (contacts push, never pull)
    torque      |tau| <= tau_max                 (imposed here; the MJCF declares none)

Cost: weighted tracking of task accelerations, plus small regularisation.

Why a QP rather than a PD loop
------------------------------
The two things that made joint PD fail on Cassie are both *inequality*
constraints: contact forces cannot pull, and torque is bounded. A QP plans within
those limits instead of discovering them by falling over. It also lets the
controller decide *where under the foot* to put the load -- with four contact
points per foot the centre of pressure is an implicit decision variable, which is
the stabilisation channel Digit's box foot has and Cassie's blade does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp

import digit_params as dp
from dynamics import CORNERS, DynamicsModel


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------


class CscPattern:
    """A fixed CSC sparsity pattern, so OSQP can be warm-updated in place.

    `scipy.sparse.csc_matrix(dense)` drops entries that happen to be exactly zero,
    so the nonzero count drifts from tick to tick as values change. OSQP's
    `update(Px=..., Ax=...)` requires the pattern to be *identical* every time and
    fails with "new number of elements out of bounds" otherwise -- which silently
    freezes the control output at whatever the last successful solve produced.

    Pinning the pattern (storing structural zeros explicitly) costs a little
    solver time and removes the failure mode entirely.
    """

    def __init__(self, shape, upper=False):
        rows, cols = shape
        self.shape = shape
        idx, ptr = [], [0]
        for j in range(cols):
            r = np.arange(min(j + 1, rows)) if upper else np.arange(rows)
            idx.append(r)
            ptr.append(ptr[-1] + len(r))
        self.indices = np.concatenate(idx).astype(np.int32)
        self.indptr = np.array(ptr, dtype=np.int32)
        counts = np.diff(self.indptr)
        self.cols = np.repeat(np.arange(cols), counts)

    def values(self, dense):
        return np.asarray(dense)[self.indices, self.cols]

    def matrix(self, dense):
        return sp.csc_matrix(
            (self.values(dense), self.indices, self.indptr), shape=self.shape
        )


@dataclass
class TaskSpec:
    """A task drives some function of q toward a setpoint at the acceleration level.

    The QP sees `J qacc = a_des - drift`, where `a_des` comes from a PD law on the
    task-space error. Weight trades tasks off against each other.
    """

    name: str
    weight: float
    kp: float
    kd: float


@dataclass
class OSCConfig:
    com: TaskSpec = field(default_factory=lambda: TaskSpec("com", 100.0, 100.0, 20.0))
    torso: TaskSpec = field(default_factory=lambda: TaskSpec("torso", 50.0, 200.0, 28.0))
    posture: TaskSpec = field(default_factory=lambda: TaskSpec("posture", 0.5, 40.0, 8.0))
    swing: TaskSpec = field(default_factory=lambda: TaskSpec("swing", 200.0, 400.0, 40.0))
    swing_ori: TaskSpec = field(default_factory=lambda: TaskSpec("swing_ori", 20.0, 200.0, 28.0))

    mu: float = 0.7                 # friction coefficient used in the QP
    w_qacc: float = 1e-4            # regularisation
    w_force: float = 1e-6
    w_tau: float = 1e-4
    min_normal_force: float = 1.0   # keep stance contacts slightly loaded [N]

    # Penalty on STANCE ankle (toe_A / toe_B) torque.
    #
    # This is not cosmetic. The flat-ground ALIP model assumes
    #     Ldot = m g x
    # with no ankle term. If the controller freely uses the stance ankle, it injects
    # a large torque about the contact point that the planner knows nothing about,
    # and the deadbeat foot placement systematically under-corrects. Measured on
    # Digit: end-of-step momentum came out 2-4x more negative than predicted, which
    # matched the impulse of a saturated 41 Nm toe motor over one step.
    #
    # Gong & Grizzle exclude the stance ankle for exactly this reason ("the small
    # ankle motor saturates easily"). Penalising it here makes the plant match the
    # reduced model the planner is using. The stairs paper takes the other route:
    # keep the ankle and add tau to the model, driven by MPC.
    w_stance_ankle: float = 0.0     # walking sets this high; standing wants it 0


class OSC:
    def __init__(self, model, feet=None, actuation=None, config: OSCConfig | None = None):
        self.model = model
        self.cfg = config or OSCConfig()
        self.feet = feet or dp.FootGeometry(model)
        self.act = actuation or dp.Actuation(model)
        self.dyn = DynamicsModel(model, self.feet, self.act)

        self.nv = model.nv
        self.nu = self.act.n
        self.n_eq_rows = 3 * model.neq

        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        # One cached solver per problem shape. Stance flips between one and two feet
        # during walking, which changes the number of force unknowns; rebuilding
        # OSQP on every flip dominates the runtime, so keep them keyed by shape.
        self._probs = {}
        self.last_solution = None
        self.solve_stats = {"solves": 0, "cold_starts": 0, "failures": 0}

        # Which actuator slots are each foot's toe motors, for the stance-ankle
        # penalty. On Digit toe_A/toe_B drive foot pitch and roll through a
        # differential, so both belong to the ankle.
        self.toe_actuators = {
            side: [k for k, nm in enumerate(self.act.names)
                   if nm.startswith(side) and ("toe_A" in nm or "toe_B" in nm)]
            for side in dp.SIDES
        }

    # -- task Jacobians -----------------------------------------------------

    def com_jacobian(self, data):
        J = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.model, data, J, 0)
        return J

    def torso_rot_jacobian(self, data):
        Jr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, data, None, Jr, self.torso_id)
        return Jr

    def foot_jacobian(self, data, side):
        """Translational Jacobian of the sole centre."""
        J_toe = self.feet.jacobian(self.model, data, side, "toe")
        J_heel = self.feet.jacobian(self.model, data, side, "heel")
        return 0.5 * (J_toe + J_heel)

    def foot_rot_jacobian(self, data, side):
        Jr = np.zeros((3, self.nv))
        mujoco.mj_jacBody(self.model, data, None, Jr, self.feet.body_id[side])
        return Jr

    def stance_jacobian(self, data, sides):
        """Rank-correct no-slip Jacobian: 6 rows per stance foot, not 3 per corner.

        A planted foot is a rigid body, so pinning it takes exactly 6 constraints.
        Constraining all four sole corners instead gives 12 rows per foot of which
        only 6 are independent. Those redundant rows are consistent in exact
        arithmetic, but the drift terms are finite-differenced and carry ~1e-3 noise,
        which makes them mutually contradictory -- and OSQP correctly reports the
        whole QP as primal infeasible.

        Contact *forces* remain per-corner, so the centre of pressure and the
        friction cones are unaffected.
        """
        rows = []
        for side in sides:
            jacp = np.zeros((3, self.nv))
            jacr = np.zeros((3, self.nv))
            mujoco.mj_jacBody(self.model, data, jacp, jacr, self.feet.body_id[side])
            rows.append(np.vstack([jacp, jacr]))
        return np.vstack(rows) if rows else np.zeros((0, self.nv))

    def stance_drift(self, data, sides):
        return self._drift(data, lambda d: self.stance_jacobian(d, sides))

    def _drift(self, data, jac_fn):
        return self.dyn._drift(data, jac_fn)

    # -- orientation error --------------------------------------------------

    @staticmethod
    def _quat_error(q_cur, q_des):
        """Rotation vector taking q_cur to q_des (world frame)."""
        neg = np.zeros(4)
        err = np.zeros(4)
        vel = np.zeros(3)
        mujoco.mju_negQuat(neg, np.asarray(q_cur, dtype=float))
        mujoco.mju_mulQuat(err, np.asarray(q_des, dtype=float), neg)
        mujoco.mju_quat2Vel(vel, err, 1.0)
        return vel

    # -- the QP -------------------------------------------------------------

    def solve(
        self,
        data,
        *,
        stance_sides,
        com_target=None,
        com_vel_target=None,
        com_axes=(0, 1, 2),
        torso_quat_target=(1.0, 0.0, 0.0, 0.0),
        posture_target=None,
        swing_side=None,
        swing_target=None,
        swing_vel_target=None,
    ):
        cfg = self.cfg
        nv, nu = self.nv, self.nu
        n_pts = len(CORNERS) * len(stance_sides)
        nc = 3 * n_pts                    # force unknowns: 3 per sole corner
        n_slip = 6 * len(stance_sides)    # no-slip rows: 6 per rigid foot
        neq_rows = self.n_eq_rows
        n = nv + nc + neq_rows + nu

        i_qacc = slice(0, nv)
        i_lc = slice(nv, nv + nc)
        i_leq = slice(nv + nc, nv + nc + neq_rows)
        i_tau = slice(nv + nc + neq_rows, n)

        M = self.dyn.mass_matrix(data)
        h = self.dyn.bias(data)
        B = self.dyn.B
        # Per-corner Jacobians map corner forces into generalised forces ...
        Jc = self.dyn.contact_jacobian(data, stance_sides)
        # ... while the no-slip constraint is imposed on the foot body (rank-correct).
        Js = self.stance_jacobian(data, stance_sides)
        drift_s = self.stance_drift(data, stance_sides)
        Jeq = self.dyn.eq_jacobian(data)
        drift_eq = self.dyn.eq_drift(data)

        # ---- equality block -----------------------------------------------
        A_dyn = np.zeros((nv, n))
        A_dyn[:, i_qacc] = M
        A_dyn[:, i_lc] = -Jc.T
        A_dyn[:, i_leq] = -Jeq.T
        A_dyn[:, i_tau] = -B
        b_dyn = -h

        A_ct = np.zeros((n_slip, n))
        A_ct[:, i_qacc] = Js
        b_ct = -drift_s

        A_lp = np.zeros((neq_rows, n))
        A_lp[:, i_qacc] = Jeq
        b_lp = -drift_eq

        # ---- inequality block ---------------------------------------------
        rows, lo, hi = [], [], []
        INF = np.inf
        for k in range(n_pts):
            base = nv + 3 * k
            fx, fy, fz = base, base + 1, base + 2
            # fz >= min_normal_force
            r = np.zeros(n); r[fz] = 1.0
            rows.append(r); lo.append(cfg.min_normal_force); hi.append(INF)
            # |fx| <= mu fz  ->  mu fz - fx >= 0 and mu fz + fx >= 0
            for axis in (fx, fy):
                r = np.zeros(n); r[fz] = cfg.mu; r[axis] = -1.0
                rows.append(r); lo.append(0.0); hi.append(INF)
                r = np.zeros(n); r[fz] = cfg.mu; r[axis] = +1.0
                rows.append(r); lo.append(0.0); hi.append(INF)

        for k in range(nu):
            r = np.zeros(n); r[nv + nc + neq_rows + k] = 1.0
            rows.append(r)
            lo.append(-self.act.torque_limit[k])
            hi.append(+self.act.torque_limit[k])

        A_ineq = np.array(rows) if rows else np.zeros((0, n))

        # ---- cost ----------------------------------------------------------
        P = np.zeros((n, n))
        q = np.zeros(n)

        def add_task(J, drift, a_des, weight):
            A = np.zeros((J.shape[0], n))
            A[:, i_qacc] = J
            b = a_des - drift
            nonlocal P, q
            P += 2.0 * weight * (A.T @ A)
            q += -2.0 * weight * (A.T @ b)

        # CoM. `com_axes` selects which components are regulated: standing uses all
        # three, but walking must use HEIGHT ONLY -- constraining horizontal CoM
        # motion would suppress the pendulum dynamics that ALIP is built on, and
        # reduce the whole thing to inverse kinematics with extra steps.
        if com_target is not None and len(com_axes):
            ax = list(com_axes)
            J = self.com_jacobian(data)[ax, :]
            drift = self._drift(data, self.com_jacobian)[ax]
            com = np.array(data.subtree_com[0])[ax]
            mujoco.mj_subtreeVel(self.model, data)
            v = np.array(data.subtree_linvel[0])[ax]
            v_des = (np.zeros(3) if com_vel_target is None
                     else np.asarray(com_vel_target))[ax]
            a = cfg.com.kp * (np.asarray(com_target)[ax] - com) + cfg.com.kd * (v_des - v)
            add_task(J, drift, a, cfg.com.weight)

        # Torso orientation
        if torso_quat_target is not None:
            J = self.torso_rot_jacobian(data)
            drift = self._drift(data, self.torso_rot_jacobian)
            e = self._quat_error(data.xquat[self.torso_id], torso_quat_target)
            w = J @ np.asarray(data.qvel)
            a = cfg.torso.kp * e - cfg.torso.kd * w
            add_task(J, drift, a, cfg.torso.weight)

        # Posture on the actuated joints
        if posture_target is not None:
            S = np.zeros((nu, nv))
            for k, dof in enumerate(self.act.dofadr):
                S[k, dof] = 1.0
            qj = np.asarray(data.qpos)[self.act.qposadr]
            vj = np.asarray(data.qvel)[self.act.dofadr]
            a = cfg.posture.kp * (np.asarray(posture_target) - qj) - cfg.posture.kd * vj
            add_task(S, np.zeros(nu), a, cfg.posture.weight)

        # Swing foot
        if swing_side is not None and swing_target is not None:
            J = self.foot_jacobian(data, swing_side)
            drift = self._drift(data, lambda d: self.foot_jacobian(d, swing_side))
            p = self.feet.contact_point(data, swing_side)
            v = J @ np.asarray(data.qvel)
            v_des = np.zeros(3) if swing_vel_target is None else np.asarray(swing_vel_target)
            a = cfg.swing.kp * (np.asarray(swing_target) - p) + cfg.swing.kd * (v_des - v)
            add_task(J, drift, a, cfg.swing.weight)

            Jr = self.foot_rot_jacobian(data, swing_side)
            drift_r = self._drift(data, lambda d: self.foot_rot_jacobian(d, swing_side))
            e = self._quat_error(data.xquat[self.feet.body_id[swing_side]],
                                 self._level_foot_quat(data, swing_side))
            w = Jr @ np.asarray(data.qvel)
            a = cfg.swing_ori.kp * e - cfg.swing_ori.kd * w
            add_task(Jr, drift_r, a, cfg.swing_ori.weight)

        # Regularisation, also makes P positive definite.
        reg = np.zeros(n)
        reg[i_qacc] = cfg.w_qacc
        reg[i_lc] = cfg.w_force
        reg[i_leq] = cfg.w_force
        reg[i_tau] = cfg.w_tau

        # Suppress stance ankle torque so the plant matches the ankle-free ALIP
        # model the planner assumes. See OSCConfig.w_stance_ankle.
        if cfg.w_stance_ankle:
            tau0 = nv + nc + neq_rows
            for side in stance_sides:
                for k in self.toe_actuators[side]:
                    reg[tau0 + k] += cfg.w_stance_ankle

        P += 2.0 * np.diag(reg)

        # ---- assemble and solve --------------------------------------------
        A_dense = np.vstack([A_dyn, A_ct, A_lp, A_ineq])
        l = np.concatenate([b_dyn, b_ct, b_lp, np.array(lo)])
        u = np.concatenate([b_dyn, b_ct, b_lp, np.array(hi)])

        key = (n, A_dense.shape[0])
        entry = self._probs.get(key)
        if entry is None:
            # OSQP wants only the upper triangle of P.
            pat_P = CscPattern((n, n), upper=True)
            pat_A = CscPattern(A_dense.shape)
            prob = self._new_problem(pat_P.matrix(P), q, pat_A.matrix(A_dense), l, u)
            entry = self._probs[key] = dict(prob=prob, pat_P=pat_P, pat_A=pat_A)
            self.solve_stats["cold_starts"] += 1
        else:
            entry["prob"].update(
                Px=entry["pat_P"].values(P), Ax=entry["pat_A"].values(A_dense),
                q=q, l=l, u=u,
            )

        self.solve_stats["solves"] += 1
        res = entry["prob"].solve()
        status = str(getattr(res.info, "status", "unknown"))

        # A warm-started solve can inherit a bad iterate and then report
        # infeasibility forever. Rebuild once from scratch before giving up --
        # and never hand back the solution vector of a failed solve, which OSQP
        # leaves as uninitialised garbage rather than None.
        if "solved" not in status:
            entry["prob"] = self._new_problem(
                entry["pat_P"].matrix(P), q, entry["pat_A"].matrix(A_dense), l, u
            )
            self.solve_stats["cold_starts"] += 1
            res = entry["prob"].solve()
            status = str(getattr(res.info, "status", "unknown"))
            if "solved" not in status:
                self.solve_stats["failures"] += 1

        z = res.x
        if "solved" not in status or z is None or not np.all(np.isfinite(z)):
            return None, dict(status=status, ok=False)

        out = dict(
            status=status,
            ok=True,
            qacc=z[i_qacc].copy(),
            lam_c=z[i_lc].copy().reshape(n_pts, 3),
            lam_eq=z[i_leq].copy(),
            tau=z[i_tau].copy(),
            stance_sides=tuple(stance_sides),
        )
        self.last_solution = out
        return out["tau"], out

    @staticmethod
    def _new_problem(P, q, A, l, u):
        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                   max_iter=4000, polish=True)
        return prob

    def _level_foot_quat(self, data, side):
        """Target orientation for the swing foot: current yaw, zero pitch and roll."""
        d = self.feet.long_direction(data, side)
        yaw = np.arctan2(d[1], d[0])
        # Rotation of `yaw` about world z, expressed relative to the foot's own
        # zero pose, whose sole normal is already handled by FootGeometry.
        R_target = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        R_body = np.array(data.xmat[self.feet.body_id[side]]).reshape(3, 3)
        # Sole-flat orientation with the same yaw: replace the body's tilt while
        # preserving how the sole frame sits inside the body frame.
        n_body = self.feet.normal_body[side]
        l_body = self.feet.long_sign[side] * self.feet.long_body[side]
        s_body = np.cross(n_body, l_body)
        R_sole_in_body = np.column_stack([l_body, s_body, n_body])
        R_des = R_target @ R_sole_in_body.T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_des.flatten())
        del R_body
        return quat
