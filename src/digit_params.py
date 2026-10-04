"""Digit model audit, actuation map, foot geometry and nominal standing pose.

The Digit counterpart of src/cassie_params.py. Differences from Cassie that matter:

  * 50 DOF / 26 actuators in this MJCF (the real robot has 20 motors -- see the
    note on redundant actuators below), against Cassie's 32 / 10.
  * The model has **no springs**: `jnt_stiffness` is zero everywhere. Newer Digit
    revisions drop the leaf springs, so the series-compliance problem that
    dominated the Cassie work simply is not present.
  * `qpos0` satisfies the loop-closure constraints **exactly**, unlike Cassie's
    `qpos_init` which sat 5.6 mm off the manifold.
  * There are only three sites in the model (imu, two hands) and **none on the
    feet**, so contact points are derived from the foot collision box and
    addressed with `mj_jac` at explicit points rather than `mj_jacSite`.
  * The foot is a genuine box, roughly 8 cm x 23.5 cm, not Cassie's blade. It has
    a real support polygon and actuated ankle pitch *and* roll.

Run:  python src/digit_params.py
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "digit" / "xml" / "digit_patched.xml"

SIDES = ("left", "right")

# Floating base is a single free joint: 3 position + 4 quaternion.
QPOS_BASE_POS = slice(0, 3)
QPOS_BASE_QUAT = slice(3, 7)
NUM_BASE_DOF = 6

# The six motors per leg on the real robot. `tarsus`, `toe_pitch` and `toe_roll`
# also carry actuators in this MJCF, but on hardware they are *dependent*
# coordinates: tarsus is passive, and toe pitch/roll are driven through the
# toe_A / toe_B differential. Commanding them directly is physically fictitious,
# so they are excluded here and the leg is controlled through the real six.
LEG_MOTORS = (
    "hip_roll",
    "hip_yaw",
    "hip_pitch",
    "knee",
    "toe_A",
    "toe_B",
)
ARM_MOTORS = ("shoulder_roll", "shoulder_pitch", "shoulder_yaw", "elbow")

# Joints that exist in the MJCF as actuators but are dependent on hardware.
FICTITIOUS_ACTUATORS = ("tarsus", "toe_pitch", "toe_roll")

# Peak torque limits for Digit's motors [Nm], from Agility's published figures.
# The MJCF ships ctrlrange = [0, 0] with ctrllimited = False, i.e. *unbounded*
# torque, which would quietly let a controller cheat. These are imposed instead.
TORQUE_LIMIT = {
    "hip_roll": 126.0,
    "hip_yaw": 79.0,
    "hip_pitch": 216.0,
    "knee": 231.0,
    "toe_A": 41.0,
    "toe_B": 41.0,
    "shoulder_roll": 126.0,
    "shoulder_pitch": 126.0,
    "shoulder_yaw": 79.0,
    "elbow": 126.0,
}

FOOT_BODY = {side: f"{side}_toe_roll" for side in SIDES}


# ---------------------------------------------------------------------------


def load(path=None, colored: bool = False):
    """Load the Digit model.

    `colored=True` uses the palette in src/digit_visual.py. It changes only the
    rgba of visual (non-colliding) geoms, so dynamics are bit-identical -- it is
    purely for rendering.
    """
    if path is None and colored:
        from digit_visual import write_colored

        path = write_colored()
    model = mujoco.MjModel.from_xml_path(str(path or MODEL_PATH))
    return model, mujoco.MjData(model)


def _name(model, objtype, i):
    return mujoco.mj_id2name(model, objtype, i)


# ---------------------------------------------------------------------------
# Foot geometry
# ---------------------------------------------------------------------------


class FootGeometry:
    """Contact points on the sole, derived from the foot's collision box.

    The model gives us a box geom on `<side>_toe_roll`. Its four bottom corners
    are the support polygon; the midpoints of the two short edges act as the
    'toe' and 'heel' equivalents. Everything is expressed in the foot body frame
    so it can be handed to `mj_jac`, which accepts an arbitrary point.

    CRITICAL: the box carries a rotation as well as a position. In this model it is
    `euler='-1.047197551 0 0'`, i.e. 60 degrees about x, so the sole normal in the
    *body* frame is (0, 0.866, 0.5) and not body +z. Reading only `geom_pos` and
    `geom_size` and assuming the sole is the body's xy-plane leaves the foot tilted
    by ~50 degrees, which is exactly the bug this class was written to avoid.
    Everything below is therefore constructed in the *geom* frame and rotated into
    the body frame by `geom_quat`.
    """

    def __init__(self, model):
        self.model = model
        self.body_id = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FOOT_BODY[s])
                        for s in SIDES}
        self.local = {}          # sole points, in the FOOT BODY frame
        self.half = {}           # box half-extents, in the GEOM frame
        self.normal_body = {}    # sole normal, in the FOOT BODY frame
        self.long_body = {}      # heel->toe direction, in the FOOT BODY frame
        for side in SIDES:
            gid = self._collision_geom(side)
            size = np.array(model.geom_size[gid])
            pos = np.array(model.geom_pos[gid])
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[gid])
            R = R.reshape(3, 3)          # geom -> body

            self.half[side] = size.copy()
            long_axis = int(np.argmax(size[:2]))
            self.long_axis = long_axis
            L, W = size[long_axis], size[1 - long_axis]

            def pt(along, across):
                p = np.zeros(3)
                p[long_axis] = along * L
                p[1 - long_axis] = across * W
                p[2] = -size[2]          # the sole is the box's -z face
                return pos + R @ p       # into the body frame

            self.local[side] = {
                "toe": pt(+1.0, 0.0),
                "heel": pt(-1.0, 0.0),
                "center": pt(0.0, 0.0),
                "corner_tl": pt(+1.0, +1.0),
                "corner_tr": pt(+1.0, -1.0),
                "corner_hl": pt(-1.0, +1.0),
                "corner_hr": pt(-1.0, -1.0),
            }
            # +z of the geom points away from the sole, i.e. "up" when standing.
            self.normal_body[side] = R[:, 2]
            self.long_body[side] = R[:, long_axis]

        # The geom eulers are mirrored between sides (-60 deg left, +60 deg right),
        # so the geom's +y axis points in OPPOSITE world directions on the two feet.
        # It is therefore not "heel to toe" until the sign is fixed per side. Left
        # unset it makes a yaw target of 0 demand a 180 deg flip on one foot, which
        # saturates the hip yaw joint. `calibrate` resolves it from an actual pose.
        self.long_sign = {s: +1.0 for s in SIDES}
        self._calibrated = False

    def _collision_geom(self, side):
        """The foot geom that actually collides (contype != 0)."""
        best = None
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != self.body_id[side]:
                continue
            if self.model.geom_contype[g] == 0:
                continue
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            best = g
        if best is None:
            raise RuntimeError(f"no colliding box geom on {FOOT_BODY[side]}")
        return best

    def world(self, data, side, which="center"):
        """World position of a named sole point."""
        b = self.body_id[side]
        R = np.array(data.xmat[b]).reshape(3, 3)
        return np.array(data.xpos[b]) + R @ self.local[side][which]

    def jacobian(self, model, data, side, which="center"):
        """Translational Jacobian of that sole point (3 x nv)."""
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, self.world(data, side, which),
                      self.body_id[side])
        return jacp

    def sole_normal(self, data, side):
        """World-frame sole normal, pointing away from the ground when flat."""
        R = np.array(data.xmat[self.body_id[side]]).reshape(3, 3)
        return R @ self.normal_body[side]

    def calibrate(self, data):
        """Fix the heel/toe sign per side using an upright pose.

        Called once with the robot standing: whichever end of the long axis has the
        larger world x is the toe. Position and tilt tasks are unaffected (the
        contact point is a midpoint, and the normal has no sign ambiguity) -- only
        the yaw task and the toe/heel labels depend on this.
        """
        for side in SIDES:
            R = np.array(data.xmat[self.body_id[side]]).reshape(3, 3)
            if (R @ self.long_body[side])[0] < 0:
                self.long_sign[side] = -1.0
                loc = self.local[side]
                loc["toe"], loc["heel"] = loc["heel"], loc["toe"]
                loc["corner_tl"], loc["corner_hl"] = loc["corner_hl"], loc["corner_tl"]
                loc["corner_tr"], loc["corner_hr"] = loc["corner_hr"], loc["corner_tr"]
        self._calibrated = True
        return self.long_sign

    def long_direction(self, data, side):
        """World-frame heel-to-toe direction (requires `calibrate` for the sign)."""
        R = np.array(data.xmat[self.body_id[side]]).reshape(3, 3)
        return self.long_sign[side] * (R @ self.long_body[side])

    def corner_heights(self, data, side):
        """Heights of the four sole corners; all equal means the foot is flat."""
        return np.array([
            self.world(data, side, k)[2]
            for k in ("corner_tl", "corner_tr", "corner_hl", "corner_hr")
        ])

    def contact_point(self, data, side):
        """Effective contact point: midpoint of toe and heel."""
        return 0.5 * (self.world(data, side, "toe") + self.world(data, side, "heel"))

    def dimensions(self, side="left"):
        h = self.half[side]
        long_len = 2 * h[self.long_axis]
        short_len = 2 * h[1 - self.long_axis]
        return dict(length=long_len, width=short_len, long_axis="xyz"[self.long_axis])


# ---------------------------------------------------------------------------
# Actuation
# ---------------------------------------------------------------------------


class Actuation:
    """Motor bookkeeping, restricted to the physically real actuators.

    Unlike Cassie's MJCF, all gears here are 1.0, so `data.ctrl` *is* joint
    torque directly. Torque limits are applied from TORQUE_LIMIT because the
    model declares none.
    """

    def __init__(self, model, include_arms: bool = True, real_only: bool = True):
        self.model = model
        self.include_arms = include_arms

        wanted = []
        for i in range(model.nu):
            jid = model.actuator_trnid[i, 0]
            jname = _name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if real_only and any(f in jname for f in FICTITIOUS_ACTUATORS):
                continue
            is_arm = any(a in jname for a in ARM_MOTORS)
            if is_arm and not include_arms:
                continue
            wanted.append(i)

        self.act_ids = np.array(wanted)
        self.joint_ids = np.array([model.actuator_trnid[i, 0] for i in wanted])
        self.names = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joint_ids]
        self.qposadr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dofadr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.gear = np.array([model.actuator_gear[i, 0] for i in wanted])

        self.torque_limit = np.array([self._limit(n) for n in self.names])

        self.B = np.zeros((model.nv, len(wanted)))
        for k, dof in enumerate(self.dofadr):
            self.B[dof, k] = 1.0

    @staticmethod
    def _limit(joint_name):
        for key, val in TORQUE_LIMIT.items():
            if key in joint_name:
                return val
        return 50.0

    @property
    def n(self):
        return len(self.act_ids)

    def apply_torque(self, data, tau):
        """Write torques to the selected actuators, clipped to TORQUE_LIMIT."""
        tau = np.clip(np.asarray(tau, dtype=float), -self.torque_limit, self.torque_limit)
        data.ctrl[:] = 0.0
        data.ctrl[self.act_ids] = tau / self.gear
        return tau

    def joint_pd(self, data, q_target, kp, kd, qd_target=None):
        q = data.qpos[self.qposadr]
        qd = data.qvel[self.dofadr]
        qd_t = np.zeros_like(qd) if qd_target is None else np.asarray(qd_target)
        return kp * (np.asarray(q_target) - q) + kd * (qd_t - qd)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def equality_rows(model, data):
    """Indices of `efc` rows belonging to equality constraints.

    Cassie's model let us assume every row was an equality once contacts were
    disabled. Digit has active joint limits at typical poses, so the rows must be
    filtered by type -- silently mixing limit rows into a loop-closure residual
    would corrupt any IK built on it.
    """
    n = data.nefc
    if n == 0:
        return np.zeros(0, dtype=int)
    types = np.asarray(data.efc_type[:n])
    return np.flatnonzero(types == mujoco.mjtConstraint.mjCNSTR_EQUALITY)


def measure(model, data, feet: FootGeometry):
    total_mass = float(np.sum(model.body_mass))
    com = np.array(data.subtree_com[0])
    left = feet.contact_point(data, "left")
    right = feet.contact_point(data, "right")
    ground = 0.5 * (left[2] + right[2])
    return {
        "total_mass": total_mass,
        "com": com,
        "com_height_above_feet": float(com[2] - ground),
        "left_foot": left,
        "right_foot": right,
        "foot_separation_y": float(abs(left[1] - right[1])),
        "base_to_com": com - np.array(data.qpos[QPOS_BASE_POS]),
    }


def alip_params(T: float = 0.35, qpos=None):
    """Build an `AlipParams` for Digit from the model and nominal pose."""
    import alip

    model, data = load()
    feet = FootGeometry(model)
    data.qpos[:] = nominal_pose(model, data, feet) if qpos is None else qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    m = measure(model, data, feet)
    return alip.AlipParams(
        m=m["total_mass"],
        H=m["com_height_above_feet"],
        T=T,
        W=m["foot_separation_y"],
    )


# ---------------------------------------------------------------------------
# Nominal standing pose
# ---------------------------------------------------------------------------

_NOMINAL_CACHE = ROOT / "models" / "digit" / "nominal_qpos.npy"


def nominal_pose(model, data, feet: FootGeometry | None = None,
                 com_height: float = 0.80, recompute: bool = False):
    """Standing pose with both feet flat, at nominal width, under the torso.

    `qpos0` splays the legs sideways (feet at y = +-0.40 m, z = 0.21 m), so a
    standing configuration has to be solved for. Rather than hardcode joint
    angles of unknown provenance, this runs the IK from `qpos0` with the foot
    targets walked gradually to the desired standing positions -- a homotopy,
    which avoids needing a good initial guess.

    Cached to models/digit/nominal_qpos.npy since it is deterministic.
    """
    if _NOMINAL_CACHE.exists() and not recompute:
        return np.load(_NOMINAL_CACHE)

    from digit_ik import DigitIK  # imported lazily to avoid a cycle

    ik = DigitIK()
    q = solve_nominal(ik, com_height=com_height)
    np.save(_NOMINAL_CACHE, q)
    return q


MAX_LEG_REACH = 0.9105  # hip_roll origin to sole, straight leg (measured at qpos0)


def _homotopy(ik, base_z, half_width=0.13, n=20, tol=1e-6):
    """Walk the feet from their qpos0 positions to a flat stance, at base height.

    `qpos0` splays the legs out sideways (feet at y = +-0.42 m), which is nowhere
    near a standing pose, so a direct solve has no good initial guess. Stepping the
    targets gradually keeps every intermediate solve well conditioned.
    """
    model = ik.model
    q = model.qpos0.copy()
    ik.data.qpos[:] = q
    mujoco.mj_forward(model, ik.data)
    start = {s: ik.feet.contact_point(ik.data, s).copy() for s in SIDES}
    goal = {"left": np.array([0.0, +half_width, 0.0]),
            "right": np.array([0.0, -half_width, 0.0])}
    base_z0 = float(model.qpos0[2])

    res = None
    for k in range(1, n + 1):
        a = k / n
        targets = {s: (1 - a) * start[s] + a * goal[s] for s in SIDES}
        base = np.array([0.0, 0.0, (1 - a) * base_z0 + a * base_z])
        res = ik.solve(q, targets, base_pos=base, base_quat=(1, 0, 0, 0),
                       flat_feet=True, max_iter=150, tol=tol)
        q = res.qpos
    return q, res


def solve_nominal(ik, com_height=0.80, half_width=0.13, n_homotopy=20,
                  verbose=False, iters=4):
    """Standing pose whose CoM sits `com_height` above the soles.

    The base height that produces a given CoM height is not known in advance -- the
    offset depends on the leg configuration, so it cannot be read off `qpos0`. A
    couple of fixed-point iterations converge it quickly, since d(comH)/d(base_z)
    is very close to 1.
    """
    base_z = com_height + 0.02  # first guess; refined below
    q = res = None
    for i in range(iters):
        q, res = _homotopy(ik, base_z, half_width, n_homotopy)
        ik.data.qpos[:] = q
        mujoco.mj_forward(ik.model, ik.data)
        ground = 0.5 * sum(ik.feet.contact_point(ik.data, s)[2] for s in SIDES)
        comH = float(ik.data.subtree_com[0][2]) - ground
        err = com_height - comH
        if verbose:
            print(f"  pass {i+1}: base_z {base_z:.4f} -> comH {comH:.4f} "
                  f"(target {com_height:.4f}, err {err*1000:+.2f} mm)  "
                  f"task {res.task_error*1e3:.5f} mm  closure {res.closure_error*1e3:.5f} mm")
        if abs(err) < 1e-5:
            break
        base_z += err
    return q


# ---------------------------------------------------------------------------


def report():
    model, data = load()
    feet = FootGeometry(model)
    act_all = Actuation(model, real_only=False)
    act = Actuation(model)
    act_legs = Actuation(model, include_arms=False)

    print(f"model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} "
          f"neq={model.neq} nsite={model.nsite}")
    print(f"total mass = {np.sum(model.body_mass):.3f} kg   "
          f"(torso alone {model.body_mass[1]:.2f} kg)")
    print()

    print("--- actuators ---")
    print(f"  in MJCF                : {act_all.n}")
    print(f"  physically real         : {act.n}   (excludes {', '.join(FICTITIOUS_ACTUATORS)})")
    print(f"  legs only               : {act_legs.n}")
    print(f"  all gears == 1.0        : {np.allclose(act_all.gear, 1.0)}  -> ctrl IS torque")
    print(f"  ctrllimited in model    : {bool(np.any(model.actuator_ctrllimited))}"
          "   <-- False means the model imposes NO torque limit")
    print()
    for nm, lim in zip(act.names, act.torque_limit):
        print(f"    {nm:<28} limit {lim:>6.1f} Nm")
    print()

    print("--- springs ---")
    stiff = [i for i in range(model.njnt) if model.jnt_stiffness[i] > 0]
    print(f"  joints with stiffness > 0: {len(stiff)}"
          f"{'  -> no series compliance, unlike Cassie' if not stiff else ''}")
    print()

    print("--- foot geometry ---")
    dims = feet.dimensions()
    print(f"  sole box: {dims['length']*100:.1f} cm long x {dims['width']*100:.1f} cm wide"
          f"   (long axis = local {dims['long_axis']})")
    print(f"  support polygon area ~ {dims['length']*dims['width']*1e4:.0f} cm^2"
          "   (Cassie's blade foot has none)")
    print()

    print("--- loop closure ---")
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    rows = equality_rows(model, data)
    print(f"  equality constraints: {model.neq}  -> {len(rows)} efc rows")
    print(f"  total efc rows at qpos0: {data.nefc}  "
          f"({data.nefc - len(rows)} are limits/contacts, so rows MUST be filtered)")
    viol = np.max(np.abs(np.asarray(data.efc_pos)[rows])) if len(rows) else 0.0
    print(f"  violation at qpos0: {viol*1000:.6f} mm")
    print()

    print("--- nominal standing pose ---")
    q = nominal_pose(model, data, feet)
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    m = measure(model, data, feet)
    import alip

    p = alip.AlipParams(m=m["total_mass"], H=m["com_height_above_feet"],
                        W=m["foot_separation_y"])
    print(f"  m = {p.m:.4f} kg")
    print(f"  H = {p.H:.4f} m  (CoM above foot contact)")
    print(f"  W = {p.W:.4f} m")
    print(f"  omega = {p.omega:.4f} rad/s")
    print(f"  base pos {np.array2string(np.array(q[QPOS_BASE_POS]), precision=4)}")
    print(f"  CoM      {np.array2string(m['com'], precision=4)}")
    print(f"  base->CoM offset {np.array2string(m['base_to_com'], precision=4)}")
    print(f"  left foot  {np.array2string(m['left_foot'], precision=4)}")
    print(f"  right foot {np.array2string(m['right_foot'], precision=4)}")
    rows = equality_rows(model, data)
    print(f"  closure violation at nominal pose: "
          f"{np.max(np.abs(np.asarray(data.efc_pos)[rows]))*1000:.6f} mm")
    return p


if __name__ == "__main__":
    report()
