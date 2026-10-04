"""Step 1: ground the ALIP parameters in Cassie's actual model.

`AlipParams` defaults were guessed (m=31, H=0.85). Everything downstream scales
off mass and CoM height, so pull the real numbers out of cassie.xml.

Also records the qpos layout, which the IK and animation stages both need.

Run:  python src/cassie_params.py
"""

from __future__ import annotations

import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cassie-mujoco-sim" / "model" / "cassie.xml"

# Floating base: 3 slide joints (x, y, z) then a ball joint (quaternion).
QPOS_PELVIS_POS = slice(0, 3)
QPOS_PELVIS_QUAT = slice(3, 7)

# Actuated joint names, in actuator order.
MOTOR_JOINTS = (
    "hip-roll",
    "hip-yaw",
    "hip-pitch",
    "knee",
    "foot",
)

# Nominal standing pose from cassie-mujoco-sim/src/cassiemujoco.c (qpos_init).
QPOS_INIT = np.array(
    [
        0, 0, 1.01, 1, 0, 0, 0,
        0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
        -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
        -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
        -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
    ],
    dtype=float,
)


def load():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def nominal_pose(model, data):
    """Set the documented standing pose and run forward kinematics."""
    data.qpos[:] = QPOS_INIT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def site_xpos(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"no site named {name!r}")
    return np.array(data.site_xpos[sid])


def foot_center(model, data, side):
    """Midpoint of the toe and heel contact sites -- the effective contact point."""
    return 0.5 * (site_xpos(model, data, f"{side}-toe") + site_xpos(model, data, f"{side}-heel"))


def measure(model, data):
    total_mass = float(np.sum(model.body_mass))
    com = np.array(data.subtree_com[0])  # whole-model CoM

    left = foot_center(model, data, "left")
    right = foot_center(model, data, "right")
    ground_z = 0.5 * (left[2] + right[2])

    # Hip roll joint origins set the nominal lateral foot spacing.
    hip_width = abs(left[1] - right[1])

    return {
        "total_mass": total_mass,
        "com": com,
        "com_height_above_feet": float(com[2] - ground_z),
        "left_foot": left,
        "right_foot": right,
        "foot_separation_y": hip_width,
        "pelvis_to_com": com - data.qpos[QPOS_PELVIS_POS],
    }


class Actuation:
    """Index bookkeeping for the 10 motors, shared by every controller.

    cassie.xml uses `motor` actuators with a gear ratio, so the joint torque is
    `gear * ctrl` and `ctrlrange` is in pre-gear units. Anything computing
    torques needs to divide by gear before writing to `data.ctrl`, and the torque
    limit is `gear * ctrlrange`.
    """

    def __init__(self, model):
        self.model = model
        jids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
        self.joint_ids = np.array(jids)
        self.names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids
        ]
        self.qposadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dofadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.gear = np.array(model.actuator_gear[:, 0])
        self.ctrlrange = np.array(model.actuator_ctrlrange)
        self.torque_limit = self.gear * self.ctrlrange[:, 1]

        # B matrix: maps motor torques to generalised forces.
        self.B = np.zeros((model.nv, model.nu))
        for i, dof in enumerate(self.dofadr):
            self.B[dof, i] = 1.0

    def apply_torque(self, data, tau):
        """Write joint torques [Nm] to data.ctrl, clipped to the actuator limits."""
        u = np.clip(np.asarray(tau) / self.gear, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        data.ctrl[:] = u
        return u * self.gear  # the torque actually applied

    def joint_pd(self, data, q_target, kp, kd, qd_target=None):
        """Simple joint-space PD. Useful as a baseline and for posture terms."""
        q = data.qpos[self.qposadr]
        qd = data.qvel[self.dofadr]
        qd_t = np.zeros_like(qd) if qd_target is None else np.asarray(qd_target)
        return kp * (np.asarray(q_target) - q) + kd * (qd_t - qd)


def alip_params(T: float = 0.35):
    """Build an `AlipParams` from the model rather than from hardcoded guesses."""
    import alip  # local import so this module stays usable without alip.py

    model, data = load()
    nominal_pose(model, data)
    m = measure(model, data)
    return alip.AlipParams(
        m=m["total_mass"],
        H=m["com_height_above_feet"],
        T=T,
        W=m["foot_separation_y"],
    )


def report():
    model, data = load()
    nominal_pose(model, data)
    m = measure(model, data)

    print(f"model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")
    print()
    print("--- ALIP parameters, measured at the nominal standing pose ---")
    print(f"total mass          m = {m['total_mass']:.4f} kg")
    print(f"CoM height          H = {m['com_height_above_feet']:.4f} m  (above foot contact)")
    print(f"foot separation     W = {m['foot_separation_y']:.4f} m")
    w = np.sqrt(9.81 / m["com_height_above_feet"])
    print(f"=> omega = sqrt(g/H)  = {w:.4f} rad/s")
    print()
    print("--- geometry ---")
    print(f"CoM (world)            {np.array2string(m['com'], precision=4)}")
    print(f"CoM relative to pelvis {np.array2string(m['pelvis_to_com'], precision=4)}")
    print(f"left  foot center      {np.array2string(m['left_foot'], precision=4)}")
    print(f"right foot center      {np.array2string(m['right_foot'], precision=4)}")
    print()
    print("--- actuated joints (actuator order) ---")
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = model.jnt_range[jid]
        gear = model.actuator_gear[i, 0]
        cr = model.actuator_ctrlrange[i]
        print(
            f"  u[{i}] {name:<17} qpos[{model.jnt_qposadr[jid]:>2}] "
            f"range [{np.degrees(lo):>7.1f},{np.degrees(hi):>7.1f}] deg  "
            f"gear {gear:>5.1f}  max torque {gear * cr[1]:>6.1f} Nm"
        )

    print()
    print("--- passive springs and loop closures ---")
    for i in range(model.njnt):
        if model.jnt_stiffness[i] > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  spring  {name:<20} stiffness {model.jnt_stiffness[i]:.0f}")
    for i in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        print(f"  equality {name}")

    return m


if __name__ == "__main__":
    report()
