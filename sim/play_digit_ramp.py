"""Play a slope walk on a REAL tilted ramp in MuJoCo, with gravity pointing down.

Why this file exists
--------------------
The controller runs in the *slope frame*: a level floor with gravity rotated by
gamma. That is exactly equivalent to a tilted floor with vertical gravity -- it is
the same physical problem viewed from a rotated frame -- and it keeps the nominal
pose, the flat-foot target and all the contact geometry unchanged. But it *looks*
level, which is useless for showing someone a robot climbing a hill.

Rotating the entire system (robot + floor + gravity) is a symmetry of the problem,
so the trajectory can be moved into the gravity-vertical frame exactly, with no
approximation:

    R = R_y(-gamma)        maps slope frame -> world frame
    base position  p_W = R p_S
    base orientation q_W = quat(R) * q_S
    joint angles          unchanged

and the floor is given `euler="0 -gamma 0"`, whose normal becomes
(-sin gamma, 0, cos gamma) -- i.e. rising in +x for a positive (uphill) grade.

Usage
-----
    python sim/digit_walk.py --seconds 10 --speed 0.4 --slope-deg 22 \
        --out digit_up22_qpos.npy
    mjpython sim/play_digit_ramp.py --qpos digit_up22_qpos.npy --slope-deg 22

Downhill is the same with a negative angle.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import digit_params as dp  # noqa: E402

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
XML_DIR = ROOT / "digit-model" / "xml"
DT = 1.0 / 100.0
TINT_SIDES = False   # set by --tint-sides


def make_ramp_xml(slope_deg: float, ramp_len: float = 60.0,
                  offwidth: int = 1920, offheight: int = 1080) -> pathlib.Path:
    """Write a model whose floor is a plane tilted by `slope_deg` about y.

    The file is written beside the original so the relative `meshdir` still
    resolves. A positive angle gives ground that rises in +x.
    """
    from digit_visual import colorize

    src = XML_DIR / "digit_patched.xml"
    # Colour the robot: the MJCF paints every mesh the same grey, so she otherwise
    # renders as one undifferentiated mass.
    text = colorize(src.read_text(), tint_sides=TINT_SIDES)

    # MuJoCo's offscreen framebuffer defaults to 640x480, so any render wider than
    # that fails with "Image width > framebuffer width". The model has no <visual>
    # block at all, so one is inserted.
    if "offwidth" not in text:
        text = text.replace(
            "<mujoco model=\"digit\">",
            f'<mujoco model="digit">\n'
            f'       <visual><global offwidth="{offwidth}" offheight="{offheight}"/></visual>',
            1,
        )

    gamma = np.radians(slope_deg)
    old = '<geom name="floor" pos="0 0 0" type="plane" size="0 0 1" material="MatPlane" />'
    if old not in text:
        raise RuntimeError("floor geom not found; the model layout has changed")

    # euler is in radians for this model (compiler angle="radian").
    new = (
        f'<geom name="floor" pos="0 0 0" type="plane" size="{ramp_len} {ramp_len} 1" '
        f'euler="0 {-gamma:.9f} 0" material="MatPlane" />'
    )
    out = XML_DIR / f"digit_ramp_{slope_deg:+.1f}.xml".replace("+", "p").replace("-", "m")
    out.write_text(text.replace(old, new))
    return out


def rotate_trajectory(qpos: np.ndarray, slope_deg: float) -> np.ndarray:
    """Move a slope-frame trajectory into the gravity-vertical frame."""
    gamma = np.radians(slope_deg)
    c, s = np.cos(-gamma), np.sin(-gamma)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    q_rot = np.zeros(4)
    mujoco.mju_mat2Quat(q_rot, R.flatten())

    out = qpos.copy()
    out[:, 0:3] = qpos[:, 0:3] @ R.T
    for i in range(len(qpos)):
        res = np.zeros(4)
        mujoco.mju_mulQuat(res, q_rot, qpos[i, 3:7])
        out[i, 3:7] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", default="digit_slope_up10_qpos.npy")
    ap.add_argument("--slope-deg", type=float, required=True)
    ap.add_argument("--loop", type=int, default=4)
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate")
    ap.add_argument("--tint-sides", action="store_true",
                    help="warm the left leg, cool the right, to follow the gait")
    args = ap.parse_args()

    global TINT_SIDES
    TINT_SIDES = args.tint_sides

    path = OUT / args.qpos
    if not path.exists():
        sys.exit(f"{path} not found. Generate it with:\n"
                 f"  python sim/digit_walk.py --seconds 10 --speed 0.4 "
                 f"--slope-deg {args.slope_deg} --out {args.qpos}")

    qpos = rotate_trajectory(np.load(path), args.slope_deg)
    xml = make_ramp_xml(args.slope_deg)
    print(f"ramp model: {xml.name}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    grade = 100.0 * np.tan(np.radians(args.slope_deg))
    print(f"{qpos.shape[0]} frames, {qpos.shape[0] * DT:.1f} s per loop")
    print(f"slope {args.slope_deg:+.1f} deg ({grade:+.1f}% grade), "
          f"{'UPHILL' if args.slope_deg > 0 else 'DOWNHILL'}")
    print("kinematic playback of a physics run -- gravity is vertical, ground is tilted")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -8

        for _ in range(args.loop):
            t0 = time.time()
            for i in range(len(qpos)):
                if not viewer.is_running():
                    return
                data.qpos[:] = qpos[i]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                sleep = t0 + (i + 1) * DT / args.speed - time.time()
                if sleep > 0:
                    time.sleep(sleep)


if __name__ == "__main__":
    main()
