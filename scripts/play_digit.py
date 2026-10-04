"""Play the Digit ALIP walking trajectory in the MuJoCo viewer.

Kinematic playback: qpos is written directly, so there is no physics -- no contact
forces, no torques, she cannot fall and cannot slip.

macOS needs mjpython for the interactive viewer:
    source .venv/bin/activate
    python scripts/digit_animate.py      # writes out/digit_walk_qpos.npy
    mjpython scripts/play_digit.py
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

QPOS = ROOT / "out" / "digit_walk_qpos.npy"
DT = 1.0 / 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    if not QPOS.exists():
        sys.exit(f"{QPOS} not found. Run: python scripts/digit_animate.py")

    qpos = np.load(QPOS)
    model, data = dp.load(colored=True)
    print(f"{qpos.shape[0]} frames, {qpos.shape[0] * DT:.2f} s per loop")
    print("kinematic playback -- no physics, no contact forces")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -12

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
