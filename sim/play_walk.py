"""Play the solved ALIP walking trajectory in the real MuJoCo viewer.

This is kinematic playback: qpos is written directly each frame, so there is no
physics. Cassie cannot fall over here, and she cannot slip -- the feet are exactly
where the planner put them. Contact forces come in the next stage.

macOS needs mjpython for the interactive viewer:
    source .venv/bin/activate
    mjpython sim/play_walk.py

Options:
    --loop N     repeat the trajectory N times (default 3)
    --speed S    playback rate, 1.0 = real time (default 1.0)
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

MODEL = ROOT / "cassie-mujoco-sim" / "model" / "cassie.xml"
QPOS = ROOT / "out" / "walk_qpos.npy"
DT = 1.0 / 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    if not QPOS.exists():
        sys.exit(f"{QPOS} not found. Run: python sim/alip_animate.py")

    qpos = np.load(QPOS)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)

    print(f"{qpos.shape[0]} frames, {qpos.shape[0] * DT:.2f} s per loop")
    print("kinematic playback -- no physics, no contact forces")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Track the pelvis so she does not walk out of frame.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "cassie-pelvis"
        )
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -15

        for _ in range(args.loop):
            t_start = time.time()
            for i in range(len(qpos)):
                if not viewer.is_running():
                    return
                data.qpos[:] = qpos[i]
                data.qvel[:] = 0.0
                # Kinematics only: no mj_step, so nothing integrates or collides.
                mujoco.mj_forward(model, data)
                viewer.sync()

                target = t_start + (i + 1) * DT / args.speed
                sleep = target - time.time()
                if sleep > 0:
                    time.sleep(sleep)


if __name__ == "__main__":
    main()
