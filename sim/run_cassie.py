"""Load the Cassie model from cassie-mujoco-sim and open the MuJoCo viewer.

Usage (macOS requires mjpython, not python, for the interactive viewer):
    source .venv/bin/activate
    mjpython sim/run_cassie.py
"""

import pathlib
import sys

import mujoco
import mujoco.viewer

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = ROOT / "cassie-mujoco-sim" / "model" / "cassie.xml"


def main() -> None:
    if not MODEL.exists():
        sys.exit(f"Model not found at {MODEL}\nRun: bash setup.sh")

    # The MJCF uses relative paths for meshes, so load from its own directory.
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)

    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # ALIP controller goes here: read state from `data`, write torques
            # to data.ctrl[:]. Left as zeros for now, so Cassie will collapse.
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
