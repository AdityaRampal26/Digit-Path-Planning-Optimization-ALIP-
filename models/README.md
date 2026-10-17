# models/

The simulated robot: everything MuJoCo needs to know about Digit's shape, mass and motors.

| File | What it is |
|---|---|
| `digit/xml/digit_patched.xml` | The Digit model the controller loads: every body, joint, motor, mass and collision shape, in MuJoCo's XML format (MJCF) |
| `digit/xml/digit.xml` | The original model file it was made from |
| `digit/xml/digit_colored.xml` | Same robot with a cleaner color scheme, for videos (made by `src/digit_visual.py`) |
| `digit/xml/digit_ramp_*.xml` | Digit placed on a real tilted ramp (−22°, 0°, +22°, +50°), for the slope videos |
| `digit/xml/meshes/` | 28 3D part shapes (`.stl`): shins, hips, feet, arms |
| `digit/nominal_qpos.npy` | The saved standing pose every run starts from |

The model has 50 degrees of freedom and 26 actuators. Its feet are real boxes, about 8 × 23.5 cm.

`setup.sh` also clones the Cassie model into `cassie-mujoco-sim/` here, for the earlier Cassie stage. It isn't stored in this repo.
