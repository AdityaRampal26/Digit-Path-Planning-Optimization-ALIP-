# src/

The controller. Scripts in `scripts/` import these modules.

| File | What it does |
|---|---|
| `alip.py` | The ALIP model and footstep math, including the slope extension |
| `walk_controller.py` | The walking loop: plans each step, swings the foot, detects touchdown, switches legs |
| `osc.py` | The whole-body controller: the QP that turns goals into joint torques |
| `dynamics.py` | Pulls the robot's equations of motion out of MuJoCo for the QP |
| `digit_params.py` | Digit's measurements: mass, foot shape, motor limits, standing pose |
| `digit_state.py` | Works out the ALIP state from the full simulation, and senses which foot is on the ground |
| `digit_ik.py` | Inverse kinematics: finds joint angles that put a foot at a given spot |
| `digit_visual.py` | Recolors the model for nicer videos |
| `gait.py` | Turns an ALIP plan into a full-body reference motion, used for the pre-planned playbacks |
| `cassie_*.py` | The earlier stage of the project, on Agility's Cassie robot |

**Where to start reading:** `alip.py` for the idea, then `walk_controller.py` for how it's used every tick.
