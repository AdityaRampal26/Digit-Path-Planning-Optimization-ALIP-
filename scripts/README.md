# scripts/

Things you can run, from the repo root, after `bash setup.sh && source .venv/bin/activate`. On macOS, use `mjpython` for anything that opens the viewer.

**Watch it**

| Script | What it does |
|---|---|
| `play_digit_physics.py` | Opens the MuJoCo viewer and lets you watch Digit walk live |
| `play_digit_ramp.py` | Same, on a real tilted ramp |

**Experiments**

| Script | What it does |
|---|---|
| `digit_stand.py` | Can Digit just stand? Includes a push test |
| `digit_one_leg.py` | Can it balance on one leg? |
| `digit_walk.py` | Runs a walk with no window and prints speed, height, tilt and torque stats |
| `sweep_envelope.py` | Tries a range of slopes and scores each run pass or fail |
| `tune_touchdown.py` | Tunes how hard a foot must press before it counts as landed |

**Make the visuals**

| Script | What it does |
|---|---|
| `record_video.py` | Records the MuJoCo GIF at the top of the main page |
| `make_gifs.py` | Makes the ALIP-vs-no-ALIP and three-terrain animations on the main page, as 2D side views and MuJoCo renders |
| `render_digit.py`, `render_still.py`, `render_skeleton.py` | Stills and stick-figure renders, no GPU needed |

**Earlier stages**

| Script | What it does |
|---|---|
| `alip_pointmass.py` | ALIP on a single point mass: the idea with no robot attached |
| `run_cassie.py`, `physics_rollout.py`, `play_walk.py`, `alip_animate.py` | The Cassie stage |
| `digit_animate.py`, `play_digit.py` | Pre-planned (not live) Digit walk playback |
