#!/usr/bin/env bash
# Setup for ALIP walking on Digit (and the earlier Cassie stage), macOS/Linux
# Run from the repo root:   bash setup.sh
set -euo pipefail

cd "$(dirname "$0")"

# 1. Clone the OSU DRL repo (we use its MJCF model, not its C build system)
if [ ! -d cassie-mujoco-sim ]; then
  git clone https://github.com/osudrl/cassie-mujoco-sim.git
else
  echo "cassie-mujoco-sim already present, skipping clone"
fi

# 2. Python env with the official MuJoCo bindings
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
# mujoco: sim + mjpython launcher
# osqp/scipy: the OSC quadratic program  (full list: requirements.txt)
# matplotlib: skeleton renders (no OpenGL needed)
pip install --quiet -r requirements.txt

echo
echo "Done. Activate first:  source .venv/bin/activate"
echo
echo "Digit (current work):"
echo "  mjpython sim/play_digit_physics.py --speed 0.5   # live closed-loop ALIP walk"
echo "  python   sim/digit_walk.py --seconds 20 --speed 0.5   # headless, with metrics"
echo "  python   sim/digit_stand.py --seconds 30 --push 120   # standing + push recovery"
echo "  python   sim/record_video.py --all --gif                 # GIFs (flat, +22, -22)"
echo
echo "Cassie (earlier stage, kinematic only):"
echo "  python   sim/alip_animate.py && mjpython sim/play_walk.py"
echo
echo "Tests:"
echo "  for t in tests/test_*.py; do python \$t; done"
echo
echo "Note: mjpython (not python) is required for the interactive viewer on macOS."
