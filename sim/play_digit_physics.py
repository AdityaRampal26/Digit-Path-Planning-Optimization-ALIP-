"""Watch the ALIP walk in MuJoCo's own renderer -- live physics, not playback.

Unlike sim/play_digit.py (which replays a precomputed qpos trajectory with
mj_forward and no dynamics), this runs the actual closed loop inside the viewer:
mj_step every tick, torques from the OSC, footsteps planned from measured angular
momentum. What you see is the controller working, and she can fall.

macOS needs mjpython for the interactive viewer:
    source .venv/bin/activate
    mjpython sim/play_digit_physics.py --speed 0.5

Keys/mouse are the normal MuJoCo viewer controls. Options:
    --speed S       commanded forward speed [m/s]
    --control-hz H  controller rate (default 500)
    --realtime      throttle to wall clock (off by default: the QP is slower than
                    real time in Python, so it runs as fast as it can)
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

import alip  # noqa: E402
import digit_params as dp  # noqa: E402
from osc import OSC, OSCConfig  # noqa: E402
from walk_controller import WalkCommand, WalkController  # noqa: E402

SINK = 0.001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--control-hz", type=float, default=500.0)
    ap.add_argument("--step-T", type=float, default=0.35)
    ap.add_argument("--swing-height", type=float, default=0.07)
    ap.add_argument("--ramp-after", type=float, default=1.0)
    ap.add_argument("--ramp-time", type=float, default=1.5)
    ap.add_argument("--slope-deg", type=float, default=0.0,
                    help="positive = uphill, negative = downhill")
    ap.add_argument("--realtime", action="store_true")
    args = ap.parse_args()

    model, data = dp.load(colored=True)
    feet = dp.FootGeometry(model)
    q_nom = dp.nominal_pose(model, data, feet)
    data.qpos[:] = q_nom
    data.qpos[2] -= SINK          # tangent soles generate no contact force
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    feet.calibrate(data)

    # Slope via rotated gravity (equivalent to tilting the floor, and it leaves the
    # nominal pose and the flat-foot target untouched). Note the viewer will still
    # draw a level floor -- she is walking on a hill in a rotated frame.
    slope = np.radians(args.slope_deg)
    g = 9.806
    model.opt.gravity[:] = [-g * np.sin(slope), 0.0, -g * np.cos(slope)]
    mu = float(model.geom_friction[0][0])
    if abs(np.tan(slope)) >= mu:
        sys.exit(f"grade {args.slope_deg} deg needs mu > {abs(np.tan(slope)):.3f}, "
                 f"floor has {mu:.3f}")

    meas = dp.measure(model, data, feet)
    params = alip.AlipParams(m=meas["total_mass"], H=meas["com_height_above_feet"],
                             T=args.step_T, W=meas["foot_separation_y"],
                             g=g, slope=slope)

    cfg = OSCConfig()
    cfg.w_stance_ankle = 20.0     # keep the plant matched to the ankle-free model
    osc = OSC(model, feet, config=cfg)
    cmd = WalkCommand(v_forward=0.0, swing_height=args.swing_height)
    ctl = WalkController(model, feet, osc, params, cmd)
    ctl.reset(data, stance_is_left=True)

    dt = model.opt.timestep
    ctrl_every = max(1, int(round(1.0 / (args.control_hz * dt))))
    tau = np.zeros(osc.act.n)

    print(f"Digit ALIP walk, live physics.  m={params.m:.2f} kg  H={params.H:.4f} m  "
          f"omega={params.omega:.3f}")
    if args.slope_deg:
        print(f"slope {args.slope_deg:+.1f} deg ({params.grade_percent:+.1f}% grade), "
              f"lean {params.x_equilibrium*100:+.1f} cm, bias {params.slope_bias:+.1f} N m")
        print("  (viewer draws a level floor: gravity is tilted instead)")
    print(f"commanding {args.speed} m/s after t={args.ramp_after}s")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -12

        step = 0
        t_wall = time.time()
        while viewer.is_running():
            t = step * dt
            if t < args.ramp_after:
                cmd.v_forward = 0.0
            elif t < args.ramp_after + args.ramp_time:
                cmd.v_forward = args.speed * (t - args.ramp_after) / args.ramp_time
            else:
                cmd.v_forward = args.speed

            if step % ctrl_every == 0:
                tau_new, info = ctl.update(data, t, ctrl_every * dt)
                if tau_new is not None and info["info"]["ok"]:
                    tau = tau_new
            osc.act.apply_torque(data, tau)
            mujoco.mj_step(model, data)

            if step % 20 == 0:
                viewer.sync()
            if step % 1000 == 0:
                mujoco.mj_subtreeVel(model, data)
                print(f"t={t:6.2f}s  step {ctl.state.step_index:>3}  "
                      f"x={data.subtree_com[0][0]:+.3f} m  "
                      f"vx={data.subtree_linvel[0][0]:+.3f} m/s  "
                      f"cmd={cmd.v_forward:.2f}  v_bias={ctl.v_bias:+.3f}")

            if data.qpos[2] < q_nom[2] - 0.25:
                print(f"\nFELL at t={t:.2f}s after {ctl.state.step_index} steps")
                break

            if args.realtime:
                target = t_wall + (step + 1) * dt
                sleep = target - time.time()
                if sleep > 0:
                    time.sleep(sleep)
            step += 1


if __name__ == "__main__":
    main()
