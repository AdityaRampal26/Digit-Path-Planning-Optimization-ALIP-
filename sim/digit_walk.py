"""Stage B gate: closed-loop ALIP walking on Digit under real physics.

Nothing here is precomputed. Every tick the controller measures angular momentum
about the stance contact point, plans a footstep, and steers the swing foot at it.

Run:  python sim/digit_walk.py [--seconds 6] [--speed 0.4]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import alip  # noqa: E402
import digit_params as dp  # noqa: E402
import digit_state as ds  # noqa: E402
from osc import OSC, OSCConfig  # noqa: E402
from walk_controller import WalkCommand, WalkController  # noqa: E402

OUT = ROOT / "out"
SINK = 0.001
LOG_HZ = 100.0


def run(seconds=6.0, control_hz=500.0, speed=0.0, ramp_after=1.0, ramp_time=1.5,
        step_T=0.35, swing_height=0.08, slope_deg=0.0, verbose=True):
    model, data = dp.load()
    feet = dp.FootGeometry(model)
    q_nom = dp.nominal_pose(model, data, feet)

    # Slope by rotating GRAVITY rather than tilting the floor.
    #
    # A uniform gravity field rotated against a flat plane is exactly equivalent to
    # a plane tilted against vertical gravity -- it is the same problem in a rotated
    # frame. Doing it this way means the world frame *is* the slope frame, so the
    # nominal pose, the flat-foot orientation target (surface normal = world +z),
    # and all the contact geometry carry over with no changes. Only the rendering
    # has to know, since she will look like she is on the flat.
    slope = np.radians(slope_deg)
    g = 9.806
    model.opt.gravity[:] = [-g * np.sin(slope), 0.0, -g * np.cos(slope)]
    mu = float(model.geom_friction[0][0])
    if abs(np.tan(slope)) >= mu:
        raise ValueError(
            f"grade {slope_deg} deg needs mu > {abs(np.tan(slope)):.3f}, "
            f"floor has {mu:.3f} -- she would simply slide")

    data.qpos[:] = q_nom
    data.qpos[2] -= SINK
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    feet.calibrate(data)

    meas = dp.measure(model, data, feet)
    params = alip.AlipParams(m=meas["total_mass"], H=meas["com_height_above_feet"],
                             T=step_T, W=meas["foot_separation_y"], g=g, slope=slope)
    # Suppress the stance ankle so the plant matches the ankle-free ALIP model the
    # planner assumes -- without this the deadbeat foot placement under-corrects and
    # she drifts backwards. See OSCConfig.w_stance_ankle.
    cfg = OSCConfig()
    cfg.w_stance_ankle = 20.0
    osc = OSC(model, feet, config=cfg)
    cmd = WalkCommand(v_forward=0.0, swing_height=swing_height)

    # Start on the SLOPE's periodic orbit, not the flat one.
    #
    # The walking-in-place orbit on a grade has the CoM sitting x* = H tan(gamma)
    # uphill of the stance foot with zero momentum. Starting from the flat-ground
    # nominal pose puts her x* off equilibrium -- 14 cm at 10 degrees -- and she
    # slides ~0.5 m downhill acquiring the lean before the gait settles.
    #
    # Worth noting: at 10 degrees x* (14.1 cm) already exceeds the foot half-length
    # (11.75 cm), so the equilibrium lies outside a feet-together support polygon.
    # Digit cannot stand still on that grade with level feet; she must stagger or
    # keep stepping.
    #
    # NB the lean has to be built by IK. Translating qpos[0] moves the whole robot,
    # feet included, leaving the CoM-to-foot offset unchanged -- it does nothing.
    # The base must move while the feet stay planted, i.e. she actually leans.
    if slope:
        from digit_ik import DigitIK

        ik = DigitIK()
        ik.data.qpos[:] = data.qpos
        mujoco.mj_forward(ik.model, ik.data)
        ik.feet.calibrate(ik.data)
        targets = {s: ik.feet.contact_point(ik.data, s).copy() for s in dp.SIDES}
        base = np.array(data.qpos[0:3])
        base[0] += params.x_equilibrium
        res = ik.solve(np.array(data.qpos), targets, base_pos=base,
                       base_quat=(1, 0, 0, 0), flat_feet=True,
                       yaw_target={"left": 0.0, "right": 0.0}, max_iter=200, tol=1e-6)
        if res.converged:
            data.qpos[:] = res.qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        elif verbose:
            print(f"  WARNING: could not build the {params.x_equilibrium*100:.1f} cm "
                  f"uphill lean (task err {res.task_error*1e3:.2f} mm); "
                  "starting off-equilibrium")

    ctl = WalkController(model, feet, osc, params, cmd)
    ctl.reset(data, stance_is_left=True)

    dt = model.opt.timestep
    n_steps = int(seconds / dt)
    ctrl_every = max(1, int(round(1.0 / (control_hz * dt))))
    log_every = max(1, int(round(1.0 / (LOG_HZ * dt))))
    weight = ds.total_mass(model) * 9.806

    if verbose:
        print(f"Digit ALIP walking: m={params.m:.2f} kg H={params.H:.4f} "
              f"W={params.W:.4f} T={params.T} omega={params.omega:.3f}")
        if slope_deg:
            print(f"slope: {slope_deg:+.1f} deg ({params.grade_percent:+.1f}% grade)  "
                  f"{'UPHILL' if slope_deg > 0 else 'DOWNHILL'}")
            print(f"  equilibrium lean = {params.x_equilibrium*100:+.1f} cm uphill of the foot")
            print(f"  constant gravity torque = {params.slope_bias:+.1f} N m")
            print(f"  friction needed {abs(np.tan(slope)):.3f}, available {mu:.3f}")
        print(f"command: 0 m/s, ramping to {speed} m/s over {ramp_time}s after t={ramp_after}s")

    tau = np.zeros(osc.act.n)
    log, qpos_log = [], []
    failures = 0
    fell_at = None

    for step in range(n_steps):
        t = step * dt

        # Speed command schedule.
        if t < ramp_after:
            cmd.v_forward = 0.0
        elif t < ramp_after + ramp_time:
            cmd.v_forward = speed * (t - ramp_after) / ramp_time
        else:
            cmd.v_forward = speed

        if step % ctrl_every == 0:
            tau_new, info = ctl.update(data, t, ctrl_every * dt)
            if tau_new is None or not info["info"]["ok"]:
                failures += 1
            else:
                tau = tau_new
        osc.act.apply_torque(data, tau)

        if step % log_every == 0:
            st = ctl.last_info.get("alip")
            cs = ctl.det.state(data)
            mujoco.mj_subtreeVel(model, data)
            log.append(dict(
                t=t,
                com=np.array(data.subtree_com[0]),
                vx=float(data.subtree_linvel[0][0]),
                vy=float(data.subtree_linvel[0][1]),
                com_z=float(data.subtree_com[0][2]),
                base_z=float(data.qpos[2]),
                Lx=st.Lx if st else 0.0, Ly=st.Ly if st else 0.0,
                x=st.x if st else 0.0, y=st.y if st else 0.0,
                stance=ctl.state.stance, step=ctl.state.step_index,
                phase=ctl.last_info.get("phase", 0.0),
                tilt=float(np.degrees(np.arccos(np.clip(
                    np.array(data.xmat[osc.torso_id]).reshape(3, 3)[2, 2], -1, 1)))),
                fz=cs.force_left + cs.force_right,
                sat=float(np.max(np.abs(tau) / osc.act.torque_limit)),
                cmd_v=cmd.v_forward,
            ))
            qpos_log.append(data.qpos.copy())

        mujoco.mj_step(model, data)
        if fell_at is None and data.qpos[2] < q_nom[2] - 0.25:
            fell_at = t
            if verbose:
                print(f"  FELL at t={t:.2f}s after {ctl.state.step_index} steps")
            break

    out = dict(log=log, qpos=np.array(qpos_log), failures=failures, fell_at=fell_at,
               steps=ctl.state.step_index, touchdowns=ctl.state.touchdowns,
               params=params, weight=weight)
    if verbose:
        _report(out, seconds)
    return out


def _report(out, seconds):
    log = out["log"]
    print(f"\nsteps taken: {out['steps']}   QP failures: {out['failures']}")
    print(f"{'t':>6} {'stance':>7} {'step':>5} {'comZ':>7} {'vx':>7} {'cmd':>6} "
          f"{'Ly':>8} {'tilt':>6} {'Fz/W':>6} {'sat':>5}")
    for d in log[:: max(1, len(log) // 15)]:
        print(f"{d['t']:6.2f} {d['stance']:>7} {d['step']:>5} {d['com_z']:7.4f} "
              f"{d['vx']:7.3f} {d['cmd_v']:6.2f} {d['Ly']:8.2f} {d['tilt']:6.2f} "
              f"{d['fz']/out['weight']:6.2f} {d['sat']:5.2f}")

    if out["touchdowns"]:
        tds = out["touchdowns"]
        dts = np.diff([td["t"] for td in tds]) if len(tds) > 1 else np.array([np.nan])
        xs = [td["landed"][0] for td in tds]
        print(f"\nstep durations: mean {np.nanmean(dts):.3f} s "
              f"(commanded T={out['params'].T})")
        if len(xs) > 2:
            print(f"step lengths  : mean {np.mean(np.diff(xs)):.4f} m")

    com = np.array([d["com"] for d in log])
    vx = np.array([d["vx"] for d in log])
    tilt = np.array([d["tilt"] for d in log])
    print(f"\ndistance travelled : {com[-1,0]-com[0,0]:+.3f} m")
    print(f"CoM height         : {com[:,2].min():.4f} .. {com[:,2].max():.4f} m")
    print(f"max torso tilt     : {tilt.max():.2f} deg")
    print(f"max torque satur.  : {max(d['sat'] for d in log)*100:.0f}%")
    cruise = [d for d in log if d["cmd_v"] > 0.05]
    if cruise:
        cv = np.mean([d["cmd_v"] for d in cruise[len(cruise)//2:]])
        mv = np.mean([d["vx"] for d in cruise[len(cruise)//2:]])
        print(f"cruise speed       : commanded {cv:.3f}, measured {mv:.3f} m/s")

    if out["fell_at"] is not None:
        print(f"\nVERDICT: FELL at t={out['fell_at']:.2f} s after {out['steps']} steps")
    else:
        print(f"\nVERDICT: WALKED {out['steps']} steps for {seconds:.0f} s without falling")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--control-hz", type=float, default=500.0)
    ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--step-T", type=float, default=0.35)
    ap.add_argument("--swing-height", type=float, default=0.08)
    ap.add_argument("--slope-deg", type=float, default=0.0,
                    help="positive = uphill, negative = downhill")
    ap.add_argument("--out", default="digit_walk_physics_qpos.npy")
    args = ap.parse_args()

    out = run(args.seconds, args.control_hz, args.speed, step_T=args.step_T,
              swing_height=args.swing_height, slope_deg=args.slope_deg)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / args.out, out["qpos"])
    print(f"\nwrote {OUT / args.out}  shape {out['qpos'].shape}")
