"""Record MuJoCo video of Digit walking — flat, uphill and downhill.

Renders offscreen with mujoco.Renderer (needs a working GL stack, which macOS has)
and writes mp4. Uses the coloured model and a real tilted ramp, so the slope reads
as a slope rather than as a level floor.

    pip install imageio imageio-ffmpeg

    python scripts/record_video.py --all --gif        # flat, +22, -22 as GIFs (half speed)
    python scripts/record_video.py --all --gif --slowmo 3   # slower still
    python scripts/record_video.py --all --gif --slowmo 1   # real time
    python scripts/record_video.py --all              # same, as mp4
    python scripts/record_video.py --all --gif --side # locked side view (slope reads better)

Outputs out/digit_flat.gif, digit_up22.gif, digit_down22.gif (or .mp4 without --gif).
GIFs are downscaled and frame-dropped, since a full-rate GIF is enormous.

Note: this does NOT need mjpython — offscreen rendering has no window, so plain
`python` is correct here.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from play_digit_ramp import make_ramp_xml, rotate_trajectory  # noqa: E402

OUT = ROOT / "out"
DT = 1.0 / 100.0

JOBS = [
    ("flat",   0.0, "digit_walk_long_qpos.npy"),
    ("up22",  22.0, "digit_stageC_up22_qpos.npy"),
    ("down22", -22.0, "digit_stageC_down22_qpos.npy"),
]


def record(name, slope_deg, qpos_file, width=1280, height=720, fps=50,
           side_view=False, distance=4.0, elevation=-10.0, azimuth=90.0,
           gif=False, max_seconds=None, slowmo=1.0):
    """`fps` sets how densely the trajectory is sampled; `slowmo` sets how slowly it
    plays back. They are separate on purpose - lowering fps alone just drops frames
    and the motion still runs at real time, only choppier."""
    path = OUT / qpos_file
    if not path.exists():
        print(f"  SKIP {name}: {qpos_file} not found")
        return None

    xml = make_ramp_xml(slope_deg)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    qpos = np.load(path)
    if slope_deg:
        qpos = rotate_trajectory(qpos, slope_deg)

    # Render every Nth frame so playback lands at `fps`.
    stride = max(1, int(round((1.0 / DT) / fps)))
    n = len(qpos) if max_seconds is None else min(len(qpos), int(max_seconds / DT))
    idx = range(0, n, stride)

    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    if side_view:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = torso
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = []
    with mujoco.Renderer(model, height, width) as r:
        for i in idx:
            data.qpos[:] = qpos[i]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if side_view:
                # Follow along the walk, but keep the horizon fixed.
                cam.lookat[:] = [data.xpos[torso][0], 0.0, data.xpos[torso][2] - 0.2]
            r.update_scene(data, camera=cam, scene_option=opt)
            frames.append(r.render())

    import imageio.v2 as imageio

    play_fps = fps / slowmo

    if gif:
        out = OUT / f"digit_{name}.gif"
        # imageio wants milliseconds per frame here.
        imageio.mimwrite(out, frames, duration=1000.0 / play_fps, loop=0)
        mb = out.stat().st_size / 1e6
        print(f"  wrote {out}  ({len(frames)} frames, {len(frames)/play_fps:.1f} s "
              f"at {slowmo:g}x slow-motion, {mb:.1f} MB)")
        if mb > 12:
            print("  large for a slide - try --gif-width 640 --fps 20")
        return out

    out = OUT / f"video_{name}.mp4"
    try:
        imageio.mimwrite(out, frames, fps=play_fps, quality=8, macro_block_size=1)
    except Exception as exc:                      # noqa: BLE001
        png_dir = OUT / f"frames_{name}"
        png_dir.mkdir(exist_ok=True)
        for k, f in enumerate(frames):
            imageio.imwrite(png_dir / f"f{k:05d}.png", f)
        print(f"  mp4 write failed ({exc}); wrote PNGs to {png_dir}")
        print(f"  make the video with:\n"
              f"    ffmpeg -framerate {play_fps:g} -i {png_dir}/f%05d.png "
              f"-pix_fmt yuv420p {out}")
        return png_dir

    print(f"  wrote {out}  ({len(frames)} frames, {len(frames)/play_fps:.1f} s "
          f"at {slowmo:g}x slow-motion)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slope-deg", type=float, default=None)
    ap.add_argument("--qpos", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--side", action="store_true", help="locked-horizon side view")
    ap.add_argument("--gif", action="store_true", help="write GIF instead of mp4")
    ap.add_argument("--gif-width", type=int, default=800,
                    help="render width when --gif (keeps file size sane)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="trim to the first N seconds of simulated time")
    ap.add_argument("--slowmo", type=float, default=2.0,
                    help="playback slowdown; 1 = real time, 2 = half speed")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-10.0)
    ap.add_argument("--distance", type=float, default=4.0)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    jobs = JOBS if args.all or args.slope_deg is None else [
        (args.name or f"slope{args.slope_deg:+.0f}", args.slope_deg,
         args.qpos or "digit_walk_long_qpos.npy")]

    # GIFs default to smaller and slower: a 50 fps 1280px GIF is hundreds of MB.
    w = args.gif_width if args.gif else args.width
    h = int(round(w * args.height / args.width))
    fps = 20 if (args.gif and args.fps == 50) else args.fps

    for name, deg, f in jobs:
        print(f"{name} ({deg:+.0f}°):")
        record(name, deg, f, width=w, height=h, fps=fps,
               side_view=args.side, distance=args.distance,
               elevation=args.elevation, azimuth=args.azimuth,
               gif=args.gif, max_seconds=args.seconds, slowmo=args.slowmo)
