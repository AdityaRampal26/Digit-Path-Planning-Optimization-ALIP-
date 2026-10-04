"""Sweep the contact-touchdown threshold.

The old timer fired at exactly phase = 1, when the swing foot had just arrived at
ground level with ~0 N on it. A contact trigger therefore has to fire on *first*
contact, not on a fully loaded foot: MuJoCo's soft contacts need real penetration
to build tens of newtons, so a high threshold silently drags the step well past T.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "sim"))

import digit_walk  # noqa: E402
import walk_controller as wc  # noqa: E402

_ORIG = wc.WalkController.__init__


def with_settings(force=None, overtime=None, reach=None, ds=None):
    def patched(self, *a, **k):
        _ORIG(self, *a, **k)
        if force is not None:
            self.touchdown_force = force
        if overtime is not None:
            self.max_overtime = overtime
        if reach is not None:
            self.reach_rate = reach
        if ds is not None:
            self.double_support_time = ds
    wc.WalkController.__init__ = patched


def restore():
    wc.WalkController.__init__ = _ORIG


def trial(deg, seconds=8.0, speed=0.4):
    o = digit_walk.run(seconds=seconds, control_hz=500, speed=speed, ramp_after=0.8,
                       ramp_time=1.2, slope_deg=deg, verbose=False)
    tds = o["touchdowns"]
    ov = np.array([t.get("overtime", 0.0) for t in tds])[3:]
    dts = np.diff([t["t"] for t in tds])[3:] if len(tds) > 4 else np.array([np.nan])
    xs = np.array([d["com"][0] for d in o["log"]])
    late = [d for d in o["log"] if d["t"] > seconds * 0.65]
    sv = ((late[-1]["com"][0] - late[0]["com"][0]) / (late[-1]["t"] - late[0]["t"])
          if len(late) > 2 else float("nan"))
    return dict(
        steps=o["steps"], dist=float(xs[-1] - xs[0]),
        meanT=float(np.mean(dts)) if len(dts) else float("nan"),
        overtime_ms=float(np.mean(ov) * 1000) if len(ov) else float("nan"),
        late_frac=float(np.mean(ov > 0)) if len(ov) else float("nan"),
        speed=sv, fell=o["fell_at"],
    )


if __name__ == "__main__":
    print(f'{"Fthr":>6} {"grade":>7} {"steps":>6} {"dist":>8} {"V":>7} '
          f'{"meanT":>8} {"over_ms":>8} {"late%":>6}  verdict')
    for fthr in (5.0, 15.0, 30.0):
        with_settings(force=fthr)
        for deg in (0.0, -22.0):
            r = trial(deg)
            v = "WALKED" if r["fell"] is None else f'FELL@{r["fell"]:.1f}s'
            print(f'{fthr:>6.0f} {deg:>+6.1f}d {r["steps"]:>6} {r["dist"]:>+8.3f} '
                  f'{r["speed"]:>7.3f} {r["meanT"]:>8.4f} {r["overtime_ms"]:>+8.1f} '
                  f'{100*r["late_frac"]:>5.0f}%  {v}')
        restore()
