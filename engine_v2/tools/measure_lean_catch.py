"""Does a leaning body catch itself with the leg on the leaning side?

    python tools/measure_lean_catch.py

Standing still, one sideways push, and the same push again with the lean
reflex silenced. Reported: how far the upright axis went over, whether the
body ever lost it, and how much of it was back at the end. Nothing here is on
the control path.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.embodied import EmbodiedController      # noqa: E402
from born_wired.go2_body import Go2Body                 # noqa: E402
from born_wired.stereo_senses import RawEyes            # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
CONFIG = ROOT / "live_config.json"
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
STEPS, PUSH_UNTIL, TOTAL = 400, 25, 400


def push(newtons, direction=(0., 1., 0.), hold=STEPS, **extra):
    parameters = dict(json.loads(CONFIG.read_text(encoding="utf-8-sig"))["parameters"])
    parameters.update(extra)
    # The sizes named below are this probe's own; the live window's copies of
    # the same knobs would otherwise be passed twice, which is an error rather
    # than an override.
    for name in ("eye_width", "eye_height", "motor_units", "proprio_units",
                 "association_units"):
        parameters.pop(name, None)
    body = Go2Body(model_path=ARENA)
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    observation = body.reset(seed=0)
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    least, laid, steady = 1., False, []
    for step in range(TOTAL):
        body.set_body_force(np.asarray(direction)*newtons if step < PUSH_UNTIL else np.zeros(3))
        target, activation = brain.step(observation, environment=environment,
                                        eye_pixels=eyes.observe_raw(), dt=.01, learn=False)
        body.command_eyes(brain.eye_command())
        observation = body.step(target, duration=.01, activation=activation)
        upright = float(body.data.xmat[base].reshape(3, 3)[2, 2])
        least = min(least, upright)
        laid = laid or upright < .7
        if step >= hold:
            steady.append(upright)
    eyes.close()
    return dict(least=least, laid=laid, end=float(np.mean(steady)) if steady else float(upright))


def main():
    print("standing still, one sideways push at t=0, 4 s of response, live parameters")
    print("%-22s %10s %12s %10s %10s %8s" % (
        "lean reflex", "push (N)", "min up_z", "lost upright", "end up_z", "legs safe"))
    for label, extra in (("on (live)", dict()), ("off", dict(steady_gain=0.))):
        for newtons in (20., 40., 60., 80.):
            report = push(newtons, **extra)
            print("%-22s %10.0f %12.4f %10s %10.4f %8s" % (
                label, newtons, report["least"], report["laid"], report["end"], "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
