"""Does the body still walk, and what does it do at a wall?

    python tools/measure_wall_contact.py

Two scenes, both with the parameters the live window runs. First the furnished
arena for 15 s, to see that wiring the eyes into the protective cells did not
cost the gait. Then the bare home straight ahead of the east wall, driven
forward for 30 s, to see whether the body keeps pushing against it or turns
away. Nothing here is on the control path; it only reads the body and the
reflex cells.
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
from born_wired.reflex_senses import ReflexSenses       # noqa: E402
from born_wired.stereo_senses import RawEyes            # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
CONFIG = ROOT / "live_config.json"
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
START = 2.2          # so the inner face of east_wall is 1.22 m ahead


def build(body, parameters):
    # The sizes named below are this probe's own; the live window's copies of
    # the same knobs would otherwise be passed twice, which is an error rather
    # than an override.
    parameters = dict(parameters)
    for name in ("eye_width", "eye_height", "motor_units", "proprio_units",
                 "association_units"):
        parameters.pop(name, None)
    return EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                              motor_units=20, proprio_units=8, association_units=8,
                              eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                              **parameters)


def walk(steps=1500, locomotion=.6, seed=0, start=None, **extra):
    parameters = dict(json.loads(CONFIG.read_text(encoding="utf-8-sig"))["parameters"])
    parameters.update(extra)
    body = Go2Body(model_path=ARENA)
    brain = build(body, parameters)
    eyes, senses = (RawEyes(body, width=brain.eye_width, height=brain.eye_height),
                    ReflexSenses(body))
    observation = body.reset(seed=seed)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if start is not None:
        body.data.qpos[body._base_qpos:body._base_qpos + 3] = start
        mujoco.mj_forward(body.model, body.data)
        observation = body.observe()
    low, upright, touch = 1., 1., 0.
    contact, hard = 0, 0
    brake, near, wall, path = [], [], [], []
    for step in range(steps):
        environment = senses.observe()
        target, activation = brain.step(observation, environment=environment,
                                        eye_pixels=eyes.observe_raw(), dt=.01, learn=False,
                                        locomotion=locomotion)
        body.command_eyes(brain.eye_command())
        observation = body.step(target, duration=.01, activation=activation)
        reflex = brain.diagnostics()["reflex_activity"]
        if step > 100:
            low = min(low, float(body.data.xpos[base][2]))
            upright = min(upright, float(body.data.xmat[base].reshape(3, 3)[2, 2]))
            strongest = float(environment["body_touch"].max())
            touch = max(touch, strongest)
            contact += int(strongest > .05)
            hard += int(strongest > .5)
            brake.append(np.mean(reflex["brake"]))
            near.append(np.mean(reflex["near"]))
            wall.append(np.mean(reflex["wall_contact"]))
        if step % 250 == 0:
            position = np.asarray(body.data.xpos[base])
            path.append((step*.01, float(position[0]), float(position[1])))
    eyes.close()
    rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
    x, y = float(body.data.xpos[base][0]), float(body.data.xpos[base][1])
    return dict(x=x, y=y, low=low, upright=upright, touch=touch,
                yaw=float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
                wall=float(np.mean(wall)), brake=float(np.mean(brake)),
                near=float(np.mean(near)), contact=contact, hard=hard, path=path)


def main():
    print("15 s in the furnished arena and 45 s driving at the east wall")
    print("contact = ticks with any body sector touching, hard = ticks over half scale")
    print("%-26s %7s %7s %7s %8s %7s %6s %6s %6s %7s %6s" % (
        "scene", "x(m)", "y(m)", "min h", "min up_z", "wall", "brake", "near", "contact",
        "hard", "yaw(deg)"))
    for label, arguments in (
            ("arena 15 s (live params)", dict(steps=1500)),
            ("arena, no eye route", dict(steps=1500, eye_near_gain=0., eye_stereo_gain=0.)),
            ("arena, no steady reflex", dict(steps=1500, steady_gain=0.)),
            ("arena, no wall reflex", dict(steps=1500, wall_gain=0.)),
            ("wall 45 s (live params)", dict(steps=4500, start=[START, 0., .35])),
            ("wall 45 s, no wall reflex", dict(steps=4500, start=[START, 0., .35], wall_gain=0.)),
            ("wall 45 s, no eye route", dict(steps=4500, start=[START, 0., .35],
                                             eye_near_gain=0., eye_stereo_gain=0.))):
        report = walk(**arguments)
        print("%-26s %7.3f %7.3f %7.3f %8.4f %7.3f %6.3f %6.3f %6d %7d %6.1f" % (
            label, report["x"], report["y"], report["low"], report["upright"],
            report["wall"], report["brake"], report["near"], report["contact"],
            report["hard"], report["yaw"]))
        print("      x: " + " ".join("%.2f" % p[1] for p in report["path"])
              + "   y: " + " ".join("%.2f" % p[2] for p in report["path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
