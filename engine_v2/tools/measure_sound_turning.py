"""What a sound does to the eyes and the orienting cells, in front of the head
and behind it, with the outer ear's front/back route on and off.

    python tools/measure_sound_turning.py

One 262 Hz source, retinas dark, legs held at rest, 500 steps. The eye yaw is
the mean of the two eyes in radians, and "orienting" is the left minus right
orienting cell. The only thing that separates the two rows of a pair is whether
the source sits in front of the ears or behind them.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from born_wired.binaural_senses import BinauralSenses      # noqa: E402
from born_wired.embodied import EmbodiedController         # noqa: E402
from born_wired.go2_body import Go2Body                    # noqa: E402
from born_wired.stereo_senses import RawEyes               # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
SCENERY = ("red_pillar", "blue_box", "front_block", "curb", "low_step", "platform",
           "ramp", "passage_a", "passage_b", "sound_low", "sound_high")
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
STEPS = 500


def body_with_source(x, y):
    body = Go2Body(model_path=ARENA)
    for name in SCENERY:
        geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            body.model.geom_pos[geom] = [60., 60., -8.]
    geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "sound_low")
    body.model.geom_pos[geom] = [x, y, .30]
    mujoco.mj_forward(body.model, body.data)
    return body


def listen(body, **parameters):
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes = RawEyes(body)
    ears = BinauralSenses(body, window_samples=160)
    observation = body.observe()
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    for _ in range(STEPS):
        target, activation = brain.step(observation, environment=environment,
                                        ear_waveform=ears.observe(), dt=.01, learn=False)
        body.command_eyes(brain.eye_command())
        observation = body.step(np.asarray(body.home_angles), duration=.01,
                                activation=activation)
    eyes.close()
    command = brain.eye_command()
    orienting = brain.network.activity[brain.groups["orienting"]]
    return .5*(command[0] + command[2]), float(orienting[0] - orienting[1])


def main():
    print("eye yaw and left-minus-right orienting, 500 steps, retinas dark")
    print("%-14s %6s %12s %12s" % ("source", "gain", "eye yaw", "orienting"))
    for label, x, y in (("front-left", 1.10, .60), ("behind-left", -.70, .60),
                        ("front-right", 1.10, -.60), ("behind-right", -.70, -.60),
                        ("dead front", 1.10, 0.), ("dead behind", -.70, 0.)):
        for gain in (0., 1.2):
            yaw, orienting = listen(body_with_source(x, y), pinna_orient_gain=gain)
            print("%-14s %6.1f %+12.4f %+12.4f" % (label, gain, yaw, orienting))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
