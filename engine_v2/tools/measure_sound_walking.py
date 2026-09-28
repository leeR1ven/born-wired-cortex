"""Does the outer ear's behind-sound route disturb walking?

    python tools/measure_sound_walking.py

The live window always feeds the ears, so the new edge runs whenever a sound
sits behind the animal. Three seeds walk 15 s in the furnished arena with eyes
and ears both live, once with the route on and once with it silenced. Reported:
how far the body got, how low it went after the first second, and whether the
upright axis was lost.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.binaural_senses import BinauralSenses      # noqa: E402
from born_wired.embodied import EmbodiedController         # noqa: E402
from born_wired.go2_body import Go2Body                    # noqa: E402
from born_wired.stereo_senses import RawEyes               # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
CONFIG = ROOT / "live_config.json"
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
STEPS, LOCOMOTION = 1500, .6


def walk(seed, gain):
    body = Go2Body(model_path=ARENA)
    # The parameters the live window actually runs, so this measures the real
    # configuration rather than a stripped-down one.
    parameters = dict(json.loads(CONFIG.read_text(encoding="utf-8-sig"))["parameters"])
    parameters.update(seed=seed, pinna_orient_gain=gain)
    # The sizes named below are this probe's own; the live window's copies of
    # the same knobs would otherwise be passed twice, which is an error rather
    # than an override.
    for name in ("eye_width", "eye_height", "motor_units", "proprio_units",
                 "association_units"):
        parameters.pop(name, None)
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    eyes, ears = (RawEyes(body, width=brain.eye_width, height=brain.eye_height),
                  BinauralSenses(body, window_samples=160))
    observation = body.reset(seed=seed)
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    low, upright = 1., 1.
    for step in range(STEPS):
        target, activation = brain.step(observation, environment=environment,
                                        eye_pixels=eyes.observe_raw(), ear_waveform=ears.observe(),
                                        dt=.01, learn=False, locomotion=LOCOMOTION)
        body.command_eyes(brain.eye_command())
        observation = body.step(target, duration=.01, activation=activation)
        if step > 100:
            low = min(low, float(body.data.xpos[base][2]))
            upright = min(upright, float(body.data.xmat[base].reshape(3, 3)[2, 2]))
    eyes.close()
    return float(body.data.xpos[base][0]), low, upright


def main():
    print("15 s of walking in the furnished arena, eyes and ears both live")
    print("%6s %10s %12s %12s %12s" % ("seed", "route", "walked (m)", "min height", "min up_z"))
    for seed in (0, 1, 2):
        for gain in (0., 1.2):
            forward, low, upright = walk(seed, gain)
            print("%6d %10s %12.3f %12.3f %12.4f" % (seed, "on" if gain else "off",
                                                     forward, low, upright))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
