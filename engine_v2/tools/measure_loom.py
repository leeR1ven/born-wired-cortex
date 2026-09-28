"""Is a picture that is opening outwards read as "this is coming at me"?

    python tools/measure_loom.py

Four scenes, and the same three cells are read in each:

  still     - parked in front of the striped wall, nothing moving. There is
              nothing to answer.
  sweep     - the two eyes are turned by hand, so every edge in the picture
              travels the same way at once. That is what an eye movement does,
              and it must not read as something coming closer.
  approach  - one small ball walked in from 3.0 m to 0.35 m, at three speeds.
              This is what the cells are for.
  walking   - the animal walks in the furnished arena, which moves every edge
              in its whole view. Measured so the cost of the reflex is known.

The cells themselves are built in born_wired/embodied.py. Nothing here is on
the control path. "outward" is an edge that has moved away from the middle of
the picture, "inward" one that has moved towards it, "growing" is both halves
of one picture opening outwards together, and "looming" is what reaches the
brake.
"""
import argparse
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
SCENERY = ("red_pillar", "blue_box", "front_block", "left_block", "right_block",
           "green_target", "curb", "low_step", "platform", "ramp", "passage_a",
           "passage_b", "sound_low", "sound_high")
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
WALL_FACE = 3.5 - .08
HEAD = ("scene", "outward", "inward", "looming")


def settings(extra=None):
    parameters = dict(json.loads(CONFIG.read_text(encoding="utf-8-sig"))["parameters"])
    parameters.update(extra or {})
    return parameters


def brain_of(body, **parameters):
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


def empties(body, keep=()):
    for name in SCENERY:
        if name in keep:
            continue
        geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            body.model.geom_pos[geom] = [60., 60., -8.]


def park(body, distance):
    body.data.qpos[body._base_qpos] = WALL_FACE - float(distance) - .30
    body.data.qpos[body._base_qpos + 1] = 0.
    mujoco.mj_forward(body.model, body.data)


def rest_run(steps, setup=None, eye_sweep=None, ball=None, **parameters):
    """The loop stepped with the legs held at rest, reading the three cells."""
    body = Go2Body(model_path=ARENA)
    if setup is not None:
        setup(body)
    target = None
    if ball is not None:
        target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
        body.model.geom_size[target] = [.06]*3
    brain = brain_of(body, **parameters)
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    observation = body.observe()
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    rows = []
    for step in range(steps):
        if ball is not None:
            body.model.geom_pos[target] = [.30 + ball(step), 0., .32]
            mujoco.mj_forward(body.model, body.data)
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=.01, learn=False)[1]
        body.command_eyes(brain.eye_command() if eye_sweep is None
                          else np.array([eye_sweep(step), 0., eye_sweep(step), 0.]))
        observation = body.step(np.asarray(body.home_angles), duration=.01, activation=activation)
        loom = brain.diagnostics()["eye_loom"]
        rows.append((loom["outward"], loom["inward"],
                     float(np.mean(loom["growing"])), loom["looming"]))
    eyes.close()
    return np.array(rows)


def walk_run(steps, **parameters):
    """The animal walking in the furnished arena, reading the three cells."""
    body = Go2Body(model_path=ARENA)
    brain = brain_of(body, **parameters)
    eyes, senses = (RawEyes(body, width=brain.eye_width, height=brain.eye_height),
                    ReflexSenses(body))
    observation = body.reset(seed=0)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    upright, rows = 1., []
    for step in range(steps):
        environment = senses.observe()
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=.01, learn=False,
                                locomotion=.6)[1]
        body.command_eyes(brain.eye_command())
        observation = body.step(np.asarray(body.home_angles), duration=.01,
                                activation=activation)
        if step > 100:
            upright = min(upright, float(body.data.xmat[base].reshape(3, 3)[2, 2]))
            loom = brain.diagnostics()["eye_loom"]
            rows.append((loom["outward"], loom["inward"],
                     float(np.mean(loom["growing"])), loom["looming"]))
    eyes.close()
    return np.array(rows), upright, float(body.data.xpos[base][0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", default="{}",
                        help="controller parameters to override, as JSON")
    options = parser.parse_args(argv)
    extra = json.loads(options.parameters)

    print("nothing moving, the two eyes held still and the legs at rest")
    print("%-30s %8s %8s %8s   %s" % (HEAD + ("growing",)))
    for distance in (.35, .80, 1.60, 3.00):
        setup = lambda body, d=distance: (empties(body), park(body, d))
        rows = rest_run(220, setup=setup, **extra)[-100:]
        mean = rows.mean(axis=0)
        print("%-30s %8.3f %8.3f %8.3f   %.3f (peak %.3f)" % (
            "  wall %.2f m ahead" % distance, mean[0], mean[1], mean[3],
            mean[2], rows[:, 2].max()))
    print()

    print("the eyes turned by hand, one smooth sweep each way")
    for rate in (.6, 3.0):
        setup = lambda body: (empties(body), park(body, 3.0))
        rows = rest_run(400, setup=setup,
                        eye_sweep=lambda s, r=rate: .5*np.sin(r*.01*s), **extra)[-200:]
        mean = rows.mean(axis=0)
        print("%-30s %8.3f %8.3f %8.3f   %.3f" % ("  sweep %.1f rad/s" % rate,
                                                  mean[0], mean[1], mean[3], mean[2]))
    setup = lambda body: (empties(body), park(body, 3.0))
    rows = rest_run(400, setup=setup, eye_sweep=lambda s: .5, **extra)[-200:]
    mean = rows.mean(axis=0)
    print("%-30s %8.3f %8.3f %8.3f   %.3f" % ("  held hard over", mean[0], mean[1],
                                              mean[3], mean[2]))
    print()

    print("one ball walked in from 3.0 m to 0.35 m")
    print("%-30s %8s %8s %8s" % HEAD)
    for speed in (.5, 1.5, 3.0):
        travel = 3.0 - .35
        span = max(2, int(round(travel/(speed*.01))) - 1)
        rows = rest_run(span + 120, setup=empties,
                        ball=lambda s: 3.0 - travel*min(1., s/float(span)), **extra)
        third = max(1, len(rows)//3)
        for index, name in enumerate(("closing in", "arriving", "arrived")):
            block = rows[index*third:(index + 1)*third]
            mean = block.mean(axis=0)
            print("%-30s %8.3f %8.3f %8.3f   %.3f" % ("  %.1f m/s, %s" % (speed, name),
                                                      mean[0], mean[1], mean[3], mean[2]))
    print()

    print("the animal walking in the furnished arena")
    print("%-30s %8s %8s %8s   %s" % (HEAD + ("gait",)))
    for label, overrides in (("live", {}), ("brake route off", {"eye_loom_gain": 0.}),
                             ("cells off", {"eye_loom_threshold": 1e9})):
        rows, upright, x = walk_run(1200, **settings(dict(extra, **overrides)))
        mean = rows.mean(axis=0)
        print("%-30s %8.3f %8.3f %8.3f   up_z %.4f, x %.3f m" % (
            "  " + label, mean[0], mean[1], mean[3], upright, x))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
