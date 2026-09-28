"""Find the numbers that make the eyes follow a moving thing, by generating a
population of candidate brains and keeping the ones that follow it best.

    python tools/search_eye_tracking.py                 # short run
    python tools/search_eye_tracking.py --long          # longer run

The graph is fixed: every candidate has exactly the same cells and the same
edges as every other. Only the numbers below are drawn new, so a result can be
written back into the innate wiring as numbers. Nothing here is on the control
path and nothing is learned during a trial: each candidate is dropped into a
clean room and read out.

The score of a candidate is how far the two eyes sit from the thing they are
supposed to be looking at, in radians, averaged over the time each scene is
run and over six scenes: the ball crossing left to right at three heights, the
ball crossing top to bottom at two places across, and the ball held still. The
still scene is a control: a brain that twitches the eyes at nothing pays for it
there.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.embodied import EmbodiedController      # noqa: E402
from born_wired.go2_body import Go2Body                 # noqa: E402
from born_wired.stereo_senses import RawEyes            # noqa: E402

ARENA = ROOT / "models" / "reflex_arena.xml"
OUT = ROOT / "artifacts" / "eye_tracking_search.json"
SCENERY = ("red_pillar", "blue_box", "front_block", "curb", "low_step", "platform",
           "ramp", "passage_a", "passage_b", "sound_low", "sound_high")
ENVIRONMENT = ("body_touch", "foot_obstacle", "foot_load", "foot_slip")
DT, BALL, DISTANCE = .01, .06, .90

# name: (low, high, whole_number). Log spans are drawn evenly in the exponent for
# the wide ones, so a factor of ten is not one in a hundred draws.
KNOBS = (
    ('eye_change_gain', 2., 200., False),
    ('eye_change_threshold', .02, 60., False),
    ('eye_change_trace_time', .01, .08, False),
    ('eye_change_common', 0., 12., False),
    ('eye_change_relay_gain', 0., 4., False),
    ('eye_change_relay_steps', 1, 20, True),
    ('eye_row_gain', 2., 200., False),
    ('eye_row_threshold', .02, 60., False),
    ('eye_row_trace_time', .01, .08, False),
    ('eye_row_common', 0., 12., False),
    ('eye_row_relay_gain', 0., 4., False),
    ('eye_gaze_gain', 0., 1., False),
    ('eye_track_gain', 0., 8., False),
    ('eye_pitch_gain', 0., 8., False),
    ('eye_relay_gain', 0., 30., False),
)
LOGARITHMIC = ('eye_change_gain', 'eye_change_threshold', 'eye_row_gain', 'eye_row_threshold')
DEFAULTS = dict(eye_change_gain=48., eye_change_threshold=.5, eye_change_trace_time=.01,
                eye_change_common=3., eye_change_relay_gain=.5, eye_change_relay_steps=10,
                eye_row_gain=48., eye_row_threshold=.5, eye_row_trace_time=.01,
                eye_row_common=3., eye_row_relay_gain=.5, eye_gaze_gain=1.,
                eye_track_gain=2., eye_pitch_gain=4., eye_relay_gain=8.)

# (axis, lateral or vertical offset of the ball's line, half travel)
SCENES = (('y', 0., .30), ('y', .25, .30), ('y', -.25, .30),
          ('z', 0., .30), ('z', .25, .30), ('still', 0., 0.))


def draw(rng, whole):
    return rng.integers(whole[0], whole[1] + 1) if whole and False else None


def sample(rng):
    candidate = {}
    for name, low, high, whole in KNOBS:
        if whole:
            candidate[name] = int(rng.integers(int(low), int(high) + 1))
        elif name in LOGARITHMIC:
            candidate[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            candidate[name] = float(rng.uniform(low, high))
    return candidate


def mutate(rng, parent, scale=.35):
    child = dict(parent)
    for name, low, high, whole in KNOBS:
        if rng.random() > .45:
            continue
        if whole:
            step = max(1, int(round(scale*rng.normal()*4)))
            child[name] = int(np.clip(parent[name] + rng.choice([-1, 1])*step, low, high))
        elif name in LOGARITHMIC:
            child[name] = float(np.clip(np.exp(np.log(parent[name]) + scale*rng.normal()),
                                        low, high))
        else:
            span = high - low
            child[name] = float(np.clip(parent[name] + scale*rng.normal()*.25*span, low, high))
    return child


def trial(body, brain, eyes, observation, environment, base, target, scene, steps):
    axis, offset, half = scene
    base_rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
    errors = []
    for step in range(steps):
        if axis == 'still':
            position = [.30 + DISTANCE, 0., .32]
        else:
            travel = half - 2.*half*step/float(steps - 1)
            position = ([.30 + DISTANCE, travel, .32 + offset] if axis == 'y'
                        else [.30 + DISTANCE, offset, .32 + travel])
        body.model.geom_pos[target] = position
        mujoco.mj_forward(body.model, body.data)
        activation = brain.step(observation, environment=environment,
                                eye_pixels=eyes.observe_raw(), dt=DT, learn=False)[1]
        body.command_eyes(brain.eye_command())
        if step > steps//5:
            offset_world = (np.asarray(body.data.geom_xpos[target])
                            - np.asarray(body.data.xpos[base]))
            rotation = np.asarray(body.data.xmat[base]).reshape(3, 3)
            bearing = np.arctan2(offset_world @ rotation[:, 1], offset_world @ rotation[:, 0])
            elevation = np.arctan2(offset_world @ rotation[:, 2], offset_world @ rotation[:, 0])
            command = np.asarray(brain.eye_command())
            yaw = .5*(command[0] + command[2])
            pitch = .5*(command[1] + command[3])
            errors.append(abs(yaw - bearing) + abs(-pitch - elevation))
    observation = body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
    return float(np.mean(errors)), observation


def evaluate(candidate, steps):
    body = Go2Body(model_path=ARENA)
    for name in SCENERY:
        geom = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            body.model.geom_pos[geom] = [60., 60., -8.]
    target = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "green_target")
    body.model.geom_size[target] = [BALL]*3
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               motor_units=20, proprio_units=8, association_units=8,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **candidate)
    eyes = RawEyes(body)
    environment = {name: np.zeros(4) for name in ENVIRONMENT}
    environment["foot_support"] = np.zeros(4, dtype=bool)
    base = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    scores = []
    for scene in SCENES:
        observation = body.observe()
        score, _ = trial(body, brain, eyes, observation, environment, base, target, scene, steps)
        scores.append(score)
    eyes.close()
    return float(np.mean(scores)), [float(s) for s in scores]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--population', type=int, default=12)
    parser.add_argument('--steps', type=int, default=140)
    parser.add_argument('--keep', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args(argv)
    if args.long:
        args.generations, args.population, args.steps = 6, 24, 250
    rng = np.random.default_rng(args.seed)
    history = []
    print('scene order: %s' % (', '.join('%s%+.2f' % (axis, offset) for axis, offset, _ in SCENES),))
    started = time.time()

    def judge(candidate, note):
        score, per_scene = evaluate(candidate, args.steps)
        history.append(dict(note=note, score=score, scenes=per_scene, knobs=candidate))
        print('  %-18s %8.4f   %s' % (note, score, ' '.join('%.3f' % s for s in per_scene)),
              flush=True)
        return score

    score = judge(dict(DEFAULTS), 'defaults')
    best = [(score, dict(DEFAULTS))]
    for generation in range(args.generations):
        print('generation %d (elapsed %.0fs)' % (generation, time.time() - started), flush=True)
        population = []
        if generation == 0:
            population.append(dict(DEFAULTS))
            population.extend(sample(rng) for _ in range(args.population - 1))
        else:
            elite = [knobs for _, knobs in best[:args.keep]]
            population.append(dict(elite[0]))
            population.extend(mutate(rng, elite[rng.integers(len(elite))])
                              for _ in range(args.population//2 - 1))
            population.extend(sample(rng) for _ in range(args.population - len(population)))
        scored = []
        for index, candidate in enumerate(population):
            scored.append((judge(candidate, 'g%d/%d' % (generation, index)), candidate))
        scored.sort(key=lambda pair: pair[0])
        best = scored[:args.keep]
        print('  best of generation %d: %.4f' % (generation, best[0][0]), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(dict(best=best[0][1], best_score=best[0][0], defaults=DEFAULTS,
                   history=history, knobs=[list(knob) for knob in KNOBS],
                   scenes=[list(scene) for scene in SCENES], steps=args.steps,
                   seconds=time.time() - started),
              open(OUT, 'w', encoding='utf-8'), indent=1)
    print('best score %.4f' % best[0][0])
    for name, value in best[0][1].items():
        print('  %-26s %s' % (name, value))
    print('wrote %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
