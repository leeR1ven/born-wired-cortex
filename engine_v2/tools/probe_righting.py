"""Measure whether a fallen robot gets back on its feet; selects no action.

Starts either already down (a body tilt) or upright and then pushed over, and
records the body up-axis, the height, and the activity of the reflex groups.
Everything here is read-only instrumentation; the controller only ever sees the
same environment dictionary the other runners use.

    python tools/probe_righting.py --output artifacts/righting.json --seeds 0 1
"""
import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.validate_reflex_v3 import controller_parameters, floor_z, load_live_parameters
from born_wired.embodied import EmbodiedController
from born_wired.go2_body import Go2Body
from born_wired.reflex_senses import ReflexSenses

NEURAL_DT = 0.01
PHYSICS_DT = 0.002
SAMPLE_INTERVAL = 0.05
DRIVE = 0.65
GROUPS = ('protective_tilt', 'body_touch', 'righting', 'righting_push', 'phase', 'withdrawal')
POSES = {'on_back': (np.pi, 0.), 'on_side': (np.pi / 2, 0.), 'nose_up': (0., np.pi / 2),
         'nose_down': (0., -np.pi / 2)}
IMPACT = dict(start=1.5, force=(0., 140., 0.), duration=.2)
FORCE = [140.]
UP_Z = .8
HEIGHT = .15
HOLD = .5


def parse_seeds(value):
    return [int(item) for item in value.split(',') if item.strip()]


def group_value(controller, name):
    ids = controller.groups.get(name)
    return None if ids is None else np.asarray(controller.network.activity[ids], dtype=float)


def run_seed(*, seed, pose, duration, model_path, parameters, impact, force=None):
    steps = int(round(duration / NEURAL_DT))
    body = Go2Body(model_path=model_path, timestep=PHYSICS_DT)
    tilt = POSES.get(pose)
    body.reset(pose='crouch' if tilt else 'stand', seed=seed, joint_noise=.01, tilt=tilt)
    senses = ReflexSenses(body)
    if tilt is not None:
        robot = np.zeros(body.model.ngeom, dtype=bool)
        robot[senses._robot_geoms] = True
        lowest = float(np.min(body.data.geom_xpos[robot, 2] - body.model.geom_rbound[robot]))
        body.data.qpos[body._base_qpos + 2] += float(body.model.geom_pos[body._floor, 2]) + .002 - lowest
        mujoco.mj_forward(body.model, body.data)
    controller = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                                    seed=seed, **parameters)
    observation = body.observe()
    environment = senses.observe()
    applied = False
    trace = []
    up_z = np.zeros(steps + 1)
    height = np.zeros(steps + 1)
    error = None
    started = time.perf_counter()
    try:
        for index in range(steps + 1):
            up_z[index] = float(observation['body_up'][2])
            height[index] = float(observation['body_height'])
            if index % int(SAMPLE_INTERVAL / NEURAL_DT) == 0:
                sample = dict(time_s=index * NEURAL_DT, up_z=up_z[index],
                              height=height[index],
                              body_touch=np.asarray(environment['body_touch']).tolist(),
                              foot_support=np.asarray(environment['foot_support']).astype(int).tolist())
                for name in GROUPS:
                    values = group_value(controller, name)
                    if values is not None:
                        sample[name] = values.tolist()
                trace.append(sample)
            if index == steps:
                break
            if impact and not applied and index * NEURAL_DT >= IMPACT['start']:
                shove = FORCE[0] if force is None else float(force)
                body.apply_force((IMPACT['force'][0], shove, IMPACT['force'][2]), IMPACT['duration'])
                applied = True
            target, activation = controller.step(observation, environment=environment, autonomy=False,
                                                 locomotion=DRIVE, dt=NEURAL_DT, learn=True)
            observation = body.step(target, duration=NEURAL_DT, activation=activation)
            environment = senses.observe()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    upright = up_z >= UP_Z
    window = int(HOLD / NEURAL_DT)
    down = np.flatnonzero(up_z < .5)
    # Only a stretch after the body has actually been on the ground counts as
    # getting back up; standing before a push is not a recovery.
    after = 0 if not down.size else int(down[0])
    sustained = [index for index in range(after + window, steps + 1)
                 if upright[index - window:index + 1].all()]
    last = slice(max(0, steps - int(1.0 / NEURAL_DT)), steps + 1)
    return dict(seed=seed, pose=pose, duration_seconds=duration, impact=bool(impact),
                status='ok' if error is None else 'error', error=error, drive=DRIVE,
                metrics=dict(minimum_up_z=float(up_z.min()), maximum_up_z=float(up_z.max()),
                             final_up_z=float(up_z[last].mean()),
                             final_height=float(height[last].mean()),
                             recovered=bool(sustained),
                             recovery_s=None if not sustained else sustained[0] * NEURAL_DT,
                             time_to_feet_s=None if not sustained else (sustained[0] - after) * NEURAL_DT,
                             on_feet_share=float(np.mean(height > HEIGHT)),
                             wall_seconds=time.perf_counter() - started),
                trace=trace)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=parse_seeds, default=[0, 1])
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--poses', default='on_back,on_side,nose_up')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/reflex_arena.xml')
    parser.add_argument('--impact-force', type=float, default=140.)
    parser.add_argument('--righting-gain', type=float, default=None,
                        help='override the burst gain; 0 disables the pathway')
    args = parser.parse_args(argv)
    FORCE[0] = args.impact_force
    parameters = controller_parameters(load_live_parameters())
    if args.righting_gain is not None:
        parameters['righting_gain'] = args.righting_gain
    cases = [('impact', True)] if args.poses == 'impact' else [(p, False) for p in args.poses.split(',')]
    results = [run_seed(seed=seed, pose=pose, duration=args.duration, model_path=args.model,
                        parameters=parameters, impact=impact)
               for pose, impact in cases for seed in args.seeds]
    report = dict(model=str(args.model), drive=DRIVE, righting_gain=parameters.get('righting_gain'),
                  poses=dict(POSES), impact=dict(IMPACT, force=(0., FORCE[0], 0.)) if args.poses == 'impact' else None,
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    for result in results:
        metrics = result['metrics']
        print(f"{result['pose']:8s} seed {result['seed']}: min_up_z={metrics['minimum_up_z']:5.2f} "
              f"final_up_z={metrics['final_up_z']:5.2f} final_h={metrics['final_height']:.3f} "
              f"recovered={metrics['recovered']} t_up={metrics['time_to_feet_s']} err={result['error']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
