"""Measure the paw paths around one low obstacle; selects no action.

The controller only receives the same environment dictionary the other runners
use. Everything recorded here (paw height, geom-to-geom contact with the curb,
group activities) is read-only instrumentation.

    python tools/probe_obstacle_step.py --output artifacts/obstacle_probe.json \
        --seeds 0 1 --duration 12
"""
import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.validate_reflex_v3 import (CONTROLLER_DEFAULTS, base_position, controller_parameters,
                                      floor_z, load_live_parameters)
from born_wired.embodied import EmbodiedController
from born_wired.go2_body import Go2Body
from born_wired.reflex_senses import ReflexSenses

NEURAL_DT = 0.01
PHYSICS_DT = 0.002
SAMPLE_INTERVAL = 0.1
CURB = 'curb'
DRIVE = 0.65
START = (0.85, 0.0)
GROUPS = ('foot_obstacle', 'withdrawal', 'stumble', 'clearance', 'phase', 'recruitment')


def parse_seeds(value):
    return [int(item) for item in value.split(',') if item.strip()]


def group_value(controller, name):
    ids = controller.groups.get(name)
    return None if ids is None else np.asarray(controller.network.activity[ids], dtype=float)


def run_seed(*, seed, duration, model_path, parameters):
    steps = int(round(duration / NEURAL_DT))
    body = Go2Body(model_path=model_path, timestep=PHYSICS_DT)
    body.reset(seed=seed, joint_noise=.01)
    body.data.qpos[:2] = START
    mujoco.mj_forward(body.model, body.data)
    curb = int(mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, CURB))
    if curb < 0:
        raise ValueError(f'the model has no {CURB} geom')
    curb_far_x = float(body.model.geom_pos[curb, 0] + body.model.geom_size[curb, 0])
    feet = [int(geom) for geom in body._feet]
    foot_radius = np.array([float(body.model.geom_size[geom, 0]) for geom in feet])
    senses = ReflexSenses(body)
    controller = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                                    seed=seed, **parameters)
    observation = body.observe()
    environment = senses.observe()
    floor = floor_z(body)

    clearance = np.zeros((steps + 1, 4))
    hits = np.zeros((steps + 1, 4), dtype=bool)
    timeline = []
    maximum_x = float(base_position(body)[0])
    previous = base_position(body)
    started = time.perf_counter()
    error = None
    try:
        for index in range(steps + 1):
            contact = body.observe()['foot_position'][:, 2] - foot_radius - floor_z(body)
            clearance[index] = contact
            for item in body.data.contact[:body.data.ncon]:
                pair = {int(item.geom1), int(item.geom2)}
                if curb in pair and item.dist <= 0:
                    for leg, geom in enumerate(feet):
                        if geom in pair:
                            hits[index, leg] = True
            if index == steps:
                break
            if index % int(SAMPLE_INTERVAL / NEURAL_DT) == 0:
                sample = dict(time_s=index * NEURAL_DT, base_x=float(base_position(body)[0]),
                              clearance=clearance[index].tolist(),
                              foot_obstacle=np.asarray(environment['foot_obstacle']).tolist(),
                              hits=hits[index].astype(int).tolist())
                for name in GROUPS:
                    values = group_value(controller, name)
                    if values is not None:
                        sample[name] = values.tolist()
                timeline.append(sample)
            target, activation = controller.step(
                observation, environment=environment, autonomy=False, locomotion=DRIVE,
                dt=NEURAL_DT, learn=True)
            observation = body.step(target, duration=NEURAL_DT, activation=activation)
            environment = senses.observe()
            position = base_position(body)
            maximum_x = max(maximum_x, float(position[0]))
            previous = position
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'

    final = clearance[:steps + 1]
    summary = []
    for leg in range(4):
        first = int(np.argmax(hits[:, leg])) if hits[:, leg].any() else -1
        window = slice(first, min(steps + 1, first + int(1.0 / NEURAL_DT))) if first >= 0 else slice(0, 0)
        summary.append(dict(
            leg=leg,
            curb_hits=int(hits[:, leg].sum()),
            first_hit_s=None if first < 0 else first * NEURAL_DT,
            clearance_max_m=float(final[:, leg].max()),
            clearance_before_hit_m=float(final[:first, leg].max()) if first > 0 else None,
            clearance_after_hit_1s_m=float(final[window, leg].max()) if first >= 0 else None,
        ))
    return dict(
        seed=seed, duration_seconds=duration, status='ok' if error is None else 'error', error=error,
        drive=DRIVE, curb_far_x=curb_far_x, hits_total=int(hits.sum()),
        metrics=dict(maximum_x=maximum_x, final_x=float(previous[0]),
                     crossed=bool(maximum_x > curb_far_x + .15),
                     final_clearance_m=summary,
                     minimum_up_z=float(np.min([o for o in [observation['body_up'][2]]])),
                     wall_seconds=time.perf_counter() - started),
        timeline=timeline)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=parse_seeds, default=[0, 1])
    parser.add_argument('--duration', type=float, default=12.0)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/reflex_arena.xml')
    parser.add_argument('--clearance-lift', type=float, default=None,
                        help='override the contact-to-lift gain; 0 disables the pathway')
    args = parser.parse_args(argv)
    parameters = controller_parameters(load_live_parameters())
    if args.clearance_lift is not None:
        parameters['clearance_lift'] = args.clearance_lift
    results = [run_seed(seed=seed, duration=args.duration, model_path=args.model, parameters=parameters)
               for seed in args.seeds]
    report = dict(model=str(args.model), drive=DRIVE, start=list(START),
                  clearance_lift=parameters.get('clearance_lift'),
                  curb=dict(name=CURB, height_m=0.06), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    for result in results:
        metrics = result['metrics']
        print(f"seed {result['seed']}: hits={result['hits_total']} max_x={metrics['maximum_x']:.3f} "
              f"crossed={metrics['crossed']} err={result['error']}")
        for leg in metrics['final_clearance_m']:
            print(f"   leg{leg['leg']} hits={leg['curb_hits']:4d} first={leg['first_hit_s']} "
                  f"max={leg['clearance_max_m']:.4f} before={leg['clearance_before_hit_m']} "
                  f"after1s={leg['clearance_after_hit_1s_m']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
