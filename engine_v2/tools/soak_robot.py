"""Continuous mixed-stimulus learning stress run; never resets brain or body."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
from born_wired.innate import InnateController
from evaluate_robot import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--output', default='artifacts/mixed_1hour.json')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 120 or args.seconds % .01 > 1e-6 and .01-args.seconds % .01 > 1e-6:
        parser.error('duration must be >= 120 and a multiple of .01')
    started = time.perf_counter()
    hashes = source_hashes()
    body = Go2Body()
    observation = body.reset(seed=args.seed, joint_noise=.01)
    brain = InnateController(body.home_angles, body.lower_limits, body.upper_limits, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    samples = []
    failure = None
    steps = round(args.seconds / .01)
    max_budget = max_bound_violation = 0.
    max_weight_change = 0.
    for tick in range(steps):
        t = tick * .01
        phase = tick % 12000
        # The timetable defines only stimuli, interventions and disturbances.
        # It never writes a motor value, weight, activity or simulation state.
        locomotion = .8 if 1500 <= phase < 3500 else (.4 if 10500 <= phase < 11500 else 0)
        flexion = .8 if 4500 <= phase < 5500 else 0
        cue = [float(4500 <= phase < 5500 or 6500 <= phase < 7500),
               float(8500 <= phase < 9500), 0., 0.]
        if phase == 10000:
            body.apply_force([0, 30 * (-1 if (tick // 12000) % 2 else 1), 0], .2)
        sensed = dict(observation)
        sensed['joint_position'] = observation['joint_position'] + rng.normal(0, .003, 12)
        sensed['gravity_direction'] = observation['gravity_direction'] + rng.normal(0, .001, 3)
        sensed['gravity_direction'] /= np.linalg.norm(sensed['gravity_direction'])
        try:
            target, activation = brain.step(sensed, dt=.01, flexion=flexion, locomotion=locomotion,
                                            cues=cue, feedback=not (11500 <= phase < 11800))
            observation = body.step(target, duration=.01, activation=activation)
            if tick % 100 == 99:
                syn = brain.synapses
                weights = syn.weights
                totals = syn.budget_totals()
                ratio = max(float(np.max(total / syn.budgets)) for total in totals)
                violation = max(float(np.max(syn.lower-weights)), float(np.max(weights-syn.w_max)), 0.)
                max_budget = max(max_budget, ratio)
                max_bound_violation = max(max_bound_violation, violation)
                max_weight_change = max(max_weight_change, float(np.max(np.abs(weights-brain.initial_weights))))
                samples.append(dict(time=observation['time'], phase_seconds=phase/100,
                                    height=observation['body_height'], up_z=float(observation['body_up'][2]),
                                    velocity=observation['body_linear_velocity'].tolist(),
                                    cue=cue[:2], flexion=flexion, locomotion=locomotion,
                                    flexion_activity=float(brain.network.activity[brain.groups['flexion']][0]),
                                    memory_weights=weights[syn.tether == 0].tolist(), budget_ratio=ratio))
                if violation > 1e-12 or ratio > 1+1e-12:
                    raise RuntimeError('weight invariant violation')
            if t > 2 and (observation['body_height'] < .10 or observation['body_up'][2] < .65):
                raise RuntimeError('fall criterion: height < .10 or up_z < .65')
        except Exception as exc:
            failure = dict(time=float(body.data.time), error=f'{type(exc).__name__}: {exc}')
            break
        if tick % 60000 == 59999:
            print(json.dumps(dict(simulated_seconds=observation['time'], height=observation['body_height'],
                                  budget_ratio_max=max_budget)), flush=True)
    payload = dict(config=dict(seconds=args.seconds, seed=args.seed, neural_dt=.01, physics_dt=.002,
                               joint_noise_std=.003, gravity_noise_std=.001, cycle_seconds=120,
                               learning=True, resets_during_run=0),
                   source_sha256=hashes, source_unchanged_during_run=hashes == source_hashes(),
                   python=sys.version, numpy=np.__version__, mujoco=mujoco.__version__,
                   simulated_seconds=float(body.data.time), failure=failure,
                   max_budget_ratio_at_1s_checks=max_budget,
                   max_edge_bound_violation_at_1s_checks=max_bound_violation,
                   max_weight_change_at_1s_checks=max_weight_change,
                   diagnostics=brain.diagnostics(), samples=samples,
                   wall_seconds=time.perf_counter()-started)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('samples', 'source_sha256', 'python')}), flush=True)
    return int(failure is not None)


if __name__ == '__main__':
    raise SystemExit(main())
