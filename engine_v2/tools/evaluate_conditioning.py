"""Physical paired/unpaired/no-learning controls; stimuli never set joint angles."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
from born_wired.innate import InnateController
from evaluate_robot import source_hashes


def run(condition, seed):
    body = Go2Body()
    observation = body.reset(seed=seed, joint_noise=.01)
    brain = InnateController(body.home_angles, body.lower_limits, body.upper_limits, seed=seed)
    samples = []
    for tick in range(5500):
        t = tick * .01
        training_cue = 5 <= t < (15 if condition == 'unpaired' else 25)
        test_cue = 35 <= t < 45
        cue = float(training_cue or test_cue)
        flexion = .8 if (15 if condition == 'unpaired' else 5) <= t < 25 else 0
        target, activation = brain.step(observation, flexion=flexion, cues=[cue, 0, 0, 0],
                                        dt=.01, learn=condition != 'frozen')
        observation = body.step(target, duration=.01, activation=activation)
        if tick % 10 == 9:
            samples.append(dict(time=observation['time'], height=observation['body_height'],
                                up_z=float(observation['body_up'][2]),
                                flexion_activity=float(brain.network.activity[brain.groups['flexion']][0]),
                                rear_calf=float(observation['joint_position'][8])))
    def mean(key, start, end):
        return float(np.mean([s[key] for s in samples if start < s['time'] <= end]))
    metrics = dict(baseline_height=mean('height', 1, 5), cue_only_height=mean('height', 38, 45),
                   recovery_height=mean('height', 50, 55),
                   cue_only_flexion=mean('flexion_activity', 38, 45),
                   cue_only_rear_calf=mean('rear_calf', 38, 45),
                   minimum_up_z=min(s['up_z'] for s in samples))
    return dict(condition=condition, seed=seed, metrics=metrics,
                diagnostics=brain.diagnostics(), samples=samples)


if __name__ == '__main__':
    started = time.perf_counter()
    hashes = source_hashes()
    results = []
    for condition in ('paired', 'unpaired', 'frozen'):
        for seed in range(5):
            result = run(condition, seed)
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('samples', 'diagnostics')}), flush=True)
    report = dict(source_sha256=hashes, source_unchanged_during_run=hashes == source_hashes(),
                  seconds_per_trial=55, dt=.01, seeds=list(range(5)),
                  wall_seconds=time.perf_counter()-started, results=results,
                  limits='Four cue channels, one conditioned drive. No reversal, delayed reward or lifelong-memory claim.')
    (ROOT/'artifacts/conditioning.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
