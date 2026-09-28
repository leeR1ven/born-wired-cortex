"""Repeated hidden-feature conditioning with learning always enabled."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.go2_body import Go2Body
from born_wired.feature_routed import FeatureRoutedController


def main():
    sources = list((ROOT/'born_wired').glob('*.py')) + [Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    start = time.perf_counter()
    body = Go2Body()
    observation = body.reset(seed=11, joint_noise=.01)
    brain = FeatureRoutedController(body.home_angles, body.lower_limits, body.upper_limits, seed=11)
    records = []
    error = None
    for tick in range(60000):
        phase = tick % 6000
        teaching = 500 <= phase < 1500
        stimulus_a = teaching or 3500 <= phase < 4500
        stimulus_b = 2000 <= phase < 3000 or 5000 <= phase < 6000
        features = [float(stimulus_a), 0, 0, float(stimulus_b)]
        try:
            target, activation = brain.step(observation, features=features,
                                            flexion=.8 if teaching else 0, dt=.01, learn=True)
            observation = body.step(target, duration=.01, activation=activation)
            if tick % 10 == 9:
                records.append(dict(time=observation['time'], cycle=tick//6000, phase=phase/100,
                                    teacher=teaching, height=observation['body_height'],
                                    up_z=float(observation['body_up'][2]),
                                    flexion=float(brain.network.activity[brain.groups['flexion']][0])))
            if tick > 200 and (observation['body_height'] < .10 or observation['body_up'][2] < .65):
                raise RuntimeError('fall criterion')
        except Exception as exc:
            error = dict(time=observation['time'], error=f'{type(exc).__name__}: {exc}')
            break
    summary = []
    for cycle in sorted({r['cycle'] for r in records}):
        def mean(key, lo, hi):
            values = [r[key] for r in records if r['cycle'] == cycle and lo <= r['phase'] < hi]
            return float(np.mean(values)) if values else None
        summary.append(dict(cycle=cycle, a_height=mean('height', 39, 45), b_height=mean('height', 54, 60),
                            a_flexion=mean('flexion', 39, 45), b_flexion=mean('flexion', 54, 60)))
    report = dict(seconds=observation['time'], seed=11, learning_always_enabled=True,
                  neural_dt=.01, physics_dt=.002, body_resets_during_run=0, brain_resets_during_run=0,
                  source_sha256=hashes, source_unchanged=all(hashes[str(p.relative_to(ROOT))] == hashlib.sha256(p.read_bytes()).hexdigest() for p in sources),
                  failure=error, cycles=summary, diagnostics=brain.diagnostics(),
                  wall_seconds=time.perf_counter()-start, samples=records)
    (ROOT/'artifacts/online_features_10min.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples', 'source_sha256')}))
    return int(error is not None)


if __name__ == '__main__':
    raise SystemExit(main())
