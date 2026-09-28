"""Split one brain step into its device phases, with the device synchronised.

The device queues work asynchronously, so a stopwatch around a call returns
almost immediately unless something waits for the answer.  Each wrapper here
synchronises on both sides of the call it measures, so the milliseconds printed
are the milliseconds that call actually occupied the card.

    python tools/probe_brain_phases.py --repeat 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from born_wired.go2_body import Go2Body
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes
from born_wired.binaural_senses import BinauralSenses
from born_wired import torch_execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--config', type=Path, default=ROOT / 'live_config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    body = Go2Body(config.get('model', ROOT / 'models/reflex_arena.xml'))
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **config.get('parameters', {}))
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    ears = BinauralSenses(body, window_samples=160)
    device = brain.network._device
    if device is None:
        raise SystemExit('this probe needs the device engine; set BORN_WIRED_DEVICE=cuda')
    torch = device.torch
    sync = torch.cuda.synchronize
    totals = {}
    calls = {}

    def instrument(owner, name):
        original = getattr(owner, name)

        def wrapper(*arguments, **keywords):
            sync()
            stamp = time.perf_counter()
            result = original(*arguments, **keywords)
            sync()
            totals[name] = totals.get(name, 0.) + (time.perf_counter() - stamp)
            calls[name] = calls.get(name, 0) + 1
            return result
        setattr(owner, name, wrapper)

    for name in ('currents', 'update', 'project'):
        instrument(device.synapses, name)
    for name in ('advance', 'record', 'recorded_step', 'commit', 'host_rate'):
        instrument(device, name)
    instrument(brain.auditory, 'step')
    instrument(brain.network, '_step_device')

    environment = {name: np.zeros(4) for name in ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
    observation = body.observe()

    def run():
        return brain.step(observation, environment=environment, eye_pixels=eyes.observe_raw(),
                          ear_waveform=ears.observe(), dt=.01, autonomy=True)

    for _ in range(args.warmup):
        run()
    for name in list(totals):
        totals[name] = 0.
        calls[name] = 0
    sync()
    stamp = time.perf_counter()
    for _ in range(args.repeat):
        run()
    sync()
    whole = (time.perf_counter() - stamp)/args.repeat*1000
    print('neurons %d  edges %d  repeat %d' % (brain.network.n_neurons, len(brain.synapses.src), args.repeat))
    print('%-14s %10s %8s' % ('device phase', 'mean ms', 'per step'))
    for name in sorted(totals, key=lambda key: -totals[key]):
        print('%-14s %10.2f %8.2f' % (name, 1000*totals[name]/args.repeat, calls[name]/args.repeat))
    measured = sum(totals.values())/args.repeat*1000
    print('%-14s %10.2f' % ('whole step', whole))
    print('%-14s %10.2f' % ('inside device', measured))
    print('%-14s %10.2f' % ('outside device', whole - measured))
    eyes.close()


if __name__ == '__main__':
    main()
