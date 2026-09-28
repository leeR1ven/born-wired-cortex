"""Where does a live control step spend its milliseconds?

Times the pieces of the loop in tools/live_dog.py separately, at the shipped
configuration, so that a speed change can be argued from numbers instead of a
guess about which part is slow.  Nothing here changes the brain: it builds the
same body, senses and controller the window builds and reports the cost of
each call the loop makes.

    python tools/profile_live_step.py [--repeat 40] [--width 160] [--height 120]
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
from born_wired.reflex_senses import ReflexSenses
from born_wired.stereo_senses import RawEyes
from born_wired.binaural_senses import BinauralSenses
from born_wired import torch_execution

PHASES = ('body_senses', 'eye_render', 'ear_relay', 'brain', 'physics', 'whole_step')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=6)
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--config', type=Path, default=ROOT / 'live_config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    parameters = dict(config.get('parameters', {}))
    if args.width:
        parameters['eye_width'] = args.width
    if args.height:
        parameters['eye_height'] = args.height
    body = Go2Body(config.get('model', ROOT / 'models/reflex_arena.xml'))
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               **parameters)
    senses = ReflexSenses(body)
    eyes = RawEyes(body, width=brain.eye_width, height=brain.eye_height)
    ears = BinauralSenses(body, window_samples=160)
    seen = [body.observe()]
    print('engine     %s' % torch_execution.name_of(torch_execution.resolve()))
    print('neurons    %d' % brain.network.n_neurons)
    print('edges      %d' % len(brain.synapses.src))
    print('retina     %dx%d  drawn %dx%d' % (brain.eye_width, brain.eye_height,
                                             eyes.width*eyes.oversample, eyes.height*eyes.oversample))
    accumulators = {name: [] for name in PHASES}

    def one_step(record):
        stamp = time.perf_counter()
        environment = senses.observe()
        mark = time.perf_counter()
        eye_pixels = eyes.observe_raw()
        mark2 = time.perf_counter()
        ear_waveform = ears.observe()
        mark3 = time.perf_counter()
        target, activation = brain.step(seen[0], environment=environment, eye_pixels=eye_pixels,
                                        ear_waveform=ear_waveform, dt=.01, autonomy=True)
        mark4 = time.perf_counter()
        if brain.eye_encoder is not None:
            body.command_eyes(brain.eye_command())
        seen[0] = body.step(target, duration=.01, activation=activation)
        mark5 = time.perf_counter()
        if record:
            for name, value in (('body_senses', mark-stamp), ('eye_render', mark2-mark),
                                ('ear_relay', mark3-mark2), ('brain', mark4-mark3),
                                ('physics', mark5-mark4), ('whole_step', mark5-stamp)):
                accumulators[name].append(1000.*value)

    for _ in range(args.warmup):
        one_step(False)
    for _ in range(args.repeat):
        one_step(True)
    print('%-12s %10s %10s %10s' % ('phase', 'mean ms', 'min ms', 'max ms'))
    measured = 0.
    for name in PHASES[:-1]:
        values = accumulators[name]
        measured += float(np.mean(values))
        print('%-12s %10.2f %10.2f %10.2f' % (name, np.mean(values), np.min(values), np.max(values)))
    whole = accumulators['whole_step']
    print('%-12s %10.2f' % ('sum of parts', measured))
    print('%-12s %10.2f %10.2f %10.2f' % ('whole_step', np.mean(whole), np.min(whole), np.max(whole)))
    print('steps/s    %.1f' % (1000./np.mean(whole)))
    eyes.close()


if __name__ == '__main__':
    main()
