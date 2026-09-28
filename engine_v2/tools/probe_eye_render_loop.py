"""Why does the eye render cost ten times more inside the loop than alone?

The same two cameras are drawn alone and then with the rest of the loop around
them, one suspect at a time, so the extra milliseconds can be attributed to a
call rather than to the drawing.

    python tools/probe_eye_render_loop.py
"""
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


def stage(name, milliseconds):
    print('%-34s %8.3f' % (name, milliseconds))


def main():
    body = Go2Body(ROOT / 'models/reflex_arena.xml')
    senses = ReflexSenses(body)
    eyes = RawEyes(body, width=160, height=120)
    ears = BinauralSenses(body, window_samples=160)

    def renders(count=40):
        for _ in range(count):
            eyes.observe_raw()

    renders(5)
    stamp = time.perf_counter()
    for _ in range(40):
        eyes.observe_raw()
    stage('render alone', (time.perf_counter()-stamp)/40*1000)

    stamp = time.perf_counter()
    for _ in range(40):
        senses.observe()
        eyes.observe_raw()
    stage('after ReflexSenses.observe', (time.perf_counter()-stamp)/40*1000)

    stamp = time.perf_counter()
    for _ in range(40):
        ears.observe()
        eyes.observe_raw()
    stage('after BinauralSenses.observe', (time.perf_counter()-stamp)/40*1000)

    stamp = time.perf_counter()
    for _ in range(40):
        body.step(np.asarray(body.data.ctrl[:12]).copy(), duration=.01)
        eyes.observe_raw()
    stage('after body.step', (time.perf_counter()-stamp)/40*1000)

    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               eye_limits=(body.eye_lower_limits, body.eye_upper_limits),
                               motor_units=200, proprio_units=64, association_units=256)
    stamp = time.perf_counter()
    for _ in range(40):
        eyes.observe_raw()
    stage('render alone, after brain built', (time.perf_counter()-stamp)/40*1000)
    environment = {name: np.zeros(4) for name in ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
    observation = body.observe()
    stamp = time.perf_counter()
    for _ in range(40):
        brain.step(observation, environment=environment, eye_pixels=eyes.observe_raw(),
                   ear_waveform=ears.observe(), dt=.01, autonomy=True)
    stage('render inside brain.step', (time.perf_counter()-stamp)/40*1000)
    print('engine %s' % torch_execution.name_of(torch_execution.resolve()))
    eyes.close()


if __name__ == '__main__':
    main()
