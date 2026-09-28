"""录下来的步和一步步发的步：结果是否逐位相同，各花多少时间。

用法：python tools/probe_recorded_step.py [步数] [--cells N]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired import torch_execution
from born_wired.adaptive import AdaptiveNetwork
from born_wired.regulation import RegulatedSynapses


def build(cells, seed=5):
    rng = np.random.default_rng(seed)
    count = 40 * cells
    src = rng.integers(0, cells, 2 * count)
    dst = rng.integers(0, cells, 2 * count)
    apart = src != dst
    src, dst = src[apart], dst[apart]
    # One row per directed pair: the graph holds at most one edge between two
    # cells, so the probe draws pairs and keeps the first of each.
    _, first = np.unique(src*cells + dst, return_index=True)
    first = np.sort(first)[:count]
    src, dst = src[first], dst[first]
    return RegulatedSynapses(src, dst, rng.uniform(0, .5, count),
                             np.where(rng.random(cells) < .8, 1, -1), cells,
                             lower=0, upper=.8, budgets=40., plasticity=.4, tether=.05,
                             learning_rate=.12, target_activity=.15)


def run(recorded, steps, cells, seed=5, dt=.01, learning_interval=.05):
    os.environ[torch_execution.RECORDING] = 'on' if recorded else 'off'
    synapses = build(cells, seed)
    network = AdaptiveNetwork(synapses, tau=.035, adaptation_tau=.4, adaptation_gain=2.5,
                              bias=0., learning_interval=learning_interval)
    rng = np.random.default_rng(seed + 1)
    inputs = [(rng.uniform(0, 1., cells), rng.uniform(0, .3, cells) if step % 3 == 0 else None)
              for step in range(steps)]
    started = time.perf_counter()
    for exc, inh in inputs:
        network.step(exc, inh, dt=dt, learn=False)
    for exc, inh in inputs:
        network.step(exc, inh, dt=dt)
    elapsed = time.perf_counter() - started
    device = network._device
    return dict(seconds=elapsed, steps=2*steps, activity=network.activity,
                voltage=network.voltage, adaptation=network.adaptation,
                weights=synapses.weights,
                path='recorded' if device is not None and device.recorded_available()
                     else ('eager' if device is not None else 'host'))


def main():
    arguments = list(sys.argv[1:])
    steps, cells = 300, 4000
    while arguments:
        argument = arguments.pop(0)
        if argument == '--cells':
            cells = int(arguments.pop(0))
        else:
            steps = int(argument)
    recorded = run(True, steps, cells)
    eager = run(False, steps, cells)
    print(f"{cells} 细胞   {recorded['steps']} 步   录制路径 {recorded['path']}   "
          f"逐步路径 {eager['path']}")
    print(f"  逐步 {1000*eager['seconds']/eager['steps']:7.3f} ms/步    "
          f"录制 {1000*recorded['seconds']/recorded['steps']:7.3f} ms/步    "
          f"快 {eager['seconds']/recorded['seconds']:.2f} 倍")
    for field in ('activity', 'voltage', 'adaptation', 'weights'):
        left, right = np.asarray(recorded[field]), np.asarray(eager[field])
        print(f"  {field:10s} 逐位相同 {np.array_equal(left, right)}   "
              f"最大差 {np.max(np.abs(left - right)):.3e}")


if __name__ == '__main__':
    main()
