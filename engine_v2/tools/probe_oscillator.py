"""Reproduce a two-cell autonomous inhibitory oscillator; no periodic input."""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from born_wired.adaptive import AdaptiveNetwork
from born_wired.synapses import BoundedSynapses


def measure_case(*, symmetric=False, learn=True, adaptation_gain=2.5, seconds=12, dt=.002):
    synapses = BoundedSynapses([0, 1], [1, 0], [2.5, 2.5], [-1, -1], 2,
                              w_max=3, budgets=3, learning_rate=.01, decay=.001)
    net = AdaptiveNetwork(synapses, bias=.8, initial_voltage=[0, 0] if symmetric else [.001, 0],
                          tau=.08, adaptation_tau=.4, adaptation_gain=adaptation_gain)
    count = round(seconds / dt)
    rates = np.empty((count, 2))
    min_voltage, max_voltage = np.inf, -np.inf
    min_adaptation, max_adaptation = np.inf, -np.inf
    started = time.perf_counter()
    for step in range(count):
        rates[step] = net.step(0, dt=dt, learn=learn)
        min_voltage = min(min_voltage, float(net.voltage.min()))
        max_voltage = max(max_voltage, float(net.voltage.max()))
        min_adaptation = min(min_adaptation, float(net.adaptation.min()))
        max_adaptation = max(max_adaptation, float(net.adaptation.max()))
    tail = rates[round(5 / dt):]
    difference = tail[:, 0] - tail[:, 1]
    crossings = np.flatnonzero((difference[:-1] < 0) & (difference[1:] >= 0))
    periods = np.diff(crossings) * dt
    standard_deviations = np.std(tail, axis=0)
    correlation = float(np.corrcoef(tail.T)[0, 1]) if np.all(standard_deviations > 1e-10) else None
    return dict(symmetric_initial_state=symmetric, learn=learn, adaptation_gain=adaptation_gain,
                steps=count, simulated_seconds=seconds, observation_window_seconds=[5, seconds],
                rate_min=tail.min(axis=0).tolist(), rate_max=tail.max(axis=0).tolist(),
                peak_to_peak=np.ptp(tail, axis=0).tolist(), correlation=correlation,
                positive_crossings_after_five_seconds=len(crossings),
                frequency_hz=float(1 / periods.mean()) if len(periods) else None,
                voltage_range=[min_voltage, max_voltage], adaptation_range=[min_adaptation, max_adaptation],
                initial_weights=[2.5, 2.5], final_weights=synapses.weights.tolist(),
                sampled_activity_after_five_seconds=tail[::round(.1 / dt)].tolist(),
                wall_seconds=time.perf_counter() - started)


def main():
    cases = [measure_case(), measure_case(learn=False), measure_case(symmetric=True),
             measure_case(adaptation_gain=0)]
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  command="python tools/probe_oscillator.py", deterministic=True,
                  config=dict(n_neurons=2, signs=[-1, -1], src=[0, 1], dst=[1, 0], weights=2.5,
                              bias=.8, tau=.08, adaptation_tau=.4, adaptation_gain=2.5,
                              dt=.002, external_exc=0, external_inh=0, modulator=1,
                              asymmetric_initial_voltage=[.001, 0], initial_adaptation=0,
                              w_max=3, budgets=3, learning_rate=.01, decay=.001, plasticity=1),
                  cases=cases,
                  scope="Autonomous rate oscillation only, not a robot gait or a guarantee of long-term stability. Exact symmetric initial states can remain symmetric and stop alternating.")
    output = Path(__file__).resolve().parents[1] / "artifacts" / "oscillator_probe.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for case in cases:
        print({key: case[key] for key in ("learn", "symmetric_initial_state", "adaptation_gain", "peak_to_peak", "frequency_hz", "correlation")})


if __name__ == "__main__":
    main()
