"""Synthetic cue pairing: compare compressed-only and hidden-layer routes."""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from born_wired.distributed import DistributedFeatureNetwork


STIMULI = (np.array([1., 0, 0, 0]), np.array([0., 0, 0, 1.]))
CONFIG = dict(dt=.005, trial_seconds=.4, rest_seconds=.2, training_pairs=30,
              motor_teaching_strength=1.2, evaluation_seconds=.6, evaluation_tail_seconds=.15)


def measure_responses(model):
    responses = []
    for features in STIMULI:
        model.reset_state()
        records = []
        for _ in range(round(CONFIG["evaluation_seconds"] / CONFIG["dt"])):
            model.step(features, dt=CONFIG["dt"], learn=False)
            records.append(model.network.activity)
        mean = np.mean(records[-round(CONFIG["evaluation_tail_seconds"] / CONFIG["dt"]):], axis=0)
        responses.append({name: mean[ids].tolist() for name, ids in model.groups.items()})
    return responses


def evaluate_case(seed, hidden_routes, learn, motor_mapping=(0, 1)):
    if sorted(motor_mapping) != [0, 1]:
        raise ValueError("motor_mapping must be a permutation of (0, 1)")
    model = DistributedFeatureNetwork(hidden_routes=hidden_routes, seed=seed)
    before = measure_responses(model)
    model.reset_state()
    rng = np.random.default_rng(seed)
    sequence = []
    started = time.perf_counter()
    for _ in range(CONFIG["training_pairs"]):
        for stimulus_index in rng.permutation(2):
            sequence.append(int(stimulus_index))
            teacher = np.zeros(2)
            teacher[motor_mapping[stimulus_index]] = CONFIG["motor_teaching_strength"]
            for _ in range(round(CONFIG["trial_seconds"] / CONFIG["dt"])):
                model.step(STIMULI[stimulus_index], teacher, dt=CONFIG["dt"], learn=learn)
            for _ in range(round(CONFIG["rest_seconds"] / CONFIG["dt"])):
                model.step(np.zeros(4), dt=CONFIG["dt"], learn=learn)
    after = measure_responses(model)
    motor = np.array([response["motor"] for response in after])
    contrast = float((motor[0, 0] - motor[0, 1] + motor[1, 1] - motor[1, 0]) / 2)
    paired_contrast = float(np.mean([motor[i, motor_mapping[i]] - motor[i, 1-motor_mapping[i]] for i in range(2)]))
    return dict(seed=seed, hidden_routes=hidden_routes, learn=learn, config=CONFIG,
                stimuli=[v.tolist() for v in STIMULI], training_sequence=sequence,
                motor_teaching_mapping=list(motor_mapping), paired_motor_contrast=paired_contrast,
                training_steps=CONFIG["training_pairs"] * 2 * round((CONFIG["trial_seconds"] + CONFIG["rest_seconds"]) / CONFIG["dt"]),
                training_simulated_seconds=CONFIG["training_pairs"] * 2 * (CONFIG["trial_seconds"] + CONFIG["rest_seconds"]),
                motor_teaching_input_at_evaluation=[0, 0], before=before, after=after,
                motor_contrast=contrast, max_weight_change=float(np.max(np.abs(model.synapses.weights - model.initial_weights))),
                graph=model.graph_snapshot(), wall_seconds=time.perf_counter() - started)


def main():
    started = time.perf_counter()
    cases = [evaluate_case(seed, hidden, learn) for seed in (0, 1, 2)
             for hidden, learn in ((False, True), (True, True), (True, False))]
    cases.append(evaluate_case(0, True, True, motor_mapping=(1, 0)))
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  command="python tools/evaluate_hidden_routes.py", cases=cases,
                  wall_seconds=time.perf_counter() - started,
                  scope="Two synthetic feature patterns with externally paired unconditioned motor input during training. No image recognition, face identity, learned gait, or embodied capability is demonstrated.")
    output = Path(__file__).resolve().parents[1] / "artifacts" / "hidden_routes.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for case in cases:
        print({key: case[key] for key in ("seed", "hidden_routes", "learn", "motor_teaching_mapping", "motor_contrast", "max_weight_change")},
              "responses", [item["motor"] for item in case["after"]])


if __name__ == "__main__":
    main()
