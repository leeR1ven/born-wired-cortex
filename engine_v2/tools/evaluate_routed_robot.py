"""Continuous-body synthetic feature conditioning, retaining all failures."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from born_wired.feature_routed import FeatureRoutedController
from born_wired.go2_body import Go2Body

FEATURES = dict(A=np.array([1., 0, 0, 0]), B=np.array([0., 0, 0, 1.]), neutral=np.zeros(4))
CONFIG = dict(dt=.01, motor_units=100, pairs=6, teaching_flexion=.8,
              training_stimulus_seconds=2., training_rest_seconds=.5,
              compressed_route_cap=.10, hidden_route_cap=.55, new_route_plasticity=3.,
              initial_joint_noise=.01)


def evaluate_case(seed, hidden_routes, learn):
    body = Go2Body()
    body.reset(seed=seed, joint_noise=CONFIG["initial_joint_noise"])
    brain = FeatureRoutedController(body.home_angles, body.lower_limits, body.upper_limits,
                                    seed=seed, hidden_routes=hidden_routes, motor_units=CONFIG["motor_units"])
    observation = body.observe()
    stages = []
    started = time.perf_counter()

    def stage(name, cue, seconds, flexion=0, learning=False):
        nonlocal observation
        rows = []
        for _ in range(round(seconds / CONFIG["dt"])):
            target, activation = brain.step(observation, features=FEATURES[cue],
                                            flexion=flexion, dt=CONFIG["dt"], learn=learning)
            observation = body.step(target, duration=CONFIG["dt"], activation=activation)
            activity = brain.network.activity
            rows.append([observation["body_height"], observation["body_up"][2],
                         activity[brain.groups["flexion"][0]], activity[brain.groups["locomotion"][0]],
                         activity[brain.groups["feature_compressed"][0]],
                         float(np.mean(observation["joint_position"][[7, 10]])),
                         float(np.mean(activation))])
        tail = np.asarray(rows)[-min(len(rows), round(1 / CONFIG["dt"])):]
        mean = tail.mean(axis=0)
        stages.append(dict(name=name, cue=cue, seconds=seconds, end_time=observation["time"],
                           external_flexion=flexion, learning=learning, tail_seconds=len(tail)*CONFIG["dt"],
                           mean_height=float(mean[0]), min_height=float(np.min(np.asarray(rows)[:, 0])),
                           mean_up_z=float(mean[1]), flexion_activity=float(mean[2]),
                           locomotion_activity=float(mean[3]), compressed_activity=float(mean[4]),
                           mean_rear_thigh_angle=float(mean[5]), mean_recruitment=float(mean[6])))

    failure = None
    try:
        stage("settle", "neutral", 1)
        for cue in ("A", "B"):
            stage("before_"+cue, cue, 2)
            stage("before_rest_"+cue, "neutral", 1)
        for pair in range(CONFIG["pairs"]):
            stage(f"train_{pair}_A", "A", CONFIG["training_stimulus_seconds"], CONFIG["teaching_flexion"], learn)
            stage(f"train_{pair}_rest_A", "neutral", CONFIG["training_rest_seconds"], learning=learn)
            stage(f"train_{pair}_B", "B", CONFIG["training_stimulus_seconds"], learning=learn)
            stage(f"train_{pair}_rest_B", "neutral", CONFIG["training_rest_seconds"], learning=learn)
        for index, cue in enumerate(("B", "A", "B")):
            stage(f"after_{index}_"+cue, cue, 3)
            if index < 2:
                stage("after_rest_"+str(index), "neutral", 2)
    except Exception as error:
        failure = repr(error)
    edge_slice = slice(brain.original_n_edges, None)
    syn = brain.synapses
    final = {s["name"]: s for s in stages if s["name"].startswith("after_") and "rest" not in s["name"]}
    return_metrics = None
    if len(final) == 3:
        first_b, response_a, last_b = (final[name] for name in ("after_0_B", "after_1_A", "after_2_B"))
        return_metrics = dict(b_height_return_error=abs(last_b["mean_height"]-first_b["mean_height"]),
                              b_flexion_return_error=abs(last_b["flexion_activity"]-first_b["flexion_activity"]),
                              a_minus_b_flexion=response_a["flexion_activity"]-last_b["flexion_activity"],
                              b_minus_a_height=last_b["mean_height"]-response_a["mean_height"])
    return dict(seed=seed, hidden_routes=hidden_routes, learning_during_training=learn, config=CONFIG,
                failure=failure, simulated_seconds=float(body.data.time), wall_seconds=time.perf_counter()-started,
                neurons=syn.n_neurons, edges=len(syn.src), base_neurons=brain.original_n_neurons,
                base_edges=brain.original_n_edges, continuous_neural_and_body_states=True,
                stages=stages, return_metrics=return_metrics, diagnostics=brain.diagnostics(),
                added_edges=dict(src=syn.src[edge_slice].tolist(), dst=syn.dst[edge_slice].tolist(),
                                 initial_weights=brain.initial_weights[edge_slice].tolist(), weights=syn.weights[edge_slice].tolist(),
                                 lower=syn.lower[edge_slice].tolist(), upper=syn.w_max[edge_slice].tolist(),
                                 plasticity=syn.plasticity[edge_slice].tolist(), tether=syn.tether[edge_slice].tolist()))


def main():
    parser = argparse.ArgumentParser()
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=0)
    seed_group.add_argument("--seeds", type=str)
    args = parser.parse_args()
    try:
        seeds = [args.seed] if args.seeds is None else [int(value) for value in args.seeds.split(",")]
        if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be distinct nonnegative integers")
    except ValueError as error:
        parser.error(str(error))
    sources = ["born_wired/feature_routed.py", "born_wired/innate.py", "born_wired/adaptive.py",
               "born_wired/regulation.py", "born_wired/go2_body.py", "born_wired/synapses.py",
               "born_wired/encoding.py", "tools/evaluate_routed_robot.py"]
    hashes = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources}
    cases = []
    for seed in seeds:
        for hidden, learn in ((False, True), (True, True), (True, False)):
            case = evaluate_case(seed, hidden, learn)
            cases.append(case)
            print(dict(seed=seed, hidden_routes=hidden, learning=learn, failure=case["failure"],
                       return_metrics=case["return_metrics"]), flush=True)
    after_hashes = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources}
    command = f"python tools/evaluate_routed_robot.py --seeds {','.join(map(str, seeds))}"
    report = dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                  source_sha256=hashes, source_sha256_after=after_hashes, source_unchanged=hashes==after_hashes,
                  command=command, seeds=seeds,
                  cases=cases, scope="Synthetic four-channel current input, not camera vision. Continuous Go2 simulation with no neural/body reset between stages; test teacher input is zero. Structural cap .10 on the shared compressed route is an explicit prior, not learned or hidden.")
    (ROOT/"artifacts/routed_robot.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
