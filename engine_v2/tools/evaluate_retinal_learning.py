"""Offline stimulus protocol for local retinal learning; never a live policy."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from born_wired.embodied import EmbodiedController
from born_wired.go2_body import Go2Body


PROTOCOL = dict(dt=.01, learning_interval=.05, baseline_seconds=2.,
                training_rounds=5, blue_seconds_per_round=2., black_seconds_per_round=2.,
                washout_seconds=2., probe_seconds=2., readout_tail_seconds=.5,
                stimulus=dict(rows=[8, 15], columns=[15, 23], blue=255, other_channels=0,
                              eyes="both identical raw RGB", contact_front=.8),
                autonomy=False, probe_learning=False,
                controller=dict(motor_units=20, proprio_units=8, association_units=8))
SOURCE_FILES = ("born_wired/embodied.py", "born_wired/reflex_controller.py", "born_wired/innate.py",
                "born_wired/adaptive.py", "born_wired/regulation.py", "born_wired/synapses.py",
                "tools/evaluate_retinal_learning.py")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def controller(body, seed, small=True):
    brain = EmbodiedController(body.home_angles, body.lower_limits, body.upper_limits,
                               seed=seed, **(PROTOCOL["controller"] if small else {}))
    if not np.isclose(brain.network.learning_interval, PROTOCOL["learning_interval"]):
        raise ValueError("source learning_interval does not match protocol")
    return brain


def blank_environment():
    return {key: np.zeros(4) for key in ("body_touch", "foot_obstacle", "foot_load", "foot_slip")}


def stimuli():
    blue = np.zeros(EmbodiedController.eye_shape, dtype=np.uint8)
    blue[:, 8:15, 15:23, 2] = 255
    return blue, np.zeros_like(blue)


def run_phase(brain, observation, image, touch, seconds, learn):
    environment = blank_environment()
    environment["body_touch"][0] = touch
    tail_count = round(PROTOCOL["readout_tail_seconds"] / PROTOCOL["dt"])
    samples, targets = [], []
    for _ in range(round(seconds / PROTOCOL["dt"])):
        target, _ = brain.step(observation, environment=environment, eye_pixels=image,
                               dt=PROTOCOL["dt"], learn=learn, autonomy=False)
        rates = brain.network.activity
        samples.append([rates[brain.groups["aversive"]][1],
                        rates[brain.groups["appetitive"]][0],
                        rates[brain.groups["retinal_memory"]].max()])
        targets.append(target)
    samples, targets = np.asarray(samples), np.asarray(targets)
    return dict(aversive_center_tail_mean=float(samples[-tail_count:, 0].mean()),
                appetitive_tail_mean=float(samples[-tail_count:, 1].mean()),
                memory_peak_tail_mean=float(samples[-tail_count:, 2].mean()),
                target_tail_mean=targets[-tail_count:].mean(axis=0).tolist(),
                target_tail_std=targets[-tail_count:].std(axis=0).tolist(),
                aversive_center_peak=float(samples[:, 0].max()),
                all_finite=bool(np.isfinite(samples).all() and np.isfinite(targets).all()))


def memory_snapshot(brain, indices):
    values = brain.synapses.weights[indices]
    return dict(weights=values.tolist(), minimum=float(values.min()), maximum=float(values.max()),
                total=float(values.sum()), nonzero=int(np.count_nonzero(values)))


def one_case(body, seed, condition):
    started = time.perf_counter()
    brain = controller(body, seed)
    observation = body.observe()
    blue, black = stimuli()
    memory_ids = brain.groups["retinal_memory"]
    indices = np.flatnonzero(np.isin(brain.synapses.src, memory_ids))
    mapping = {int(cell): index for index, cell in enumerate(memory_ids)}
    targets = {int(cell): f"aversive_{index}" for index, cell in enumerate(brain.groups["aversive"])}
    targets[int(brain.groups["appetitive"][0])] = "appetitive"
    result = dict(seed=seed, condition=condition, status="running", neurons=brain.network.n_neurons,
                  edges=len(brain.synapses.src), parameter_sha256=fingerprint(dict(PROTOCOL, seed=seed, condition=condition)),
                  memory_edges=[dict(edge_index=int(index), memory_index=mapping[int(brain.synapses.src[index])],
                                     target=targets[int(brain.synapses.dst[index])], cap=float(brain.synapses.w_max[index]))
                                for index in indices])
    result["memory_before"] = memory_snapshot(brain, indices)
    result["baseline"] = run_phase(brain, observation, blue, 0., 2., False)
    training = []
    for index in range(PROTOCOL["training_rounds"]):
        contact_blue = .8 if condition != "unpaired" else 0.
        contact_black = .8 if condition == "unpaired" else 0.
        training.append(dict(round=index, blue=run_phase(brain, observation, blue, contact_blue, 2., condition != "learn_off"),
                             black=run_phase(brain, observation, black, contact_black, 2., condition != "learn_off")))
    result["training"] = training
    result["memory_after_training"] = memory_snapshot(brain, indices)
    result["washout"] = run_phase(brain, observation, black, 0., 2., False)
    result["probe"] = run_phase(brain, observation, blue, 0., 2., False)
    result["memory_after_probe"] = memory_snapshot(brain, indices)
    result["aversive_center_delta"] = (result["probe"]["aversive_center_tail_mean"]
                                       - result["baseline"]["aversive_center_tail_mean"])
    dq = np.asarray(result["probe"]["target_tail_mean"]) - result["baseline"]["target_tail_mean"]
    result["target_mean_delta"] = dq.tolist()
    result["target_max_abs_mean_delta_rad"] = float(np.max(np.abs(dq)))
    weights = brain.synapses.weights
    result["bounds_valid"] = bool(np.all(weights >= brain.synapses.lower) and np.all(weights <= brain.synapses.w_max))
    result["body_time_unchanged"] = bool(body.data.time == observation["time"])
    result["status"] = "complete"
    result["wall_seconds"] = time.perf_counter() - started
    return result


def latency(body, small):
    brain = controller(body, 0, small=small)
    pixels = np.random.default_rng(391).integers(0, 256, EmbodiedController.eye_shape, dtype=np.uint8)
    observation, environment = body.observe(), blank_environment()
    times = []
    for _ in range(100):
        started = time.perf_counter()
        brain.step(observation, environment=environment, eye_pixels=pixels, dt=.01, learn=True, autonomy=False)
        times.append(1000 * (time.perf_counter() - started))
    return dict(configuration="small" if small else "default", neurons=brain.network.n_neurons,
                edges=len(brain.synapses.src), steps=100, learning_interval=brain.network.learning_interval,
                mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times, 95)), maximum_ms=float(np.max(times)),
                over_10ms=int(np.count_nonzero(np.asarray(times) > 10)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/retinal_learning_v3.json")
    parser.add_argument("--resume", action="store_true", help="Keep completed cases if protocol/source hashes still match")
    args = parser.parse_args()
    hashes = source_hashes()
    report = dict(status="running", python=sys.version, python_executable=sys.executable,
                  platform=platform.platform(), numpy=np.__version__, mujoco=mujoco.__version__,
                  protocol=PROTOCOL, protocol_sha256=fingerprint(PROTOCOL), source_sha256=hashes, cases=[])
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous["source_sha256"] != hashes or previous["protocol_sha256"] != fingerprint(PROTOCOL):
            raise ValueError("cannot resume across changed code or protocol")
        report["cases"] = previous["cases"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    body = Go2Body()
    for seed in args.seeds:
        for condition in ("paired", "unpaired", "learn_off"):
            if any(case["seed"] == seed and case["condition"] == condition for case in report["cases"]):
                continue
            try:
                case = one_case(body, seed, condition)
            except Exception:
                case = dict(seed=seed, condition=condition, status="failed", traceback=traceback.format_exc())
            report["cases"].append(case)
            save()
            print(json.dumps({key: case.get(key) for key in ("seed", "condition", "status", "aversive_center_delta",
                                                           "target_max_abs_mean_delta_rad", "wall_seconds")}), flush=True)
    report["latency"] = [latency(body, True), latency(body, False)]
    report["source_sha256_at_end"] = source_hashes()
    report["source_unchanged"] = report["source_sha256_at_end"] == hashes
    report["status"] = "complete" if all(case["status"] == "complete" for case in report["cases"]) else "completed_with_failures"
    save()


if __name__ == "__main__":
    main()
