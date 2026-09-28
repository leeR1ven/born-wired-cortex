#!/usr/bin/env python3
"""Closed-loop Go2/InnateController validation with raw sample output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from born_wired.go2_body import Go2Body
from born_wired.innate import InnateController


DT = 0.01
SAMPLE_INTERVAL = 0.1
JOINT_NOISE = 0.01
TILT_LOW = -0.025
TILT_HIGH = 0.025
FALL_HEIGHT = 0.10
FALL_UP_Z = 0.65
DEFAULT_CASES = ("stand", "rise", "flex", "perturb", "rhythm", "silenced")
SOURCE_FILES = ("go2_body.py", "innate.py", "adaptive.py", "regulation.py", "synapses.py", "encoding.py")

# Only initial state, timed drive inputs, and external forces are declared here.
SCENARIOS = {
    "stand": {"pose": "stand"},
    "rise": {"pose": "crouch"},
    "flex": {
        "pose": "stand",
        "drives": {"flexion": ((5.0, 0.60, 0.8),)},
    },
    "perturb": {
        "pose": "stand",
        "forces": ((10.0, (0.0, 30.0, 0.0), 0.2),),
    },
    "rhythm": {
        "pose": "stand",
        "drives": {"locomotion": ((5.0, None, 0.8),)},
    },
    "silenced": {"pose": "stand", "silence_motor": True},
}


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("value must not be empty")
    return items


def parse_seeds(value: str) -> list[int]:
    try:
        return [int(item) for item in parse_csv(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    hashes = {}
    for name in SOURCE_FILES:
        path = ROOT / "born_wired" / name
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return hashes


def timed_value(windows, elapsed: float, total_seconds: float) -> float:
    value = 0.0
    for start, end_fraction, scheduled in windows:
        end = None if end_fraction is None else end_fraction * total_seconds
        if elapsed >= start and (end is None or elapsed < end):
            value = scheduled
    return value


def scenario_inputs(name: str, elapsed: float, total_seconds: float) -> tuple[float, float]:
    drives = SCENARIOS[name].get("drives", {})
    return (
        timed_value(drives.get("flexion", ()), elapsed, total_seconds),
        timed_value(drives.get("locomotion", ()), elapsed, total_seconds),
    )


def phase_activity(controller: InnateController) -> list[float]:
    activity = np.asarray(controller.network.activity)
    return activity[controller.groups["phase"]].tolist()


def make_sample(body: Go2Body, controller: InnateController, phase: np.ndarray,
                case: str, rear_indices: list[int]) -> dict:
    observation = body.observe()
    elapsed = float(observation["time"])
    if case == "silenced":
        fallen = None
    else:
        start = 2.0 if case == "rise" else 1.0
        fallen = bool(
            elapsed >= start
            and (
                observation["body_height"] < FALL_HEIGHT
                or observation["body_up"][2] < FALL_UP_Z
            )
        )
    ctrl = np.asarray(body.data.ctrl)
    return {
        "time": elapsed,
        "body_height": float(observation["body_height"]),
        "body_up_z": float(observation["body_up"][2]),
        "body_linear_velocity": np.asarray(
            observation["body_linear_velocity"], dtype=float
        ).tolist(),
        "foot_contact_count": int(np.count_nonzero(observation["foot_contact"])),
        "foot_contact": observation["foot_contact"].tolist(),
        "body_position": body.data.qpos[:3].tolist(),
        "torque_peak": float(np.max(np.abs(ctrl))) if ctrl.size else 0.0,
        "phase_activity": np.asarray(phase, dtype=float).tolist(),
        "rear_leg_joint_angles": np.asarray(
            observation["joint_position"][rear_indices], dtype=float
        ).tolist(),
        "fallen": fallen,
    }


def mean_or_none(values: np.ndarray):
    return None if values.size == 0 else np.mean(values, axis=0).tolist()


def summarize(samples: list[dict], case: str, total_seconds: float) -> dict:
    post_1s = [sample for sample in samples if sample["time"] >= 1.0]
    min_height_sample = min(post_1s, key=lambda sample: sample["body_height"]) if post_1s else None
    min_up_sample = min(post_1s, key=lambda sample: sample["body_up_z"]) if post_1s else None

    late_start = max(0.0, total_seconds - 5.0)
    late = [sample for sample in samples if sample["time"] >= late_start]
    early = [sample for sample in samples if sample["time"] <= 1.0]

    late_height = np.asarray([sample["body_height"] for sample in late], dtype=float)
    late_up_z = np.asarray([sample["body_up_z"] for sample in late], dtype=float)
    late_velocity = np.asarray(
        [sample["body_linear_velocity"] for sample in late], dtype=float
    )
    early_joints = np.asarray(
        [sample["rear_leg_joint_angles"] for sample in early], dtype=float
    )
    late_joints = np.asarray(
        [sample["rear_leg_joint_angles"] for sample in late], dtype=float
    )

    if early_joints.size and late_joints.size:
        early_mean_array = np.mean(early_joints, axis=0)
        late_mean_array = np.mean(late_joints, axis=0)
        early_mean = early_mean_array.tolist()
        late_mean = late_mean_array.tolist()
        joint_change = (late_mean_array - early_mean_array).tolist()
        joint_change_l2 = float(np.linalg.norm(late_mean_array - early_mean_array))
    else:
        early_mean = []
        late_mean = []
        joint_change = []
        joint_change_l2 = None

    fall_values = [sample["fallen"] for sample in samples if sample["fallen"] is not None]
    return {
        "fall_start_seconds": None if case == "silenced" else (2.0 if case == "rise" else 1.0),
        "fall_criteria": {
            "body_height_below": FALL_HEIGHT,
            "body_up_z_below": FALL_UP_Z,
        },
        "min_body_height_after_1s": (
            None if min_height_sample is None else min_height_sample["body_height"]
        ),
        "min_body_height_time": None if min_height_sample is None else min_height_sample["time"],
        "min_body_up_z_after_1s": None if min_up_sample is None else min_up_sample["body_up_z"],
        "min_body_up_z_time": None if min_up_sample is None else min_up_sample["time"],
        "last_5s_mean_body_height": None if late_height.size == 0 else float(late_height.mean()),
        "last_5s_mean_body_up_z": None if late_up_z.size == 0 else float(late_up_z.mean()),
        "last_5s_mean_linear_velocity": mean_or_none(late_velocity),
        "last_5s_mean_speed": (
            None if late_velocity.size == 0 else float(np.linalg.norm(late_velocity, axis=1).mean())
        ),
        "fallen_any": None if case == "silenced" or not fall_values else bool(any(fall_values)),
        "early_rear_joint_angle_mean": early_mean,
        "late_rear_joint_angle_mean": late_mean,
        "late_minus_early_rear_joint_angle": joint_change,
        "late_minus_early_rear_joint_angle_l2": joint_change_l2,
    }


def run_case(case: str, seed: int, seconds: float) -> dict:
    started = time.perf_counter()
    body = None
    controller = None
    samples: list[dict] = []
    result = {
        "case": case,
        "seed": seed,
        "simulated_seconds": seconds,
        "samples": samples,
    }

    try:
        body = Go2Body()
        rng = np.random.default_rng(seed)
        tilt = rng.uniform(TILT_LOW, TILT_HIGH, size=2)
        observation = body.reset(
            pose=SCENARIOS[case]["pose"],
            seed=seed,
            joint_noise=JOINT_NOISE,
            tilt=tilt,
        )
        controller = InnateController(
            body.home_angles,
            body.lower_limits,
            body.upper_limits,
            seed=seed,
        )
        rear_names = ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint")
        rear_indices = [body.joint_names.index(name) for name in rear_names]
        phase = np.asarray(phase_activity(controller), dtype=float)
        samples.append(make_sample(body, controller, phase, case, rear_indices))

        steps = int(round(seconds / DT))
        sample_stride = int(round(SAMPLE_INTERVAL / DT))
        force_applied = [False] * len(SCENARIOS[case].get("forces", ()))
        for tick in range(1, steps + 1):
            elapsed = (tick - 1) * DT
            for index, (start, force, duration) in enumerate(
                SCENARIOS[case].get("forces", ())
            ):
                if not force_applied[index] and elapsed >= start:
                    body.apply_force(force, duration)
                    force_applied[index] = True

            flexion, locomotion = scenario_inputs(case, elapsed, seconds)
            target, activation = controller.step(
                observation,
                flexion=flexion,
                locomotion=locomotion,
                dt=DT,
                learn=True,
                silence_motor=SCENARIOS[case].get("silence_motor", False),
            )
            phase = np.asarray(phase_activity(controller), dtype=float)
            observation = body.step(target, duration=DT, activation=activation)
            if tick % sample_stride == 0:
                samples.append(make_sample(body, controller, phase, case, rear_indices))

        result["metrics"] = summarize(samples, case, seconds)
        result["diagnostics"] = controller.diagnostics()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if samples:
            try:
                result["metrics"] = summarize(samples, case, seconds)
            except Exception as metric_exc:
                result["metrics_error"] = f"{type(metric_exc).__name__}: {metric_exc}"
        if controller is not None:
            try:
                result["diagnostics"] = controller.diagnostics()
            except Exception as diagnostic_exc:
                result["diagnostics_error"] = (
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                )

    result["wall_clock_seconds"] = time.perf_counter() - started
    result["sample_count"] = len(samples)
    return result


def progress_line(result: dict) -> str:
    if "error" in result:
        payload = {
            "case": result["case"],
            "seed": result["seed"],
            "status": "error",
            "error": result["error"],
        }
    else:
        metrics = result["metrics"]
        payload = {
            "case": result["case"],
            "seed": result["seed"],
            "status": "ok",
            "seconds": result["simulated_seconds"],
            "fallen": metrics["fallen_any"],
            "min_height": metrics["min_body_height_after_1s"],
            "late_mean_up_z": metrics["last_5s_mean_body_up_z"],
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2,3,4"))
    parser.add_argument("--cases", type=parse_csv, default=parse_csv(",".join(DEFAULT_CASES)))
    parser.add_argument("--output", default="artifacts/task05_robot.json")
    args = parser.parse_args()

    if args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be a positive finite number")
    if not np.isclose(args.seconds / DT, round(args.seconds / DT), rtol=0, atol=1e-9):
        parser.error("--seconds must be a multiple of 0.01")
    unknown = sorted(set(args.cases) - set(SCENARIOS))
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")

    started = time.perf_counter()
    hashes_at_start = source_hashes()
    results = []
    for case in args.cases:
        for seed in args.seeds:
            result = run_case(case, seed, args.seconds)
            results.append(result)
            print(progress_line(result), flush=True)

    output = Path(args.output)
    payload = {
        "versions": {
            "python": sys.version.split()[0],
            "python_full": sys.version,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
        },
        "config": {
            "seconds": args.seconds,
            "seeds": args.seeds,
            "cases": args.cases,
            "dt": DT,
            "sample_interval": SAMPLE_INTERVAL,
            "joint_noise": JOINT_NOISE,
            "tilt_range": [TILT_LOW, TILT_HIGH],
        },
        "source_sha256": hashes_at_start,
        "source_unchanged_during_run": hashes_at_start == source_hashes(),
        "wall_clock_seconds": None,
        "results": results,
    }
    payload["wall_clock_seconds"] = time.perf_counter() - started
    write_output(output, payload)
    return 1 if any("error" in result for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
