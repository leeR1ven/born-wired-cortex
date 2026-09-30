#!/usr/bin/env python3
"""Run a bounded closed-loop validation of the reflex controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from born_wired.go2_body import DEFAULT_MODEL, Go2Body
from born_wired.reflex_controller import ReflexController
from born_wired.reflex_senses import ReflexSenses
from born_wired.embodied import EmbodiedController
from born_wired.stereo_senses import RawEyes
from born_wired.binaural_senses import BinauralSenses


NEURAL_DT = 0.01
PHYSICS_DT = 0.002
SAMPLE_INTERVAL = 0.5
# How far it has to move before the clock on "is it still going anywhere" is
# reset.  Smaller than this and a shivering animal reads as travelling; bigger
# and a slow one reads as stopped.
PROGRESS_STEP_M = 0.05
CONTROLLER_DEFAULTS = {
    "motor_units": 200,
    "proprio_units": 64,
    "association_units": 256,
    "balanced_gait": True,
    "avoidance_gain": 0.25,
    "withdrawal_gain": 0.35,
    "startle_gain": 0.18,
    "support_calf": -1.5,
}
SOURCE_FILES = (
    "born_wired/embodied.py",
    "born_wired/auditory_neurons.py",
    "born_wired/stereo_senses.py",
    "born_wired/binaural_senses.py",
    "born_wired/reflex_controller.py",
    "born_wired/reflex_senses.py",
    "born_wired/go2_body.py",
    "born_wired/feature_routed.py",
    "born_wired/innate.py",
    "born_wired/adaptive.py",
    "born_wired/regulation.py",
    "born_wired/synapses.py",
    "born_wired/encoding.py",
)
SCENARIOS = {
    'motor_probe': {'autonomy': False, 'locomotion': .65},
    "autonomous": {"autonomy": True, "locomotion": 0.0},
    "driven": {"autonomy": False, "locomotion": 0.65},
    "rest": {"autonomy": False, "locomotion": 0.0},
    "amble": {"autonomy": False, "locomotion": 0.35},
    "sprint": {"autonomy": False, "locomotion": 1.0},
}

# How far past its own starting line the animal has to get before it counts
# as having crossed the thing in the way.  The number is measured along the
# direction the animal was facing at the start, so a scene that begins facing
# west is asked the same question as one that begins facing east.
CROSSED_PROGRESS = {
    'passage': 1.70, 'step': 1.10, 'ramp': 1.25, 'curb': .75,
    'platform': .35, 'wall': .98, 'ramp_top': .50, 'step_top': .30,
    'platform_top': .65, 'blocked': 1.45,
}


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("seeds must not be empty")
    return seeds


def positive_duration(value: str) -> float:
    try:
        duration = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("duration must be a number") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise argparse.ArgumentTypeError("duration must be positive and finite")
    return duration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def load_live_parameters() -> dict:
    with (ROOT / "live_config.json").open(encoding="utf-8-sig") as source:
        config = json.load(source)
    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("live_config.json parameters must be an object")
    return dict(parameters)


def as_list(value) -> list:
    return np.asarray(value).tolist()


def finite(value) -> bool:
    return bool(np.isfinite(np.asarray(value, dtype=float)).all())


def group_mean(controller: ReflexController, name: str) -> float:
    ids = np.asarray(controller.groups[name], dtype=int)
    return float(np.mean(controller.network.activity[ids]))


def base_position(body: Go2Body) -> np.ndarray:
    base_id = int(body.model.body("base").id)
    return body.data.xpos[base_id].copy()


def floor_z(body: Go2Body) -> float:
    floor_id = int(body.model.geom("floor").id)
    return float(body.data.geom_xpos[floor_id, 2])


def make_controller(
    body: Go2Body,
    *,
    seed: int,
    controller_parameters: dict,
) -> ReflexController:
    parameters = dict(controller_parameters)
    if getattr(body, "has_eyes", False) and "eye_limits" not in parameters:
        # The eyes are muscles, and a controller built without eye limits has
        # no eye motor cells at all: ``eye_command`` raises, so the walking
        # brain could not point its eyes no matter what its visual cortex saw.
        # The same limits the standing gaze exams pass.
        parameters["eye_limits"] = (body.eye_lower_limits, body.eye_upper_limits)
    return EmbodiedController(
        body.home_angles,
        body.lower_limits,
        body.upper_limits,
        seed=seed,
        **parameters,
    )


def controller_parameters(config_parameters: dict) -> dict:
    parameters = dict(CONTROLLER_DEFAULTS)
    parameters.update(config_parameters)
    return parameters


def initial_sample(
    *,
    time_s: float,
    position: np.ndarray,
    controller: ReflexController,
    environment: dict,
) -> dict:
    # One readout, not four: each call copies the weight table, and a sample
    # is taken twenty times a walk, so the three extra calls were three times
    # the cost of a sample for no extra information.
    reading = controller.diagnostics()
    diagnostics = reading["reflex_activity"]
    return {
        "time": float(time_s),
        "position": as_list(position),
        "drive": group_mean(controller, "rhythm_recruitment"),
        "forward_drive": group_mean(controller, "locomotion"),
        "curiosity": float(diagnostics["curiosity"][0]),
        "fatigue": float(diagnostics["fatigue"][0]),
        "rest": float(diagnostics["rest"][0]),
        "initiation": float(diagnostics["initiation"][0]),
        "retinal_activity": reading.get('retinal_activity'),
        "binocular_activity": reading.get('binocular_population_activity'),
        "auditory_activity": reading.get('auditory_activity'),
        "reflex_activity": diagnostics,
        "motor_effort": float(environment['motor_effort']),
        "body_touch": as_list(environment['body_touch']),
        "foot_obstacle": as_list(environment['foot_obstacle']),
    }


def sample(
    *,
    time_s: float,
    position: np.ndarray,
    controller: ReflexController,
    environment: dict,
) -> dict:
    return initial_sample(
        time_s=time_s,
        position=position,
        controller=controller,
        environment=environment,
    )


def run_seed(
    *,
    seed: int,
    duration: float,
    model_path: Path,
    scenario: str,
    controller_parameters_for_seed: dict,
    terrain_start: str = 'origin',
    injected: dict = None,
    start_yaw: float = None,
    props: dict = None,
    startle=None,
    distance_goal_m: float = None,
    pace_gate=None,
    stall_seconds: float = None,
) -> dict:
    """One closed-loop run.

    ``distance_goal_m`` ends the run the moment it has travelled that far, so a
    walker that gets there in forty seconds does not cost the four hundred a
    slow one would have.  ``pace_gate`` is ``(seconds, metres)``: at that time,
    if it has not covered that far, it never will in useful time and the run
    stops there.  ``stall_seconds`` stops a run that has not moved
    ``PROGRESS_STEP_M`` in that long.  All three are off unless asked for, so
    every existing exam reads exactly as it did.
    """
    scenario_inputs = SCENARIOS[scenario]
    steps = int(round(duration / NEURAL_DT))
    if not math.isclose(steps * NEURAL_DT, duration, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("duration must be an integer multiple of 0.01 seconds")

    body = Go2Body(model_path=model_path, timestep=PHYSICS_DT)
    # Perturb the physical start, not merely unused association weights.
    initial_yaw = float(np.random.default_rng(seed).uniform(-.12, .12))
    start_heading = float(initial_yaw + (0. if start_yaw is None else start_yaw))
    body.reset(seed=seed, joint_noise=.01, tilt=(0.,0.,start_heading))
    starts = {'origin': (0.,0.), 'passage': (-2.7,-1.05), 'step': (.55,-1.), 'ramp': (.25,-2.1),
              'curb': (.35,0.), 'platform': (.90,-2.1), 'wall': (2.6,0.),
              'ramp_top': (1.20,-2.1,.040), 'step_top': (1.35,-1.,.025),
              'platform_top': (1.90,-2.1,.060),
              'blocked': (.90,.20)}
    start_place = starts[terrain_start]
    body.data.qpos[:2] = start_place[:2]
    if len(start_place) > 2:
        # A scene that begins on top of a prop has to begin on top of it.
        body.data.qpos[body._base_qpos + 2] += float(start_place[2])
    if props:
        for prop_name, place in props.items():
            prop = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, prop_name)
            if prop < 0:
                raise ValueError('no such prop: %s' % prop_name)
            body.model.geom_pos[prop] = list(place)
    mujoco.mj_forward(body.model, body.data)
    facing = np.array([math.cos(start_heading), math.sin(start_heading)])
    senses = ReflexSenses(body)
    controller = make_controller(
        body,
        seed=seed,
        controller_parameters=controller_parameters_for_seed,
    )
    observation = body.observe()

    def sensed():
        values = senses.observe()
        if injected:
            for name, extra in injected.items():
                values[name] = np.clip(np.asarray(values[name], dtype=float)
                                       + np.broadcast_to(np.asarray(extra, dtype=float), (4,)),
                                       0., 1.)
        return values

    environment = sensed()
    eyes = (
        RawEyes(body, width=controller.eye_width, height=controller.eye_height)
        if body.model.ncam >= 2 else None
    )
    ears = BinauralSenses(body, window_samples=160)
    pixels = None

    position = base_position(body)
    initial_position = position.copy()
    previous_position = position.copy()
    horizontal_path = 0.0
    maximum_x = float(position[0])
    maximum_progress = 0.0
    minimum_progress = 0.0
    previous_yaw = float(np.arctan2(body.data.xmat[body._base].reshape(3,3)[1,0],
                                     body.data.xmat[body._base].reshape(3,3)[0,0]))
    smooth_yaw = previous_yaw
    sampled_yaw = previous_yaw
    yaw_smoothing = 1. - math.exp(-NEURAL_DT/.20)
    yaw_path = 0.0
    speeds = []
    up_z = []
    heights = []
    drive_values = []
    contact_slip = [[] for _ in range(4)]
    clearance_gt_2mm = [[] for _ in range(4)]
    timeline = [
        sample(
            time_s=0.0,
            position=position,
            controller=controller,
            environment=environment,
        )
    ]
    next_sample_time = SAMPLE_INTERVAL
    max_scaffold_relative_change = 0.0
    checks = {
        "all_finite": True,
        "targets_within_limits": True,
        "activation_in_range": True,
        "weights_finite": True,
        "weights_within_bounds": True,
        "activity_in_range": True,
        "adaptation_in_range": True,
        "observations_finite": True,
        "time_monotonic": True,
    }
    previous_time = float(body.data.time)
    started = time.perf_counter()
    error = None
    eye_travel_rad = 0.0
    eye_angle_rad = None
    travelled_m = 0.0
    elapsed_loop = 0.0
    reached_goal = False
    stalled_early = False
    too_slow = False
    progress_position = initial_position.copy()
    progress_time = 0.0

    try:
        for step_index in range(steps):
            if eyes is not None and scenario != 'motor_probe' and step_index % 10 == 0:
                pixels = eyes.observe_raw()
            startle_now = (float(startle) if isinstance(startle, (int, float))
                           else (0. if startle is None else float(startle(step_index*NEURAL_DT))))
            target, activation = controller.step(
                observation,
                environment=environment,
                autonomy=scenario_inputs["autonomy"],
                locomotion=scenario_inputs["locomotion"],
                dt=NEURAL_DT,
                learn=True,
                startle=startle_now,
                eye_pixels=pixels,
                ear_waveform=ears.observe() if scenario != 'motor_probe' else np.zeros((2,160)),
            )
            # The eyes are muscles, and the brain already says where to point
            # them: the walking loop simply never asked.  Uncommanded, the eye
            # motors held whatever angle they were left at, so a walking animal
            # could not look at anything no matter what its visual cortex did.
            # This is the same command the standing gaze exams use, sent every
            # step while it walks.
            if getattr(controller, "eye_encoder", None) is not None:
                eye_command = controller.eye_command()
                body.command_eyes(eye_command)
                eye_angle_now = .5 * (float(eye_command[0]) + float(eye_command[2]))
                if eye_angle_rad is not None:
                    eye_travel_rad += abs(eye_angle_now - eye_angle_rad)
                eye_angle_rad = eye_angle_now

            solved = body.step(target, duration=NEURAL_DT, activation=activation)
            observation = solved
            environment = sensed()

            target = np.asarray(target, dtype=float)
            activation = np.asarray(activation, dtype=float)
            weights = controller.synapses.weights
            checks["all_finite"] = checks["all_finite"] and bool(
                finite(target) and finite(activation) and finite(weights)
            )
            checks["targets_within_limits"] = checks["targets_within_limits"] and bool(
                np.all(target >= controller.lower) and np.all(target <= controller.upper)
            )
            checks["activation_in_range"] = checks["activation_in_range"] and bool(
                np.all((activation >= 0) & (activation <= 1))
            )
            checks["weights_finite"] = checks["weights_finite"] and finite(weights)
            checks["weights_within_bounds"] = checks["weights_within_bounds"] and bool(
                np.all(weights >= controller.synapses.lower)
                and np.all(weights <= controller.synapses.w_max)
            )
            checks["activity_in_range"] = bool(
                np.all((controller.network.activity >= 0) & (controller.network.activity <= 1))
            ) and checks["activity_in_range"]
            checks["adaptation_in_range"] = bool(
                np.all((controller.network.adaptation >= 0) & (controller.network.adaptation <= 1))
            ) and checks["adaptation_in_range"]
            checks["observations_finite"] = checks["observations_finite"] and bool(
                finite(solved["body_height"])
                and finite(solved["body_up"])
                and finite(solved["joint_position"])
                and finite(solved["joint_velocity"])
                and finite(environment["foot_slip_mps"])
            )

            current_time = float(body.data.time)
            checks["time_monotonic"] = checks["time_monotonic"] and current_time > previous_time
            previous_time = current_time

            # The whole dictionary was asked for here every step, and only
            # this one entry was read out of it: the other four cost about
            # 4 ms a step and went in the bin.  Same number, asked for by name.
            max_scaffold_relative_change = max(
                max_scaffold_relative_change,
                controller.scaffold_drift(weights),
            )

            position = base_position(body)
            elapsed_loop = (step_index + 1) * NEURAL_DT
            travelled_m = float(np.linalg.norm(position[:2] - initial_position[:2]))
            if distance_goal_m is not None and travelled_m >= distance_goal_m:
                reached_goal = True
                break
            if float(np.linalg.norm(position[:2] - progress_position[:2])) >= PROGRESS_STEP_M:
                progress_position = position.copy()
                progress_time = elapsed_loop
            if stall_seconds is not None and elapsed_loop - progress_time > stall_seconds:
                stalled_early = True
                break
            if (pace_gate is not None and elapsed_loop >= pace_gate[0]
                    and travelled_m < pace_gate[1]):
                too_slow = True
                break
            maximum_x = max(maximum_x, float(position[0]))
            horizontal_path += float(np.linalg.norm(position[:2] - previous_position[:2]))
            progress = float((position[:2] - initial_position[:2]) @ facing)
            maximum_progress = max(maximum_progress, progress)
            minimum_progress = min(minimum_progress, progress)
            rotation_now = body.data.xmat[body._base].reshape(3, 3)
            yaw_now = float(np.arctan2(rotation_now[1,0], rotation_now[0,0]))
            previous_yaw = yaw_now
            smooth_yaw += yaw_smoothing * math.atan2(math.sin(yaw_now - smooth_yaw),
                                                     math.cos(yaw_now - smooth_yaw))
            if (step_index + 1) % 5 == 0:
                yaw_path += abs(float(np.angle(np.exp(1j*(smooth_yaw - sampled_yaw)))))
                sampled_yaw = smooth_yaw
            speeds.append(float(np.linalg.norm(np.asarray(solved['body_linear_velocity'])[:2])))
            previous_position = position.copy()
            up_z.append(float(solved["body_up"][2]))
            heights.append(float(solved["body_height"]))
            drive_values.append(group_mean(controller, "rhythm_recruitment"))

            for foot in range(4):
                if bool(environment["foot_contact"][foot]):
                    contact_slip[foot].append(float(environment["foot_slip_mps"][foot]))
                clearance_gt_2mm[foot].append(
                    bool(float(solved["foot_position"][foot, 2])
                         - float(body.model.geom_size[body._feet[foot], 0]) - floor_z(body) > 0.002)
                )

            elapsed = (step_index + 1) * NEURAL_DT
            if elapsed + 1e-9 >= next_sample_time:
                entry = sample(
                    time_s=elapsed,
                    position=position,
                    controller=controller,
                    environment=environment,
                )
                entry["eye_angle_rad"] = eye_angle_rad
                timeline.append(entry)
                next_sample_time += SAMPLE_INTERVAL
    except Exception as exc:  # Preserve failed seeds in the report.
        error = f"{type(exc).__name__}: {exc}"

    wall_duration = time.perf_counter() - started
    if eyes is not None:
        eyes.close()
    final_position = previous_position
    slip_summary = [
        {
            "mean_mps": None if not values else float(np.mean(values)),
            "max_mps": None if not values else float(np.max(values)),
            "contact_samples": len(values),
        }
        for values in contact_slip
    ]
    clearance_summary = [
        {
            "fraction_gt_2mm": None if not values else float(np.mean(values)),
            "samples": len(values),
        }
        for values in clearance_gt_2mm
    ]
    return {
        "seed": seed,
        "initial_joint_noise_rad": .01,
        "initial_yaw_rad": initial_yaw,
        "terrain_start": terrain_start,
        "scenario": scenario,
        "status": "ok" if error is None else "error",
        "behavior_pass": bool(error is None and all(checks.values()) and up_z
                              and min(up_z) > .5 and min(heights) > .15),
        "error": error,
        "steps": steps,
        "wall_duration_seconds": wall_duration,
        "metrics": {
            "minimum_up_z": None if not up_z else float(np.min(up_z)),
            "minimum_height_m": None if not heights else float(np.min(heights)),
            "total_displacement_m": float(np.linalg.norm(final_position - initial_position)),
            "horizontal_path_m": horizontal_path,
            "maximum_world_x": maximum_x,
            "crossed_test_marker": (None if terrain_start not in CROSSED_PROGRESS
                                     else maximum_progress > CROSSED_PROGRESS[terrain_start]),
            "final_yaw_rad": float(np.arctan2(
                body.data.xmat[body._base].reshape(3,3)[1,0],
                body.data.xmat[body._base].reshape(3,3)[0,0])) - start_heading,
            "maximum_progress_m": float(maximum_progress),
            "minimum_progress_m": float(minimum_progress),
            "yaw_path_rad": float(yaw_path),
            "lateral_m": float((final_position[:2] - initial_position[:2])
                                @ np.array([-facing[1], facing[0]])),
            "straightness": (float(np.linalg.norm(final_position[:2] - initial_position[:2])
                                    / horizontal_path) if horizontal_path > 0 else 0.),
            "max_speed_mps": None if not speeds else float(np.max(speeds)),
            "mean_speed_mps": None if not speeds else float(np.mean(speeds)),
            "fraction_drive_gt_0_35": None if not drive_values else float(np.mean(np.asarray(drive_values) > 0.35)),
            "fraction_drive_lt_0_03": None if not drive_values else float(np.mean(np.asarray(drive_values) < 0.03)),
            "contact_conditioned_foot_slip": slip_summary,
            "foot_clearance_gt_2mm_fraction": clearance_summary,
            "maximum_scaffold_relative_change": max_scaffold_relative_change,
            "eye_travel_rad": float(eye_travel_rad),
            "final_eye_angle_rad": eye_angle_rad,
        },
        "ended": {
            "reached_goal": bool(reached_goal),
            "stalled": bool(stalled_early),
            "too_slow": bool(too_slow),
            "travelled_m": float(travelled_m),
            "elapsed_s": float(elapsed_loop),
        },
        "checks": checks,
        "no_resets": True,
        "explicit_resets": 0,
        "timeline": timeline,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument("--duration", type=positive_duration, default=60.0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="autonomous")
    parser.add_argument('--terrain-start', choices=('origin','passage','step','ramp'), default='origin')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = (args.model if args.model is not None else Path(DEFAULT_MODEL)).resolve()
    config_parameters = load_live_parameters()
    source_sha256 = source_hashes()
    results = []
    for seed in args.seeds:
        parameters = controller_parameters(config_parameters)
        try:
            result = run_seed(
                seed=seed,
                duration=args.duration,
                model_path=model_path,
                scenario=args.scenario,
                controller_parameters_for_seed=parameters,
                terrain_start=args.terrain_start,
            )
        except Exception as exc:
            result = {
                "seed": seed,
                "scenario": args.scenario,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": None,
                "checks": None,
                "no_resets": None,
                "timeline": [],
            }
        result["controller_parameters"] = dict(parameters, seed=seed)
        results.append(result)
        print(json.dumps({
            "seed": seed,
            "scenario": args.scenario,
            "status": result["status"],
            "error": result["error"],
        }, ensure_ascii=False))

    payload = {
        "scenario": args.scenario,
        "duration_seconds": args.duration,
        "neural_dt_seconds": NEURAL_DT,
        "physics_dt_seconds": PHYSICS_DT,
        "learn": True,
        "learning_interval_seconds": .05,
        "drive_metric": "mean activity of shared rhythm_recruitment neurons, includes orienting",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "model_source": "cli" if args.model is not None else "default",
        "live_config_parameters": config_parameters,
        "controller_defaults": CONTROLLER_DEFAULTS,
        "source_sha256": source_sha256,
        "versions": {
            "python": platform.python_version(),
            "python_full": sys.version,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if all(r.get('behavior_pass', False) for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
