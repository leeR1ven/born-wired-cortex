"""Reproducible quantitative experiment for task 04 encoders."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from born_wired.encoding import RecruitmentEncoder, TuningEncoder


N_CHANNELS = 12
NEURON_COUNTS = (10, 25, 50, 100)
INPUT_SAMPLES = 1001
NOISE_STDS = (0.0, 0.01, 0.05)
NOISE_OBSERVATIONS = 200
SEEDS = (0, 1, 2, 3, 4)
DECISION_BOUNDARY = 0.505


def encoder_for(name: str, neurons: int) -> RecruitmentEncoder | TuningEncoder:
    if name == "recruitment":
        return RecruitmentEncoder(N_CHANNELS, neurons)
    if name == "tuning":
        return TuningEncoder(N_CHANNELS, neurons)
    raise ValueError(f"unknown encoder: {name}")


def ideal_scalar(name: str, values: np.ndarray, neurons: int) -> np.ndarray:
    if name == "recruitment":
        return np.rint(values * neurons) / neurons
    return values


def encode_batch(
    encoder: RecruitmentEncoder | TuningEncoder,
    values: np.ndarray,
) -> np.ndarray:
    return np.stack([encoder.encode(row) for row in values], axis=0)


def decode_batch(
    encoder: RecruitmentEncoder | TuningEncoder,
    activity: np.ndarray,
) -> np.ndarray:
    return np.stack([encoder.decode(row) for row in activity], axis=0)


def roundtrip_metrics(
    encoder: RecruitmentEncoder | TuningEncoder,
    values: np.ndarray,
    name: str,
) -> dict[str, Any]:
    activity = encode_batch(encoder, values)
    decoded = decode_batch(encoder, activity)
    error = decoded - values
    ideal = ideal_scalar(name, values, encoder.neurons_per_channel)
    ideal_error = decoded - ideal
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs_error_from_scalar_readout": float(
            np.max(np.abs(ideal_error))
        ),
        "exact_fraction_atol_1e-12": float(
            np.mean(np.isclose(error, 0.0, atol=1e-12, rtol=0.0))
        ),
        "all_finite": bool(np.all(np.isfinite(decoded))),
    }


def experiment_1() -> dict[str, Any]:
    uniform = np.linspace(0.0, 1.0, INPUT_SAMPLES)
    mixed = np.empty((INPUT_SAMPLES, N_CHANNELS), dtype=np.float64)
    for channel in range(N_CHANNELS):
        mixed[:, channel] = np.roll(uniform, channel * 97)

    results: dict[str, Any] = {
        "uniform_value_count": INPUT_SAMPLES,
        "mixed_channel_value_count": INPUT_SAMPLES,
        "encoders": {},
    }
    for name in ("recruitment", "tuning"):
        encoder_results: dict[str, Any] = {}
        for neurons in NEURON_COUNTS:
            encoder = encoder_for(name, neurons)
            uniform_values = np.repeat(uniform[:, np.newaxis], N_CHANNELS, axis=1)
            encoder_results[str(neurons)] = {
                "uniform_channels": roundtrip_metrics(
                    encoder, uniform_values, name
                ),
                "mixed_channels": roundtrip_metrics(encoder, mixed, name),
            }
        results["encoders"][name] = encoder_results
    return results


def pattern_difference_report(
    encoder: RecruitmentEncoder | TuningEncoder,
) -> dict[str, Any]:
    probes = np.full((4, N_CHANNELS), 0.5, dtype=np.float64)
    probes[:, 0] = (0.50, 0.51, 0.52, 0.54)
    activity = encode_batch(encoder, probes)
    decoded = decode_batch(encoder, activity)
    probe_values = probes[:, 0]
    comparisons: list[dict[str, Any]] = []
    for left in range(2):
        for right in range(left + 1, 4):
            difference = activity[left] - activity[right]
            comparisons.append(
                {
                    "left": float(probe_values[left]),
                    "right": float(probe_values[right]),
                    "decoded_left_channel_0": float(decoded[left, 0]),
                    "decoded_right_channel_0": float(decoded[right, 0]),
                    "different_units_channel_0": int(
                        np.count_nonzero(difference[0])
                    ),
                    "different_units_all_channels": int(
                        np.count_nonzero(difference)
                    ),
                    "l1_channel_0": float(np.sum(np.abs(difference[0]))),
                    "max_abs_channel_0": float(np.max(np.abs(difference[0]))),
                }
            )
    return {"probing_values": [0.50, 0.51, 0.52, 0.54], "pairs": comparisons}


def experiment_2() -> dict[str, Any]:
    conditions: list[dict[str, Any]] = []
    for name in ("recruitment", "tuning"):
        for neurons in NEURON_COUNTS:
            encoder = encoder_for(name, neurons)
            pattern_050 = np.full(N_CHANNELS, 0.5, dtype=np.float64)
            pattern_051 = pattern_050.copy()
            pattern_051[0] = 0.51

            for noise_std in NOISE_STDS:
                seed_records: list[dict[str, Any]] = []
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    observations: list[dict[str, Any]] = []
                    for value in (0.50, 0.51):
                        base = encoder.encode(
                            pattern_050 if value == 0.50 else pattern_051
                        )
                        activity = np.repeat(
                            base[np.newaxis, :, :],
                            NOISE_OBSERVATIONS,
                            axis=0,
                        )
                        if noise_std > 0.0:
                            activity = activity + rng.normal(
                                0.0, noise_std, size=activity.shape
                            )
                        activity = np.clip(activity, 0.0, 1.0)

                        decoded = np.full(NOISE_OBSERVATIONS, np.nan)
                        invalid = 0
                        for observation, row in enumerate(activity):
                            try:
                                decoded[observation] = encoder.decode(row)[0]
                            except ValueError:
                                invalid += 1

                        valid = np.isfinite(decoded)
                        predictions = np.full(NOISE_OBSERVATIONS, -1, dtype=np.int8)
                        predictions[valid] = (
                            decoded[valid] >= DECISION_BOUNDARY
                        ).astype(np.int8)
                        expected = 1 if value == 0.51 else 0
                        accuracy = float(np.mean(predictions == expected))
                        errors = np.abs(decoded[valid] - value)
                        observations.append(
                            {
                                "value": value,
                                "observations": NOISE_OBSERVATIONS,
                                "accuracy": accuracy,
                                "mae_valid": (
                                    float(np.mean(errors))
                                    if errors.size
                                    else None
                                ),
                                "rmse_valid": (
                                    float(np.sqrt(np.mean(np.square(errors))))
                                    if errors.size
                                    else None
                                ),
                                "invalid_observations": invalid,
                                "invalid_fraction": invalid
                                / NOISE_OBSERVATIONS,
                            }
                        )

                    low, high = observations
                    seed_records.append(
                        {
                            "seed": seed,
                            "accuracy_0_50": low["accuracy"],
                            "accuracy_0_51": high["accuracy"],
                            "balanced_accuracy": 0.5
                            * (low["accuracy"] + high["accuracy"]),
                            "balanced_mae_valid": _mean_optional(
                                low["mae_valid"], high["mae_valid"]
                            ),
                            "invalid_fraction": (
                                low["invalid_fraction"]
                                + high["invalid_fraction"]
                            )
                            / 2.0,
                            "classes": observations,
                        }
                    )

                conditions.append(
                    {
                        "encoder": name,
                        "neurons_per_channel": neurons,
                        "noise_std": noise_std,
                        "seeds": list(SEEDS),
                        "aggregate": {
                            "balanced_accuracy_mean": float(
                                np.mean(
                                    [
                                        item["balanced_accuracy"]
                                        for item in seed_records
                                    ]
                                )
                            ),
                            "balanced_accuracy_min": float(
                                np.min(
                                    [
                                        item["balanced_accuracy"]
                                        for item in seed_records
                                    ]
                                )
                            ),
                            "balanced_mae_valid_mean": float(
                                np.mean(
                                    [
                                        item["balanced_mae_valid"]
                                        for item in seed_records
                                        if item["balanced_mae_valid"] is not None
                                    ]
                                )
                            ),
                            "invalid_fraction_mean": float(
                                np.mean(
                                    [
                                        item["invalid_fraction"]
                                        for item in seed_records
                                    ]
                                )
                            ),
                        },
                        "per_seed": seed_records,
                    }
                )

    return {
        "decision_boundary": DECISION_BOUNDARY,
        "observations_per_class": NOISE_OBSERVATIONS,
        "noise_model": (
            "iid N(0, std^2) added independently to every (channel, unit) "
            "activity after encoding, then clipped to [0, 1]"
        ),
        "invalid_observation": (
            "a decoded row with zero total activity; counted as incorrect"
        ),
        "conditions": conditions,
    }


def _mean_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return float(np.mean(values)) if values else None


def experiment_3() -> dict[str, Any]:
    values = np.linspace(0.0, 1.0, INPUT_SAMPLES)
    inputs = np.repeat(values[:, np.newaxis], N_CHANNELS, axis=1)
    budgets: list[dict[str, Any]] = []
    for name in ("recruitment", "tuning"):
        for neurons in NEURON_COUNTS:
            encoder = encoder_for(name, neurons)
            activity = encode_batch(encoder, inputs)
            total_activity = np.sum(activity, axis=(1, 2))
            budgets.append(
                {
                    "encoder": name,
                    "neurons_per_channel": neurons,
                    "population_neurons": encoder.size,
                    "mean_total_activity_per_input": float(
                        np.mean(total_activity)
                    ),
                    "mean_activity_per_channel": float(
                        np.mean(total_activity) / N_CHANNELS
                    ),
                    "mean_active_fraction": float(np.mean(activity)),
                }
            )
    return {
        "input_distribution": "1001 uniform values repeated on 12 channels",
        "noise_model": (
            "budget comparison is noiseless; experiment 2 uses iid Gaussian "
            "unit noise followed by clipping"
        ),
        "budgets": budgets,
        "interpretation_limit": (
            "Both encoders use the same neurons per channel, but their mean "
            "activity differs. This experiment does not equalize biological "
            "cost and cannot establish an optimum under equal cost."
        ),
    }


def main() -> None:
    started = time.perf_counter()

    section_started = time.perf_counter()
    experiment_one = experiment_1()
    experiment_one["pattern_differences"] = {
        name: {
            str(neurons): pattern_difference_report(
                encoder_for(name, neurons)
            )
            for neurons in NEURON_COUNTS
        }
        for name in ("recruitment", "tuning")
    }
    time_one = time.perf_counter() - section_started

    section_started = time.perf_counter()
    experiment_two = experiment_2()
    time_two = time.perf_counter() - section_started

    section_started = time.perf_counter()
    experiment_three = experiment_3()
    time_three = time.perf_counter() - section_started

    elapsed = time.perf_counter() - started
    output = {
        "task": "04_fine_motor_encoding",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "python_full": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": elapsed,
            "section_elapsed_seconds": {
                "experiment_1": time_one,
                "experiment_2": time_two,
                "experiment_3": time_three,
            },
        },
        "configuration": {
            "n_channels": N_CHANNELS,
            "neurons_per_channel": list(NEURON_COUNTS),
            "roundtrip_input_values": INPUT_SAMPLES,
            "noise_stds": list(NOISE_STDS),
            "noise_observations_per_class": NOISE_OBSERVATIONS,
            "seeds": list(SEEDS),
            "decision_boundary": DECISION_BOUNDARY,
        },
        "experiment_1_roundtrip_and_patterns": experiment_one,
        "experiment_2_noisy_measurement": experiment_two,
        "experiment_3_budget": experiment_three,
        "failures": [],
        "limitations": [
            (
                "The tuning curve is a rate approximation and does not claim "
                "infinite physical resolution for every floating-point value."
            ),
            (
                "The 0.505 boundary is an offline diagnostic only and is not "
                "connected to control."
            ),
            (
                "Mean activity differs between encoders, so this experiment "
                "does not compare biological cost or optimality."
            ),
            (
                "This module encodes and decodes signals; it does not claim "
                "control of a physical robot or MuJoCo body."
            ),
        ],
    }

    output_path = ROOT / "artifacts" / "task04_encoding.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    old_recruitment = experiment_one["encoders"]["recruitment"]["10"][
        "uniform_channels"
    ]
    fine_recruitment = experiment_one["encoders"]["recruitment"]["100"][
        "uniform_channels"
    ]
    old_cross = experiment_one["encoders"]["recruitment"]["10"][
        "mixed_channels"
    ]
    print(f"artifact={output_path}")
    print(
        "recruitment K10 max/mean error "
        f"{old_recruitment['max_abs_error']:.6f}/"
        f"{old_recruitment['mean_abs_error']:.6f}; "
        f"mixed max {old_cross['max_abs_error']:.6f}; "
        f"K100 max {fine_recruitment['max_abs_error']:.6f}"
    )
    print(f"elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
