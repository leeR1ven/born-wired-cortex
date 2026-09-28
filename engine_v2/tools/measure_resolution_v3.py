"""Measure raw resolution and fixed-threshold noise behavior of encoders."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from born_wired.encoding import RecruitmentEncoder, TuningEncoder


TOOL_VERSION = "resolution-v3-1"
N_CHANNELS = 12
CHANGED_CHANNEL = 0
BASE_VALUE = 0.5
INPUT_VALUES = (0.500, 0.502, 0.505, 0.510)
RECRUITMENT_SIZES = (10, 100, 200, 400)
TUNING_SIZES = (32, 64, 128)
NOISE_STDS = (0.0, 0.01, 0.05)
NOISE_INPUTS = (0.500, 0.505)
SEEDS = tuple(range(10))
SAMPLES_PER_INPUT_PER_SEED = 1000
DECODE_THRESHOLD = 0.5025
CROSSTALK_NOISE_STD = 0.05
ARTIFACT_PATH = ROOT / "artifacts" / "resolution_v3.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_encoder(
    family: str, neurons_per_channel: int, n_channels: int
) -> RecruitmentEncoder | TuningEncoder:
    if family == "recruitment":
        return RecruitmentEncoder(
            n_channels=n_channels,
            neurons_per_channel=neurons_per_channel,
        )
    if family == "tuning":
        return TuningEncoder(
            n_channels=n_channels,
            neurons_per_channel=neurons_per_channel,
        )
    raise ValueError(f"unknown encoder family: {family}")


def make_input_array(value: float) -> np.ndarray:
    values = np.full(N_CHANNELS, BASE_VALUE, dtype=np.float64)
    values[CHANGED_CHANNEL] = value
    return values


def encode_batch(
    encoder: RecruitmentEncoder | TuningEncoder, values: np.ndarray
) -> np.ndarray:
    """Vectorized single-channel encoding used by the noise experiment."""
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if np.any(values_array < 0.0) or np.any(values_array > 1.0):
        raise ValueError("values must be in [0, 1]")

    size = encoder.neurons_per_channel
    if isinstance(encoder, RecruitmentEncoder):
        counts = np.rint(values_array * float(size)).astype(np.intp)
        counts = np.clip(counts, 0, size)
        return (
            np.arange(size, dtype=np.intp)[np.newaxis, :]
            < counts[:, np.newaxis]
        ).astype(np.float64)

    centers = encoder.centers[np.newaxis, :]
    distance = np.abs(values_array[:, np.newaxis] - centers)
    return np.maximum(1.0 - distance * float(size - 1), 0.0)


def decode_batch(
    encoder: RecruitmentEncoder | TuningEncoder, activity: np.ndarray
) -> np.ndarray:
    """Vectorized single-channel decoding; all-zero tuning rows decode as 0."""
    activity_array = np.asarray(activity, dtype=np.float64)
    if activity_array.ndim != 2:
        raise ValueError("activity must have shape (samples, neurons)")
    if activity_array.shape[1] != encoder.neurons_per_channel:
        raise ValueError("activity width does not match encoder")

    if isinstance(encoder, RecruitmentEncoder):
        return np.mean(activity_array, axis=1)

    total = np.sum(activity_array, axis=1)
    decoded = np.zeros(activity_array.shape[0], dtype=np.float64)
    active = total > 0.0
    weighted = np.sum(
        activity_array[active] * encoder.centers[np.newaxis, :],
        axis=1,
    )
    decoded[active] = weighted / total[active]
    return decoded


def vectorized_encoding_matches(
    family: str, neurons_per_channel: int
) -> bool:
    encoder = make_encoder(family, neurons_per_channel, 1)
    values = np.asarray(INPUT_VALUES, dtype=np.float64)
    batch = encode_batch(encoder, values)
    references = np.stack(
        [encoder.encode(np.asarray([value])) for value in values]
    )[:, 0, :]
    return bool(np.array_equal(batch, references))


def pairwise_l2(activities: np.ndarray) -> list[dict[str, Any]]:
    distances: list[dict[str, Any]] = []
    for left in range(len(INPUT_VALUES)):
        for right in range(left + 1, len(INPUT_VALUES)):
            distance = float(
                np.linalg.norm(activities[left] - activities[right])
            )
            distances.append(
                {
                    "left": INPUT_VALUES[left],
                    "right": INPUT_VALUES[right],
                    "l2": distance,
                }
            )
    return distances


def no_noise_resolution(
    family: str, neurons_per_channel: int
) -> dict[str, Any]:
    encoder = make_encoder(family, neurons_per_channel, N_CHANNELS)
    activities = np.stack(
        [encoder.encode(make_input_array(value)) for value in INPUT_VALUES]
    )
    decoded = np.stack(
        [encoder.decode(activity) for activity in activities]
    )
    unique_count = int(np.unique(activities, axis=0).shape[0])

    readbacks: list[dict[str, Any]] = []
    for index, value in enumerate(INPUT_VALUES):
        per_channel_total = np.sum(activities[index], axis=1)
        other_error = np.abs(
            decoded[index, np.arange(N_CHANNELS) != CHANGED_CHANNEL]
            - BASE_VALUE
        )
        readbacks.append(
            {
                "input": value,
                "decoded_channel_0": float(decoded[index, CHANGED_CHANNEL]),
                "decoded_other_channels_max_abs_error": float(
                    np.max(other_error)
                ),
                "per_channel_total_activity": [
                    float(item) for item in per_channel_total
                ],
                "total_activity": float(np.sum(per_channel_total)),
            }
        )

    totals = np.asarray(
        [item["total_activity"] for item in readbacks], dtype=np.float64
    )
    return {
        "family": family,
        "neurons_per_channel": neurons_per_channel,
        "size": int(encoder.size),
        "unique_encoding_count": unique_count,
        "all_input_pairs_distinguishable": unique_count == len(INPUT_VALUES),
        "pairwise_l2": pairwise_l2(activities),
        "no_noise_readback": readbacks,
        "raw_total_activity_min": float(np.min(totals)),
        "raw_total_activity_max": float(np.max(totals)),
        "raw_total_activity_max_minus_min": float(
            np.max(totals) - np.min(totals)
        ),
        "vectorized_encoding_matches_encoder": vectorized_encoding_matches(
            family, neurons_per_channel
        ),
    }


def crosstalk_validation(
    family: str, neurons_per_channel: int
) -> dict[str, Any]:
    encoder = make_encoder(family, neurons_per_channel, N_CHANNELS)
    one_channel = make_encoder(family, neurons_per_channel, 1)
    target = 0.505

    low_context = np.full(N_CHANNELS, 0.0, dtype=np.float64)
    low_context[CHANGED_CHANNEL] = target
    high_context = np.full(N_CHANNELS, 1.0, dtype=np.float64)
    high_context[CHANGED_CHANNEL] = target
    low_activity = encoder.encode(low_context)
    high_activity = encoder.encode(high_context)
    low_decoded = encoder.decode(low_activity)
    high_decoded = encoder.decode(high_activity)
    canonical_activities = np.stack(
        [encoder.encode(make_input_array(value)) for value in INPUT_VALUES]
    )
    canonical_decoded = np.stack(
        [encoder.decode(activity) for activity in canonical_activities]
    )

    base_activity = encoder.encode(make_input_array(target))
    rng = np.random.default_rng(0)
    noisy_activity = np.clip(
        np.broadcast_to(
            base_activity,
            (SAMPLES_PER_INPUT_PER_SEED, N_CHANNELS, neurons_per_channel),
        ).copy()
        + rng.normal(
            0.0,
            CROSSTALK_NOISE_STD,
            size=(
                SAMPLES_PER_INPUT_PER_SEED,
                N_CHANNELS,
                neurons_per_channel,
            ),
        ),
        0.0,
        1.0,
    )
    multi_decoded = np.zeros(
        (SAMPLES_PER_INPUT_PER_SEED, N_CHANNELS),
        dtype=np.float64,
    )
    nonzero_rows = np.flatnonzero(
        np.any(noisy_activity > 0.0, axis=(1, 2))
    )
    for sample_index in nonzero_rows:
        multi_decoded[sample_index] = encoder.decode(
            noisy_activity[sample_index]
        )
    independently_decoded = np.stack(
        [
            decode_batch(one_channel, noisy_activity[:, channel, :])
            for channel in range(N_CHANNELS)
        ],
        axis=1,
    )

    return {
        "target_context_invariance": {
            "target_input": target,
            "non_target_contexts": [0.0, 1.0],
            "target_activity_identical": bool(
                np.array_equal(
                    low_activity[CHANGED_CHANNEL],
                    high_activity[CHANGED_CHANNEL],
                )
            ),
            "target_decode_max_abs_difference": float(
                np.max(
                    np.abs(
                        low_decoded[CHANGED_CHANNEL]
                        - high_decoded[CHANGED_CHANNEL]
                    )
                )
            ),
        },
        "canonical_non_target_check": {
            "input_values": list(INPUT_VALUES),
            "non_target_activity_identical_across_inputs": bool(
                np.all(
                    canonical_activities[:, 1:, :]
                    == canonical_activities[0:1, 1:, :]
                )
            ),
            "non_target_readback_max_abs_error": float(
                np.max(
                    np.abs(
                        canonical_decoded[:, 1:]
                        - BASE_VALUE
                    )
                )
            ),
        },
        "noise_columnwise_check": {
            "noise_std": CROSSTALK_NOISE_STD,
            "seed": 0,
            "samples": SAMPLES_PER_INPUT_PER_SEED,
            "all_channel_decode_max_abs_error_vs_independent": float(
                np.max(np.abs(multi_decoded - independently_decoded))
            ),
        },
    }


def run_noise_trials(
    family: str, neurons_per_channel: int
) -> list[dict[str, Any]]:
    encoder = make_encoder(family, neurons_per_channel, 1)
    results: list[dict[str, Any]] = []

    for noise_std in NOISE_STDS:
        for seed in SEEDS:
            decoded_by_input: dict[float, np.ndarray] = {}
            all_zero_by_input: dict[float, np.ndarray] = {}
            for input_value in NOISE_INPUTS:
                values = np.full(
                    SAMPLES_PER_INPUT_PER_SEED,
                    input_value,
                    dtype=np.float64,
                )
                activity = encode_batch(encoder, values)
                rng = np.random.default_rng(seed)
                noisy = np.clip(
                    activity
                    + rng.normal(
                        0.0,
                        noise_std,
                        size=activity.shape,
                    ),
                    0.0,
                    1.0,
                )
                decoded_by_input[input_value] = decode_batch(encoder, noisy)
                all_zero_by_input[input_value] = np.all(
                    noisy == 0.0, axis=1
                )

            low_decoded = decoded_by_input[NOISE_INPUTS[0]]
            high_decoded = decoded_by_input[NOISE_INPUTS[1]]
            low_all_zero = all_zero_by_input[NOISE_INPUTS[0]]
            high_all_zero = all_zero_by_input[NOISE_INPUTS[1]]

            low_predictions = low_decoded > DECODE_THRESHOLD
            high_predictions = high_decoded > DECODE_THRESHOLD
            true_negative_rate = float(np.mean(~low_predictions))
            true_positive_rate = float(np.mean(high_predictions))
            balanced_accuracy = 0.5 * (
                true_negative_rate + true_positive_rate
            )
            mse_low = float(
                np.mean(
                    np.square(low_decoded - NOISE_INPUTS[0])
                )
            )
            mse_high = float(
                np.mean(
                    np.square(high_decoded - NOISE_INPUTS[1])
                )
            )
            all_zero_rate = 0.5 * (
                float(np.mean(low_all_zero))
                + float(np.mean(high_all_zero))
            )

            results.append(
                {
                    "family": family,
                    "neurons_per_channel": neurons_per_channel,
                    "noise_std": noise_std,
                    "seed": seed,
                    "samples_per_input": SAMPLES_PER_INPUT_PER_SEED,
                    "balanced_accuracy": balanced_accuracy,
                    "true_positive_rate": true_positive_rate,
                    "true_negative_rate": true_negative_rate,
                    "mse": 0.5 * (mse_low + mse_high),
                    "mse_input_0_500": mse_low,
                    "mse_input_0_505": mse_high,
                    "all_zero_rate": all_zero_rate,
                    "all_zero_rate_input_0_500": float(
                        np.mean(low_all_zero)
                    ),
                    "all_zero_rate_input_0_505": float(
                        np.mean(high_all_zero)
                    ),
                }
            )
    return results


def summarize_noise(
    trials: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for family in ("recruitment", "tuning"):
        sizes = (
            RECRUITMENT_SIZES
            if family == "recruitment"
            else TUNING_SIZES
        )
        for neurons_per_channel in sizes:
            for noise_std in NOISE_STDS:
                selected = [
                    item
                    for item in trials
                    if item["family"] == family
                    and item["neurons_per_channel"] == neurons_per_channel
                    and item["noise_std"] == noise_std
                ]
                balanced = np.asarray(
                    [item["balanced_accuracy"] for item in selected],
                    dtype=np.float64,
                )
                mse = np.asarray(
                    [item["mse"] for item in selected],
                    dtype=np.float64,
                )
                all_zero = np.asarray(
                    [item["all_zero_rate"] for item in selected],
                    dtype=np.float64,
                )
                summaries.append(
                    {
                        "family": family,
                        "neurons_per_channel": neurons_per_channel,
                        "noise_std": noise_std,
                        "seed_count": len(selected),
                        "balanced_accuracy_mean": float(np.mean(balanced)),
                        "balanced_accuracy_std": float(np.std(balanced)),
                        "balanced_accuracy_min": float(np.min(balanced)),
                        "balanced_accuracy_max": float(np.max(balanced)),
                        "mse_mean": float(np.mean(mse)),
                        "mse_std": float(np.std(mse)),
                        "all_zero_rate_mean": float(np.mean(all_zero)),
                        "all_zero_rate_max": float(np.max(all_zero)),
                    }
                )
    return summaries


def build_artifact() -> dict[str, Any]:
    resolution: list[dict[str, Any]] = []
    crosstalk: list[dict[str, Any]] = []
    configs = (
        [("recruitment", size) for size in RECRUITMENT_SIZES]
        + [("tuning", size) for size in TUNING_SIZES]
    )
    for family, neurons_per_channel in configs:
        resolution.append(
            no_noise_resolution(family, neurons_per_channel)
        )
        crosstalk.append(
            {
                "family": family,
                "neurons_per_channel": neurons_per_channel,
                **crosstalk_validation(family, neurons_per_channel),
            }
        )

    trials: list[dict[str, Any]] = []
    for family, neurons_per_channel in configs:
        trials.extend(
            run_noise_trials(family, neurons_per_channel)
        )

    script_path = Path(__file__).resolve()
    encoding_path = ROOT / "born_wired" / "encoding.py"
    return {
        "schema_version": 1,
        "tool": "measure_resolution_v3",
        "tool_version": TOOL_VERSION,
        "parameters": {
            "n_channels": N_CHANNELS,
            "changed_channel": CHANGED_CHANNEL,
            "base_value": BASE_VALUE,
            "input_values": list(INPUT_VALUES),
            "recruitment_neurons_per_channel": list(RECRUITMENT_SIZES),
            "tuning_neurons_per_channel": list(TUNING_SIZES),
            "decode_threshold": DECODE_THRESHOLD,
            "decision_rule": {
                "label_rule": "high if decoded > decode_threshold else low",
                "positive_input": NOISE_INPUTS[1],
                "negative_input": NOISE_INPUTS[0],
            },
        },
        "noise_model": {
            "distribution": "independent identically distributed Gaussian",
            "domain": "neural activity",
            "std_values": list(NOISE_STDS),
            "clip_min": 0.0,
            "clip_max": 1.0,
            "inputs": list(NOISE_INPUTS),
            "seeds": list(SEEDS),
            "samples_per_input_per_seed": SAMPLES_PER_INPUT_PER_SEED,
            "all_zero_tuning_decode_fallback": 0.0,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_sha256": {
            "born_wired/encoding.py": sha256_file(encoding_path),
            "tools/measure_resolution_v3.py": sha256_file(script_path),
        },
        "resolution": resolution,
        "crosstalk_validation": crosstalk,
        "noise_trials": trials,
        "noise_summary": summarize_noise(trials),
        "limitations": [
            "Raw total activity is not normalized; encoder family and "
            "neurons_per_channel have different raw activity totals.",
            "L2 distances and readback values are raw finite-rate-vector "
            "measurements, not claims about metabolic cost or biological "
            "resolution.",
        ],
    }


def main() -> None:
    artifact = build_artifact()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
