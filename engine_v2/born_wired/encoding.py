"""Population encoders for bounded continuous motor commands."""

from __future__ import annotations

import numpy as np


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _population_size(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("neurons_per_channel must be an integer")
    result = int(value)
    if result < 2:
        raise ValueError("neurons_per_channel must be at least 2")
    return result


def _as_bounded_array(
    values: object, shape: tuple[int, ...], name: str
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an array-like numeric value") from exc

    if np.issubdtype(raw.dtype, np.bool_) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values")

    try:
        array = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc

    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array


class _BaseEncoder:
    def __init__(self, n_channels: int, neurons_per_channel: int) -> None:
        self.n_channels = _positive_int("n_channels", n_channels)
        self.neurons_per_channel = _population_size(neurons_per_channel)

    @property
    def size(self) -> int:
        return self.n_channels * self.neurons_per_channel

    def _encode_values(self, values: object) -> np.ndarray:
        return _as_bounded_array(
            values, (self.n_channels,), "values"
        ).astype(np.float64, copy=True)

    def _decode_activity(self, activity: object) -> np.ndarray:
        return _as_bounded_array(
            activity,
            (self.n_channels, self.neurons_per_channel),
            "activity",
        ).astype(np.float64, copy=True)

    def encode(self, values: object) -> np.ndarray:
        raise NotImplementedError

    def decode(self, activity: object) -> np.ndarray:
        raise NotImplementedError


class RecruitmentEncoder(_BaseEncoder):
    """Encode each channel by recruiting a rounded number of active units."""

    def __init__(
        self, n_channels: int = 12, neurons_per_channel: int = 100
    ) -> None:
        super().__init__(n_channels, neurons_per_channel)

    def encode(self, values: object) -> np.ndarray:
        values_array = self._encode_values(values)
        counts = np.rint(
            values_array * float(self.neurons_per_channel)
        ).astype(np.intp)
        counts = np.clip(counts, 0, self.neurons_per_channel)

        activity = np.zeros(
            (self.n_channels, self.neurons_per_channel), dtype=np.float64
        )
        for channel, count in enumerate(counts):
            activity[channel, : int(count)] = 1.0
        return activity

    def decode(self, activity: object) -> np.ndarray:
        activity_array = self._decode_activity(activity)
        return np.mean(activity_array, axis=1)


class TuningEncoder(_BaseEncoder):
    """Encode commands through overlapping triangular tuning curves."""

    def __init__(
        self, n_channels: int = 12, neurons_per_channel: int = 32
    ) -> None:
        super().__init__(n_channels, neurons_per_channel)
        self._centers = np.linspace(
            0.0, 1.0, self.neurons_per_channel, dtype=np.float64
        )

    @property
    def centers(self) -> np.ndarray:
        return self._centers.copy()

    def encode(self, values: object) -> np.ndarray:
        values_array = self._encode_values(values)
        distance = np.abs(
            values_array[:, np.newaxis] - self._centers[np.newaxis, :]
        )
        return np.maximum(
            1.0
            - distance * float(self.neurons_per_channel - 1),
            0.0,
        )

    def decode(self, activity: object) -> np.ndarray:
        activity_array = self._decode_activity(activity)
        total = np.sum(activity_array, axis=1)
        if np.any(total <= 0.0):
            raise ValueError(
                "activity rows must have positive total activity"
            )
        weighted = np.sum(
            activity_array * self._centers[np.newaxis, :], axis=1
        )
        return weighted / total
