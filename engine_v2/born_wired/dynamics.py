"""Minimal conductance-style rate dynamics for the bounded synapse model.

This module connects activity to local plasticity. It is a numerical rate
approximation, not a validated neuron or a robot controller.
"""
from __future__ import annotations

import numbers

import numpy as np

from born_wired.synapses import BoundedSynapses


ROUNDING_TOLERANCE = 1e-14


def _scalar(value, name, *, positive=False, nonnegative=False, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real scalar")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return value


def _vector(value, size, name, *, broadcast=False, low=None, high=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if broadcast and result.ndim == 0:
        result = np.full(size, result.item(), dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    if low is not None and np.any(result < low):
        raise ValueError(f"{name} is below {low}")
    if high is not None and np.any(result > high):
        raise ValueError(f"{name} is above {high}")
    return result


def _unit_vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{name} is non-finite")
    if np.any(result < -ROUNDING_TOLERANCE) or np.any(result > 1.0 + ROUNDING_TOLERANCE):
        raise FloatingPointError(f"{name} is outside [0, 1]")
    return np.clip(result, 0.0, 1.0)


class ConductanceNetwork:
    """A bounded rate network with optional learning through its synapses."""

    def __init__(
        self,
        synapses: BoundedSynapses,
        *,
        tau=0.02,
        leak=1.0,
        threshold=0.2,
        initial_voltage=0.0,
    ):
        if not isinstance(synapses, BoundedSynapses):
            raise TypeError("synapses must be a BoundedSynapses instance")
        self.synapses = synapses
        self.n_neurons = synapses.n_neurons
        self.tau = _scalar(tau, "tau", positive=True)
        self.leak = _scalar(leak, "leak", positive=True)
        self.threshold = _scalar(
            threshold, "threshold", nonnegative=True, upper=1.0
        )
        if self.threshold >= 1.0:
            raise ValueError("threshold must be < 1")
        self._voltage = _vector(
            initial_voltage,
            self.n_neurons,
            "initial_voltage",
            broadcast=True,
            low=0,
            high=1,
        )
        self._activity = self._decode(self._voltage)

    def _decode(self, voltage):
        activity = np.maximum(voltage - self.threshold, 0.0) / (1.0 - self.threshold)
        return _unit_vector(activity, "activity")

    @property
    def voltage(self):
        result = self._voltage.copy()
        result.flags.writeable = False
        return result

    @property
    def activity(self):
        result = self._activity.copy()
        result.flags.writeable = False
        return result

    def step(
        self,
        external_exc,
        external_inh=None,
        *,
        dt=0.002,
        learn=True,
        modulator=1.0,
    ):
        external_exc = _vector(
            external_exc, self.n_neurons, "external_exc", low=0
        )
        if external_inh is None:
            external_inh = np.zeros(self.n_neurons, dtype=np.float64)
        else:
            external_inh = _vector(
                external_inh, self.n_neurons, "external_inh", low=0
            )
        dt = _scalar(dt, "dt", positive=True)
        if not isinstance(learn, (bool, np.bool_)):
            raise ValueError("learn must be a bool")
        learn = bool(learn)
        modulation = _vector(
            modulator,
            self.n_neurons,
            "modulator",
            broadcast=True,
            low=-1,
            high=1,
        )

        old_activity = self._activity
        old_voltage = self._voltage
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            recurrent_exc, recurrent_inh = self.synapses.currents(old_activity)
            ge = external_exc + recurrent_exc
            gi = external_inh + recurrent_inh
            conductance = self.leak + ge + gi
            if (
                not np.all(np.isfinite(ge))
                or not np.all(np.isfinite(gi))
                or not np.all(np.isfinite(conductance))
            ):
                raise FloatingPointError("non-finite conductance")
            target = ge / conductance
            exponent = dt * conductance / self.tau
            if not np.all(np.isfinite(target)) or not np.all(np.isfinite(exponent)):
                raise FloatingPointError("non-finite dynamics intermediate")
            retention = np.exp(-exponent)
            update_fraction = -np.expm1(-exponent)
            new_voltage = retention * old_voltage + update_fraction * target
            new_activity = self._decode(new_voltage)
            new_voltage = _unit_vector(new_voltage, "voltage")
            new_activity = _unit_vector(new_activity, "activity")

        # Both state values are validated before this atomic synapse update.
        if learn:
            self.synapses.update(
                old_activity,
                new_activity,
                modulator=modulation,
                dt=dt,
            )
        self._voltage = new_voltage
        self._activity = new_activity
        return new_activity.copy()
