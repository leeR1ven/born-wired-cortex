"""Bounded local correlation learning: a candidate component, not a complete brain.

Each postsynaptic cell has separate excitatory and inhibitory resource budgets.
These bounds establish numerical invariants, not stability of robot behaviour.
"""
from __future__ import annotations

import numbers
import numpy as np

from .torch_execution import SynapseTensors


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} must be a real scalar')
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def _vector(value, size, name, *, broadcast=False, low=None, high=None):
    original = np.asarray(value)
    if original.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be real numeric values')
    result = np.array(original, dtype=np.float64, copy=True)
    if broadcast and result.ndim == 0:
        result = np.full(size, result.item(), dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a finite vector of length {size}')
    if low is not None and np.any(result < low):
        raise ValueError(f'{name} is below {low}')
    if high is not None and np.any(result > high):
        raise ValueError(f'{name} is above {high}')
    return result


def _indices(value, n, name):
    arr = np.asarray(value)
    # [] is an unambiguous empty list even though numpy defaults it to float.
    if arr.ndim != 1 or (arr.size and arr.dtype.kind not in 'iu'):
        raise ValueError(f'{name} must be a one-dimensional integer array')
    if np.any(arr < 0) or np.any(arr >= n):
        raise ValueError(f'{name} contains an out-of-range endpoint')
    return np.array(arr, dtype=np.int64, copy=True)


def _checked_in_place(value, size, name, *, low=None, high=None):
    """The checks of _vector for an array the caller owns and will write into.

    A network built out of layers adds one contribution per layer to the same
    vector of external excitation before it reaches the cells, and each layer
    used to take its own checked copy of it first. That is a full copy of the
    sheet of cells per layer per step, for an array the outermost layer built
    in the first place. Python cannot hide that cost the way it hides the
    arithmetic: the copy is the cost. This runs the same checks where the
    array lies, so the layers can add into one vector instead of copying it
    again, and callers that have not built the vector themselves keep using
    _vector and its copy.
    """
    result = np.asarray(value)
    if result.dtype != np.float64 or result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a finite float64 vector of length {size}')
    if low is not None and np.any(result < low):
        raise ValueError(f'{name} is below {low}')
    if high is not None and np.any(result > high):
        raise ValueError(f'{name} is above {high}')
    return result


class BoundedSynapses:
    """Sparse same-network innate and acquired synapses with local constraints.

    Plasticity zero is available for ablations. Normal operation defaults to
    plasticity one. Learning is per-second and therefore explicitly uses dt.
    """

    def __init__(self, src, dst, weights, signs, n_neurons, *, w_max=1.0,
                 budgets=2.0, learning_rate=0.01, decay=0.001, plasticity=1.0):
        if isinstance(n_neurons, (bool, np.bool_)) or not isinstance(n_neurons, numbers.Integral) or n_neurons < 1:
            raise ValueError('n_neurons must be a positive integer')
        self.n_neurons = int(n_neurons)
        # The host engine is the default; RegulatedSynapses may replace this
        # with a device copy of the same graph.
        self._execution = None
        self.src = _indices(src, self.n_neurons, 'src')
        self.dst = _indices(dst, self.n_neurons, 'dst')
        if self.src.shape != self.dst.shape or np.any(self.src == self.dst):
            raise ValueError('edge lengths must agree and self-connections are excluded')
        if len(set(zip(self.src.tolist(), self.dst.tolist()))) != len(self.src):
            raise ValueError('duplicate edges are excluded')
        size = len(self.src)
        self.signs = _vector(signs, self.n_neurons, 'signs')
        if np.any((self.signs != 1) & (self.signs != -1)):
            raise ValueError('each source cell must have sign +1 or -1')
        self.w_max = _vector(w_max, size, 'w_max', broadcast=True, low=0)
        self.budgets = _vector(budgets, self.n_neurons, 'budgets', broadcast=True, low=0)
        self.plasticity = _vector(plasticity, size, 'plasticity', broadcast=True, low=0)
        self.learning_rate = _scalar(learning_rate, 'learning_rate')
        self.decay = _scalar(decay, 'decay')
        self._frozen = self.plasticity == 0
        self._positive = self.signs[self.src] > 0
        # Inhibitory edges are counted into a second block of the same array,
        # so one pass over the edges gives both postsynaptic sums. The block is
        # a fixed address, not a second synapse.
        self._dst_block = np.where(self._positive, self.dst,
                                   self.dst + self.n_neurons)
        self._dst_block.flags.writeable = False
        initial = _vector(weights, size, 'weights', low=0)
        if np.any(initial[self._frozen] > self.w_max[self._frozen]):
            raise ValueError('frozen weights exceed their individual bounds')
        self._groups = []
        for group in (self._positive, ~self._positive):
            fixed = group & self._frozen
            mutable = group & ~self._frozen
            with np.errstate(over='raise', invalid='raise'):
                used = np.bincount(self.dst[fixed], weights=initial[fixed], minlength=self.n_neurons)
            if not np.all(np.isfinite(used)) or np.any(used > self.budgets):
                raise ValueError('frozen weights exceed a postsynaptic budget')
            self._groups.append((mutable, self.budgets - used))
        self._weights = self._project(initial)
        for arr in (self.src, self.dst, self.signs, self.w_max, self.budgets, self.plasticity):
            arr.flags.writeable = False

    @property
    def weights(self):
        execution = self._execution
        if execution is not None:
            execution.refresh_host_weights()
        # Independent read-only snapshot: public callers cannot corrupt updates.
        result = self._weights.copy()
        result.flags.writeable = False
        return result

    def _project(self, proposed):
        result = np.minimum(np.maximum(proposed, 0), self.w_max)
        for mutable, available in self._groups:
            sums = np.bincount(self.dst[mutable], weights=result[mutable], minlength=self.n_neurons)
            if not np.all(np.isfinite(sums)):
                raise FloatingPointError('non-finite synaptic budget sum')
            scale = np.ones(self.n_neurons)
            exceeded = sums > available
            scale[exceeded] = available[exceeded] / sums[exceeded]
            result[mutable] *= scale[self.dst[mutable]]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('non-finite projected weights')
        return result

    def currents(self, activity):
        activity = _vector(activity, self.n_neurons, 'activity', low=0, high=1)
        execution = self._execution
        if execution is not None:
            return execution.host_currents(activity)
        with np.errstate(over='raise', invalid='raise'):
            flow = self._weights * activity[self.src]
            both = np.bincount(self._dst_block, weights=flow, minlength=2*self.n_neurons)
        if both.dtype != np.float64:
            # An empty edge set makes bincount answer with its default integer
            # type. A current is a real number either way, and the caller is
            # allowed to add into the array it gets back.
            both = both.astype(np.float64)
        currents = both[:self.n_neurons], both[self.n_neurons:]
        if not all(np.all(np.isfinite(x)) for x in currents):
            raise FloatingPointError('non-finite synaptic current')
        return currents

    def update(self, pre, post, modulator=1.0, dt=0.02):
        pre = _vector(pre, self.n_neurons, 'pre', low=0, high=1)
        post = _vector(post, self.n_neurons, 'post', low=0, high=1)
        modulation = _vector(modulator, self.n_neurons, 'modulator', broadcast=True, low=-1, high=1)
        dt = _scalar(dt, 'dt', positive=True)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            drive = modulation[self.dst] * pre[self.src] * post[self.dst]
            delta = dt * self.plasticity * (self.learning_rate * drive - self.decay * self._weights)
            proposal = self._weights + delta
            if not np.all(np.isfinite(proposal)):
                raise FloatingPointError('non-finite learning proposal')
            result = self._project(proposal)
        # Only commit once every check has succeeded.
        self._weights = result

    def budget_totals(self):
        """Separate sums for diagnostics, not an alternative learning path."""
        execution = self._execution
        if execution is not None:
            execution.refresh_host_weights()
        return tuple(np.bincount(self.dst[g], weights=self._weights[g], minlength=self.n_neurons)
                     for g in (self._positive, ~self._positive))
