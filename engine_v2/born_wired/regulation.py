"""Local plasticity with explicit structural ranges and incoming resource budgets.

Ranges are a developmental constraint, not proof of reliable behaviour. Memory
edges may use [0, cap] while innate edges retain a nonzero, plastic scaffold.
"""
import numpy as np
from .synapses import BoundedSynapses, _vector, _scalar
from .torch_execution import SynapseTensors, resolve


class RegulatedSynapses(BoundedSynapses):
    def __init__(self, src, dst, weights, signs, n_neurons, *, lower=0,
                 upper=1, budgets=10, plasticity=1, learning_rate=.05,
                 tether=.05, target_activity=.15, device=None):
        # Set before the base class runs: it calls _project, which dispatches on
        # this attribute, and the graph itself has to exist before a device copy
        # of it can be taken.
        self._execution = None
        size = len(src)
        self._lower = _vector(lower, size, 'lower', broadcast=True, low=0)
        upper = _vector(upper, size, 'upper', broadcast=True, low=0)
        initial = _vector(weights, size, 'weights', low=0)
        if np.any(self._lower > upper) or np.any(initial < self._lower) or np.any(initial > upper):
            raise ValueError('initial weights must lie inside lower/upper bounds')
        plasticity = _vector(plasticity, size, 'plasticity', broadcast=True, low=0)
        # Frozen edges reserve their exact value, just as in the base class.
        self._lower[plasticity == 0] = initial[plasticity == 0]
        self.anchor = initial.copy()
        self.tether = _vector(tether, size, 'tether', broadcast=True, low=0)
        self.target_activity = _vector(target_activity, n_neurons, 'target_activity',
                                       broadcast=True, low=0, high=1)
        super().__init__(src, dst, initial, signs, n_neurons, w_max=upper,
                         budgets=budgets, plasticity=plasticity,
                         learning_rate=learning_rate, decay=0)
        for a in (self._lower, self.anchor, self.tether, self.target_activity):
            a.flags.writeable = False
        chosen = resolve(device)
        if chosen is not None:
            self._execution = SynapseTensors(self, chosen)

    @property
    def lower(self):
        result = self._lower.copy()
        result.flags.writeable = False
        return result

    def _structure(self):
        """Per-group addresses and reserves that never change after building.

        Which edges a group holds, which of them the local rule may move, what
        they already reserve and what is therefore available are all fixed once
        the graph exists. Measuring them here costs the same as measuring them
        on every write, and this way it happens once. A rebuild constructs a new
        object, so the cache cannot outlive the graph it describes.
        """
        prepared = getattr(self, '_prepared', None)
        if prepared is None:
            prepared = []
            for group in (self._positive, ~self._positive):
                reserve = np.bincount(self.dst[group], weights=self._lower[group],
                                      minlength=self.n_neurons)
                if not np.isfinite(reserve).all() or np.any(reserve > self.budgets):
                    raise ValueError('structural lower bounds exceed an incoming budget')
                movable = np.flatnonzero(group & ~self._frozen)
                prepared.append((movable, self.dst[movable], self._lower[movable],
                                 self.budgets - reserve))
            prepared = tuple(prepared)
            for _, dst, lower, available in prepared:
                dst.flags.writeable = False
                lower.flags.writeable = False
                available.flags.writeable = False
            self._prepared = prepared
        return prepared

    def _project(self, proposed):
        execution = self._execution
        if execution is not None:
            return execution.host_project(proposed)
        result = np.clip(proposed, self._lower, self.w_max)
        for movable, dst, lower, available in self._structure():
            if not movable.size:
                continue
            excess = result[movable]
            excess -= lower
            sums = np.bincount(dst, weights=excess, minlength=self.n_neurons)
            if not np.isfinite(sums).all():
                raise FloatingPointError('non-finite budget sum')
            scale = np.ones(self.n_neurons)
            over = sums > available
            scale[over] = available[over] / sums[over]
            result[movable] = lower + excess*scale[dst]
        if not np.isfinite(result).all():
            raise FloatingPointError('non-finite projected weights')
        return result

    def update(self, pre, post, modulator=1, dt=.002):
        pre = _vector(pre, self.n_neurons, 'pre', low=0, high=1)
        post = _vector(post, self.n_neurons, 'post', low=0, high=1)
        modulation = _vector(modulator, self.n_neurons, 'modulator', broadcast=True, low=-1, high=1)
        dt = _scalar(dt, 'dt', positive=True)
        execution = self._execution
        if execution is not None:
            execution.host_update(pre, post, modulation, dt)
            return
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            # Each edge sees only its two cells, a postsynaptic target and a
            # local anchor. For inhibitory edges high post rate increases I.
            # An edge whose source cell is quiet cannot correlate this write, so
            # for it the only change left is the pull back towards its anchor;
            # that is one expression for every edge. The correlation itself is
            # then evaluated on the edges whose source is actually firing, which
            # is the same sum, in the same order, over fewer edges.
            change = self.anchor - self._weights
            change *= self.tether
            source_rate = pre[self.src]
            driven = np.flatnonzero(source_rate)
            if driven.size:
                local = post[self.dst[driven]] - self.target_activity[self.dst[driven]]
                local *= source_rate[driven]
                change[driven] = (self.learning_rate * modulation[self.dst[driven]] * local
                                  - self.tether[driven]*(self._weights[driven] - self.anchor[driven]))
            proposal = np.multiply(self.plasticity, dt)
            proposal *= change
            proposal += self._weights
            if not np.isfinite(proposal).all():
                raise FloatingPointError('non-finite plasticity proposal')
            result = self._project(proposal)
        self._weights = result
