"""Event-driven execution of the same cells with the same equations.

AdaptiveNetwork multiplies every synapse on every step, so what a step costs is
set by how many synapses exist rather than by how many cells are busy. Measured
on one graph of a hundred thousand cells, a step cost the same thirteen
milliseconds whether one cell was speaking or eighty-six thousand were.

The rule a step follows is
    current = bias + recurrent_excitation - recurrent_inhibition
              + external - gain*adaptation
    voltage, adaptation <- one step of leakage towards that
and the recurrent part of the current is a sum over the synapses whose source
cell is speaking. A cell that no speaking cell reaches therefore receives no
synaptic current at all: its own step depends on nothing but its bias and its
adaptation, and those two are a linear recursion with constant coefficients, so
the result of leaving it alone for any number of steps can be worked out
directly instead of stepping it through them. That is what this module does. A
cell is touched only when it is speaking, when a speaking cell reaches it, when
it is driven from outside, or when its own bias could lift it back over the
cutoff; anything else is left where it was, and its state is brought forward
exactly when it is next touched.

The one departure from AdaptiveNetwork is `cutoff`: a rate at or below it is
treated as silence. AdaptiveNetwork computes every speech, however faint, so
with cutoff zero the two agree exactly, cell for cell and step for step.
"""
import numpy as np

from .synapses import _indices, _scalar, _vector


class SparseNetwork:
    """Same cells, same equations, work proportional to speech rather than size."""

    def __init__(self, synapses, tau=.08, adaptation_tau=.4, adaptation_gain=2.5,
                 bias=0, initial_voltage=0, initial_adaptation=0, cutoff=1e-3):
        self.synapses = synapses
        self.n_neurons = int(synapses.n_neurons)
        n = self.n_neurons
        self._tau = _vector(tau, n, 'tau', broadcast=True, low=0)
        self._adaptation_tau = _vector(adaptation_tau, n, 'adaptation_tau', broadcast=True, low=0)
        if np.any(self._tau == 0) or np.any(self._adaptation_tau == 0):
            raise ValueError('time constants must be positive')
        self._gain = _vector(adaptation_gain, n, 'adaptation_gain', broadcast=True, low=0)
        self._bias = _vector(bias, n, 'bias', broadcast=True)
        self._voltage = _vector(initial_voltage, n, 'initial_voltage', broadcast=True)
        self._adaptation = _vector(initial_adaptation, n, 'initial_adaptation',
                                   broadcast=True, low=0, high=1)
        self.cutoff = float(_scalar(cutoff, 'cutoff'))
        if self.cutoff < 0:
            raise ValueError('cutoff must be non-negative')
        self._weights = synapses._weights
        self._src = synapses.src
        self._dst = synapses.dst
        self._signs = synapses.signs
        order = np.argsort(self._src, kind='stable')
        counts = np.bincount(self._src, minlength=n)
        self._order = order
        self._counts = counts
        self._starts = np.r_[0, np.cumsum(counts)][:-1]
        # Whether an edge excites or inhibits is a property of the cell it
        # starts from, so it is settled once here instead of every step.
        self._edge_excites = self._signs[self._src] > 0
        self._ticks = 0
        self._last = np.zeros(n, dtype=np.int64)
        self._last_dt = None
        self._rate = np.clip(self._voltage, 0, 1)
        # With no input a cell moves between where it is now and its bias, so a
        # bias above the cutoff is exactly when a quiet cell can lift itself
        # back into speech. Those cells have to be stepped every tick, or the
        # engine would never notice the moment they wake: in a graph where a
        # cell inhibits itself this happens as soon as its own rate decays.
        self._watch = np.flatnonzero(self._bias > self.cutoff)
        self._speaking = np.flatnonzero(self._rate > self.cutoff)
        self._learning_elapsed = 0.
        self.learning_interval = float(getattr(synapses, 'learning_interval', 0.))

    @staticmethod
    def _frozen(array):
        view = array.view()
        view.flags.writeable = False
        return view

    @property
    def activity(self):
        return self._frozen(self._rate)

    @property
    def voltage(self):
        return self._frozen(self._voltage)

    @property
    def adaptation(self):
        return self._frozen(self._adaptation)

    @property
    def speaking(self):
        """How many cells are above the cutoff right now."""
        return int(self._speaking.size)

    def step(self, external_exc, external_inh=None, dt=.002, learn=False, modulator=1.):
        """One tick, with the outside drive given as a full-length vector."""
        drive = self._dense(external_exc, 'external_exc')
        brake = None if external_inh is None else self._dense(external_inh, 'external_inh')
        return self._tick(drive, brake, dt, learn)

    def step_sparse(self, external_exc=None, external_inh=None, dt=.002, learn=False,
                    modulator=1.):
        """One tick, with the outside drive given as a (cells, currents) pair.

        Nothing here walks the population, so a tick in which two cells are busy
        costs what two cells cost however many cells the network holds.
        """
        drive = None if external_exc is None else self._sparse(external_exc, 'external_exc')
        brake = None if external_inh is None else self._sparse(external_inh, 'external_inh')
        return self._tick(drive, brake, dt, learn)

    def catch_up(self, dt=None):
        """Bring every stored state forward to the current tick.

        `step` leaves quiet cells where they were, so a quiet cell's state is
        exact but dated; this is how a caller reads the whole population.
        """
        if dt is None:
            if self._last_dt is None:
                raise ValueError('no step has been taken yet, so dt must be given')
            dt = self._last_dt
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        every = np.arange(self.n_neurons)
        self._advance(every, self._ticks - self._last[every], dt)
        self._last[every] = self._ticks
        self._rate[every] = np.clip(self._voltage[every], 0, 1)

    def _tick(self, drive, brake, dt, learn):
        if learn:
            raise NotImplementedError(
                'sparse learning is not written yet; run this network with learn=False')
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self._last_dt = dt
        self._ticks += 1
        speaking = self._speaking
        positions, source = self._ragged(speaking)
        rate = self._rate[speaking]
        pieces = [speaking, self._watch]
        if positions.size:
            pieces.append(self._dst[positions])
        for extra in (drive, brake):
            if extra is not None and extra[0].size:
                pieces.append(extra[0])
        touched = np.unique(np.concatenate(pieces))
        # `_last` is the tick a cell's stored state is valid at, and the step
        # below is itself the move from the previous tick to this one, so the
        # catch-up stops one tick short of now.
        self._advance(touched, self._ticks - 1 - self._last[touched], dt)
        self._last[touched] = self._ticks
        old_rate = np.clip(self._voltage[touched], 0, 1)
        self._rate[touched] = old_rate
        excitation = np.zeros(touched.size)
        inhibition = np.zeros(touched.size)
        if positions.size:
            target = np.searchsorted(touched, self._dst[positions])
            flow = self._weights[positions]*rate[source]
            excites = self._edge_excites[positions]
            excitation = np.bincount(target[excites], weights=flow[excites],
                                     minlength=touched.size)
            if not excites.all():
                inhibition = np.bincount(target[~excites], weights=flow[~excites],
                                         minlength=touched.size)
        current = (self._bias[touched] + excitation - inhibition
                   - self._gain[touched]*self._adaptation[touched])
        if drive is not None and drive[0].size:
            current += self._spread(touched, drive)
        if brake is not None and brake[0].size:
            current -= self._spread(touched, brake)
        with np.errstate(over='ignore', under='ignore'):
            blend = -np.expm1(-dt/self._tau[touched])
            adaptation_blend = -np.expm1(-dt/self._adaptation_tau[touched])
        voltage = (1 - blend)*self._voltage[touched] + blend*current
        adaptation = ((1 - adaptation_blend)*self._adaptation[touched]
                      + adaptation_blend*old_rate)
        self._voltage[touched] = voltage
        self._adaptation[touched] = adaptation
        self._rate[touched] = np.clip(voltage, 0, 1)
        self._speaking = touched[self._rate[touched] > self.cutoff]
        return self.activity

    def _advance(self, indices, lag, dt):
        """Exactly what `lag` steps of untouched leakage would have produced."""
        moving = lag > 0
        if not np.any(moving):
            return
        indices = indices[moving]
        lag = lag[moving].astype(np.float64)
        tau = self._tau[indices]
        adaptation_tau = self._adaptation_tau[indices]
        with np.errstate(over='ignore', under='ignore'):
            blend = -np.expm1(-dt/tau)
            adaptation_blend = -np.expm1(-dt/adaptation_tau)
            # exp(-lag*dt/tau) is the same as (1-blend)**lag but has no
            # cancellation when the lag is long and the blend is small.
            decay = np.exp(-lag*dt/tau)
            settled = np.exp(-lag*dt/adaptation_tau)
            geometric = np.where(blend > 1e-12, (1. - decay)/np.where(blend > 1e-12, blend, 1.),
                                 lag)
            gap = adaptation_blend - blend
            cross = np.where(np.abs(gap) <= 1e-12, lag*decay/(1. - blend),
                             (decay - settled)/np.where(np.abs(gap) <= 1e-12, 1., gap))
        voltage = self._voltage[indices]
        adaptation = self._adaptation[indices]
        self._voltage[indices] = (decay*voltage
                                  + blend*(self._bias[indices]*geometric
                                           - self._gain[indices]*adaptation*cross))
        self._adaptation[indices] = settled*adaptation

    def _ragged(self, sources):
        counts = self._counts[sources]
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        offsets = np.cumsum(counts) - counts
        positions = self._order[np.repeat(self._starts[sources], counts)
                                + (np.arange(total) - np.repeat(offsets, counts))]
        return positions, np.repeat(np.arange(sources.size), counts)

    @staticmethod
    def _spread(touched, piece):
        cells, amounts = piece
        return np.bincount(np.searchsorted(touched, cells), weights=amounts,
                           minlength=touched.size)

    def _dense(self, value, name):
        vector = np.asarray(value, dtype=np.float64).ravel()
        if vector.shape != (self.n_neurons,):
            raise ValueError('%s must have one entry per cell' % name)
        cells = np.flatnonzero(vector)
        amounts = vector[cells]
        if amounts.size and (amounts.min() < 0 or not np.isfinite(amounts).all()):
            raise ValueError('%s must be finite and non-negative' % name)
        return cells, amounts

    def _sparse(self, value, name):
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError('%s must be a (cells, currents) pair' % name)
        cells = _indices(value[0], self.n_neurons, name)
        amounts = _vector(value[1], cells.size, name, broadcast=True, low=0)
        return cells, amounts
