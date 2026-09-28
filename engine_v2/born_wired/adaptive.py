"""Rate neurons with local adaptation and transactional synaptic learning."""

import numpy as np

from .synapses import _checked_in_place, _indices, _scalar, _vector
from .torch_execution import CellTensors, to_device


class AdaptiveNetwork:
    """Finite-input voltage dynamics; only firing rates and adaptation lie in [0, 1]."""

    def __init__(self, synapses, tau=.08, adaptation_tau=.4, adaptation_gain=2.5,
                 bias=0, initial_voltage=0, initial_adaptation=0, learning_interval=0.):
        self.synapses = synapses
        self.n_neurons = synapses.n_neurons
        self._tau = self._parameter(tau, "tau", low=0)
        self._adaptation_tau = self._parameter(adaptation_tau, "adaptation_tau", low=0)
        if np.any(self._tau == 0) or np.any(self._adaptation_tau == 0):
            raise ValueError("time constants must be positive")
        self._gain = self._parameter(adaptation_gain, "adaptation_gain", low=0)
        self._bias = self._parameter(bias, "bias")
        self._voltage = self._parameter(initial_voltage, "initial_voltage")
        self._adaptation = self._parameter(initial_adaptation, "initial_adaptation", low=0, high=1)
        self.learning_interval = _scalar(learning_interval, 'learning_interval')
        self._learning_elapsed = 0.
        # Two blends depend on dt and tau alone, and one step over every cell
        # writes into four working arrays. Neither the blends nor the shape of
        # the arithmetic is new on the next step, so both are kept and the
        # expression itself is evaluated in place instead of building a dozen
        # temporary arrays per step. Keeping them is why a bigger sheet of cells
        # costs time in proportion to the cells, and not more.
        self._blends = None
        self._working = None
        # A network whose graph has a device copy runs its whole step there.
        # The host arrays above stay the reader-facing state either way.
        self._device = None
        execution = getattr(synapses, '_execution', None)
        if execution is not None and execution.n_neurons == self.n_neurons:
            self._device = CellTensors(self, execution)

    def _parameter(self, value, name, **bounds):
        return _vector(value, self.n_neurons, name, broadcast=True, **bounds)

    def _blend_pair(self, dt):
        """Voltage and adaptation blends for this dt, measured once per dt."""
        cached = self._blends
        if cached is None or cached[0] != dt:
            with np.errstate(over='ignore', under='ignore'):
                voltage_blend = -np.expm1(-dt / self._tau)
                adaptation_blend = -np.expm1(-dt / self._adaptation_tau)
            cached = (dt, voltage_blend, 1 - voltage_blend,
                      adaptation_blend, 1 - adaptation_blend)
            for array in cached[1:]:
                array.flags.writeable = False
            self._blends = cached
        return cached[1:]

    def _buffers(self):
        """Working arrays for one step, allocated once at the size of the graph."""
        working = self._working
        if working is None:
            n = self.n_neurons
            working = dict(rate=np.empty(n), rate_next=np.empty(n), term=np.empty(n),
                           voltage=np.empty(n), adaptation=np.empty(n))
            self._working = working
        return working

    @staticmethod
    def _snapshot(value):
        snapshot = value.copy()
        snapshot.flags.writeable = False
        return snapshot

    @property
    def activity(self):
        device = self._device
        if device is not None:
            return device.host_rate()
        return self._snapshot(np.clip(self._voltage, 0, 1))

    def rates_at(self, indices):
        """The named cells' firing rates, the numbers `activity` answers for them.

        A reader that wants a handful of groups should not pay for the whole
        sheet of cells to cross the bus; this is the same read, narrowed to the
        cells that were asked for.
        """
        device = self._device
        if device is not None:
            return device.rates_at(indices)
        return self.activity[_indices(indices, self.n_neurons, "indices")]

    @property
    def voltage(self):
        self._refresh()
        return self._snapshot(self._voltage)

    @property
    def adaptation(self):
        self._refresh()
        return self._snapshot(self._adaptation)

    def _refresh(self):
        """Bring the host arrays up to date before a reader looks at them."""
        device = self._device
        if device is not None and device.stale:
            device.write_back(self)

    def _modulation(self, value):
        """A validated modulator. A scalar stays a scalar so it can broadcast."""
        if np.ndim(value) == 0:
            return float(_vector(value, 1, 'modulator', broadcast=True, low=-1, high=1)[0])
        return self._parameter(value, 'modulator', low=-1, high=1)

    def step(self, external_exc, external_inh=None, dt=.002, learn=True, modulator=1,
             wanted=None):
        """Update using old firing rates; learning failure leaves both cell states unchanged."""
        exc = self._parameter(external_exc, "external_exc", low=0)
        return self.step_owned(exc, external_inh, dt=dt, learn=learn, modulator=modulator,
                               wanted=wanted, check=False)

    def step_owned(self, exc, external_inh=None, dt=.002, learn=True, modulator=1,
                   wanted=None, check=True):
        """The same step, for an excitation vector the caller already owns.

        `step` takes a checked copy of the excitation so that a caller's array
        is never written to. A layered network adds one contribution per layer
        to a single vector before it reaches the cells, and a second and third
        copy of that vector for each layer is time spent in proportion to the
        sheet of cells with none of it being new work. This entry point checks
        the vector where it lies instead, so the layers above can add into the
        one array the outermost of them already built.

        `wanted` names the cells whose rates the caller will actually read. A
        controller reads a few thousand of four hundred thousand of them, and a
        reader that says which ones does not pay for the rest to travel; `None`
        hands back the whole sheet, which is what every reader that does not
        say keeps getting.

        `check` is for the layers below a layer that has already checked this
        same vector. A controller stacks three or four of them, and each used to
        run the whole sheet's worth of finiteness and bound tests over an array
        that only the layer above it had just touched. The outermost layer
        checks once - `step` checks the copy it takes - and hands the vector
        down with the answer already known.
        """
        if check:
            exc = _checked_in_place(exc, self.n_neurons, "external_exc", low=0)
        # A caller that asks for no inhibition should not pay for a sheet of
        # zeros; zero is what the arithmetic below subtracts either way.
        inh = None if external_inh is None else self._parameter(external_inh, "external_inh", low=0)
        dt = _scalar(dt, "dt", positive=True)
        modulation = self._modulation(modulator)
        if not isinstance(learn, (bool, np.bool_)):
            raise ValueError("learn must be boolean")
        wanted = None if wanted is None else _indices(wanted, self.n_neurons, "wanted")
        if self._device is not None:
            return self._step_device(exc, inh, dt, bool(learn), modulation, wanted)
        if inh is None:
            inh = 0.
        # A very large dt/tau means complete relaxation, not an invalid state.
        voltage_blend, voltage_keep, adaptation_blend, adaptation_keep = self._blend_pair(dt)
        working = self._buffers()
        # The driving rates are read three ways below and nothing writes to
        # them, so they are clipped into one working array, not two.
        rate = np.clip(self._voltage, 0, 1, out=working['rate'])
        rec_exc, rec_inh = self.synapses.currents(rate)
        with np.errstate(over="raise", invalid="raise"):
            current = rec_exc
            np.add(self._bias, current, out=current)
            current += exc
            current -= rec_inh
            current -= inh
            np.multiply(self._gain, self._adaptation, out=working['term'])
            current -= working['term']
            voltage = working['voltage']
            np.multiply(voltage_keep, self._voltage, out=voltage)
            np.multiply(voltage_blend, current, out=current)
            voltage += current
            adaptation = working['adaptation']
            np.multiply(adaptation_keep, self._adaptation, out=adaptation)
            np.multiply(adaptation_blend, rate, out=working['rate_next'])
            adaptation += working['rate_next']
        # Nothing is committed until every check has passed, so a failed step
        # leaves the cell states exactly as the next step would find them.
        if not (np.isfinite(current).all() and np.isfinite(voltage).all() and np.isfinite(adaptation).all()):
            raise FloatingPointError("non-finite candidate neuron state")
        if np.any(adaptation < 0) or np.any(adaptation > 1):
            raise FloatingPointError("adaptation left its invariant interval")
        new_rate = np.clip(voltage, 0, 1, out=working['rate_next'])
        elapsed = self._learning_elapsed + dt if learn else self._learning_elapsed
        if learn and elapsed + 1e-12 >= self.learning_interval:
            self.synapses.update(rate, new_rate, modulator=modulation, dt=elapsed)
            elapsed = 0.
        np.copyto(self._voltage, voltage)
        np.copyto(self._adaptation, adaptation)
        self._learning_elapsed = elapsed
        activity = self.activity
        return activity if wanted is None else activity[wanted]

    def _step_device(self, exc, inh, dt, learn, modulation, wanted=None):
        """The same step as above, evaluated on the device.

        The candidates are computed first and committed last, so a failed check
        or a failed learning write leaves the cells exactly as they were. Where
        the device can record a step, that whole sheet of arithmetic is one
        launch and the predicates travel back with the candidates; where it
        cannot, they are still asked of the device one round trip at a time.
        """
        device = self._device
        torch = device.torch
        if device.recorded_available():
            rate, voltage, adaptation, flags = device.recorded_step(exc, inh, dt)
            if not flags.all():
                if not flags[0]:
                    raise FloatingPointError('non-finite synaptic current')
                if not flags[1]:
                    raise FloatingPointError("non-finite candidate neuron state")
                raise FloatingPointError("adaptation left its invariant interval")
        else:
            excitatory = to_device(torch, exc, device.device)
            inhibitory = None if inh is None else to_device(torch, inh, device.device)
            rate, current, voltage, adaptation = device.advance(excitatory, inhibitory, dt)
            # The same three predicates, combined on the device instead of in
            # Python: `and` stops at the first one and pays a round trip to the
            # device for each, while `&` evaluates them together and the single
            # bool() below waits once for the answer.
            finite = (torch.isfinite(current).all() & torch.isfinite(voltage).all()
                      & torch.isfinite(adaptation).all())
            if not bool(finite):
                raise FloatingPointError("non-finite candidate neuron state")
            if bool((adaptation < 0).any() | (adaptation > 1).any()):
                raise FloatingPointError("adaptation left its invariant interval")
        elapsed = self._learning_elapsed + dt if learn else self._learning_elapsed
        if learn and elapsed + 1e-12 >= self.learning_interval:
            new_rate = torch.clamp(voltage, 0, 1)
            if np.ndim(modulation) == 0:
                modulator = torch.full((self.n_neurons,), float(modulation),
                                       dtype=torch.float64, device=device.device)
            else:
                modulator = to_device(torch, modulation, device.device)
            device.synapses.update(rate, new_rate, modulator, elapsed)
            elapsed = 0.
        device.commit(voltage, adaptation)
        self._learning_elapsed = elapsed
        return self.activity if wanted is None else device.rates_at(wanted)
