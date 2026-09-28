"""A small fixed delayed-synapse auditory circuit, not a sound locator.

Raw normalized audio is the only runtime input. Delays are innate axons, not
measured arrival times. No FFT, correlation, frequency/angle estimate or action
selection occurs. The nonlinear temporal tuning is intentionally approximate.
"""
import numbers

import numpy as np

# The circuit below advances one audio sample at a time, and at 16 kHz that is
# a hundred and sixty interpreter-visible array calls for every tenth of a
# second of sound. Compiling the same recurrence removes the interpreter from
# between the samples; the arithmetic it performs is the arithmetic written
# out below it, term for term and in the same order. numba is a way to run this
# module faster, never a way to run a different one, so a host without it keeps
# the plain loop.
try:
    from numba import njit
except Exception:  # pragma: no cover - depends on the host
    njit = None

if njit is not None:
    @njit(cache=True, fastmath=False, boundscheck=False)
    def _advance_samples(voltage, adaptation, flat, cursor, taps, dst, signed, bias,
                         gain, decay, decay_new, adaptation_keep, adaptation_new,
                         external, rate, arrivals, weighted, current):
        """The per-sample recurrence of step(), compiled.

        Every statement mirrors the host line it replaces, including which
        operand comes first, because a float64 sum depends on the order of its
        terms and this circuit is compared against the host sample for sample.
        `fastmath` stays off for the same reason: it would let the compiler
        reassociate.
        """
        neurons = voltage.size
        ring = flat.size // neurons
        edges = dst.size
        samples = external.shape[0]
        for index in range(samples):
            for cell in range(neurons):
                value = voltage[cell]
                if value > 1.:
                    value = 1.
                elif value < 0.:
                    value = 0.
                rate[cell] = value
                flat[cursor*neurons + cell] = value
            for edge in range(edges):
                arrivals[edge] = flat[taps[cursor, edge]]
            for edge in range(edges):
                weighted[edge] = signed[edge]*arrivals[edge]
            for cell in range(neurons):
                current[cell] = 0.
            for edge in range(edges):
                current[dst[edge]] += weighted[edge]
            for cell in range(neurons):
                current[cell] = bias[cell] + current[cell]
            for cell in range(neurons):
                current[cell] = current[cell] + external[index, cell]
            for cell in range(neurons):
                current[cell] = current[cell] - gain[cell]*adaptation[cell]
                current[cell] = current[cell]*decay_new[cell]
                voltage[cell] = decay[cell]*voltage[cell] + current[cell]
                adaptation[cell] = adaptation[cell]*adaptation_keep + adaptation_new*rate[cell]
            cursor += 1
            if cursor == ring:
                cursor = 0
        return cursor


# Keys that step() returns. EmbodiedController reads rates and spatial_rates;
# changing either side of that pair on its own is rejected when the modules load.
OUTPUT_KEYS = ('rates', 'orientation', 'spatial_rates', 'pinna_rates')


class AuditoryNeurons:
    """Persistent 16 kHz circuit; ears L/R, bands nominally 262/523/880 Hz.

    step(raw[2,N]) returns rates[2,3] and orientation[2], all in [0,1].
    Orientation is left/right *cell activity*, not a decoded location. Fixed
    transmitter signs apply to every outgoing edge. Matched E/I copies share
    receptors but remain explicit neurons with their own states.
    """

    def __init__(self, sample_rate=16000):
        if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, numbers.Integral) or sample_rate != 16000:
            raise ValueError("this fixed axonal circuit requires sample_rate=16000")
        self.sample_rate = int(sample_rate)
        self._receptor_gain = 24.
        self.nominal_frequencies = (262, 523, 880)
        self.period_samples = np.array([61, 31, 18], dtype=np.int64)
        half_period = np.array([30, 16, 9], dtype=np.int64)
        names, signs, tau, bias, adaptation_gain = [], [], [], [], []

        def cells(label, shape, sign=1, time=.00008, base=0., fatigue=.03):
            ids = np.arange(len(names), len(names)+int(np.prod(shape))).reshape(shape)
            for coordinate in np.ndindex(shape):
                names.append(label+"_"+"_".join(map(str, coordinate)))
                signs.append(float(np.broadcast_to(sign, shape)[coordinate]))
                tau.append(time); bias.append(base); adaptation_gain.append(fatigue)
            return ids

        self._hair = cells("hair_ear_polarity_transmitter", (2, 2, 2), sign=[1, -1], time=.00004, base=-.15, fatigue=.02)
        band = cells("band_ear_frequency_polarity", (2, 3, 2), base=-.03, fatigue=.04)
        self._pool = cells("band_pool_ear_frequency", (2, 3), time=.020, fatigue=.03)
        coincidence = cells("coincidence_side_polarity_transmitter", (2, 2, 2), sign=[1, -1], base=-1.25, fatigue=.03)
        energy = cells("energy_ear_transmitter", (2, 2), sign=[1, -1], time=.010, fatigue=.03)
        self._orientation = cells("orientation_side", (2,), time=.020, fatigue=.15)
        frequency_coincidence = cells('frequency_coincidence', (2,3,2,2), sign=[1,-1], base=-.4, fatigue=.03)
        frequency_energy = cells('frequency_energy', (2,3,2), sign=[1,-1], time=.010)
        self._spatial = cells('spatial_ear_frequency', (2,3), time=.040, fatigue=.12)
        self._pinna = cells('pinna_ear_direction_transmitter', (2,2,2), sign=[1,-1],
                            base=-.02, time=.035, fatigue=.06)
        self._pinna_mean = cells('pinna_ear_mean_band', (2,), time=.035, fatigue=.06)
        # Opponent E/I copies form damped recurrent neural resonators. Their
        # fixed connections preserve frequency-specific phase before the
        # rectifying band cells. No measured frequency or direction is decoded.
        self._resonator = cells('resonator_ear_frequency_axis_transmitter', (2,3,4,2),
                                sign=[1,-1], time=1e-8, base=.5, fatigue=0.)
        src, dst, weights, delays = [], [], [], []

        def edge(source, target, weight, delay=0):
            src.append(int(source)); dst.append(int(target))
            weights.append(float(weight)); delays.append(int(delay))

        for ear in range(2):
            for polarity in range(2):
                he, hi = self._hair[ear, polarity]
                for frequency, period in enumerate(self.period_samples):
                    target = band[ear, frequency, polarity]
                    positive, negative = (0,1) if polarity == 0 else (1,0)
                    edge(self._resonator[ear,frequency,positive,0], target, 2.)
                    edge(self._resonator[ear,frequency,negative,1], target, 2.)
                    edge(target, self._pool[ear, frequency], 1.)
                for target in energy[ear]:
                    edge(he, target, 1.)
                # A left-preferring cell receives delayed left and immediate
                # right excitation. Its mirrored partner reverses these axons.
                for target in coincidence[ear, polarity]:
                    edge(he, target, 1., 3)
                    edge(self._hair[1-ear, polarity, 0], target, 1.)
                edge(coincidence[ear, polarity, 0], self._orientation[ear], 3.)
                edge(coincidence[1-ear, polarity, 1], self._orientation[ear], 3.)
            edge(energy[ear, 0], self._orientation[ear], .7)
            edge(energy[1-ear, 1], self._orientation[ear], .4)
            for frequency in range(3):
                for polarity in range(2):
                    source = band[ear,frequency,polarity]
                    for target in frequency_coincidence[ear,frequency,polarity]:
                        edge(source, target, 1., 3)
                        edge(band[1-ear,frequency,polarity], target, 1.)
                    for target in frequency_energy[ear,frequency]:
                        edge(source, target, 1.)
                    edge(frequency_coincidence[ear,frequency,polarity,0], self._spatial[ear,frequency], 3.)
                    edge(frequency_coincidence[1-ear,frequency,polarity,1], self._spatial[ear,frequency], 3.)
                edge(frequency_energy[ear,frequency,0], self._spatial[ear,frequency], .7)
                edge(frequency_energy[1-ear,frequency,1], self._spatial[ear,frequency], .4)

        # Two arrival times cannot separate a sound in front of the head from its
        # mirror image behind it, because the ears are a pair of points and both
        # sources sit the same distance from each of them. What does separate
        # them is what the outer ear does to the sound on the way in: it holds the
        # top of a sound from behind back a little more than the top of one from
        # in front, so the same noise arrives duller from behind and brighter from
        # in front. These cells answer that shape. Each is excited by one band and
        # held down by the average of all three through its own inhibitory copy,
        # so the top-band cell is left firing by a brighter-than-average top and
        # the bottom-band cell by a duller-than-average top. Comparing a band with
        # the average compares shapes and not levels, so a loud far sound and a
        # quiet near one reach them alike. A sound carrying only one band has no
        # shape to compare: it fires its own band's cell by the same amount from
        # either direction, so a pure tone is left unplaced rather than
        # misplaced - which is what happens in a real ear too.
        for ear in range(2):
            low, mid, high = self._pool[ear, 0], self._pool[ear, 1], self._pool[ear, 2]
            mean = self._pinna_mean[ear]
            ahead, ahead_i = self._pinna[ear, 0]
            behind, behind_i = self._pinna[ear, 1]
            for band_cell in (low, mid, high):
                edge(band_cell, mean, 1./3.)
            edge(high, ahead, 1.)
            edge(mean, ahead_i, 1.45)
            edge(ahead_i, ahead, 1.)
            edge(low, behind, 1.)
            edge(mean, behind_i, 1.43)
            edge(behind_i, behind, 1.)

        for ear in range(2):
            for frequency, hz in enumerate(self.nominal_frequencies):
                damping = np.exp(-2*np.pi*20/self.sample_rate)
                angle = 2*np.pi*hz/self.sample_rate
                c, s = damping*np.cos(angle)/2, damping*np.sin(angle)/2
                # Axes: x+, x-, y+, y-. Each row is a fixed synaptic stencil.
                stencil = np.array([[c,-c,-s,s],[-c,c,s,-s],[s,-s,c,-c],[-s,s,-c,c]])
                for target_axis, source_axis in np.ndindex(4,4):
                    coefficient = stencil[target_axis,source_axis]
                    source = self._resonator[ear,frequency,source_axis,0 if coefficient>=0 else 1]
                    for target in self._resonator[ear,frequency,target_axis]:
                        edge(source, target, abs(coefficient))

        self.names = tuple(names)
        self.n_neurons = len(names)
        self._signs = np.asarray(signs)
        self._src, self._dst = np.asarray(src), np.asarray(dst)
        self._weights, self._delays = np.asarray(weights), np.asarray(delays)
        self._bias, self._gain = np.asarray(bias), np.asarray(adaptation_gain)
        self._decay = np.exp(-1/(self.sample_rate*np.asarray(tau)))
        self._adaptation_decay = float(np.exp(-1/(self.sample_rate*.150)))
        self._signed_weights = self._weights*self._signs[self._src]
        # The two ear-polarity weights, the resonator drive scale and the one
        # minus each decay are fixed numbers; the inner loop is short enough
        # that building them per sample is a measurable share of it.
        self._ear_sign = np.array([1., -1.])
        self._resonator_gain = 1/30.
        self._decay_new = 1. - self._decay
        self._adaptation_keep = self._adaptation_decay
        self._adaptation_new = 1. - self._adaptation_decay
        for value in (self.period_samples, self._signs, self._src, self._dst, self._weights,
                      self._delays, self._bias, self._gain, self._decay, self._signed_weights):
            value.flags.writeable = False
        self._ear_sign.flags.writeable = False
        # Where the two receptor groups sit once a chunk's currents are laid
        # out as whole rows rather than rebuilt sample by sample. Both are
        # structure - the ids never move - so the flattening is done once.
        self._hair_ids = self._hair.ravel()
        self._resonator_plus = self._resonator[:, :, 0, :].ravel()
        self._resonator_minus = self._resonator[:, :, 1, :].ravel()
        self.reset()

    def reset(self):
        """Explicit experimental reset; step never resets between audio chunks."""
        self._voltage = np.zeros(self.n_neurons)
        self._voltage[self._resonator] = .5
        self._adaptation = np.zeros(self.n_neurons)
        ring = int(self._delays.max())+1
        self._history = np.zeros((ring, self.n_neurons))
        self._cursor = 0
        self.sample_count = 0
        # Where each edge reads, flattened, for every position the cursor can
        # be in. The edge delays never change, so this is arithmetic the loop
        # below would otherwise repeat once per edge per sample.
        self._taps = (((np.arange(ring)[:, None] - self._delays[None, :]) % ring)*self.n_neurons
                      + self._src[None, :]).astype(np.intp)
        # Working arrays, sized once: the loop writes into these rather than
        # asking the allocator for a fresh array twenty times per sample.
        self._rate_buffer = np.empty(self.n_neurons)
        self._drive_buffer = np.empty(self.n_neurons)
        self._current_buffer = np.empty(self.n_neurons)
        self._arrivals = np.empty(self._src.size)
        self._weighted_buffer = np.empty(self._src.size)

    @property
    def weights(self):
        result = self._weights.copy()
        result.flags.writeable = False
        return result

    def snapshot(self):
        """Independent copies include every neuron's state and delayed axon state."""
        external_low, external_high = np.zeros(self.n_neurons), np.zeros(self.n_neurons)
        external_high[self._hair] = self._receptor_gain
        external_low[self._resonator[:,:,:2,:]] = -1/30
        external_high[self._resonator[:,:,:2,:]] = 1/30
        return dict(names=self.names, sample_count=self.sample_count, sample_rate=self.sample_rate,
                    external_low=external_low, external_high=external_high,
                    resonator_ids=self._resonator.copy(), spatial_output_ids=self._spatial.copy(),
                    receptor_gain=self._receptor_gain,
                    period_samples=self.period_samples.copy(), src=self._src.copy(), dst=self._dst.copy(),
                    weights=self._weights.copy(), signs=self._signs.copy(), delay_samples=self._delays.copy(),
                    bias=self._bias.copy(), adaptation_gain=self._gain.copy(),
                    membrane_decay=self._decay.copy(), adaptation_decay=self._adaptation_decay,
                    voltage=self._voltage.copy(), adaptation=self._adaptation.copy(),
                    activity=np.clip(self._voltage, 0., 1.), axon_history=self._history.copy(),
                    axon_cursor=self._cursor, receptor_ids=self._hair.copy(), band_output_ids=self._pool.copy(),
                    orientation_output_ids=self._orientation.copy())

    def step(self, raw):
        """Consume one finite normalized waveform chunk without mutating it.

        Each sample writes old rates into the axon ring, sums signed incoming
        synaptic currents, and applies membrane leak and local adaptation.
        All input validation precedes computation; state commits after success.
        """
        if not isinstance(raw, np.ndarray) or raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[0] != 2 or raw.shape[1] == 0:
            raise ValueError("raw must be a real ndarray with shape (2,N), N>0")
        if not np.isfinite(raw).all() or np.any(raw < -1) or np.any(raw > 1):
            raise ValueError("raw audio must be finite and normalized to [-1,1]")
        raw = raw.astype(np.float64, copy=True)
        samples = raw.shape[1]
        # Receptor currents are a function of the waveform alone: no cell state
        # enters them. Laying the whole chunk out in one pass therefore does the
        # same arithmetic the loop used to repeat per sample - four fancy-index
        # assignments, a maximum and a product - once for the chunk instead.
        external = np.zeros((samples, self.n_neurons))
        external[:, self._hair_ids] = np.repeat(
            self._receptor_gain*np.maximum(raw.T[:, :, None]*self._ear_sign, 0.).reshape(samples, 4), 2, axis=1)
        # Signed ear pressure is a receptor current, not an extracted spectrum.
        # Opponent receptor currents drive x+ and x- cells.
        pressure = self._resonator_gain*raw.T
        external[:, self._resonator_plus] = np.repeat(pressure, 6, axis=1)
        external[:, self._resonator_minus] = np.repeat(-pressure, 6, axis=1)
        voltage, adaptation, history = self._voltage.copy(), self._adaptation.copy(), self._history.copy()
        flat = history.ravel()
        cursor = self._cursor
        rate, term = self._rate_buffer, self._drive_buffer
        arrivals, weighted = self._arrivals, self._weighted_buffer
        decay_new, adaptation_new = self._decay_new, self._adaptation_new
        if njit is not None:
            cursor = _advance_samples(voltage, adaptation, flat, cursor, self._taps,
                                      self._dst, self._signed_weights, self._bias,
                                      self._gain, self._decay, decay_new,
                                      self._adaptation_keep, adaptation_new,
                                      external, rate, arrivals, weighted,
                                      self._current_buffer)
        else:
            cursor = self._advance_on_host(voltage, adaptation, history, flat, cursor,
                                           external, rate, term, arrivals, weighted,
                                           decay_new, adaptation_new)
        if not (np.isfinite(voltage).all() and np.isfinite(adaptation).all()):
            raise FloatingPointError("non-finite auditory neural state")
        self._voltage, self._adaptation, self._history = voltage, adaptation, history
        self._cursor = cursor
        self.sample_count += raw.shape[1]
        rates = np.clip(voltage, 0., 1.)
        return dict(rates=rates[self._pool].copy(), orientation=rates[self._orientation].copy(),
                    spatial_rates=rates[self._spatial].copy(),
                    pinna_rates=rates[self._pinna[:,:,0]].copy())

    def _advance_on_host(self, voltage, adaptation, history, flat, cursor, external,
                         rate, term, arrivals, weighted, decay_new, adaptation_new):
        """The same recurrence, sample by sample, in NumPy.

        This is what runs wherever numba is absent. It is also the definition
        the compiled copy above answers to: the two are compared sample for
        sample, so a change here belongs in both.
        """
        samples = external.shape[0]
        ring = history.shape[0]
        with np.errstate(over="raise", invalid="raise"):
            for index in range(samples):
                # min-then-max is the same clamp as np.clip without the
                # fromnumeric wrapper the caller pays for on every sample.
                np.minimum(voltage, 1., out=rate)
                np.maximum(rate, 0., out=rate)
                history[cursor] = rate
                np.take(flat, self._taps[cursor], out=arrivals)
                np.multiply(self._signed_weights, arrivals, out=weighted)
                current = np.bincount(self._dst, weights=weighted, minlength=self.n_neurons)
                np.add(self._bias, current, out=current)
                np.add(current, external[index], out=current)
                np.multiply(self._gain, adaptation, out=term)
                np.subtract(current, term, out=current)
                np.multiply(current, decay_new, out=current)
                np.multiply(self._decay, voltage, out=voltage)
                np.add(voltage, current, out=voltage)
                np.multiply(adaptation, self._adaptation_keep, out=adaptation)
                np.multiply(rate, adaptation_new, out=term)
                np.add(adaptation, term, out=adaptation)
                cursor = (cursor+1) % ring
        return cursor
