"""Device execution for the regulated synapse graph and its sheet of cells.

The arithmetic here is the arithmetic of synapses.py, regulation.py and
adaptive.py. Only the place it runs changes: the same expressions, over the same
numbers, evaluated on a graphics device instead of the host. Three consequences
are stated here rather than hidden.

* A postsynaptic sum over edges is a scatter. The obvious device scatter uses
  atomics, whose order depends on how the blocks happen to finish: two runs of
  one model then disagree in the last bits, and a rerun is not the same
  experiment. `Segmented` therefore expresses the same sum as a gather into
  fixed-width rows plus a reduction along that axis, which torch evaluates in a
  fixed order. The result is reproducible run to run but is not the host's
  edge-order sum to the last bit, so tools/compare_engines.py measures the gap.
* The host arrays on the owning objects stay the public face of the model.
  `weights`, `voltage` and `adaptation` keep answering with host snapshots; the
  device holds the working copy and writes it back when a host reader asks.
* Only the reduced quantities differ from the host at all, and only below the
  fifteenth significant digit. Every other operation is elementwise.

The device is chosen by the environment variable BORN_WIRED_DEVICE: `auto`
(default) takes CUDA when torch is importable and a device is present, `cpu`
keeps the host engine, and any other value is handed to torch. A machine without
torch, or without a GPU, keeps the host engine, because a missing accelerator is
not an error.
"""

from __future__ import annotations

import os

import numpy as np


ENVIRONMENT = 'BORN_WIRED_DEVICE'

# A recorded step is the same arithmetic, in the same order, as an eager step;
# the switch exists so the two paths can be compared against each other.
RECORDING = 'BORN_WIRED_RECORDING'

_MISSING = object()
_resolved = _MISSING


class Unavailable(RuntimeError):
    """The requested device cannot be used in this process."""


def torch_module():
    """Import torch on first use: the import is heavy and often unnecessary."""
    try:
        import torch
    except Exception as error:  # pragma: no cover - depends on the host
        raise Unavailable(str(error)) from error
    return torch


def resolve(preference=None):
    """The torch device to evaluate on, or None to keep the host engine."""
    global _resolved
    if preference is None and _resolved is not _MISSING:
        return _resolved
    choice = (preference or os.environ.get(ENVIRONMENT) or 'auto').strip().lower()
    device = None
    if choice not in ('cpu', 'off', 'none', 'numpy'):
        try:
            torch = torch_module()
            if choice in ('auto', 'gpu'):
                device = torch.device('cuda') if torch.cuda.is_available() else None
            else:
                device = torch.device(choice)
        except Unavailable:
            device = None
        except (RuntimeError, AssertionError, ValueError):
            # A machine that advertises a device it cannot open, or a driver
            # older than the wheel, falls back to the host engine too.
            device = None
    if preference is None:
        _resolved = device
    return device


def set_default(preference):
    """Fix the process default for graphs built from now on; None means host."""
    global _resolved
    _resolved = _MISSING
    _resolved = resolve(preference)
    return _resolved


def name_of(device):
    """A short label for reports; `host` is the NumPy engine."""
    return 'host' if device is None else str(device)


def recording_requested():
    """Whether a device step may be recorded once and replayed as one launch."""
    choice = (os.environ.get(RECORDING) or 'on').strip().lower()
    return choice not in ('0', 'off', 'false', 'no', 'eager', 'numpy')


def to_device(torch, array, device):
    """A float64 device copy of one host array."""
    array = np.asarray(array)
    if array.dtype != np.float64 or not array.flags.writeable or not array.flags.c_contiguous:
        array = np.array(array, dtype=np.float64, copy=True)
    return torch.as_tensor(array, device=device)


_float = to_device


def _integer(torch, array, device):
    return torch.as_tensor(np.array(array, dtype=np.int64, copy=True), device=device)


def _boolean(torch, array, device):
    return torch.as_tensor(np.array(array, dtype=bool, copy=True), device=device)


class Segmented:
    """A per-destination sum, written so that a rerun reproduces it exactly.

    Counting starts indexed by destination, so the edges of one destination are
    a contiguous run. The runs are cut into rank groups whose widths grow by
    three, which keeps the padding below four times the edge count while leaving
    few enough groups to launch. Each group becomes a (cells, width) table of
    edge addresses plus a mask for the padding, and the sum over a row is a
    reduction along a fixed axis.
    """

    def __init__(self, torch, device, n_neurons, dst, base=None):
        self.torch = torch
        self.device = device
        self.n_neurons = n_neurons
        counts = torch.bincount(dst, minlength=n_neurons)
        order = torch.argsort(dst, stable=True)
        if base is not None:
            order = base[order]
        zeros = torch.zeros(1, dtype=torch.int64, device=device)
        starts = torch.cat([zeros, counts.cumsum(0)[:-1]])
        self.levels = []
        self.slots = 0
        low, width = 0, 4
        while True:
            cells = torch.nonzero(counts > low, as_tuple=False).flatten()
            if cells.numel() == 0:
                break
            present = counts[cells]
            # Ranks are absolute: this group continues where the last one ended.
            rank = low + torch.arange(width, device=device)
            usable = rank.unsqueeze(0) < present.unsqueeze(1)
            ranks = torch.clamp(rank.unsqueeze(0), max=(present - 1).unsqueeze(1))
            self.levels.append((cells, order[starts[cells].unsqueeze(1) + ranks], usable))
            self.slots += cells.numel()*width
            low += width
            width *= 3

    def reduce(self, values):
        """Sum `values` (one entry per counted edge) into one entry per cell."""
        torch = self.torch
        total = torch.zeros(self.n_neurons, dtype=torch.float64, device=self.device)
        for cells, edges, usable in self.levels:
            part = (values[edges] * usable).sum(dim=1)
            total.index_put_((cells,), total.index_select(0, cells) + part)
        return total


class SynapseTensors:
    """Device copy of one regulated graph: edges, ranges and the write path."""

    def __init__(self, synapses, device):
        torch = torch_module()
        self.torch = torch
        self.device = device
        self.owner = synapses
        self.n_neurons = synapses.n_neurons
        self.learning_rate = float(synapses.learning_rate)
        self.src = _integer(torch, synapses.src, device)
        self.dst = _integer(torch, synapses.dst, device)
        self.lower = _float(torch, synapses._lower, device)
        self.upper = _float(torch, synapses.w_max, device)
        self.plasticity = _float(torch, synapses.plasticity, device)
        self.anchor = _float(torch, synapses.anchor, device)
        self.tether = _float(torch, synapses.tether, device)
        self.target_activity = _float(torch, synapses.target_activity, device)
        self.weights = _float(torch, synapses._weights, device)
        # Whether the owner's host copy still holds the values the device does.
        # A write moves the device array; handing eight megabytes back across
        # the bus on every write, for a reader that may not exist, is work the
        # model should only do when someone actually reads.
        self.host_stale = False
        # The two sign blocks are addresses into one edge array on the host. A
        # device keeps the same split as two index lists, so one pass per block
        # still produces one postsynaptic sum per block.
        positive = _boolean(torch, synapses._positive, device)
        order = torch.arange(synapses.src.size, device=device)
        self.exc_edges = order[positive]
        self.inh_edges = order[~positive]
        self.exc_dst = self.dst[self.exc_edges]
        self.inh_dst = self.dst[self.inh_edges]
        self.exc_segments = Segmented(torch, device, self.n_neurons, self.exc_dst, self.exc_edges)
        self.inh_segments = Segmented(torch, device, self.n_neurons, self.inh_dst, self.inh_edges)
        self.groups = tuple(
            (_integer(torch, movable, device), self.dst[_integer(torch, movable, device)],
             _float(torch, lower, device), _float(torch, available, device),
             Segmented(torch, device, self.n_neurons, _integer(torch, dst, device)))
            for movable, dst, lower, available in synapses._structure())
        self.slots = self.exc_segments.slots + self.inh_segments.slots

    def empty(self):
        """A zero vector over cells, the shape every postsynaptic sum has."""
        return self.torch.zeros(self.n_neurons, dtype=self.torch.float64, device=self.device)

    def currents(self, activity):
        """Postsynaptic excitatory and inhibitory sums for a device activity."""
        excitatory, inhibitory = self._currents_unchecked(activity)
        # Combined on the device: two predicates, one wait, where `and` would
        # stop at the first and pay a round trip for each.
        if not self.finite_pair(excitatory, inhibitory):
            raise FloatingPointError('non-finite synaptic current')
        return excitatory, inhibitory

    def _currents_unchecked(self, activity):
        """The same two sums, without the wait that cannot be recorded."""
        flow = self.weights * activity[self.src]
        return self.exc_segments.reduce(flow), self.inh_segments.reduce(flow)

    def finite_pair(self, excitatory, inhibitory):
        """Finiteness of two sums as one predicate, so a reader waits once."""
        torch = self.torch
        return bool(torch.isfinite(excitatory).all() & torch.isfinite(inhibitory).all())

    def host_currents(self, activity):
        excitatory, inhibitory = self.currents(_float(self.torch, activity, self.device))
        return excitatory.cpu().numpy(), inhibitory.cpu().numpy()

    def project(self, proposed):
        """Clip each edge to its own range, then share the cell's remaining budget."""
        torch = self.torch
        result = torch.clamp(proposed, min=self.lower, max=self.upper)
        for edges, dst, lower, available, segments in self.groups:
            if edges.numel() == 0:
                continue
            sums = segments.reduce(result[edges] - lower)
            if not bool(torch.isfinite(sums).all()):
                raise FloatingPointError('non-finite budget sum')
            scale = torch.ones(self.n_neurons, dtype=torch.float64, device=self.device)
            over = sums > available
            scale[over] = available[over] / sums[over]
            result.index_copy_(0, edges, lower + (result[edges] - lower)*scale[dst])
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('non-finite projected weights')
        return result

    def host_project(self, proposed):
        return self.project(_float(self.torch, proposed, self.device)).cpu().numpy()

    def update(self, pre, post, modulator, dt):
        """One local plasticity write, the expression RegulatedSynapses uses."""
        torch = self.torch
        weights = self.weights
        source_rate = pre[self.src]
        local = (post[self.dst] - self.target_activity[self.dst]) * source_rate
        correlated = ((self.learning_rate * modulator[self.dst]) * local
                      - self.tether * (weights - self.anchor))
        change = torch.where(source_rate > 0, correlated,
                             (self.anchor - weights) * self.tether)
        proposal = ((self.plasticity * dt) * change) + weights
        if not bool(torch.isfinite(proposal).all()):
            raise FloatingPointError('non-finite plasticity proposal')
        result = self.project(proposal)
        # Only commit once every check has succeeded, and into the array that
        # is already there: rebinding would leave a recorded step reading the
        # buffer it was recorded against, so the write lands in place and the
        # address every reader already holds is the one carrying the new
        # values.
        self.weights.copy_(result)
        self.host_stale = True
        return self.weights

    def refresh_host_weights(self):
        """Put this graph's values into the host array readers look at.

        The write lands in the array that is already there rather than in a
        fresh one, so anything holding that array - a diagnostic that took it
        at build time - stays pointed at the values as they move.
        """
        if self.host_stale:
            np.copyto(self.owner._weights, self.weights.detach().cpu().numpy())
            self.host_stale = False

    def host_update(self, pre, post, modulator, dt):
        torch = self.torch
        result = self.update(_float(torch, pre, self.device), _float(torch, post, self.device),
                             _float(torch, modulator, self.device), dt)
        return result.cpu().numpy()

    def totals(self):
        """Separate sums per sign block, the diagnostic budget_totals returns."""
        return (self.exc_segments.reduce(self.weights).cpu().numpy(),
                self.inh_segments.reduce(self.weights).cpu().numpy())


class CellTensors:
    """Device copy of one network's cells, and the step that advances them."""

    def __init__(self, network, synapses):
        torch = synapses.torch
        self.torch = torch
        self.device = synapses.device
        self.synapses = synapses
        self.n_neurons = network.n_neurons
        self.tau = _float(torch, network._tau, self.device)
        self.adaptation_tau = _float(torch, network._adaptation_tau, self.device)
        self.gain = _float(torch, network._gain, self.device)
        self.bias = _float(torch, network._bias, self.device)
        self.voltage = _float(torch, network._voltage, self.device)
        self.adaptation = _float(torch, network._adaptation, self.device)
        self._blends = None
        # Set by commit; a host reader of voltage/adaptation asks for the write
        # back instead of paying for it on every step.
        self.stale = False
        # Host copy of the committed rates. One transfer serves every reader of
        # this step's activity, and the mirrors above stay behind until someone
        # actually asks for the raw voltage.
        self._rate = None
        # Page-locked landing space for the rates on their way to the host. The
        # driver fills it directly; it is never handed out, so overwriting it on
        # the next step cannot be seen by anyone.
        self._staging = None
        # The recorded step. One step of this sheet is some sixty kernels whose
        # arithmetic costs a fraction of what issuing them costs, so the step is
        # recorded once per dt and replayed as a single launch. What the
        # recording reads has to keep its address for the life of the sheet -
        # the two exogenous vectors below, the weights, the cell state - and
        # what it writes lands in candidate buffers of its own, so a failed
        # check still leaves the committed state untouched.
        self._graph = None
        self._graph_dt = None
        self._inputs = None
        self._outputs = None
        self._inhibition_written = False

    def blends(self, dt):
        """Voltage and adaptation blends for this dt, measured once per dt."""
        cached = self._blends
        if cached is None or cached[0] != dt:
            torch = self.torch
            voltage_blend = -torch.expm1(-dt / self.tau)
            adaptation_blend = -torch.expm1(-dt / self.adaptation_tau)
            cached = (dt, voltage_blend, 1 - voltage_blend,
                      adaptation_blend, 1 - adaptation_blend)
            self._blends = cached
        return cached[1:]

    def advance(self, excitatory, inhibitory, dt):
        """Candidate rates, current, voltage and adaptation; nothing is stored."""
        rate, current, voltage, adaptation, recurrent, suppression = self.candidates(
            excitatory, inhibitory, dt)
        if not self.synapses.finite_pair(recurrent, suppression):
            raise FloatingPointError('non-finite synaptic current')
        return rate, current, voltage, adaptation

    def candidates(self, excitatory, inhibitory, dt):
        """The whole step, up to the predicates the caller still owes.

        Every expression here is the expression `advance` evaluates; only the
        checks moved out, because a check that reaches the host cannot be
        recorded. A caller that runs this inside a capture reads the returns
        back through the same predicates once the recording has replayed.
        """
        torch = self.torch
        voltage_blend, voltage_keep, adaptation_blend, adaptation_keep = self.blends(dt)
        rate = torch.clamp(self.voltage, 0, 1)
        recurrent, suppression = self.synapses._currents_unchecked(rate)
        current = (recurrent + self.bias) + excitatory
        current = current - suppression
        if inhibitory is not None:
            current = current - inhibitory
        current = current - self.gain * self.adaptation
        voltage = voltage_keep * self.voltage + voltage_blend * current
        adaptation = adaptation_keep * self.adaptation + adaptation_blend * rate
        return rate, current, voltage, adaptation, recurrent, suppression

    def recorded_available(self):
        """Whether this sheet can run its step as one recorded launch."""
        return (recording_requested() and self.device.type == 'cuda'
                and self.torch.cuda.is_available())

    def _recorded_buffers(self):
        """The static device memory a recording reads and writes, built once.

        A graph addresses the same memory on every replay, so the two inputs it
        reads have to outlive the recording and the candidates it writes have
        to land somewhere the committed state does not: a failed check is
        allowed to cost the step and nothing else. The three predicates travel
        back together in one small tensor, so a reader waits once for all of
        them instead of once for each.
        """
        if self._inputs is None:
            torch = self.torch

            def vector():
                return torch.zeros(self.n_neurons, dtype=torch.float64,
                                   device=self.device)

            self._inputs = {'excitation': vector(), 'inhibition': vector()}
            self._outputs = {'rate': vector(), 'voltage': vector(),
                             'adaptation': vector(),
                             'flags': torch.zeros(3, dtype=torch.bool, device=self.device)}
        return self._inputs, self._outputs

    def _recorded_body(self, dt):
        """One step of arithmetic as the recording holds it: no host talk."""
        torch = self.torch
        inputs, outputs = self._inputs, self._outputs
        rate, current, voltage, adaptation, recurrent, suppression = self.candidates(
            inputs['excitation'], inputs['inhibition'], dt)
        finite_currents = (torch.isfinite(recurrent).all()
                           & torch.isfinite(suppression).all())
        finite_state = (torch.isfinite(current).all() & torch.isfinite(voltage).all()
                        & torch.isfinite(adaptation).all())
        in_interval = (adaptation >= 0).all() & (adaptation <= 1).all()
        outputs['rate'].copy_(rate)
        outputs['voltage'].copy_(voltage)
        outputs['adaptation'].copy_(adaptation)
        outputs['flags'].copy_(torch.stack([finite_currents, finite_state, in_interval]))

    def record(self, dt):
        """Record this dt's step once. Another dt is another step, another graph."""
        torch = self.torch
        # Measured before the capture: the blends depend on dt alone, so they
        # are constants of the recording from here on.
        self.blends(dt)
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._recorded_body(dt)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._recorded_body(dt)
        self._graph = graph
        self._graph_dt = dt

    def recorded_step(self, excitation, inhibition, dt):
        """One step for host vectors, replayed through the recording.

        Returns the candidates and the three predicates of the recorded step,
        so the caller checks and commits exactly as the eager path does.
        """
        torch = self.torch
        self._recorded_buffers()
        inputs, outputs = self._inputs, self._outputs
        inputs['excitation'].copy_(
            torch.as_tensor(np.ascontiguousarray(excitation, dtype=np.float64)))
        if inhibition is None:
            # The recording always subtracts an inhibition vector. Zero is what
            # the eager path subtracts when it has none, and the memory is only
            # cleared when an earlier step left something in it.
            if self._inhibition_written:
                inputs['inhibition'].zero_()
                self._inhibition_written = False
        else:
            inputs['inhibition'].copy_(
                torch.as_tensor(np.ascontiguousarray(inhibition, dtype=np.float64)))
            self._inhibition_written = True
        if self._graph is None or self._graph_dt != dt:
            self.record(dt)
        self._graph.replay()
        return (outputs['rate'], outputs['voltage'], outputs['adaptation'],
                outputs['flags'].cpu().numpy())

    def commit(self, voltage, adaptation):
        self.voltage.copy_(voltage)
        self.adaptation.copy_(adaptation)
        self.stale = True
        self._rate = None

    def host_rate(self):
        """Firing rates on the host, transferred once and cached for this step.

        The rates cross the bus through a page-locked buffer, which the driver
        can write straight into instead of pinning fresh pages copy by copy,
        and the clip runs on this side of the bus: the smaller of two numbers is
        the same number wherever it is taken. What is handed back is a fresh
        array, as before, so a reader that keeps one still holds this step's
        rates after the next step has overwritten the landing space.
        """
        rate = self._rate
        if rate is None:
            torch = self.torch
            if self.device.type != 'cuda':
                rate = torch.clamp(self.voltage, 0, 1).detach().cpu().numpy()
                rate.flags.writeable = False
                self._rate = rate
                return rate
            staging = self._staging
            if staging is None:
                staging = torch.empty(self.n_neurons, dtype=torch.float64, pin_memory=True)
                self._staging = staging
            staging.copy_(self.voltage, non_blocking=True)
            torch.cuda.synchronize()
            rate = np.clip(staging.numpy(), 0, 1)
            rate.flags.writeable = False
            self._rate = rate
        return rate

    def rates_at(self, indices):
        """This step's rates for the named cells only, on the host.

        A reader that wants a few groups out of a sheet of four hundred
        thousand cells should not carry the sheet across the bus. The rows are
        gathered and clipped where the sheet already lies, and only those rows
        travel; the numbers are the ones host_rate would have handed back for
        the same cells.
        """
        torch = self.torch
        index = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        return torch.clamp(self.voltage.index_select(0, index), 0, 1).cpu().numpy()

    def write_back(self, network):
        """Refresh the host arrays the reader-facing properties answer with."""
        np.copyto(network._voltage, self.voltage.detach().cpu().numpy())
        np.copyto(network._adaptation, self.adaptation.detach().cpu().numpy())
        self.stale = False
