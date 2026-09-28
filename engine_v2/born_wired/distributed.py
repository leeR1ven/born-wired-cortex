"""One-graph feature routing experiment, not a visual recognition system."""

import numbers
import numpy as np

from .adaptive import AdaptiveNetwork
from .regulation import RegulatedSynapses
from .synapses import _indices, _scalar, _vector


def bridge_edges(source_ids, target_ids, n_neurons, weight=.01):
    """Wire arbitrary global cell IDs, regardless of region or processing depth.

    Return independent src/dst/weight arrays. Overlapping populations omit self
    edges; duplicate IDs within either population are rejected.
    """
    if isinstance(n_neurons, (bool, np.bool_)) or not isinstance(n_neurons, numbers.Integral) or n_neurons < 1:
        raise ValueError("n_neurons must be a positive integer")
    source = _indices(source_ids, n_neurons, "source_ids")
    target = _indices(target_ids, n_neurons, "target_ids")
    if len(np.unique(source)) != len(source) or len(np.unique(target)) != len(target):
        raise ValueError("population IDs must be unique")
    strength = _scalar(weight, "weight")
    src, dst = np.repeat(source, len(target)), np.tile(target, len(source))
    nonself = src != dst
    return src[nonself], dst[nonself], np.full(np.count_nonzero(nonself), strength)


class DistributedFeatureNetwork:
    """Four synthetic receptors, eight tuned hidden cells, one compressed output.

    All sensory, hidden, motor and inhibitory cells live in the same sparse
    graph. Training input is ordinary external excitation, never a weight edit.
    """

    def __init__(self, *, hidden_routes=True, seed=0):
        if not isinstance(hidden_routes, (bool, np.bool_)):
            raise ValueError("hidden_routes must be boolean")
        self.hidden_routes = bool(hidden_routes)
        self.seed = seed
        self.groups = dict(sensory=np.arange(4), hidden=np.arange(4, 12),
                           compressed=np.array([12]), motor=np.array([13, 14]),
                           inhibitory=np.array([15, 16]))
        self.n_neurons = 17
        g = self.groups
        rng = np.random.default_rng(seed)
        src, dst, weights, lower, upper, plasticity, tether = [], [], [], [], [], [], []

        def add(s, d, weight, associative=False):
            src.append(int(s)); dst.append(int(d)); weights.append(float(weight))
            lower.append(0 if associative else weight * .98)
            upper.append(.8 if associative else weight * 1.02)
            plasticity.append(1 if associative else .005)
            tether.append(0 if associative else .2)

        # Smooth, overlapping receptive fields follow feature position only.
        # No motor association or stimulus label enters these connections.
        hidden_positions = np.linspace(0, 3, 8)
        for feature, source in enumerate(g["sensory"]):
            for position, target in zip(hidden_positions, g["hidden"]):
                add(source, target, 1.6 * np.exp(-.5 * ((feature - position) / .65) ** 2))
        for source in g["hidden"]:
            add(source, g["compressed"][0], .85)
        for index in range(2):
            add(g["motor"][index], g["inhibitory"][index], 1)
            add(g["inhibitory"][index], g["motor"][1-index], 1.2)
        routing_sources = np.r_[g["hidden"], g["compressed"]] if hidden_routes else g["compressed"]
        bridge_src, bridge_dst, bridge_weights = bridge_edges(routing_sources, g["motor"], self.n_neurons)
        for source, target, weight in zip(bridge_src, bridge_dst, bridge_weights):
            add(source, target, weight * rng.uniform(.9, 1.1), associative=True)
        signs = np.ones(self.n_neurons)
        signs[g["inhibitory"]] = -1
        budgets = np.full(self.n_neurons, 10.)
        budgets[g["motor"]] = 4
        self.synapses = RegulatedSynapses(src, dst, weights, signs, self.n_neurons,
                                         lower=lower, upper=upper, budgets=budgets,
                                         plasticity=plasticity, learning_rate=.35,
                                         tether=tether, target_activity=.2)
        self._bias = np.zeros(self.n_neurons)
        self._bias[g["hidden"]] = -.1
        self._bias[g["motor"]] = -.1
        self.initial_weights = self.synapses.weights
        self.reset_state()

    def reset_state(self):
        """Reset experimental initial activity only; retain the learned graph."""
        self.network = AdaptiveNetwork(self.synapses, tau=.035, adaptation_gain=0, bias=self._bias)

    def step(self, features, motor_input=None, *, dt=.005, learn=True):
        features = _vector(features, 4, "features", low=0, high=1)
        motor_input = _vector(np.zeros(2) if motor_input is None else motor_input,
                              2, "motor_input", low=0)
        external = np.zeros(self.n_neurons)
        external[self.groups["sensory"]] = features
        external[self.groups["motor"]] = motor_input
        activity = self.network.step(external, dt=dt, learn=learn)
        return activity[self.groups["motor"]].copy()

    def graph_snapshot(self):
        """Serializable explicit global-ID graph for review and reproduction."""
        syn = self.synapses
        return dict(n_neurons=self.n_neurons, n_edges=len(syn.src), hidden_routes=self.hidden_routes,
                    groups={name: ids.tolist() for name, ids in self.groups.items()},
                    src=syn.src.tolist(), dst=syn.dst.tolist(), signs=syn.signs.tolist(),
                    initial_weights=self.initial_weights.tolist(), weights=syn.weights.tolist(),
                    lower=syn.lower.tolist(), upper=syn.w_max.tolist(), budgets=syn.budgets.tolist(),
                    plasticity=syn.plasticity.tolist(), tether=syn.tether.tolist(),
                    learning_rate=syn.learning_rate, target_activity=syn.target_activity.tolist(),
                    neuron_tau=.035, adaptation_tau=.4, adaptation_gain=0, bias=self._bias.tolist(),
                    initial_voltage=0, initial_adaptation=0, modulator=1)
