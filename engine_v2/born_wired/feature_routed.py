"""Synthetic feature routes into the existing Go2 controller's single graph."""

import numpy as np

from .adaptive import AdaptiveNetwork
from .innate import InnateController
from .regulation import RegulatedSynapses
from .synapses import _checked_in_place, _indices, _vector


class FeatureInputNetwork(AdaptiveNetwork):
    """Explicit sensory-current extension, retaining AdaptiveNetwork dynamics."""

    def __init__(self, synapses, receptor_ids, **neuron_parameters):
        super().__init__(synapses, **neuron_parameters)
        self.receptor_ids = _indices(receptor_ids, self.n_neurons, "receptor_ids")
        if self.receptor_ids.size != 4 or np.unique(self.receptor_ids).size != 4:
            raise ValueError("four distinct receptor IDs are required")
        self._features = np.zeros(4)

    def set_features(self, features):
        self._features = _vector(features, 4, "features", low=0, high=1)

    def step(self, external_exc, external_inh=None, dt=.002, learn=True, modulator=1,
             wanted=None):
        external = _vector(external_exc, self.n_neurons, "external_exc", broadcast=True, low=0)
        return self.step_owned(external, external_inh, dt=dt, learn=learn, modulator=modulator,
                               wanted=wanted, check=False)

    def step_owned(self, external, external_inh=None, dt=.002, learn=True, modulator=1,
                   wanted=None, check=True):
        """As step, for the excitation vector the outermost layer assembled."""
        if check:
            external = _checked_in_place(external, self.n_neurons, "external_exc", low=0)
        with np.errstate(over="raise", invalid="raise"):
            external[self.receptor_ids] += self._features
        return super().step_owned(external, external_inh, dt=dt, learn=learn,
                                  modulator=modulator, wanted=wanted, check=False)


class FeatureRoutedController(InnateController):
    """Add 13 feature cells without changing the parent motor or sensory code.

    The compressed route has cap .10, below the local activity target .15;
    hidden routes have cap .55. These are explicit structural assumptions.
    """

    def __init__(self, home_angles, lower_limits, upper_limits, *, hidden_routes=True,
                 empty_routes=("flexion",), **parent_parameters):
        if not isinstance(hidden_routes, (bool, np.bool_)):
            raise ValueError("hidden_routes must be boolean")
        super().__init__(home_angles, lower_limits, upper_limits, **parent_parameters)
        old, state = self.synapses, self.network
        self.original_n_neurons, self.original_n_edges = old.n_neurons, len(old.src)
        self.hidden_routes = bool(hidden_routes)
        ids = np.arange(old.n_neurons, old.n_neurons + 13)
        receptor, hidden, compressed = ids[:4], ids[4:12], ids[12:]
        self.groups.update(feature_receptors=receptor, feature_hidden=hidden, feature_compressed=compressed)
        # Which innate actions a route is laid beside.  A route carries no
        # action of its own; it is the place a lesson can be written later.
        actions = []
        for name in empty_routes:
            if name not in self.groups or np.size(self.groups[name]) != 1:
                raise ValueError("each empty route must name one innate action cell")
            actions.append(int(np.ravel(self.groups[name])[0]))
        if len(set(actions)) != len(actions):
            raise ValueError("empty route targets must be distinct")
        self.empty_route_actions = tuple(actions)
        src, dst, weights, lower, upper, plasticity, tether = [], [], [], [], [], [], []

        def add(source, target, weight, cap=None):
            src.append(int(source)); dst.append(int(target)); weights.append(float(weight))
            lower.append(weight * .98 if cap is None else 0)
            upper.append(weight * 1.02 if cap is None else cap)
            plasticity.append(.005 if cap is None else 3.)
            tether.append(.2 if cap is None else 0)

        for index, source in enumerate(receptor):
            for position, target in zip(np.linspace(0, 3, 8), hidden):
                add(source, target, 1.6 * np.exp(-.5 * ((index-position)/.65)**2))
        for source in hidden:
            add(source, compressed[0], .85)
        for action in actions:
            if hidden_routes:
                for source in hidden:
                    add(source, action, .001, cap=.55)
            add(compressed[0], action, .001, cap=.10)
        self.synapses = RegulatedSynapses(
            np.r_[old.src, src], np.r_[old.dst, dst], np.r_[old.weights, weights],
            np.r_[old.signs, np.ones(13)], old.n_neurons + 13,
            lower=np.r_[old.lower, lower], upper=np.r_[old.w_max, upper],
            budgets=np.r_[old.budgets, np.full(13, 10.)],
            plasticity=np.r_[old.plasticity, plasticity], tether=np.r_[old.tether, tether],
            learning_rate=old.learning_rate, target_activity=np.r_[old.target_activity, np.full(13, .15)])
        anchor = np.r_[old.anchor, weights]
        anchor.flags.writeable = False
        self.synapses.anchor = anchor
        added_bias = np.zeros(13)
        added_bias[4:12] = -.1
        self.network = FeatureInputNetwork(
            self.synapses, receptor,
            tau=np.r_[state._tau, np.full(13, .035)],
            adaptation_tau=np.r_[state._adaptation_tau, np.full(13, .4)],
            adaptation_gain=np.r_[state._gain, np.zeros(13)], bias=np.r_[state._bias, added_bias],
            initial_voltage=np.r_[state.voltage, np.zeros(13)],
            initial_adaptation=np.r_[state.adaptation, np.zeros(13)])
        self.initial_weights = self.synapses.weights
        self._initial_voltage = self.network.voltage
        self._initial_adaptation = self.network.adaptation

    def install_weights(self, weights):
        """Put another model's weight vector on this same graph.

        The cells, the edge list and the structural bounds do not change, so a
        weight vector from another model built from the same seed lands on the
        same addresses and can be averaged, taken elementwise, or added.  The
        graph is rebuilt around the vector because the device copy of the
        weights is taken at construction, and the cells are returned to their
        initial state so that two arms start from the same body and brain.
        """
        syn = self.synapses
        merged = np.asarray(weights, dtype=float)
        if merged.shape != syn._weights.shape:
            raise ValueError("weights must match this graph's edge list")
        merged = np.clip(merged.copy(), syn.lower, syn.w_max)
        rebuilt = RegulatedSynapses(syn.src, syn.dst, merged, syn.signs, syn.n_neurons,
                                    lower=syn.lower, upper=syn.w_max, budgets=syn.budgets,
                                    plasticity=syn.plasticity,
                                    learning_rate=syn.learning_rate, tether=syn.tether,
                                    target_activity=syn.target_activity)
        rebuilt.anchor = syn.anchor
        state = self.network
        self.synapses = rebuilt
        self.network = FeatureInputNetwork(
            rebuilt, self.groups["feature_receptors"], tau=state._tau,
            adaptation_tau=state._adaptation_tau, adaptation_gain=state._gain,
            bias=state._bias, initial_voltage=self._initial_voltage,
            initial_adaptation=self._initial_adaptation)
        return rebuilt.weights.copy()

    def step(self, observation, *, features=None, **parent_step_parameters):
        features = _vector(np.zeros(4) if features is None else features, 4, "features", low=0, high=1)
        self.network.set_features(features)
        try:
            return super().step(observation, **parent_step_parameters)
        finally:
            self.network.set_features(np.zeros(4))
