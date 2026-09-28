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

    def __init__(self, home_angles, lower_limits, upper_limits, *, hidden_routes=True, **parent_parameters):
        if not isinstance(hidden_routes, (bool, np.bool_)):
            raise ValueError("hidden_routes must be boolean")
        super().__init__(home_angles, lower_limits, upper_limits, **parent_parameters)
        old, state = self.synapses, self.network
        self.original_n_neurons, self.original_n_edges = old.n_neurons, len(old.src)
        self.hidden_routes = bool(hidden_routes)
        ids = np.arange(old.n_neurons, old.n_neurons + 13)
        receptor, hidden, compressed = ids[:4], ids[4:12], ids[12:]
        self.groups.update(feature_receptors=receptor, feature_hidden=hidden, feature_compressed=compressed)
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
        if hidden_routes:
            for source in hidden:
                add(source, self.groups["flexion"][0], .001, cap=.55)
        add(compressed[0], self.groups["flexion"][0], .001, cap=.10)
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

    def step(self, observation, *, features=None, **parent_step_parameters):
        features = _vector(np.zeros(4) if features is None else features, 4, "features", low=0, high=1)
        self.network.set_features(features)
        try:
            return super().step(observation, **parent_step_parameters)
        finally:
            self.network.set_features(np.zeros(4))
