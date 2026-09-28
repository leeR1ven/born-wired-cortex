"""A developmental wiring candidate: shared joint populations, local reflexes.

No action names, keyframes, phase clock, pose lookup, or learned policy occurs
in step(). Joint operating points and reflex gains are explicit innate wiring.
"""
import numbers
import numpy as np
from .adaptive import AdaptiveNetwork
from .regulation import RegulatedSynapses
from .synapses import _vector, _scalar
from .encoding import TuningEncoder


class InnateController:
    def __init__(self, home_angles, lower_limits, upper_limits, *, seed=0,
                 motor_units=100, proprio_units=32, association_units=128,
                 balance_gain=.22, gait_gain=.65, swing_delay=.16,
                 rhythm_speed=2., rear_lift=1.8,
                 scaffold_fraction=.03, balanced_gait=False, lift_gain=.35,
                 balanced_rear_lift=1.):
        if isinstance(motor_units, bool) or not isinstance(motor_units, numbers.Integral) or motor_units < 10:
            raise ValueError('motor_units must be an integer >= 10')
        for name, value, minimum in (('proprio_units', proprio_units, 8), ('association_units', association_units, 8)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        self.home = _vector(home_angles, 12, 'home_angles')
        self.lower = _vector(lower_limits, 12, 'lower_limits')
        self.upper = _vector(upper_limits, 12, 'upper_limits')
        self.span = self.upper - self.lower
        if np.any(self.span <= 0) or np.any(self.home < self.lower) or np.any(self.home > self.upper):
            raise ValueError('invalid anatomical limits or operating points')
        balance_gain = _scalar(balance_gain, 'balance_gain')
        gait_gain = _scalar(gait_gain, 'gait_gain')
        lift_gain = _scalar(lift_gain, 'lift_gain')
        balanced_rear_lift = _scalar(balanced_rear_lift, 'balanced_rear_lift', positive=True)
        if not isinstance(balanced_gait, (bool, np.bool_)):
            raise ValueError('balanced_gait must be boolean')
        swing_delay = _scalar(swing_delay, 'swing_delay', positive=True)
        rhythm_speed = _scalar(rhythm_speed, 'rhythm_speed', positive=True)
        rear_lift = _scalar(rear_lift, 'rear_lift', positive=True)
        fraction = _scalar(scaffold_fraction, 'scaffold_fraction')
        if fraction > .25:
            raise ValueError('candidate scaffold fraction must be <= .25')
        rng = np.random.default_rng(seed)
        signs, bias, tau, adapt, initial = [], [], [], [], []
        self.groups = {}

        def cells(name, count, sign=1, base=0, time=.025, fatigue=0, start=0):
            idx = np.arange(len(signs), len(signs) + count)
            self.groups[name] = idx
            signs.extend(np.broadcast_to(sign, (count,)))
            bias.extend(np.broadcast_to(base, (count,)))
            tau.extend(np.broadcast_to(time, (count,)))
            adapt.extend(np.broadcast_to(fatigue, (count,)))
            initial.extend(np.broadcast_to(start, (count,)))
            return idx

        tonic = cells('tonic', 1, start=1)[0]
        flex = cells('flexion', 1)[0]
        flex_i = cells('flexion_inhibitory', 1, -1)[0]
        loco = cells('locomotion', 1)[0]
        cue = cells('cue', 4)
        # Two copies have fixed transmitter signs; both sense the same local
        # receptor rate. Direction selectivity is encoded in the wiring.
        vest_e = cells('vestibular_exc', 4)
        vest_i = cells('vestibular_inh', 4, -1)
        contact = cells('contact', 4)
        limit_e = cells('limit_exc', 24)
        limit_i = cells('limit_inh', 24, -1)
        proprio = cells('proprioception', 12 * proprio_units)
        phase = cells('phase', 2, time=.08 / rhythm_speed, fatigue=2.5, start=[.001, 0])
        phase_i = cells('phase_inhibitory', 2, -1, time=.008)
        phase_delay = cells('phase_delay', 2, time=swing_delay / rhythm_speed)
        phase_delay_i = cells('phase_delay_inhibitory', 2, -1, time=.008)
        centres = (np.arange(motor_units) + .5) / motor_units
        self.motor_units = motor_units
        self.motor_gain = 12.
        operating = (self.home - self.lower) / self.span
        motor_bias = np.tile(.5 - self.motor_gain * centres, 12)
        motor = cells('motor', 12 * motor_units, base=motor_bias,
                      start=motor_bias + np.repeat(self.motor_gain * operating, motor_units))
        recruit = cells('recruitment', 12, start=.85)
        association = cells('association', association_units, base=-.12,
                            sign=np.where(np.arange(association_units) % 5 == 0, -1, 1))
        src, dst, weights, plasticity, tether, lower, upper = [], [], [], [], [], [], []

        def edge(s, d, weight, memory=False):
            if weight == 0:
                return
            for target in np.atleast_1d(d):
                src.append(int(s)); dst.append(int(target)); weights.append(float(weight))
                plasticity.append(1 if memory else .02)
                tether.append(0 if memory else .2)
                lower.append(0 if memory else weight * (1 - fraction))
                upper.append(.9 if memory else weight * (1 + fraction))

        edge(flex, flex_i, 1)
        for k in range(2):
            edge(loco, phase[k], 1)
            edge(phase[k], phase_i[k], 1)
            edge(phase[k], phase_delay[k], 1)
            edge(phase_delay[k], phase_delay_i[k], 1)
            edge(phase_i[k], phase[1-k], 2.5)
        # A cue has no innate meaning. These four initially negligible edges
        # are the bounded associative pathway, in the same graph as reflexes.
        for s in cue:
            edge(s, flex, .0001, memory=True)

        # Intermediate proprioceptive cells have their own outgoing routes;
        # they need not survive association-layer compression to cause a
        # withdrawal reflex. The coefficients follow joint position tuning,
        # without assigning an action label to any hidden neuron.
        for j in range(12):
            for source, position in zip(proprio.reshape(12, proprio_units)[j], np.linspace(0, 1, proprio_units)):
                for direction, magnitude in enumerate((max((.08-position)/.08, 0),
                                                       max((position-.92)/.08, 0))):
                    edge(source, [limit_e[2*j+direction], limit_i[2*j+direction]], magnitude)

        def joint_effect(source_e, source_i, joint, radians):
            source = source_e if radians >= 0 else source_i
            edge(source, motor.reshape(12, motor_units)[joint],
                 abs(radians) * self.motor_gain / self.span[joint])

        for j in range(12):
            units = motor.reshape(12, motor_units)[j]
            edge(tonic, units, self.motor_gain * operating[j])
            edge(tonic, recruit[j], .85)
            edge(contact[j // 3], recruit[j], .15)
            # Continuous posterior flexion drive. It reuses the same motor
            # units as tonic support, tilt reflexes and rhythmic drive.
            flex_amount = (0, .7, -1.05)[j % 3] * (1 if j // 3 >= 2 else .08)
            joint_effect(flex, flex_i, j, flex_amount)
            side = 1 if j // 3 in (0, 2) else -1
            front = 1 if j // 3 < 2 else -1
            # up_x then up_y, each with positive/negative sensory cells.
            correction = (0, -1, 2)[j % 3] * balance_gain
            for axis, anatomical in ((0, front), (1, side)):
                for polarity in (0, 1):
                    index = 2 * axis + polarity
                    joint_effect(vest_e[index], vest_i[index], j,
                                 correction * anatomical * (1 if polarity == 0 else -1))
            # Proximity-to-limit withdrawal uses only that joint's receptors.
            joint_effect(limit_e[2*j], limit_i[2*j], j, .12)
            joint_effect(limit_e[2*j+1], limit_i[2*j+1], j, -.12)
            diagonal = 0 if j // 3 in (0, 3) else 1
            leg_lift = lift_gain * (balanced_rear_lift if j // 3 >= 2 else 1.)
            # Antagonistic phasic inputs. No time or phase is supplied from
            # outside the neural dynamics; sensory drive merely excites it.
            if j % 3 == 1:
                joint_effect(phase_delay[diagonal], phase_delay_i[diagonal], j, -gait_gain)
                joint_effect(phase_delay[1-diagonal], phase_delay_i[1-diagonal], j, gait_gain)
                if balanced_gait:
                    joint_effect(phase[diagonal], phase_i[diagonal], j, leg_lift)
                    joint_effect(phase[1-diagonal], phase_i[1-diagonal], j, -leg_lift)
            elif j % 3 == 2:
                if balanced_gait:
                    joint_effect(phase[diagonal], phase_i[diagonal], j, -2 * leg_lift)
                    joint_effect(phase[1-diagonal], phase_i[1-diagonal], j, 2 * leg_lift)
                else:
                    joint_effect(phase[diagonal], phase_i[diagonal], j,
                                 -2 * gait_gain * (rear_lift if j // 3 >= 2 else 1))

        # Mixed, overlapping body representation. No stand/sit identifiers
        # are supplied to these cells, or used to choose their incoming edges.
        sensory = np.r_[proprio, contact, vest_e, cue]
        for target in association:
            fan_in = min(len(sensory), 128, round(64 * proprio_units / 32))
            for source in rng.choice(sensory, fan_in, replace=False):
                edge(source, target, float(rng.uniform(.18, .30)))
        for source in association:
            candidates = association[association != source]
            for target in rng.choice(candidates, 4, replace=False):
                edge(source, target, .025)
        n = len(signs)
        budget = np.full(n, 40.)
        self.synapses = RegulatedSynapses(src, dst, weights, signs, n, lower=lower,
                                         upper=upper, budgets=budget, plasticity=plasticity,
                                         tether=tether, learning_rate=.12, target_activity=.15)
        adaptation_tau = np.full(n, .4)
        adaptation_tau[phase] /= rhythm_speed
        self.network = AdaptiveNetwork(self.synapses, tau=tau, adaptation_tau=adaptation_tau, adaptation_gain=adapt,
                                       bias=bias, initial_voltage=initial)
        self.initial_weights = self.synapses.weights
        self._initial_voltage = self.network.voltage
        self.proprio_encoder = TuningEncoder(12, proprio_units)

    def step(self, observation, *, flexion=0, locomotion=0, cues=None,
             dt=.002, learn=True, feedback=True, silence_motor=False, modulator=1.):
        drives = _vector([flexion, locomotion], 2, 'drives', low=0, high=1)
        cues = _vector(np.zeros(4) if cues is None else cues, 4, 'cues', low=0, high=1)
        q = _vector(observation['joint_position'], 12, 'joint_position')
        gravity = _vector(observation['gravity_direction'], 3, 'gravity_direction', low=-1, high=1)
        omega = _vector(observation['angular_velocity_local'], 3, 'angular_velocity_local')
        contacts = np.asarray(observation['foot_contact'])
        if contacts.shape != (4,) or not np.isin(contacts, [0, 1]).all():
            raise ValueError('invalid foot_contact')
        if any(not isinstance(flag, (bool, np.bool_)) for flag in (feedback, silence_motor)):
            raise ValueError('ablation flags must be boolean')
        ext = np.zeros(self.network.n_neurons)
        g = self.groups
        ext[g['tonic']] = 1
        ext[g['flexion']] = drives[0]
        ext[g['locomotion']] = drives[1]
        ext[g['cue']] = cues
        normalized = np.clip((q - self.lower) / self.span, 0, 1)
        if feedback:
            ext[g['proprioception']] = self.proprio_encoder.encode(normalized).ravel()
            tilt = np.clip(np.array([gravity[0] + .08 * omega[1], gravity[1] - .08 * omega[0]]) / .25, -1, 1)
            receptors = np.maximum(np.column_stack([tilt, -tilt]), 0).ravel()
            ext[g['vestibular_exc']] = receptors
            ext[g['vestibular_inh']] = receptors
            ext[g['contact']] = contacts.astype(float)
        # ext was built here and is not read again, so the network may add into
        # it in place rather than working on a second copy of the sheet.
        # These two groups are the only rates this step reads, so they are the
        # only ones asked for: on the device engine the rest of the sheet stays
        # where it is instead of crossing the bus to be thrown away.
        wanted = np.concatenate((g['motor'], g['recruitment']))
        rates = self.network.step_owned(ext, dt=dt, learn=learn, modulator=modulator,
                                        wanted=wanted)
        split = g['motor'].size
        motor = rates[:split].reshape(12, self.motor_units)
        activation = rates[split:].copy()
        if silence_motor:
            motor = np.zeros_like(motor)
            activation[:] = 0
        # A convex endpoint can exceed upper by one rounding ULP. Keep the
        # numerical readout within the same anatomical interval as its rates.
        target = np.clip(self.lower + motor.mean(axis=1) * self.span, self.lower, self.upper)
        return target, activation

    def diagnostics(self):
        rate = self.network.activity
        w = self.synapses.weights
        return dict(neurons=self.network.n_neurons, edges=len(w),
                    max_weight_change=float(np.max(np.abs(w - self.initial_weights))),
                    scaffold_max_relative_change=float(np.max(np.abs(w[self.synapses.tether > 0]
                         / self.initial_weights[self.synapses.tether > 0] - 1))),
                    memory_weights=w[self.synapses.tether == 0].tolist(),
                    mean_activity=float(rate.mean()), active_fraction=float(np.mean(rate > .01)))
