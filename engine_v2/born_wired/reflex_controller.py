"""Expanded same-graph reflex candidate. Sensors inject currents, never actions."""
import numpy as np
from .feature_routed import FeatureRoutedController, FeatureInputNetwork
from .regulation import RegulatedSynapses
from .synapses import _checked_in_place, _scalar, _vector

# The gain the whole righting action was tuned at. ``righting_gain`` is the one
# knob for all of it - the burst and the three postures - so that zero really
# means "this animal does not try to get up", which is the ablation the tests
# and the experiments switch on. The postures keep their own relative sizes; the
# master only scales them together with the burst.
RIGHTING_GAIN_NOMINAL = 2.5


class ReflexInputNetwork(FeatureInputNetwork):
    def __init__(self, synapses, receptor_ids, **parameters):
        super().__init__(synapses, receptor_ids, **parameters)
        self.reflex_input = np.zeros(self.n_neurons)

    def step(self, external_exc, *args, **kwargs):
        external = _vector(external_exc, self.n_neurons, 'external_exc', low=0)
        return self.step_owned(external, *args, check=False, **kwargs)

    def step_owned(self, external, *args, check=True, **kwargs):
        """As step, for the excitation vector the outermost layer assembled."""
        if check:
            external = _checked_in_place(external, self.n_neurons, 'external_exc', low=0)
        with np.errstate(over='raise', invalid='raise'):
            external += self.reflex_input
        return super().step_owned(external, *args, check=False, **kwargs)


class ReflexController(FeatureRoutedController):
    def __init__(self, home_angles, lower_limits, upper_limits, *, motor_units=200,
                 proprio_units=64, association_units=256, avoidance_gain=.25,
                 avoidance_mutual=.5,
                 withdrawal_gain=.35, startle_gain=.18, support_calf=-1.5, clearance_lift=.6,
                 stumble_decay=.45, clearance_threshold=.80, clearance_swing_gain=1.5, clearance_contact_gain=2.6,
                 righting_gain=2.5, righting_hip_gain=.6,
                 righting_posture_threshold=.9, righting_posture_drive=1.5,
                 righting_tuck_drive=2., righting_side_gain=.9,
                 righting_posture_fatigue=.9, righting_posture_time=.55,
                 righting_competition=1.1,
                 righting_tuck_gain=.6, righting_tuck_hip=.45,
                 righting_roll_gain=.5, righting_roll_hip=.8,
                 righting_stand_gain=.7, righting_stand_hip=.3,
                 righting_state_memory=1.2, righting_memory_time=.35,
                 righting_fallen_time=.4, righting_fallen_drive=1.5,
                 righting_fallen_threshold=.4, righting_fallen_hold_gain=1.5,
                 righting_upright_drive=.7, righting_upright_time=.5, righting_release_gain=2.,
                 righting_success_threshold=.45, righting_success_inhibition=1.,
                 righting_success_reward=1.5,
                 righting_memory_weight=.2, righting_state_gain=1.6, righting_cap=1.2,
                 righting_memory_tether=0., righting_learn=0., righting_learn_rate=100.,
                 dopamine_gain=0., touch_dopamine_gain=0., backup_routes=(), backup_cap=1.5,
                 backup_plasticity=1.,
                 balanced_gait=True, support_thigh_offset=0., vision_gain=.7, hearing_gain=.7,
                 pinna_orient_gain=1.2, wall_gain=12., wall_time=.6, wall_brake_gain=2.,
                 wall_turn_gain=1., wall_hip_gain=.4, wall_retreat_gain=.5,
                 steady_gain=.5, steady_hip_gain=.5, steady_inhibition=2., **parameters):
        working_angles = _vector(home_angles, 12, 'home_angles')
        if support_calf is not None:
            calf = _vector([support_calf], 1, 'support_calf')[0]
            working_angles[1::3] = -.5 * calf
            working_angles[2::3] = calf
        working_angles[1::3] += _vector([support_thigh_offset], 1, 'support_thigh_offset')[0]
        super().__init__(working_angles, lower_limits, upper_limits, motor_units=motor_units,
                         proprio_units=proprio_units, association_units=association_units,
                         balanced_gait=balanced_gait, **parameters)
        avoidance_gain = _scalar(avoidance_gain, 'avoidance_gain')
        avoidance_mutual = _scalar(avoidance_mutual, 'avoidance_mutual')
        withdrawal_gain = _scalar(withdrawal_gain, 'withdrawal_gain')
        startle_gain = _scalar(startle_gain, 'startle_gain')
        vision_gain = _scalar(vision_gain, 'vision_gain')
        hearing_gain = _scalar(hearing_gain, 'hearing_gain')
        pinna_orient_gain = _scalar(pinna_orient_gain, 'pinna_orient_gain')
        wall_gain = _scalar(wall_gain, 'wall_gain')
        wall_time = _scalar(wall_time, 'wall_time', positive=True)
        wall_brake_gain = _scalar(wall_brake_gain, 'wall_brake_gain')
        wall_turn_gain = _scalar(wall_turn_gain, 'wall_turn_gain')
        wall_hip_gain = _scalar(wall_hip_gain, 'wall_hip_gain')
        wall_retreat_gain = _scalar(wall_retreat_gain, 'wall_retreat_gain')
        steady_gain = _scalar(steady_gain, 'steady_gain')
        steady_hip_gain = _scalar(steady_hip_gain, 'steady_hip_gain')
        steady_inhibition = _scalar(steady_inhibition, 'steady_inhibition')
        clearance_lift = _scalar(clearance_lift, 'clearance_lift')
        stumble_decay = _scalar(stumble_decay, 'stumble_decay', positive=True)
        clearance_threshold = _scalar(clearance_threshold, 'clearance_threshold', positive=True)
        clearance_swing_gain = _scalar(clearance_swing_gain, 'clearance_swing_gain')
        clearance_contact_gain = _scalar(clearance_contact_gain, 'clearance_contact_gain')
        righting_gain = _scalar(righting_gain, 'righting_gain')
        righting_hip_gain = _scalar(righting_hip_gain, 'righting_hip_gain')
        righting_posture_drive = _scalar(righting_posture_drive, 'righting_posture_drive')
        righting_tuck_drive = _scalar(righting_tuck_drive, 'righting_tuck_drive')
        righting_side_gain = _scalar(righting_side_gain, 'righting_side_gain')
        righting_posture_time = _scalar(righting_posture_time, 'righting_posture_time')
        righting_posture_fatigue = _scalar(righting_posture_fatigue, 'righting_posture_fatigue')
        righting_competition = _scalar(righting_competition, 'righting_competition')
        righting_tuck_gain = _scalar(righting_tuck_gain, 'righting_tuck_gain')
        righting_tuck_hip = _scalar(righting_tuck_hip, 'righting_tuck_hip')
        righting_roll_gain = _scalar(righting_roll_gain, 'righting_roll_gain')
        righting_roll_hip = _scalar(righting_roll_hip, 'righting_roll_hip')
        righting_stand_gain = _scalar(righting_stand_gain, 'righting_stand_gain')
        righting_stand_hip = _scalar(righting_stand_hip, 'righting_stand_hip')
        righting_state_memory = _scalar(righting_state_memory, 'righting_state_memory')
        righting_memory_time = _scalar(righting_memory_time, 'righting_memory_time')
        righting_fallen_time = _scalar(righting_fallen_time, 'righting_fallen_time')
        righting_fallen_drive = _scalar(righting_fallen_drive, 'righting_fallen_drive')
        righting_fallen_hold_gain = _scalar(righting_fallen_hold_gain, 'righting_fallen_hold_gain')
        righting_upright_drive = _scalar(righting_upright_drive, 'righting_upright_drive')
        righting_upright_time = _scalar(righting_upright_time, 'righting_upright_time')
        righting_release_gain = _scalar(righting_release_gain, 'righting_release_gain')
        righting_fallen_threshold = _scalar(righting_fallen_threshold, 'righting_fallen_threshold')
        righting_success_inhibition = _scalar(righting_success_inhibition, 'righting_success_inhibition')
        righting_success_reward = _scalar(righting_success_reward, 'righting_success_reward')
        righting_memory_weight = _scalar(righting_memory_weight, 'righting_memory_weight')
        righting_state_gain = _scalar(righting_state_gain, 'righting_state_gain')
        righting_cap = _scalar(righting_cap, 'righting_cap')
        righting_memory_tether = _scalar(righting_memory_tether, 'righting_memory_tether')
        righting_learn_rate = _scalar(righting_learn_rate, 'righting_learn_rate')
        righting_posture_threshold = _scalar(righting_posture_threshold, 'righting_posture_threshold')
        righting_success_threshold = _scalar(righting_success_threshold, 'righting_success_threshold')
        righting_learn = float(righting_learn)
        if not np.isfinite(righting_learn) or righting_learn < 0 or righting_learn > 1:
            raise ValueError('righting_learn must lie in [0, 1]')
        # The dopamine gate slows the local rule down when nothing is being
        # taught and lets it run at its own rate while something is. It is a
        # fraction of that rate, so it cannot be larger than one, and zero is
        # the ordinary animal with no gate and no backup routes at all.
        dopamine_gain = _scalar(dopamine_gain, 'dopamine_gain')
        if dopamine_gain > 1:
            raise ValueError('dopamine_gain is a fraction of the learning rate and cannot exceed 1')
        touch_dopamine_gain = _scalar(touch_dopamine_gain, 'touch_dopamine_gain')
        backup_cap = _scalar(backup_cap, 'backup_cap')
        backup_plasticity = _scalar(backup_plasticity, 'backup_plasticity')
        backup_routes = tuple(backup_routes)
        for route in backup_routes:
            if len(route) != 2 or not all(isinstance(name, str) for name in route):
                raise ValueError('each backup route is a pair of group names')
        old, net = self.synapses, self.network
        signs, biases, taus, adaptation, adaptation_tau = [], [], [], [], []
        self.reflex_groups = {}

        def cells(name, count, sign=1, bias=0, tau=.025, fatigue=0, fatigue_tau=.4):
            ids = np.arange(old.n_neurons + len(signs), old.n_neurons + len(signs) + count)
            self.groups[name] = self.reflex_groups[name] = ids
            signs.extend(np.broadcast_to(sign, (count,)))
            biases.extend(np.broadcast_to(bias, (count,)))
            taus.extend(np.broadcast_to(tau, (count,)))
            adaptation.extend(np.broadcast_to(fatigue, (count,)))
            adaptation_tau.extend(np.broadcast_to(fatigue_tau, (count,)))
            return ids

        near = cells('near', 3)
        touch = cells('body_touch', 4)
        foot = cells('foot_obstacle', 4)
        load = cells('foot_load', 4)
        slip = cells('foot_slip', 4)
        tilt = cells('protective_tilt', 1)
        # Which way the body is being pushed off its feet, one cell for each of
        # the four ways it can go, and one leg per cell that reaches out to
        # catch it. Both are ordinary cells; the drive below is the direction
        # gravity is leaning in the body's own frame, which is a sense and not
        # an action.
        lean = cells('lean', 4)
        steady = cells('steady', 4)
        steady_i = cells('steady_inhibition', 4, sign=-1)
        steady_off = cells('steady_upright', 1, sign=-1)
        # A wall is not a stumble. The body stays against it until it has moved
        # away, so a light steady push has to count as much as a heavy one, and
        # that is what these four slow cells hold. They drive the same brake,
        # retreat and avoidance cells the instantaneous touch already drives,
        # one more voice on each.
        wall = cells('wall_contact', 4, bias=-.30, tau=wall_time)
        wall_turn = cells('wall_turn', 2)
        wall_turn_i = cells('wall_turn_inhibition', 2, sign=-1)
        abrupt = cells('abrupt_sound', 1)
        startle = cells('startle', 1, fatigue=8., fatigue_tau=1.2, tau=.015)
        startle_i = cells('startle_inhibition', 1, sign=-1)
        brake = cells('brake', 1, sign=-1)
        avoid = cells('avoidance', 2)
        avoid_i = cells('avoidance_inhibition', 2, sign=-1)
        protective = cells('protective_flexion', 1)
        protective_i = cells('protective_inhibition', 1, sign=-1)
        withdrawal = cells('withdrawal', 4)
        withdrawal_i = cells('withdrawal_inhibition', 4, sign=-1)
        impact = cells('impact_adaptation', 4, fatigue=9., fatigue_tau=.15, tau=.01)
        impact_i = cells('impact_inhibition', 4, sign=-1, tau=.01)
        # Touching a low obstacle leaves a slow per-leg state, and a second cell
        # needs that state together with the swing half of the rhythm. Neither
        # cell names an action; they are ordinary graph neurons using the same
        # update rule as the rest.
        stumble = cells('stumble', 4, bias=-.25, tau=stumble_decay, fatigue=1.2, fatigue_tau=3.)
        stumble_i = cells('stumble_inhibition', 4, sign=-1, tau=stumble_decay)
        clearance = cells('clearance', 4, bias=-clearance_threshold, tau=.08)
        clearance_i = cells('clearance_inhibition', 4, sign=-1)
        # Lying on the ground instead of the feet activates a slow state and,
        # on top of it, a burst per leg. The burst only appears when that
        # leg's swing half is present, so the legs work against the ground in
        # turn rather than all at once. They are ordinary graph neurons; the
        # rotation they add goes to the same shared motor units as a step.
        righting = cells('righting', 1, bias=-.45, tau=.35)
        righting_i = cells('righting_inhibition', 1, sign=-1)
        righting_push = cells('righting_push', 4, bias=-.75, tau=.06)
        righting_push_i = cells('righting_push_inhibition', 4, sign=-1)
        # Getting up is not one movement. Beside the burst above, the animal
        # also pulls its legs in under the body, rolls the trunk to one side
        # and pushes the legs out straight. Each is an ordinary cell on the
        # same shared motor units, each carries its own slow adaptation so a
        # posture that is not working gives way by itself, and each carries an
        # inhibitory copy that puts down the other two - the same competition
        # the two avoidance channels use. Nothing here picks a posture; the
        # drives and the strengths decide, and the one that fires when the
        # body comes back upright is the one whose connections to the fall
        # state are strengthened below.
        righting_tuck = cells('righting_tuck', 1, bias=-righting_posture_threshold,
                              tau=.05, fatigue=righting_posture_fatigue,
                              fatigue_tau=righting_posture_time)
        righting_tuck_i = cells('righting_tuck_inhibition', 1, sign=-1)
        righting_roll = cells('righting_roll', 2, bias=-righting_posture_threshold,
                              tau=.05, fatigue=righting_posture_fatigue,
                              fatigue_tau=righting_posture_time*1.5)
        righting_roll_i = cells('righting_roll_inhibition', 2, sign=-1)
        righting_stand = cells('righting_stand', 1, bias=-righting_posture_threshold,
                               tau=.05, fatigue=righting_posture_fatigue,
                               fatigue_tau=righting_posture_time*2.4)
        righting_stand_i = cells('righting_stand_inhibition', 1, sign=-1)
        # Slow copies of which way the body was tipped and which part of it was
        # on the ground: the fall state as it was a moment ago, the same idiom
        # the wall-contact cells use for a touch that outlasts the touch.
        righting_lean_memory = cells('righting_lean_memory', 4, tau=righting_state_memory)
        righting_touch_memory = cells('righting_touch_memory', 4, tau=righting_state_memory)
        # The learned edges land on the postures themselves, so the posture
        # that was firing at the moment the body came back up is the one whose
        # connections from that remembered state grow. Nothing else in the
        # graph is written by the reward below.
        # A slow copy of "the body is not on its feet" drives the postures, so
        # an action outlives the fall itself by a fraction of a second; and a
        # cell that needs that copy while the loss of the up axis is already
        # gone is the one that marks a successful righting.
        righting_memory = cells('righting_memory', 1, tau=righting_memory_time)
        righting_memory_i = cells('righting_memory_inhibition', 1, sign=-1)
        # "There is a fall in progress" is not a fading trace but a pair of
        # cells that keep each other going - the same mutually exciting pair
        # the gait recruitment uses. While the body is off its feet the pair
        # settles into its on state and stays there for as long as the animal
        # is on the ground, however long that is; it is switched off by the
        # third cell here, which answers "the body is back on its feet" and
        # takes about half a second to build up, and that lag is the window in
        # which the recovery can be credited.
        righting_fallen = cells('righting_fallen', 1, bias=-righting_fallen_threshold, tau=righting_fallen_time)
        righting_fallen_hold = cells('righting_fallen_hold', 1, bias=-righting_fallen_threshold,
                                     tau=righting_fallen_time)
        righting_upright = cells('righting_upright', 1, tau=righting_upright_time)
        righting_upright_i = cells('righting_upright_inhibition', 1, sign=-1, tau=righting_upright_time)
        righting_success = cells('righting_success', 1, bias=-righting_success_threshold, tau=.05)
        retreat = cells('retreat', 1)
        retreat_phase = cells('retreat_phase', 2, bias=-1.)
        retreat_phase_i = cells('retreat_phase_inhibition', 2, sign=-1)
        effort = cells('effort_receptor', 1)
        autonomous = cells('autonomous_receptor', 1)
        novelty = cells('novelty', 18, fatigue=12., fatigue_tau=6.)
        fatigue = cells('fatigue', 1, bias=-.22, tau=10.)
        fatigue_i = cells('fatigue_inhibition', 1, sign=-1)
        curiosity = cells('curiosity', 1, fatigue=1.4, fatigue_tau=25., tau=.25)
        rest = cells('rest', 1, bias=-.3, fatigue=2., fatigue_tau=8., tau=.4)
        rest_i = cells('rest_inhibition', 1, sign=-1)
        initiation = cells('initiation', 2, bias=-.45, tau=.15)
        retina = cells('retina', 18)  # two eyes, L/C/R sectors, R/G/B opponency
        stereo_near = cells('stereo_near', 3)
        cochlea = cells('cochlea', 6)  # left/right localization channels x frequency
        appetitive = cells('appetitive', 1, fatigue=3., fatigue_tau=20., tau=.15)
        orient = cells('orienting', 2, fatigue=.4, fatigue_tau=3.)
        orient_i = cells('orienting_inhibition', 2, sign=-1)
        aversive = cells('aversive', 3, fatigue=.5, fatigue_tau=8.)
        orient_vigor = cells('orienting_vigor', 2, bias=-.12, tau=.12)
        orient_vigor_i = cells('orienting_vigor_inhibition', 1, sign=-1, tau=.01)
        rhythm_recruit = cells('rhythm_recruitment', 2, bias=-.35, tau=.08)
        steering = cells('steering', 2, tau=.15)
        steering_i = cells('steering_inhibition', 2, sign=-1)
        auditory_spatial = cells('auditory_spatial', 6)
        # One cell per ear for a sound whose top band stands out and one for a
        # sound whose bottom band does: the outer ear brightens what comes from
        # in front and dulls what comes from behind, and these are the cells that
        # answer that. They carry no edges to any muscle yet, so they are sense
        # cells the local rule may grow from, not an action.
        auditory_pinna = cells('auditory_pinna', 4)
        auditory_approach = cells('auditory_approach', 2, bias=-.3)
        auditory_avoid = cells('auditory_avoid', 2, bias=-.3)
        # Dopamine is not a critic and carries no answer about right or wrong.
        # It is one ordinary cell whose rate says "this moment counts", and it
        # is read by learning_modulation below: while it is firing, the local
        # rule runs at its own rate everywhere; while it is quiet, the same
        # rule is slowed down by dopamine_gain. Which edges actually change is
        # still decided only by which cells were firing together.
        dopamine = cells('dopamine', 1, tau=.05)
        src, dst, weight = [], [], []
        learned, cap = [], []
        backup = []
        taken = set()

        def edge(source, targets, strength, learns=False, limit=None):
            if strength == 0:
                return
            for target in np.atleast_1d(targets):
                src.append(int(source)); dst.append(int(target)); weight.append(float(strength))
                learned.append(bool(learns))
                cap.append(float(strength if limit is None else limit))
                backup.append(False)
                taken.add((int(source), int(target)))

        def backup_edge(source, target):
            """A route that is present but carries nothing until experience writes it.

            The graph is fixed once built, so an edge that does not exist can
            never appear later. A backup edge is the other way round: it is laid
            down from the start at a weight too small to change any behaviour,
            with a lower bound of zero and a ceiling it may grow to, so the
            local rule has somewhere to write when those two populations do
            fire together. The teacher names the two populations and nothing
            else; not which cell, not how strongly, not in which direction.
            """
            if int(source) == int(target):
                return
            if (int(source), int(target)) in taken:
                return
            src.append(int(source)); dst.append(int(target)); weight.append(1e-4)
            learned.append(False)
            cap.append(float(backup_cap))
            backup.append(True)
            taken.add((int(source), int(target)))

        # Explicit innate sensory valences, not semantic object recognition.
        # Every downstream effect uses the same motor/phase cells as support.
        for eye in range(2):
            for sector in range(3):
                red, green, blue = retina.reshape(2, 3, 3)[eye, sector]
                edge(red, aversive[sector], vision_gain)
                edge(green, appetitive, vision_gain * (.8 if sector == 1 else .5))
                if sector != 1:
                    edge(green, orient[0 if sector == 0 else 1], vision_gain)
                edge(blue, curiosity, .10)
        for side in range(2):
            low, middle, high = cochlea.reshape(2, 3)[side]
            edge(low, appetitive, hearing_gain)
            edge(low, orient[side], hearing_gain)
            edge(middle, curiosity, hearing_gain * .2)
            edge(high, aversive[0 if side == 0 else 2], hearing_gain)
            edge(high, startle, hearing_gain * .3)
            edge(auditory_spatial.reshape(2,3)[side,0], auditory_approach[side], 2.)
            edge(auditory_spatial.reshape(2,3)[side,2], auditory_avoid[side], 2.)
            for ear in range(2):
                edge(cochlea.reshape(2,3)[ear,0], auditory_approach[side], .7)
                edge(cochlea.reshape(2,3)[ear,2], auditory_avoid[side], .7)
            edge(auditory_approach[side], orient[side], .8)
            edge(auditory_avoid[side], avoid[side], .8)
            # What the outer ear adds is front or back. A sound behind the
            # animal is the one it cannot see, so that is the one worth turning
            # toward: the behind cell of an ear recruits orienting on that side.
            # The ahead cell is deliberately left without an output - a sound in
            # front is already in view, and the low band above reaches orienting
            # on its own.
            edge(auditory_pinna.reshape(2, 2)[side, 1], orient[side], pinna_orient_gain)
        for side in range(2):
            edge(orient[side], orient_i[side], 1.)
            edge(orient_i[side], orient[1-side], .7)
            edge(orient_i[side], avoid[side], .25)
            edge(aversive[0 if side == 0 else 2], avoid[side], 1.3)
            edge(avoid[side], steering[side], 1.)
            edge(orient[1-side], steering[side], 1.)
            edge(steering[side], steering_i[side], 1.)
            edge(fatigue_i[0], steering[side], 1.2)
            edge(rest_i[0], steering[side], .8)
        edge(aversive[1], avoid[0], 1.)
        edge(aversive[1], avoid[1], .65)
        edge(aversive[1], brake, .6)
        edge(touch[0], aversive[1], .8)
        # Being touched is what a keeper does while teaching, so the touch
        # cells reach the dopamine cell innately. touch_dopamine_gain is zero
        # in the shipped animal and above zero in the nursery.
        for sector in range(4):
            edge(touch[sector], dopamine[0], touch_dopamine_gain)
        # The coarse hint: these two populations are worth connecting at all.
        # Nothing is decided in advance about which cell should talk to which.
        for source_name, target_name in backup_routes:
            if source_name not in self.groups or target_name not in self.groups:
                raise ValueError(f'unknown backup route {source_name} -> {target_name}')
            for source_cell in np.atleast_1d(self.groups[source_name]):
                for target_cell in np.atleast_1d(self.groups[target_name]):
                    backup_edge(source_cell, target_cell)
        edge(appetitive[0], curiosity, .5)
        edge(appetitive[0], initiation, .7)
        for sector in range(3):
            edge(stereo_near[sector], near[sector], .5)
        edge(autonomous[0], curiosity, .85)
        for target in initiation:
            edge(curiosity[0], target, 2.)
            edge(fatigue_i[0], target, 2.6)
            edge(rest_i[0], target, 1.5)
            edge(brake[0], target, 2.5)
        edge(initiation[0], initiation[1], 1.15)
        edge(initiation[1], initiation[0], 1.15)
        edge(initiation[0], self.groups['locomotion'], .75)
        edge(effort[0], fatigue, 1.7)
        edge(fatigue[0], fatigue_i, 1.)
        edge(fatigue_i[0], curiosity, 1.4)
        edge(fatigue[0], rest, 1.8)
        edge(rest[0], rest_i, 1.)
        edge(rest_i[0], curiosity, 1.1)
        edge(rest_i[0], self.groups['locomotion'], 1.1)
        for k, source in enumerate(retina):
            edge(source, novelty[k], 1.)
            edge(novelty[k], curiosity, .14)
            if (k//3) % 3 == 0:
                edge(novelty[k], avoid[1], .07)
            elif (k//3) % 3 == 2:
                edge(novelty[k], avoid[0], .07)
        edge(near[0], avoid[0], 2.)
        edge(near[2], avoid[1], 2.)
        # A slight anatomical asymmetry breaks exact frontal symmetry; it is
        # an initial connection strength, not a random action-selection script.
        edge(near[1], avoid[0], 1.8)
        edge(near[1], avoid[1], 1.2)
        edge(touch[1], avoid[0], 1.)
        edge(touch[3], avoid[1], 1.)
        # Two opposed avoidance channels: one leans the body one way, the other
        # leans it the other way. A drive that reaches both of them at once (the
        # eye's "something is coming close", whose two edges are deliberately
        # unequal - 1.8 and 1.2) is not a choice, it is both hips pushed at the
        # same time, and held for seconds that put the gait on the ground
        # (measured, up_z -0.97 driving at the east wall). An animal picks one.
        # The pair inhibit each other here, on their own weight, so the stronger
        # drive holds its channel and shuts the other down: at 0.5 a drive that
        # lights both leaves 1.0 and 0.7 (both turning), at 1.5 the second loses
        # (measured, artifacts/avoid_both_routes.log). It stays a parameter so
        # the choice can be turned off by setting it to 0.
        for i in range(2):
            edge(avoid[i], avoid_i[i], 1.)
            edge(avoid_i[i], avoid[1-i], avoidance_mutual)
            # Orienting can recruit the shared limbs while frontal advance
            # is inhibited. No external turn command or heading target exists.
            edge(avoid[i], orient_vigor, 2.)
            edge(rest_i[0], orient_vigor[i], 1.3)
        edge(orient_vigor[0], orient_vigor[1], 1.05)
        edge(orient_vigor[1], orient_vigor[0], 1.05)
        edge(orient_vigor[0], orient_vigor_i, 1.)
        edge(orient_vigor_i[0], self.groups['locomotion'], 1.2)
        edge(orient_vigor_i[0], initiation, 1.5)
        edge(orient_vigor[0], rhythm_recruit, .9)
        edge(self.groups['locomotion'][0], rhythm_recruit, .9)
        edge(retreat[0], rhythm_recruit, .9)
        edge(rhythm_recruit[0], rhythm_recruit[1], 1.1)
        edge(rhythm_recruit[1], rhythm_recruit[0], 1.1)
        edge(fatigue_i[0], rhythm_recruit, 1.2)
        edge(rest_i[0], rhythm_recruit, .7)
        for phase in self.groups['phase']:
            edge(rhythm_recruit[0], phase, .65)
        edge(near[1], brake, 1.5)
        edge(near[0], brake, .20)
        edge(near[2], brake, .20)
        edge(touch[0], brake, 2.)
        edge(tilt[0], brake, 1.)
        edge(brake[0], self.groups['locomotion'], 1.4)
        edge(abrupt[0], startle, 2.)
        edge(startle[0], startle_i, 1.)
        edge(startle_i[0], self.groups['locomotion'], .7)
        edge(tilt[0], protective, 1.)
        edge(protective[0], protective_i, 1.)
        edge(touch[0], retreat, 1.5)
        # Retreat is driven by contact; near objects alone slow/orient the
        # existing locomotor circuit. These edges share the original oscillator.
        for k in range(2):
            edge(retreat[0], retreat_phase[k], 1.25)
            edge(self.groups['phase_delay'][k], retreat_phase[k], 2.)
            edge(retreat_phase[k], retreat_phase_i[k], 1.)
        for i in range(4):
            edge(foot[i], withdrawal[i], 1.)
            edge(slip[i], withdrawal[i], .15)
            edge(withdrawal[i], withdrawal_i[i], 1.)
            edge(load[i], impact[i], 1.)
            edge(impact[i], impact_i[i], 1.)
            # A contact with a low obstacle outlasts the touch itself, so the
            # leg still knows about it when its next swing starts. Obstacle
            # strength, scraping and the withdrawal it already caused all feed
            # the same state.
            edge(foot[i], stumble[i], 1.6)
            edge(slip[i], stumble[i], .2)
            edge(withdrawal[i], stumble[i], .7)
            edge(stumble[i], stumble_i[i], 1.)
            # Lifting further is gated by the swing half of this leg's rhythm,
            # so a remembered contact raises the paw while the stance legs are
            # unaffected. The extra rotation reuses the same shared motor units.
            # The swing term only gates: on its own, even at the deepest step,
            # it stays under the threshold, so no contact means no extra lift
            # and the measured state is exactly zero. The contact term is two
            # to three times the swing term, so the lift that appears is the
            # remembered contact, not the gait.
            edge(stumble[i], clearance[i], clearance_contact_gain)
            edge(self.groups['phase'][0 if i in (0, 3) else 1], clearance[i], clearance_swing_gain)
            edge(clearance[i], clearance_i[i], 1.)
            edge(righting[0], righting_push[i], 1.4)
            edge(self.groups['phase'][0 if i in (0, 3) else 1], righting_push[i], 1.4)
            edge(righting_push[i], righting_push_i[i], 1.)
        # The body is on the ground, not the feet. The loss of balance and the
        # touch sense of the body sector that landed both feed the same state.
        edge(tilt[0], righting[0], 1.1)
        for sector in touch:
            edge(sector, righting[0], .5)
        edge(righting[0], righting_i[0], 1.)
        postures = np.r_[righting_tuck, righting_roll, righting_stand]
        postures_i = np.r_[righting_tuck_i, righting_roll_i, righting_stand_i]
        drives = (righting_tuck_drive,) + (righting_posture_drive,)*(len(postures) - 1)
        for index, posture in enumerate(postures):
            edge(righting_memory[0], posture, drives[index])
            edge(posture, postures_i[index], 1.)
            for other in range(len(postures)):
                if other != index:
                    edge(postures_i[index], postures[other], righting_competition)
        edge(tilt[0], righting_memory[0], 1.)
        edge(tilt[0], righting_fallen[0], righting_fallen_drive)
        edge(righting_fallen[0], righting_fallen_hold[0], righting_fallen_hold_gain)
        edge(righting_fallen_hold[0], righting_fallen[0], righting_fallen_hold_gain)
        # The steady excitation the whole graph carries is what this cell has
        # when nothing is wrong; the loss of the up axis pulls it down, so what
        # is left over is "the body is on its feet".
        edge(self.groups['tonic'][0], righting_upright[0], righting_upright_drive)
        edge(righting_memory_i[0], righting_upright[0], righting_upright_drive)
        edge(righting_upright[0], righting_upright_i[0], 1.)
        edge(righting_upright_i[0], righting_fallen[0], righting_release_gain)
        edge(righting_upright_i[0], righting_fallen_hold[0], righting_release_gain)
        # Which side the ground is on is a sense, not a choice, and it is what
        # breaks the tie between the postures: the side that landed gets its
        # own roll, the back gets the tuck, the front gets the push. This is
        # the same asymmetry the two near cells give the avoidance channels.
        edge(lean[3], righting_roll[0], righting_side_gain)
        edge(touch[3], righting_roll[0], righting_side_gain)
        edge(lean[2], righting_roll[1], righting_side_gain)
        edge(touch[1], righting_roll[1], righting_side_gain)
        edge(touch[2], righting_tuck[0], righting_side_gain)
        edge(touch[0], righting_stand[0], righting_side_gain)
        for sector in range(4):
            edge(lean[sector], righting_lean_memory[sector], righting_state_gain)
            edge(touch[sector], righting_touch_memory[sector], righting_state_gain)
        # These are the only edges in the graph that the reward below writes.
        # The reward gain below is large because the coincidence it looks at
        # lasts a fraction of a second: one successful righting writes about
        # 0.4 of a 1.2 ceiling, so a fall and a recovery is enough to change
        # what the animal does next time. The same gain multiplies the pull
        # back towards the innate weight, which is therefore zero: a posture
        # is corrected by the next recovery it takes part in, not by a clock.
        # a posture that was firing when the body came back up keeps more of
        # the fall state it came from, a posture that was not keeps less. With
        # the reward switched off they are ordinary edges like every other.
        for state in np.r_[righting_lean_memory, righting_touch_memory]:
            for posture in postures:
                edge(state, posture, righting_memory_weight, learns=True, limit=righting_cap)
        edge(tilt[0], righting_memory_i[0], 1.)
        edge(righting_fallen[0], righting_success[0], 1.)
        edge(righting_memory_i[0], righting_success[0], righting_success_inhibition)
        edge(righting_success[0], appetitive, righting_success_reward)
        for sector in range(4):
            edge(touch[sector], wall[sector], wall_gain)
        # Front contact is the walking case: stop the gait, take a step back
        # and turn aside. Either hand touches the front, and the innate
        # asymmetry of the near cells above decides which way it turns. A side
        # contact turns away from that side, which is the midbrain rule the
        # touch cells already follow; these are its slow copies.
        # These cells also take the eyes' reading of a wall straight ahead
        # (born_wired/embodied.py), so a wall the eyes can already see close is
        # the same thing to the body as a wall the skin is against. That route
        # is on this cell and not on the general near route above: the near
        # route shares its gain with every near object the eyes report and it
        # puts down both avoidance channels at once, which a standing animal
        # holds only for the moment of a touch. Measured, a sustained eye
        # reading there held both avoidance channels down together for seconds
        # and the gait fell over (up_z 0.99 -> -0.97 driving at the east wall);
        # on this cell the same reading turns the animal aside and the gait
        # stays up (artifacts/wall_eye_gain_sweep.log).
        edge(wall[0], brake, wall_brake_gain)
        edge(wall[0], retreat[0], wall_retreat_gain)
        edge(wall[0], avoid[0], wall_turn_gain)
        edge(wall[0], avoid[1], wall_turn_gain*.8)
        edge(wall[1], avoid[0], wall_turn_gain*.8)
        edge(wall[3], avoid[1], wall_turn_gain*.8)
        # Turning aside from a wall is its own action rather than the avoidance
        # tilt: the avoidance route shares its gain with every near object the
        # eyes report, and measured, raising that gain enough to move the body
        # away from a wall also made the gait fall over in the arena. This pair
        # reaches the hips directly, on its own gain, and only while the
        # contact lasts, which is exactly as long as the animal is against
        # something. Contact in front turns it to one hand - a standing animal
        # has to pick one, and the near cells above already carry that
        # asymmetry. Contact on a side turns away from that side.
        edge(wall[0], wall_turn[0], 1.)
        edge(wall[2], wall_turn[0], .5)
        edge(wall[1], wall_turn[0], 1.)
        edge(wall[3], wall_turn[1], 1.)
        for side in range(2):
            edge(wall_turn[side], wall_turn_i[side], 1.)
        # Each leg answers the two leans that share it, so the leg on the side
        # the body is going over reaches out, and the leg behind or in front of
        # it joins in. Extending is the mirror of the withdrawal above, on the
        # same motor units: the one that lifts a paw is the one that plants it.
        for leg, (side, fore) in enumerate(((2, 0), (3, 0), (2, 1), (3, 1))):
            edge(lean[fore], steady[leg], 1.)
            edge(lean[side], steady[leg], 1.)
            edge(steady[leg], steady_i[leg], 1.)
            # Once the body is genuinely going over, the protective and
            # righting circuits take it; this one is for the sway before that,
            # and it stands down as the tilt grows.
            edge(steady_off[0], steady_i[leg], steady_inhibition)
        edge(tilt[0], steady_off[0], 1.)

        def motor_effect(excitatory, inhibitory, joint, radians):
            units = self.groups['motor'].reshape(12, motor_units)[joint]
            edge(excitatory if radians >= 0 else inhibitory, units,
                 abs(radians) * self.motor_gain / self.span[joint])

        for j in range(12):
            leg, joint = j//3, j%3
            front = 1 if leg < 2 else -1
            if joint == 0:
                motor_effect(steering[0], steering_i[0], j, front * avoidance_gain)
                motor_effect(steering[1], steering_i[1], j, -front * avoidance_gain)
                # Widening the base is what actually stops a sideways tip, and
                # it is the hip that does it: measured on this body, turning a
                # left hip outward moves the body right, and a right hip
                # outward moves it left. So the leg on the side the body is
                # going over splays out, and the leg on the other side is left
                # alone - splaying that one would push the body further over.
                motor_effect(steady[leg], steady_i[leg], j, (1 if leg % 2 == 0 else -1)*steady_hip_gain)
                motor_effect(righting_push[leg], righting_push_i[leg], j, righting_hip_gain)
                motor_effect(wall_turn[0], wall_turn_i[0], j, front * wall_hip_gain)
                motor_effect(wall_turn[1], wall_turn_i[1], j, -front * wall_hip_gain)
            if joint in (1, 2):
                flex = 1 if joint == 1 else -2
                motor_effect(withdrawal[leg], withdrawal_i[leg], j, withdrawal_gain * flex)
                motor_effect(protective[0], protective_i[0], j, .35 * flex)
                motor_effect(startle[0], startle_i[0], j, startle_gain * flex)
                motor_effect(clearance[leg], clearance_i[leg], j, clearance_lift * flex)
                motor_effect(steady[leg], steady_i[leg], j, -steady_gain * flex)
                motor_effect(righting_push[leg], righting_push_i[leg], j, righting_gain * flex)
            edge(impact_i[leg], self.groups['recruitment'][j], .16)
            if joint == 1:
                diagonal = 0 if leg in (0, 3) else 1
                motor_effect(retreat_phase[diagonal], retreat_phase_i[diagonal], j, .65)
                motor_effect(retreat_phase[1-diagonal], retreat_phase_i[1-diagonal], j, -.65)

        # Each posture reaches the same shared units as the older reflexes.
        # Tucking pulls a leg in; rolling tucks the legs on one side and
        # extends the ones on the other, and swings every hip the same way so
        # the trunk is pushed over; pushing out straightens all four legs.
        posture_scale = righting_gain / RIGHTING_GAIN_NOMINAL
        for j in range(12):
            leg, joint = j//3, j%3
            left = 1 if leg % 2 == 0 else -1
            if joint == 0:
                motor_effect(righting_tuck[0], righting_tuck_i[0], j, left*righting_tuck_hip*posture_scale)
                motor_effect(righting_roll[0], righting_roll_i[0], j, righting_roll_hip*posture_scale)
                motor_effect(righting_roll[1], righting_roll_i[1], j, -righting_roll_hip*posture_scale)
                motor_effect(righting_stand[0], righting_stand_i[0], j, left*righting_stand_hip*posture_scale)
            else:
                flex = 1 if joint == 1 else -2
                motor_effect(righting_tuck[0], righting_tuck_i[0], j, righting_tuck_gain*flex*posture_scale)
                motor_effect(righting_roll[0], righting_roll_i[0], j, righting_roll_gain*flex*left*posture_scale)
                motor_effect(righting_roll[1], righting_roll_i[1], j, -righting_roll_gain*flex*left*posture_scale)
                motor_effect(righting_stand[0], righting_stand_i[0], j, -righting_stand_gain*flex*posture_scale)

        # The shared motor units now carry one more innate action. Overturning
        # needs a bigger rotation than any older reflex, so the incoming
        # resource of exactly those units is raised by what this action adds.
        # Every existing edge keeps its weight: the raise only removes a
        # ceiling that the new action could otherwise not fit under.
        unit_ids = self.groups['motor'].reshape(12, motor_units)
        extra = np.zeros(old.n_neurons)
        for index in range(12):
            joint = index % 3
            scale = self.motor_gain / self.span[index]
            added = righting_hip_gain if joint == 0 else righting_gain * (2 if joint == 2 else 1)
            # Every new posture can be on at the same time as the burst while
            # one hands over to the other, so the room asked for is the sum of
            # what all of them would ask for. No existing weight changes.
            if joint == 0:
                added += (righting_tuck_hip + righting_roll_hip + righting_stand_hip)*posture_scale
            else:
                added += ((righting_tuck_gain + righting_roll_gain + righting_stand_gain
                           + righting_roll_gain) * (2 if joint == 2 else 1))*posture_scale
            extra[unit_ids[index]] = added * scale
        weight = np.asarray(weight)
        learned = np.asarray(learned)
        cap = np.asarray(cap)
        backup = np.asarray(backup)
        self._righting_learn = righting_learn
        self._righting_learn_mask = postures.copy()
        self._dopamine_gain = dopamine_gain
        self._backup_routes = backup_routes
        if righting_learn > 0:
            new_lower = np.where(learned, 0., weight*.97)
            new_upper = np.where(learned, cap, weight*1.03)
            new_plasticity = np.where(learned, righting_learn_rate, .02)
            new_tether = np.where(learned, righting_memory_tether, .2)
        else:
            new_lower, new_upper = weight*.97, weight*1.03
            new_plasticity, new_tether = np.full(len(weight), .02), np.full(len(weight), .2)
        # A backup edge is allowed to grow: room below its starting weight, a
        # ceiling above it, plasticity of its own, and no pull back towards the
        # start, so a route that is used stays where it was written. Every
        # innate edge keeps the bounds it already had.
        if backup.any():
            new_lower = np.where(backup, 0., new_lower)
            new_upper = np.where(backup, backup_cap, new_upper)
            new_plasticity = np.where(backup, backup_plasticity, new_plasticity)
            new_tether = np.where(backup, 0., new_tether)
        n_new = len(signs)
        # Replace the two old unbounded-sum rhythm inputs with one shared
        # saturating recruitment neuron. Competing motivations cannot add
        # several full-strength gaits on top of each other.
        keep = ~((old.src == self.groups['locomotion'][0]) & np.isin(old.dst, self.groups['phase']))
        self._backup_mask = np.r_[np.zeros(int(keep.sum()), dtype=bool), backup]
        self.synapses = RegulatedSynapses(
            np.r_[old.src[keep], src], np.r_[old.dst[keep], dst], np.r_[old.weights[keep], weight],
            np.r_[old.signs, signs], old.n_neurons+n_new,
            lower=np.r_[old.lower[keep], new_lower], upper=np.r_[old.w_max[keep], new_upper],
            budgets=np.r_[old.budgets + extra, np.full(n_new, 40.)],
            plasticity=np.r_[old.plasticity[keep], new_plasticity],
            tether=np.r_[old.tether[keep], new_tether],
            learning_rate=old.learning_rate, target_activity=np.r_[old.target_activity, np.full(n_new, .15)])
        self.synapses.anchor = np.r_[old.anchor[keep], weight]
        self.synapses.anchor.flags.writeable = False
        self.network = ReflexInputNetwork(self.synapses, self.groups['feature_receptors'],
            tau=np.r_[net._tau, taus], adaptation_tau=np.r_[net._adaptation_tau, adaptation_tau],
            adaptation_gain=np.r_[net._gain, adaptation], bias=np.r_[net._bias, biases],
            initial_voltage=np.r_[net.voltage, biases], initial_adaptation=np.r_[net.adaptation, np.zeros(n_new)])
        self.initial_weights = self.synapses.weights
        self._initial_voltage = self.network.voltage

    def righting_modulation(self, rates=None):
        """Learning gate for the remembered righting, one entry per cell.

        It is one everywhere except on the four posture cells, where it is the
        activity of the just-got-up cell: the connections that the fall state
        and the posture were carrying at that moment are strengthened, the
        rest are left to fall back to what they started at. With the reward
        switched off the gate is exactly one and learning is what it was.
        """
        if self._righting_learn <= 0:
            return 1.
        rates = self.network.activity if rates is None else rates
        success = float(rates[self.reflex_groups['righting_success'][0]])
        modulation = np.ones(self.network.n_neurons)
        modulation[self._righting_learn_mask] = 1. + (success - 1.)*self._righting_learn
        return modulation

    def learning_modulation(self):
        """How hard the local rule writes this moment, one entry per cell.

        The dopamine cell is the only thing that says "this moment counts", and
        it says nothing about what was right. With dopamine_gain at zero this is
        exactly righting_modulation and the animal is the one it was. Above
        zero the rule runs at its own rate while dopamine is firing and is
        slowed by that fraction the rest of the time, so a route whose two ends
        are only ever together during a lesson is the route that gets written.
        """
        gain = self._dopamine_gain
        if gain <= 0:
            return self.righting_modulation()
        rates = self.network.activity
        gate = self.righting_modulation(rates)
        released = float(rates[self.groups['dopamine'][0]])
        return gate*(1. - gain*(1. - released))

    def nursery_state(self):
        """The teachable routes as they now stand: one weight per backup edge."""
        mask = self._backup_mask
        weights = self.synapses.weights
        if mask.size < weights.size:
            # The picture layer appends its own edges after this controller is
            # built; a backup edge is never one of them.
            mask = np.r_[mask, np.zeros(weights.size - mask.size, dtype=bool)]
        rates = self.network.activity
        return dict(routes=len(self._backup_routes), edges=int(mask.sum()),
                    weights=weights[mask].tolist(),
                    dopamine=float(rates[self.groups['dopamine'][0]]))

    def step(self, observation, *, environment, startle=0., reflexes=True, autonomy=True,
             auditory_activity=None, auditory_spatial_activity=None,
             auditory_pinna_activity=None, dopamine_drive=0., **parameters):
        if not isinstance(reflexes, (bool, np.bool_)):
            raise ValueError('reflexes must be boolean')
        if not isinstance(autonomy, (bool, np.bool_)):
            raise ValueError('autonomy must be boolean')
        startle = _vector([startle], 1, 'startle', low=0, high=1)
        dopamine = _vector([dopamine_drive], 1, 'dopamine_drive', low=0, high=1)
        sensor_values = {name: _vector(environment[name], 4, name, low=0, high=1)
                         for name in ('body_touch', 'foot_obstacle', 'foot_load', 'foot_slip')}
        effort = _vector([environment.get('motor_effort', 0.)], 1, 'motor_effort', low=0, high=1)
        auditory = _vector(np.zeros(6) if auditory_activity is None else np.asarray(auditory_activity).ravel(),
                           6, 'auditory_activity', low=0, high=1)
        spatial = _vector(np.zeros(6) if auditory_spatial_activity is None else np.asarray(auditory_spatial_activity).ravel(),
                          6, 'auditory_spatial_activity', low=0, high=1)
        pinna = _vector(np.zeros(4) if auditory_pinna_activity is None else np.asarray(auditory_pinna_activity).ravel(),
                        4, 'auditory_pinna_activity', low=0, high=1)
        sensed = dict(observation)
        if 'foot_support' in environment:
            support = np.asarray(environment['foot_support'])
            if support.shape != (4,) or not np.isin(support, (0, 1)).all():
                raise ValueError('invalid foot_support')
            sensed['foot_contact'] = support.astype(bool)
        external = np.zeros(self.network.n_neurons)
        if reflexes and parameters.get('feedback', True):
            g = self.groups
            for name, values in sensor_values.items():
                external[g[name]] = values
            gravity = _vector(sensed['gravity_direction'], 3, 'gravity_direction', low=-1, high=1)
            external[g['protective_tilt']] = np.clip((.70+gravity[2])/.70, 0, 1)
            # Local gravity is the world vertical in body axes, so its forward
            # and leftward parts say which way the body is leaning. Four cells,
            # four directions, no bearing is computed anywhere.
            external[g['lean']] = np.clip([gravity[0], -gravity[0], gravity[1], -gravity[1]], 0, 1)
            external[g['abrupt_sound']] = startle
            external[g['dopamine']] = dopamine
            external[g['effort_receptor']] = effort
            external[g['autonomous_receptor']] = float(autonomy)
            external[g['cochlea']] = auditory
            external[g['auditory_spatial']] = spatial
            external[g['auditory_pinna']] = pinna
        self.network.reflex_input = external
        try:
            return super().step(sensed, modulator=self.learning_modulation(), **parameters)
        finally:
            self.network.reflex_input = np.zeros(self.network.n_neurons)

    def diagnostics(self):
        result = super().diagnostics()
        rates = self.network.activity
        result['reflex_activity'] = {name: rates[self.groups[name]].tolist() for name in
            ('near', 'brake', 'avoidance', 'withdrawal', 'retreat', 'startle', 'protective_flexion',
             'curiosity', 'fatigue', 'rest', 'initiation', 'appetitive', 'orienting', 'aversive', 'orienting_vigor',
             'rhythm_recruitment', 'steering', 'stumble', 'clearance', 'righting', 'righting_push',
             'righting_tuck', 'righting_roll', 'righting_stand',
             'righting_memory', 'righting_fallen', 'righting_fallen_hold', 'righting_upright',
             'righting_success',
             'righting_lean_memory', 'righting_touch_memory',
             'lean', 'steady', 'wall_contact', 'wall_turn')}
        return result
