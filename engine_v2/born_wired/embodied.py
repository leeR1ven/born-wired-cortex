"""Raw binocular pixels enter neurons in the motor graph, without a decoder.

All receptive fields below are developmental wiring. Runtime perception is
the same E/I voltage update as the rest of the graph; no distance, direction,
image matching, object identity, or action selection is computed in step().

Four eye muscles join the same graph when the body supplies eye limits. Both
retinas feed one set of gaze cells, so where contrast sits in a picture turns
both eyes, and the same two pictures are compared side by side to drive a slow
convergence command. The angle the muscle units settle at therefore *is* the
fixation point, and a bank of cells with different thresholds reports which
distance that is, because each cell needs a different angle to turn on. No
pixel is ever converted to metres.
"""
import numpy as np
from .auditory_neurons import AuditoryNeurons, OUTPUT_KEYS
from .reflex_controller import ReflexController, ReflexInputNetwork
from .regulation import RegulatedSynapses
from .synapses import _checked_in_place, _scalar, _vector
from .encoding import TuningEncoder

# Keys this controller reads out of AuditoryNeurons.step. The pair is declared
# and checked where the modules load, so a rebuild that pairs one side of the
# interface with an older build of the other fails here, naming the key, rather
# than stopping a running window later on.
AUDITORY_KEYS_USED = ('rates', 'spatial_rates', 'pinna_rates')
if not set(AUDITORY_KEYS_USED) <= set(OUTPUT_KEYS):
    raise RuntimeError('auditory circuit does not provide '
                       + str(sorted(set(AUDITORY_KEYS_USED) - set(OUTPUT_KEYS))))

# The raw eye is EYE_HEIGHT x EYE_WIDTH per eye and everything below is derived
# from those two numbers: the contrast bank reads EYE_COLUMNS horizontal columns,
# EYE_ROWS rows spread down the picture, and the disparity bank asks about
# offsets 0..STEREO_OFFSET_MAX. Both axes are graded by distance from these
# middles, so the drive of a centred object is exactly zero. Raising the two
# numbers raises the acuity of the whole visual front end; nothing else in this
# file carries a picture size of its own.
EYE_WIDTH, EYE_HEIGHT = 48, 36
EYE_ROWS = 8
STEREO_OFFSET_MAX = 6
EYE_COLUMNS = EYE_WIDTH - 1
EYE_BINOCULAR_COLUMNS = EYE_COLUMNS - (STEREO_OFFSET_MAX + 1)
# Rows of the drawn picture the contrast bank reads. At 32x24 these are the
# original y = 3,5,...,17.
EYE_CONTRAST_ROWS = np.rint(np.linspace(3, EYE_HEIGHT - 7, EYE_ROWS)).astype(int)
COLUMN_MIDDLE, ROW_MIDDLE = (EYE_WIDTH - 1)/2., (EYE_HEIGHT - 1)/2.
# Receptor density falls off sideways from the middle of a retina, exactly as
# it does around a fovea. The gaze channels and the two comparison channels
# below are weighted by it, so the eye answers what is near where it is already
# looking instead of averaging a whole scene. It is applied across the picture
# and not down it: weighting the vertical axis the same way was tried and
# measured, and it drives the up/down pair to its own limit, because the
# picture of a room carries almost all of its contrast below the horizon (see
# docs/眼睛肌肉与自动聚焦_实现_20260923.md). That is a wiring weight, not a
# computation: no pixel is ever compared with anything.
FOVEA_WIDTH = 5.
# The weight one retinal cell carries into the three summary cells the rest of
# the graph reads, on the picture size that graph was tuned at. A bigger sheet
# has more cells per sector, so the weight per cell is divided by the count.
RETINA_SUMMARY_GAIN = .025
REFERENCE_SUMMARY_PIXELS = 36.*48./3.
# How often the local rule is allowed to rewrite a connection, in seconds of
# simulated time. One controller step is .01 s of that time, so .05 means a
# connection is reconsidered every fifth step. This is a cadence, not a
# different rule: every write still receives the whole time since the last
# write, so the same rule integrated over the same second produces the same
# kind of change whether it is applied once or five times.
LEARNING_INTERVAL = .05


def _fovea(col, middle):
    return 1./(1. + ((col-middle)/FOVEA_WIDTH)**2)
# Cell levels at which the distance bank's thresholds sit. They are read off the
# loop itself (see docs/眼睛肌肉与自动聚焦_实现_20260923.md): parallel eyes sit at
# zero, the angle between the two eyes grows as the target comes closer, and the
# bank is wired against a gain of EYE_DISTANCE_GAIN and these levels are read
# off the loop after the vergence time constant was lengthened to one second (see
# docs/?????????_??_20260923.md): a ball is seen from about 2.5 m in,
# the drive runs from 0 while the eyes are parallel up to about 1.2 rad at 0.2 m,
# and the levels below cut that run into bands of roughly even width in camera
# range. Spread over the measured working range, not over metres. The lowest
# level is below zero, so one cell stays lit while the eyes are parallel. That
# cell and the cells above it that have come on together are the pattern that
# names the band.
EYE_DISTANCE_GAIN = 2.8
DISTANCE_THRESHOLDS = (-0.10, 0.03, 0.16, 0.26, 0.37, 0.43, 0.60)


class RetinalInputNetwork(ReflexInputNetwork):
    def __init__(self, *args, pixel_ids, eye_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixel_ids = np.asarray(pixel_ids)
        self.pixel_current = np.zeros(len(self.pixel_ids))
        self.eye_ids = None if eye_ids is None else np.asarray(eye_ids)
        self.eye_current = np.zeros(0 if eye_ids is None else len(self.eye_ids))

    def step(self, external_exc, *args, **kwargs):
        external = np.array(external_exc, dtype=float, copy=True)
        return self.step_owned(external, *args, **kwargs)

    def step_owned(self, external, *args, check=True, **kwargs):
        """As step, for an excitation vector the caller already owns.

        The copy in step is what separates a caller's array from what the
        layers below add to, and it stays there. The controller builds the
        vector itself once per step and then discards it, so it does not need
        that separation: it enters here and the same two additions happen
        where its vector already lies, instead of onto a second copy of it.
        """
        if check:
            external = _checked_in_place(external, self.n_neurons, 'external_exc', low=0)
        with np.errstate(over='raise', invalid='raise'):
            external[self.pixel_ids] += self.pixel_current
            if self.eye_ids is not None:
                external[self.eye_ids] += self.eye_current
        return super().step_owned(external, *args, check=False, **kwargs)


class EmbodiedController(ReflexController):
    eye_shape = (2, EYE_HEIGHT, EYE_WIDTH, 3)

    def __init__(self, *args, eye_limits=None, eye_width=EYE_WIDTH, eye_height=EYE_HEIGHT,
                 eye_motor_units=24, eye_proprio_units=16, learning_interval=LEARNING_INTERVAL,
                 eye_distance_cells=None, eye_gain=12., eye_gaze_gain=1.,
                 eye_track_gain=2., eye_pitch_gain=4., eye_pitch_time=.03,
                 eye_gaze_inhibition=.3,
                 eye_change_gain=48., eye_change_threshold=.5, eye_change_relay_gain=.5,
                 eye_change_relay_steps=10, eye_change_trace_time=.01,
                 eye_change_common=3.,
                 eye_sound_gain=1.5, eye_memory_gain=.6,
                 eye_orient_gain=.5,
                 eye_vergence_gain=1., eye_vergence_steps=1.5, eye_vergence_inhibition=1.,
                 eye_vergence_time=1., eye_vergence_baseline=.6, eye_dominant='left',
                 eye_distance_slope=EYE_DISTANCE_GAIN, eye_distance_thresholds=None,
                 eye_binocular_baseline=-1.1, eye_relay_gain=8., eye_relay_base=.5,
                 eye_near_gain=1., eye_fusion_gain=.05, eye_stereo_gain=1., eye_wall_gain=2.,
                 eye_flow_threshold=1.1, eye_flow_time=.2, eye_grow_threshold=.5,
                 eye_loom_threshold=.5, eye_loom_gain=2.,
                 eye_band_gain=4., eye_band_threshold=.1, eye_band_common=2.,
                 eye_band_push=.25, eye_row_relay_gain=None,
                 eye_row_gain=None, eye_row_threshold=None,
                 eye_row_trace_time=None, eye_row_common=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # The picture size belongs to this animal, not to the module: everything
        # below is derived from these two numbers, and the raw eyes are built to
        # match (tools/live_dog.py reads them back off the controller). Raising
        # them raises acuity everywhere and costs cells and time in proportion
        # to the pixel count.
        eye_width, eye_height = int(eye_width), int(eye_height)
        if eye_width < 32 or eye_height < 24:
            raise ValueError('the eye must be at least 32 wide and 24 high')
        self.eye_width, self.eye_height = eye_width, eye_height
        self.eye_columns = eye_width - 1
        self.eye_binocular_columns = self.eye_columns - (STEREO_OFFSET_MAX + 1)
        if self.eye_binocular_columns < 1:
            raise ValueError('the eye is too narrow for the offset bank')
        self.eye_contrast_rows = np.rint(np.linspace(3, eye_height - 7, EYE_ROWS)).astype(int)
        self.column_middle, self.row_middle = (eye_width - 1)/2., (eye_height - 1)/2.
        self.eye_shape = (2, eye_height, eye_width, 3)
        self.learning_interval = _scalar(learning_interval, 'learning_interval')
        # One retinal cell is one vote, and a finer sheet holds more votes for
        # the same scene. Without this the same picture would drive the three
        # summary cells the rest of the graph reads harder at every increase of
        # the picture size, and the incoming budget would rescale the weights
        # instead of letting them mean what the graph was tuned against.
        self.retina_summary_gain = (RETINA_SUMMARY_GAIN*REFERENCE_SUMMARY_PIXELS
                                    / (eye_height*(eye_width/3.)))
        if eye_motor_units < 4 or eye_proprio_units < 4:
            raise ValueError('eye populations need at least four units each')
        eye_gain = float(eye_gain)
        eye_gaze_gain = float(eye_gaze_gain)
        eye_track_gain = float(eye_track_gain)
        eye_pitch_gain = float(eye_pitch_gain)
        eye_pitch_time = float(eye_pitch_time)
        eye_gaze_inhibition = float(eye_gaze_inhibition)
        eye_change_gain = float(eye_change_gain)
        eye_change_threshold = float(eye_change_threshold)
        eye_change_relay_gain = float(eye_change_relay_gain)
        eye_change_trace_time = float(eye_change_trace_time)
        eye_change_common = float(eye_change_common)
        eye_change_steps = int(eye_change_relay_steps)
        if eye_change_steps != eye_change_relay_steps or eye_change_steps < 1:
            raise ValueError('eye_change_relay_steps must be a whole number of steps, at least one')
        eye_sound_gain = float(eye_sound_gain)
        eye_memory_gain = float(eye_memory_gain)
        eye_orient_gain = float(eye_orient_gain)
        eye_vergence_gain = float(eye_vergence_gain)
        eye_vergence_steps = float(eye_vergence_steps)
        eye_vergence_inhibition = float(eye_vergence_inhibition)
        eye_vergence_time = float(eye_vergence_time)
        eye_vergence_baseline = float(eye_vergence_baseline)
        if eye_dominant not in ('left', 'right'):
            raise ValueError("eye_dominant must be 'left' or 'right'")
        eye_distance_slope = float(eye_distance_slope)
        eye_binocular_baseline = float(eye_binocular_baseline)
        eye_relay_gain = float(eye_relay_gain)
        eye_relay_base = float(eye_relay_base)
        eye_near_gain = float(eye_near_gain)
        eye_fusion_gain = float(eye_fusion_gain)
        eye_stereo_gain = float(eye_stereo_gain)
        eye_wall_gain = float(eye_wall_gain)
        eye_flow_threshold = float(eye_flow_threshold)
        eye_flow_time = float(eye_flow_time)
        eye_grow_threshold = float(eye_grow_threshold)
        eye_loom_threshold = float(eye_loom_threshold)
        eye_loom_gain = float(eye_loom_gain)
        eye_band_gain = float(eye_band_gain)
        eye_band_threshold = float(eye_band_threshold)
        eye_band_common = float(eye_band_common)
        eye_band_push = float(eye_band_push)
        eye_row_relay_gain = (eye_change_relay_gain if eye_row_relay_gain is None
                              else float(eye_row_relay_gain))
        eye_row_gain = eye_change_gain if eye_row_gain is None else float(eye_row_gain)
        eye_row_threshold = (eye_change_threshold if eye_row_threshold is None
                             else float(eye_row_threshold))
        eye_row_trace_time = (eye_change_trace_time if eye_row_trace_time is None
                              else float(eye_row_trace_time))
        eye_row_common = (eye_change_common if eye_row_common is None
                          else float(eye_row_common))
        for name, value in (('eye_binocular_baseline', eye_binocular_baseline),
                            ('eye_relay_base', eye_relay_base)):
            if not np.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if not np.isfinite(eye_relay_gain) or eye_relay_gain < 0:
            raise ValueError('eye_relay_gain must be a nonnegative finite number')
        for name, value in (('eye_gain', eye_gain), ('eye_gaze_gain', eye_gaze_gain),
                            ('eye_track_gain', eye_track_gain), ('eye_pitch_gain', eye_pitch_gain),
                            ('eye_pitch_time', eye_pitch_time),
                            ('eye_vergence_gain', eye_vergence_gain), ('eye_vergence_steps', eye_vergence_steps),
                            ('eye_vergence_time', eye_vergence_time), ('eye_vergence_baseline', eye_vergence_baseline),
                            ('eye_distance_slope', eye_distance_slope),
                            ('eye_change_threshold', eye_change_threshold),
                            ('eye_row_relay_gain', eye_row_relay_gain),
                            ('eye_row_gain', eye_row_gain),
                            ('eye_row_threshold', eye_row_threshold),
                            ('eye_row_trace_time', eye_row_trace_time),
                            ('eye_change_relay_gain', eye_change_relay_gain),
                            ('eye_change_trace_time', eye_change_trace_time),
                            ('eye_change_common', eye_change_common),
                            ('eye_band_gain', eye_band_gain), ('eye_band_threshold', eye_band_threshold),
                            ('eye_band_common', eye_band_common)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number')
        if not np.isfinite(eye_gaze_inhibition) or eye_gaze_inhibition < 0:
            raise ValueError('eye_gaze_inhibition must be a nonnegative finite number')
        # These four may be set to zero: silencing one is how the control runs
        # that show the others carry a behaviour on their own.
        for name, value in (('eye_band_push', eye_band_push),
                            ('eye_row_common', eye_row_common),
                            ('eye_change_gain', eye_change_gain), ('eye_sound_gain', eye_sound_gain),
                            ('eye_memory_gain', eye_memory_gain), ('eye_orient_gain', eye_orient_gain),
                            ('eye_near_gain', eye_near_gain), ('eye_fusion_gain', eye_fusion_gain),
                            ('eye_stereo_gain', eye_stereo_gain)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a nonnegative finite number')
        if not np.isfinite(eye_vergence_inhibition) or eye_vergence_inhibition < 0:
            raise ValueError('eye_vergence_inhibition must be a nonnegative finite number')
        thresholds = np.asarray(DISTANCE_THRESHOLDS if eye_distance_thresholds is None
                                else eye_distance_thresholds, dtype=float)
        if thresholds.ndim != 1 or thresholds.size < 2 or not np.all(np.isfinite(thresholds)):
            raise ValueError('eye_distance_thresholds must be an increasing vector')
        if np.any(np.diff(thresholds) <= 0):
            raise ValueError('eye_distance_thresholds must be increasing')
        if eye_distance_cells is None:
            eye_distance_cells = thresholds.size
        if eye_distance_cells != thresholds.size:
            raise ValueError('one threshold per distance cell')
        old, net = self.synapses, self.network
        signs, bias, tau, beta, initial, budgets = [], [], [], [], [], []

        def cells(name, count, sign=1, base=0., time=.015, adaptation=0., start=None, budget=40.):
            ids = np.arange(old.n_neurons+len(signs), old.n_neurons+len(signs)+count)
            self.groups[name] = ids
            signs.extend(np.broadcast_to(sign, (count,)).tolist())
            bias.extend(np.broadcast_to(base, (count,)).tolist())
            tau.extend(np.broadcast_to(time, (count,)).tolist())
            beta.extend(np.broadcast_to(adaptation, (count,)).tolist())
            initial.extend(np.broadcast_to(base if start is None else start, (count,)).tolist())
            budgets.extend(np.broadcast_to(budget, (count,)).tolist())
            return ids

        pixels = cells('photoreceptors', 2*self.eye_height*self.eye_width*3).reshape(self.eye_shape)
        pixel_i = cells('retinal_interneurons', pixels.size, -1, time=.01).reshape(self.eye_shape)
        opponent = cells('retinal_opponent', pixels.size, base=-.08).reshape(self.eye_shape)
        contrast = cells('retinal_contrast', 2*EYE_ROWS*self.eye_columns*2,
                         base=-.08).reshape(2, EYE_ROWS, self.eye_columns, 2)
        # The motion-onset route, laid out the way the published table lays it
        # out: one cell per column that lags its column by a moment (the trace),
        # one cell per column that answers only when its column is busier this
        # tick than last tick (the change cell, held down by the trace through an
        # inhibitory cell), and a chain of relay cells that stretches that single
        # tick into a pull long enough to move a damped eye. Each cell answers
        # what its own input is doing; no column is asked for a speed, an object
        # or a distance.
        trace = cells('retinal_trace', 2*self.eye_columns, time=eye_change_trace_time,
                      budget=1. + eye_change_gain).reshape(2, self.eye_columns)
        trace_i = cells('retinal_trace_inhibition', 2*self.eye_columns, -1, time=.01).reshape(2, self.eye_columns)
        change = cells('retinal_change', 2*self.eye_columns, base=-eye_change_threshold, time=.02,
                       budget=2. + eye_change_gain).reshape(2, self.eye_columns)
        relay = cells('retinal_change_relay', 2*self.eye_columns*eye_change_steps,
                      time=.02, budget=6.).reshape(2, self.eye_columns, eye_change_steps)
        change_common = cells('retinal_change_inhibition', 1, -1, time=.02,
                              budget=1. + 2.*self.eye_columns)[0]
        # The same three cells again, reading the picture the other way round.
        # A column is a place across the picture and says nothing about up and
        # down, so the vertical half of the motion reflex needs cells of its
        # own: for each row of the bank, one cell that carries that row a
        # moment ago (the trace), one that answers only when the row is busier
        # now than it was (held down by the trace through an inhibitory cell),
        # and a chain of relays long enough to move a damped eye. They are one
        # more voice on eye_look_up / eye_look_down and add nothing to the
        # left/right route above, which keeps its own trace, change and relay
        # cells untouched.
        trace_row = cells('retinal_trace_row', 2*EYE_ROWS, time=eye_row_trace_time,
                          budget=1. + eye_row_gain).reshape(2, EYE_ROWS)
        trace_row_i = cells('retinal_trace_row_inhibition', 2*EYE_ROWS, -1,
                            time=.01).reshape(2, EYE_ROWS)
        change_row = cells('retinal_change_row', 2*EYE_ROWS, base=-eye_row_threshold,
                           time=.02, budget=2. + eye_row_gain).reshape(2, EYE_ROWS)
        relay_row = cells('retinal_change_row_relay', 2*EYE_ROWS*eye_change_steps,
                          time=.02, budget=6.).reshape(2, EYE_ROWS, eye_change_steps)
        change_row_common = cells('retinal_change_row_inhibition', 1, -1, time=.02,
                                  budget=1. + 2.*EYE_ROWS)[0]
        # A saturated copy of both pictures, one cell for every contrast cell,
        # and the only input the disparity bank below reads. It is here because
        # the two pictures are not on the same scale as the bank's threshold: a
        # rendered edge moves a contrast cell by about 0.2 and a saturated
        # synthetic edge moves it by 1.0, so a bank that asks a *pair* of cells
        # for more than either can deliver alone either never opens on a real
        # room or opens on one cell alone. The copy reaches its ceiling for any
        # edge worth the name, so the bank asks the same question in both
        # worlds: is there an edge in this retina *and* in the other one, in
        # these two places. Measured, with the copy silent the disparity
        # population averaged 0.005 in a clean room and 0.001 in the arena,
        # which is why nothing downstream of it ever moved
        # (docs/眼睛肌肉与自动聚焦_实现_20260923.md).
        contrast_copy = cells('contrast_relay', contrast.size, base=-eye_relay_base,
                              time=.02).reshape(contrast.shape)
        # Getting bigger and getting closer are the same thing seen from two
        # moments, so the cells below are built out of two moments. Per column
        # there is a cell that answers when the column is busier than it was a
        # moment ago (the rise cell) and its mirror, a cell that answers when
        # the column is quieter than it was (the fall cell). "A moment ago" is
        # a slow trace of the column — slower than the one-tick trace the eye
        # reflex above uses, because at the speeds an animal meets things,
        # nothing moves a whole column in one tick. An edge that has left one
        # column and arrived in the one beside it lights a rise cell and a fall
        # cell at once, and that coincidence is what the outward and inward
        # cells are: an edge that arrived from the left is moving right, one
        # that arrived from the right is moving left. No column is asked for a
        # size, a distance or an object.
        #
        # An edge moving one way is what a pair of eyes does when it turns:
        # every edge in the picture goes the same way. An edge moving outwards
        # on both sides of the middle is what a picture does when the thing in
        # it is getting bigger. So the two halves are read against each other
        # and one direction on its own speaks for nothing.
        retinal_column = cells('retinal_column', 2*self.eye_columns, time=.02,
                               budget=2. + eye_change_gain).reshape(2, self.eye_columns)
        retinal_column_i = cells('retinal_column_inhibition', 2*self.eye_columns, -1,
                                 time=.01).reshape(2, self.eye_columns)
        loom_trace = cells('retinal_column_trace', 2*self.eye_columns, time=eye_flow_time,
                           budget=2. + eye_change_gain).reshape(2, self.eye_columns)
        loom_trace_i = cells('retinal_column_trace_inhibition', 2*self.eye_columns, -1,
                             time=.01).reshape(2, self.eye_columns)
        rise = cells('retinal_rise', 2*self.eye_columns, base=-eye_change_threshold, time=.02,
                     budget=2. + eye_change_gain).reshape(2, self.eye_columns)
        fall = cells('retinal_fall', 2*self.eye_columns, base=-eye_change_threshold, time=.02,
                     budget=2. + eye_change_gain).reshape(2, self.eye_columns)
        outward = cells('retinal_outward', 2*self.eye_columns, base=-eye_flow_threshold,
                        time=.02).reshape(2, self.eye_columns)
        inward = cells('retinal_inward', 2*self.eye_columns, base=-eye_flow_threshold,
                       time=.02).reshape(2, self.eye_columns)
        swell_left = cells('eye_swell_left', 2, base=-eye_grow_threshold, time=.03)
        swell_right = cells('eye_swell_right', 2, base=-eye_grow_threshold, time=.03)
        growing = cells('eye_growing', 2, base=-eye_grow_threshold, time=.03)
        cells('eye_looming', 1, base=-eye_loom_threshold, time=.03)

        # Each fixed pair joins one retinal location to one offset location in
        # the other eye. Cells are information, not a decoded depth map.
        binocular = cells('binocular', EYE_ROWS*self.eye_binocular_columns*2*(STEREO_OFFSET_MAX+1),
                          base=eye_binocular_baseline,
                          time=.025).reshape(EYE_ROWS, self.eye_binocular_columns, 2,
                                             STEREO_OFFSET_MAX+1)
        bino_i = cells('binocular_inhibition', binocular.size, -1).reshape(binocular.shape)
        memory = cells('retinal_memory', 4*4*3, base=-.15, time=.04).reshape(4,4,3)
        sources, targets, weights = [], [], []
        plastic, tether, lower, upper = [], [], [], []

        def edge(source, target, value, memory_edge=False):
            if value <= 0:
                return
            sources.append(int(source)); targets.append(int(target)); weights.append(float(value))
            plastic.append(3. if memory_edge else .005)
            tether.append(0. if memory_edge else .2)
            lower.append(0. if memory_edge else value*.97)
            upper.append(.1 if memory_edge else value*1.03)

        for eye, row, col, channel in np.ndindex(self.eye_shape):
            edge(pixels[eye,row,col,channel], pixel_i[eye,row,col,channel], 1.)
            target = opponent[eye,row,col,channel]
            edge(pixels[eye,row,col,channel], target, 2.)
            for other in range(3):
                if other != channel:
                    edge(pixel_i[eye,row,col,other], target, 1.)
            sector = min(2, col*3//self.eye_width)
            # Retinotopic convergence is synaptic summation, not image pooling.
            edge(target, self.groups['retina'].reshape(2,3,3)[eye,sector,channel],
                 self.retina_summary_gain)

        for eye, row, col, polarity in np.ndindex(contrast.shape):
            y = int(self.eye_contrast_rows[row])
            x_on, x_off = (col, col+1) if polarity == 0 else (col+1, col)
            for channel in range(3):
                edge(pixels[eye,y,x_on,channel], contrast[eye,row,col,polarity], 4.)
                edge(pixel_i[eye,y,x_off,channel], contrast[eye,row,col,polarity], 4.)

        for eye, row, col, polarity in np.ndindex(contrast.shape):
            edge(contrast[eye, row, col, polarity], contrast_copy[eye, row, col, polarity],
                 eye_relay_gain)

        # "It is getting bigger": the second way of saying "it is getting
        # closer", and the one that still works when the thing is far enough
        # away that the two eyes' pictures are the same picture. Nothing here
        # is a measurement of size — a cell only ever reports that its own
        # column is busier now than it was, or quieter, and the cells that
        # follow report two of those side by side.
        for eye, col in np.ndindex(2, self.eye_columns):
            for row, polarity in np.ndindex(EYE_ROWS, 2):
                cell = contrast[eye, row, col, polarity]
                edge(cell, retinal_column[eye, col], eye_change_gain/16.)
                edge(cell, loom_trace[eye, col], eye_change_gain/16.)
            edge(retinal_column[eye, col], retinal_column_i[eye, col], 1.)
            edge(loom_trace[eye, col], loom_trace_i[eye, col], 1.)
            edge(retinal_column[eye, col], rise[eye, col], 1.)
            edge(loom_trace_i[eye, col], rise[eye, col], 1.)
            edge(loom_trace[eye, col], fall[eye, col], 1.)
            edge(retinal_column_i[eye, col], fall[eye, col], 1.)
        for eye, col in np.ndindex(2, self.eye_columns):
            for step, target in ((1, outward), (-1, inward)):
                other = col + step
                if 0 <= other < self.eye_columns:
                    edge(rise[eye, col], target[eye, col], 1.)
                    edge(fall[eye, other], target[eye, col], 1.)
        half_width = self.column_middle + 1.
        for eye, col in np.ndindex(2, int(round(self.eye_columns*31./47.))):
            target = swell_left[eye] if col < self.column_middle else swell_right[eye]
            cell = inward[eye, col] if col < self.column_middle else outward[eye, col]
            edge(cell, target, 1./half_width)
        for eye in range(2):
            edge(swell_left[eye], growing[eye], 1.)
            edge(swell_right[eye], growing[eye], 1.)
            edge(growing[eye], self.groups['eye_looming'][0], 1.)
        # The same brake the near cells reach, so a picture opening outwards
        # slows the animal the way a wall a stride ahead slows it.
        edge(self.groups['eye_looming'][0], self.groups['brake'][0], eye_loom_gain)

        # How many offset pairs each protective sector draws on. The route
        # into those cells is a fraction of the pairs that agree at a large
        # offset, not their count: a count grows with how wide the sector is
        # and how big the picture is, and the sum then saturates whatever the
        # scene does, so the near cells could say only "near" and never "how
        # near". Summed over a fixed population it would mean the same thing at
        # every image size (docs/眼睛肌肉与自动聚焦_实现_20260923.md).
        sector_pairs = {}
        for row, col, polarity, offset in np.ndindex(binocular.shape):
            if offset >= 2:
                sector = min(2, (col + STEREO_OFFSET_MAX + 1)*3//self.eye_width)
                sector_pairs[sector] = sector_pairs.get(sector, 0) + 1
        for row, col, polarity, offset in np.ndindex(binocular.shape):
            left_col = col + STEREO_OFFSET_MAX + 1
            right_col = left_col-(offset+1)
            target = binocular[row,col,polarity,offset]
            edge(contrast_copy[0,row,left_col,polarity], target, 1.)
            edge(contrast_copy[1,row,right_col,polarity], target, 1.)
            edge(target, bino_i[row,col,polarity,offset], 1.)
            for other in range(7):
                if other != offset:
                    edge(bino_i[row,col,polarity,offset], binocular[row,col,polarity,other], .35)
            if offset >= 2:
                # Larger retinal offsets innately excite protective cells.
                # No pixel-to-metre conversion exists in the running brain.
                sector = min(2, left_col*3//self.eye_width)
                edge(target, self.groups['stereo_near'][sector], eye_stereo_gain/sector_pairs[sector])

        # Local patches keep hidden visual signals separate before broad
        # convergence. Postsynaptic valence/pressure coactivity can strengthen
        # these initially negligible routes through the existing local rule.
        for row, col, channel in np.ndindex(memory.shape):
            cell = memory[row,col,channel]
            for eye in range(2):
                for dy, dx in np.ndindex(2,2):
                    edge(opponent[eye, 3+row*5+dy, 3+col*8+dx, channel], cell, .35)
            edge(cell, self.groups['appetitive'][0], .0001, True)
            edge(cell, self.groups['aversive'][min(2,col*3//4)], .0001, True)

        self.eye_encoder = None
        self.eye_motor_units = 0
        self.eye_gain = eye_gain
        self.eye_lower = self.eye_upper = self.eye_span = np.zeros(0)
        eye_ids = None
        if eye_limits is not None:
            eye_ids = self.eye_muscles(cells, edge, eye_limits, eye_motor_units, eye_proprio_units,
                                       eye_gain, eye_gaze_gain, eye_track_gain, eye_pitch_gain,
                                       eye_pitch_time, eye_gaze_inhibition, eye_vergence_gain, eye_vergence_steps,
                                       eye_vergence_inhibition, eye_vergence_time, eye_vergence_baseline,
                                       eye_dominant, eye_distance_slope, thresholds,
                                       contrast, contrast_copy, change, trace, trace_i, relay, change_common,
                                       trace_row, trace_row_i, change_row, relay_row, change_row_common,
                                       binocular, memory,
                                       eye_change_gain, eye_change_relay_gain, eye_change_common,
                                       eye_sound_gain, eye_memory_gain, eye_orient_gain,
                                       eye_near_gain, eye_fusion_gain, eye_wall_gain,
                                       eye_loom_gain, eye_band_gain, eye_band_threshold,
                                       eye_band_common, eye_band_push, eye_row_relay_gain,
                                       eye_row_gain, eye_row_common)

        w = np.asarray(weights)
        n = len(signs)
        self.synapses = RegulatedSynapses(
            np.r_[old.src,sources], np.r_[old.dst,targets], np.r_[old.weights,w],
            np.r_[old.signs,signs], old.n_neurons+n,
            lower=np.r_[old.lower,lower], upper=np.r_[old.w_max,upper],
            budgets=np.r_[old.budgets,budgets],
            plasticity=np.r_[old.plasticity,plastic], tether=np.r_[old.tether,tether],
            learning_rate=old.learning_rate,
            target_activity=np.r_[old.target_activity,np.full(n,.15)])
        anchor = np.r_[old.anchor, w]
        anchor.flags.writeable = False
        self.synapses.anchor = anchor
        self.network = RetinalInputNetwork(self.synapses, self.groups['feature_receptors'],
            pixel_ids=pixels.ravel(), eye_ids=eye_ids,
            tau=np.r_[net._tau,tau], adaptation_tau=np.r_[net._adaptation_tau,np.full(n,.4)],
            adaptation_gain=np.r_[net._gain,beta], bias=np.r_[net._bias,bias],
            initial_voltage=np.r_[net.voltage,initial], initial_adaptation=np.r_[net.adaptation,np.zeros(n)],
            learning_interval=self.learning_interval)
        self.initial_weights = self.synapses.weights
        self._initial_voltage = self.network.voltage
        self.auditory = AuditoryNeurons()

    def eye_muscles(self, cells, edge, eye_limits, motor_units, proprio_units, gain, gaze_gain,
                    track_gain, pitch_gain, pitch_time, gaze_inhibition, vergence_gain, vergence_steps,
                    vergence_inhibition, vergence_time, vergence_baseline, dominant_side,
                    distance_slope, thresholds, contrast, contrast_copy, change, trace, trace_i, relay,
                    change_common, trace_row, trace_row_i, change_row, relay_row,
                    change_row_common, binocular, memory, change_gain, relay_gain, common_gain,
                    sound_gain,
                    memory_gain, orient_gain, near_gain, fusion_gain, wall_gain, loom_gain,
                    band_gain, band_threshold, band_common, band_push, row_relay_gain,
                    row_gain, row_common):
        """Wire the four eye muscles, their gaze drives and the distance bank.

        Every entry is an ordinary graph cell or synapse. Where a muscle settles
        is the fixation angle, and the bank below turns that angle into a
        pattern of active cells, one more of them for each distance band the
        target has come inside.
        """
        lower, upper = (_vector(np.asarray(limits, dtype=float), 4, 'eye_limits') for limits in eye_limits)
        span = upper - lower
        if np.any(span <= 0):
            raise ValueError('eye limits must be increasing')
        operating = (0.-lower)/span
        if np.any(operating < 0) or np.any(operating > 1):
            raise ValueError('the eye rest angle must lie inside its limits')
        self.eye_lower, self.eye_upper, self.eye_span = lower, upper, span
        self.eye_motor_units = int(motor_units)
        self.eye_encoder = TuningEncoder(4, int(proprio_units))
        # One eye leads and the other follows. The lead is a wiring convenience,
        # not a one-way street: both retinas feed the same cells below, so either
        # picture moves either eye, and the difference between the two pictures
        # is what turns the following eye toward the midline.
        dominant = 0 if dominant_side == 'left' else 1
        subordinate = 1 - dominant
        follower, leader = 2*subordinate, 2*dominant
        inward = 1. if subordinate else -1.
        centres = (np.arange(self.eye_motor_units) + .5)/self.eye_motor_units
        muscle_bias = np.tile(.5 - gain*centres, 4)
        resting = gain*operating
        settled = resting + 2.*gaze_gain*gain/span
        settled += vergence_steps*gain/span
        # The band reflex at the end of this method is one more voice on each
        # eye's own muscle, one command cell each way, so the resource a muscle
        # unit can hold has to cover that pair too.
        settled += 2.*band_push*gain/span
        muscle = cells('eye_motor', 4*self.eye_motor_units, base=muscle_bias,
                       start=muscle_bias + np.repeat(resting, self.eye_motor_units),
                       budget=1. + float(settled.max()))
        cells('eye_proprioception', 4*self.eye_encoder.size//4)
        # The four gaze cells now carry several parallel voices, so the
        # incoming resource each of them can hold has to cover all of them.
        look_budget = (1. + max(track_gain, gaze_inhibition) + relay_gain
                       + sound_gain + memory_gain + orient_gain)
        pitch_budget = 1. + max(pitch_gain, gaze_inhibition) + row_relay_gain
        look_left = cells('eye_look_left', 1, time=.03, budget=look_budget)[0]
        look_right = cells('eye_look_right', 1, time=.03, budget=look_budget)[0]
        look_right_i = cells('eye_look_right_inhibition', 1, -1, time=.01)[0]
        # The up/down pair carries its own time constant, and it is not the same
        # story as the left/right pair. A picture that fills both retinas - the
        # striped wall does - puts a large and unequal amount of contrast above
        # and below the middle row, so this pair is driven hard; the eye then
        # turns, the picture shifts the other way, and with the drive able to
        # follow in a thirtieth of a second the loop rings: measured, the
        # commanded pitch swings the whole 0.9 rad of its travel and spends 60%
        # of the time against a limit, and the picture changes every frame
        # because the eye is swinging (artifacts/eye_jitter_routes.log). A slower
        # pair cannot follow its own consequence, which is what an eye that
        # holds a direction does, and the left/right pair - whose target is an
        # object in the middle of a picture and not a whole wall - is left as it
        # was, because it does not ring.
        look_up = cells('eye_look_up', 1, time=pitch_time, budget=pitch_budget)[0]
        look_up_i = cells('eye_look_up_inhibition', 1, -1, time=.01)[0]
        look_down = cells('eye_look_down', 1, time=pitch_time, budget=pitch_budget)[0]
        # One cell watches how much contrast the two pictures carry together, so
        # the gaze cells answer the position of that contrast and not its size:
        # a near object fills more of a retina than a far one.
        gaze_common = cells('eye_gaze_inhibition', 1, -1, time=.02,
                            budget=1. + float(contrast.size))[0]
        # Two opposed channels answer whether the two pictures are still apart.
        # A feature left of the middle of the following eye, or right of the
        # middle of the leading eye, is the same statement: this eye has not
        # turned far enough toward the midline yet. The opposite pair says the
        # eye has turned too far, and pushes the command back.
        verge_in = cells('eye_vergence_in', 1, time=.03, budget=1.+vergence_gain)[0]
        verge_out = cells('eye_vergence_out', 1, time=.03, budget=1.+vergence_gain)[0]
        verge_out_i = cells('eye_vergence_out_inhibition', 1, -1, time=.01)[0]
        # A slow opposed pair outlasts a blink, so the command is not restarted
        # from zero every time the object is briefly hidden.
        vergence = cells('eye_vergence', 2, time=vergence_time,
                         budget=1.+max(1., vergence_inhibition)+vergence_baseline)
        vergence_i = cells('eye_vergence_inhibition', 1, -1, time=.01)[0]
        # Where each eye currently points, read out of its own muscle
        # proprioception. The two are subtracted: the right eye's yaw grows as
        # it comes in, the left eye's shrinks, so the difference is the angle
        # between the two eyes and nothing else. That matters, because the raw
        # drive to the muscles scales with whatever contrast the scene happens
        # to carry, while the angle does not.
        tilt = np.arange(proprio_units)/(proprio_units-1.)
        inward_position = cells('eye_convergence_in', 1, time=.02,
                                budget=1.+float(tilt.sum()))[0]
        outward_position = cells('eye_convergence_out', 1, -1, time=.02,
                                 budget=1.+float(tilt.sum()))[0]
        alignment = cells('binocular_pool', 7)
        cells('eye_fusion', 1)
        distance = cells('eye_distance', thresholds.size, base=-thresholds,
                         budget=2.*distance_slope+1.)

        # The rest of the body already supplies a standing current to each joint;
        # the eye muscles are no different.
        for joint in range(4):
            for unit in muscle.reshape(4, self.eye_motor_units)[joint]:
                edge(self.groups['tonic'][0], unit, resting[joint])

        # How far the contrast sits from the middle of a retina, counted from
        # both retinas at once. Left of centre pulls one way, right of centre
        # pulls the other, and a centred object pulls neither way, so the only
        # fixed point of this loop is the object sitting in the middle.
        gaze = {'left': [], 'right': [], 'up': [], 'down': []}
        for eye, row, col, polarity in np.ndindex(contrast.shape):
            cell = contrast[eye, row, col, polarity]
            y = int(self.eye_contrast_rows[row])
            gaze['left' if col < self.column_middle else 'right'].append(
                (cell, _fovea(col, self.column_middle)*abs(self.column_middle-col)/self.column_middle))
            gaze['up' if y < self.row_middle else 'down'].append(
                (cell, abs(self.row_middle-y)/self.row_middle))
        for name, target, total in (('left', look_left, track_gain), ('right', look_right, track_gain),
                                    ('up', look_up, pitch_gain), ('down', look_down, pitch_gain)):
            share = total/sum(weight for _, weight in gaze[name])
            for cell, weight in gaze[name]:
                edge(cell, target, share*weight)
        for eye, row, col, polarity in np.ndindex(contrast.shape):
            edge(contrast[eye, row, col, polarity], gaze_common, 1./float(contrast.size))
        for target in (look_left, look_right, look_up, look_down):
            edge(gaze_common, target, gaze_inhibition)

        # Four more things decide where the eyes go, each on its own edges into
        # the same four cells, so any one of them can turn the eyes by itself.
        # The parallelism is the point: none of these is a gate the others have
        # to pass through, and silencing one leaves the rest working.
        #
        # 1. What has just changed. This is the reflex the published table
        #    reports: one cell per column carries that column a moment ago (the
        #    trace), one cell per column answers only when its column is busier
        #    now than it was then (the change cell, held down by the trace
        #    through an inhibitory cell), and a chain of relay cells stretches
        #    that single tick into a pull long enough to move a damped eye. It is
        #    wired per column of the uncompressed picture, as the table is, and
        #    it is one more voice on the same gaze cells rather than a stage the
        #    others have to pass through: the static path above is what holds the
        #    eye on a thing, this one is "something moved, look at it". Its
        #    vertical half is wired the same way further down, one cell per
        #    row of the bank instead of one per column, and reaches
        #    eye_look_up / eye_look_down.
        #
        #    Two things had to be right before it earned its place, and both are
        #    measured (docs/眼睛肌肉与自动聚焦_实现_20260923.md, "眼睛该看哪儿").
        #    The trace has to lag by ONE tick, as the table's does: with a slower
        #    trace the cell answers any slow drift, including the drift of the
        #    picture caused by the eye's own turning, and the eyes then run away
        #    from a target that is standing still (0.008 -> 0.12 rad). With a
        #    one-tick trace, and a ball crossing the clean room at 0.24 m/s, the
        #    mean angle between the gaze and the ball falls from 0.0765 to 0.0383
        #    rad, and at 0.48 m/s from 0.0720 to 0.0457; a still ball is left
        #    where it is (0.0080 -> 0.0097).
        #    The columns also have to compete. Left alone, the change route
        #    answers "the picture changed", which the eye's own movement makes
        #    true everywhere, and in the furnished arena that drives the gaze
        #    off by 0.18 rad. The one inhibitory cell below sums the whole bank
        #    and holds every column down by a multiple of that sum, so a column
        #    has to out-change the rest of the picture to speak: the strongest
        #    change decides where the eyes go, and a change that is everywhere
        #    at once decides nothing. With that, the arena is left as it was
        #    (0.0949 -> 0.0969) while the clean room still improves.
        weight_by_column = np.array([_fovea(c, self.column_middle)*abs(self.column_middle-c)/self.column_middle
                                     for c in range(self.eye_columns)])
        for eye, col in np.ndindex(2, self.eye_columns):
            for row, polarity in np.ndindex(EYE_ROWS, 2):
                cell = contrast[eye, row, col, polarity]
                edge(cell, trace[eye, col], change_gain/16.)
                edge(cell, change[eye, col], change_gain/16.)
            edge(trace[eye, col], trace_i[eye, col], 1.)
            edge(trace_i[eye, col], change[eye, col], 1.)
            edge(change[eye, col], relay[eye, col, 0], 2.)
            edge(change_common, relay[eye, col, 0], 2.)
            edge(change[eye, col], change_common, common_gain/float(2*self.eye_columns))
            for step in range(relay.shape[2]-1):
                edge(relay[eye, col, step], relay[eye, col, step+1], 2.)
        moving = {'left': [], 'right': []}
        for eye, col, step in np.ndindex(2, self.eye_columns, relay.shape[2]):
            moving['left' if col < self.column_middle else 'right'].append(
                (relay[eye, col, step], weight_by_column[col]))
        for name, target in (('left', look_left), ('right', look_right)):
            share = relay_gain/sum(weight for _, weight in moving[name])
            for cell, weight in moving[name]:
                edge(cell, target, share*weight)

        # The same reflex counted down the picture instead of across it. One
        # row of the contrast bank is one band of the picture's height, and a
        # row that has just become busier is a thing that moved into it. Its
        # weight is how far the row sits from the middle of the picture, the
        # same weight the static up/down drive uses, so a row at the middle
        # asks for nothing and a row at the top or bottom asks for the most.
        # No fovea weight down the picture: weighting the vertical axis that
        # way was measured to drive this pair into its own limit (see
        # docs/眼睛肌肉与自动聚焦_实现_20260923.md).
        for eye, row in np.ndindex(2, EYE_ROWS):
            for col, polarity in np.ndindex(self.eye_columns, 2):
                cell = contrast[eye, row, col, polarity]
                edge(cell, trace_row[eye, row], row_gain/float(2*self.eye_columns))
                edge(cell, change_row[eye, row], row_gain/float(2*self.eye_columns))
            edge(trace_row[eye, row], trace_row_i[eye, row], 1.)
            edge(trace_row_i[eye, row], change_row[eye, row], 1.)
            edge(change_row[eye, row], relay_row[eye, row, 0], 2.)
            edge(change_row_common, relay_row[eye, row, 0], 2.)
            edge(change_row[eye, row], change_row_common, row_common/float(2*EYE_ROWS))
            for step in range(relay_row.shape[2]-1):
                edge(relay_row[eye, row, step], relay_row[eye, row, step+1], 2.)
        moving_row = {'up': [], 'down': []}
        for eye, row, step in np.ndindex(2, EYE_ROWS, relay_row.shape[2]):
            y = int(self.eye_contrast_rows[row])
            moving_row['up' if y < self.row_middle else 'down'].append(
                (relay_row[eye, row, step], abs(self.row_middle - y)/self.row_middle))
        for name, target in (('up', look_up), ('down', look_down)):
            total = sum(weight for _, weight in moving_row[name])
            share = row_relay_gain/total if total else 0.
            for cell, weight in moving_row[name]:
                edge(cell, target, share*weight)

        # 2. Where a sound is coming from. The ears already hold cells that
        #    prefer one side; they reach the same gaze cells the pictures do,
        #    which is what turns the eyes toward a noise behind the animal.
        spatial = self.groups['auditory_spatial'].reshape(2, 3)
        for side, target in ((0, look_left), (1, look_right)):
            for band in range(3):
                edge(spatial[side, band], target, sound_gain/3.)

        # 3. What local visual memory has come to stand for. These edges are
        #    plastic, so a patch of picture that keeps being paired with looking
        #    a certain way grows its own weight toward those cells. It is the
        #    same rule that already lets a pattern come to mean approach or
        #    avoidance, pointed at the eye muscles instead.
        for row, col, channel in np.ndindex(memory.shape):
            target = look_left if col < memory.shape[1]/2. else look_right
            edge(memory[row, col, channel], target, memory_gain/float(memory.size), True)

        # 4. The orienting cells the rest of the graph already uses, on their
        #    own edges: one more parallel voice, not a stage the others report
        #    to.
        orienting = self.groups['orienting']
        edge(orienting[0], look_left, orient_gain)
        edge(orienting[1], look_right, orient_gain)

        # The two pictures compared: each retina sends the side of its own
        # picture that means "the other eye still has to come in". Both eyes
        # feed both channels, which is what makes the pair mutual. The weights
        # add up to one, so the channel reads the offset as a fraction of a
        # half-width however much contrast the scene happens to carry.
        spread = sum(_fovea(c, self.column_middle)*abs(self.column_middle-c)/self.column_middle for c in range(self.eye_columns))
        share = vergence_gain/float(EYE_ROWS*2*spread)
        for eye, row, col, polarity in np.ndindex(contrast.shape):
            cell = contrast[eye, row, col, polarity]
            weight = share*_fovea(col, self.column_middle)*abs(self.column_middle-col)/self.column_middle
            near = col < self.column_middle
            if (eye == subordinate) == near:
                edge(cell, verge_in, weight)
            else:
                edge(cell, verge_out, weight)
        for channel in vergence:
            edge(verge_in, channel, 1.)
            edge(verge_out_i, channel, vergence_inhibition)
        edge(verge_out, verge_out_i, 1.)
        edge(vergence[0], vergence[1], vergence_baseline)
        edge(vergence[1], vergence[0], vergence_baseline)
        edge(vergence[0], vergence_i, 1.)
        edge(vergence[1], vergence_i, 1.)
        edge(look_right, look_right_i, 1.)
        edge(look_up, look_up_i, 1.)

        # How much of each offset population is currently matching. This is a
        # synaptic sum over channels that already exist, not a computed depth.
        for offset in range(STEREO_OFFSET_MAX + 1):
            share = 1./float(EYE_ROWS*self.eye_binocular_columns*2)
            for row, col, polarity in np.ndindex(EYE_ROWS, self.eye_binocular_columns, 2):
                edge(binocular[row, col, polarity, offset], alignment[offset], share)
        edge(alignment[0], self.groups['eye_fusion'][0], 1.)

        def push(joint, source, radians):
            """A cell pushes one eye joint; an inhibitory source pushes it back."""
            for unit in muscle.reshape(4, self.eye_motor_units)[joint]:
                edge(source, unit, abs(radians)*gain/span[joint])

        # Gaze (both eyes the same way) and convergence (toward each other) share
        # the same two muscles of each eye, exactly as the legs share theirs.
        for joint in (0, 2):
            push(joint, look_left, gaze_gain)
            push(joint, look_right_i, gaze_gain)
        for joint in (1, 3):
            push(joint, look_down, gaze_gain)
            push(joint, look_up_i, gaze_gain)
        # Convergence turns the two eyes toward the midline together, so the
        # gaze channel above only has to hold the average direction of the pair
        # and the two loops cannot fight each other. Which of the two cells
        # reaches which eye is anatomical: the midline lies on opposite sides of
        # the two retinas.
        push(follower, vergence[0] if inward > 0 else vergence_i, vergence_steps)
        push(leader, vergence_i if inward > 0 else vergence[0], vergence_steps)

        # The angle between the two eyes, as a fraction of a half-turn of the
        # joint. One cell reads the eye that turns in, one reads the eye that
        # turns out, and their difference is the convergence.
        proprioception = self.groups['eye_proprioception'].reshape(4, proprio_units)
        for joint, target in ((2, inward_position), (0, outward_position)):
            for unit, weight in zip(proprioception[joint], tilt):
                if weight > 0:
                    edge(unit, target, weight)

        # The distance bank. Both cells of the opposed pair reach every cell of
        # the bank, the one that reads the eye turning in as an excitatory
        # current and the one that reads the eye turning out as an inhibitory
        # one, so each cell of the bank is driven by the angle between the two
        # eyes and not by wherever the pair happens to point. They differ only
        # in how much of that angle they need, so the cells that have come on
        # are the cells for every band the eyes have come in past: the lowest
        # one stays on while the eyes are parallel, and each further cell adds
        # itself as the target comes closer. What a distance is read off is
        # therefore which cells are on, not how hard any one of them fires.
        for index in range(thresholds.size):
            edge(inward_position, distance[index], distance_slope)
            edge(outward_position, distance[index], distance_slope)

        # The bank above was a read-out that led nowhere. Two cells that the
        # rest of the graph already answers now take it, so what the eyes have
        # followed in close reaches the same "something is close" cell a
        # disparity reaches, and a pair that agrees on a target reaches the
        # same appetitive cell a green patch reaches. Close, and clearly seen,
        # are one state each however the animal arrived at them, and no pixel
        # is turned into metres anywhere on the way.
        #
        # The four bands nearest the eyes are summed with rising weights, so
        # the pull grows over the last half metre instead of switching on at a
        # cliff, and it is the pattern of bands that carries the reading. Those
        # four are the measured ones: band 3 comes on at 0.6 m, band 4 at
        # 0.5 m, band 5 at 0.3 m and band 6 at 0.2 m (see the implementation
        # report). They reach the centre cell because the bank reads an angle
        # between the eyes, which has no left and no right in it; which side
        # the obstacle is on is already carried by the picture and by the
        # disparity cells that sit beside the bank.
        #
        # What the eyes report close ahead also reaches the brake itself, on
        # its own gain, and not the two avoidance channels: measured, the
        # avoidance pair is a transient - it is what a touch does for the
        # moment of a touch - and holding both channels down together for
        # seconds while the animal is walking puts the gait on the ground
        # (up_z 0.99 -> -0.97 driving at the east wall). The brake is what a
        # slow approach needs, and slowing down is all a wall that is still a
        # stride away asks for (artifacts/wall_eye_gain_sweep.log).
        for fraction, band in zip((.15, .30, .50, .80), distance[-4:]):
            edge(band, self.groups['near'][1], near_gain*fraction)
            edge(band, self.groups['brake'][0], wall_gain*fraction)
        # Two eyes settled on one target is something standing in plain view.
        # The edge is plastic and starts small, so the local rule decides
        # whether a clear picture comes to mean approach, exactly as it does
        # for a local patch of retina.
        edge(self.groups['eye_fusion'][0], self.groups['appetitive'][0], fusion_gain, True)
        edge(self.groups['eye_fusion'][0], self.groups['curiosity'][0], fusion_gain*.5, True)
        # Two eyes, one reflex, and the one thing neither eye can answer on
        # its own. Both retinas are cut into the same five bands across, and
        # each band of each retina has its own cell, so the left eye's bands and
        # the right eye's bands are two of the same set: A,B,C,D,E and a,b,c,d,e
        # with A and a the same band in the two pictures. One retina's five
        # bands are five cells, and the picture drives each in proportion to how
        # much of that band carries an edge. The five are siblings and not
        # rivals: nothing here suppresses one band because
        # another is busy, and a picture with edges in two bands lights two
        # bands. The bands read the saturated copy of the picture, for the same
        # reason the disparity bank does: a rendered edge moves a raw contrast
        # cell by about a fifth of its range, so a threshold asked of a whole
        # band would otherwise fit one rendering and not another.
        #
        # A cell that both a left band and a right band reach fires when one eye
        # is busy in one band and the other eye is busy in another: the two eyes
        # are on different things. That cell turns each eye the way that carries
        # its own busy band toward the middle band, and the pull adds up over
        # every pair that is lit, so what each eye answers is the centre of mass
        # of its own picture: the pair whose band is already the middle one
        # contributes nothing, and a picture whose edges straddle the middle
        # pulls evenly both ways and is left where it is. B in the left eye
        # and d in the right eye drive the left eye toward C and the right eye
        # toward c. A and a together ask for no correction and are read as the
        # pair having settled on one point. Nothing here is measured, shifted or
        # compared: a cell fires because two bands drive it at once, and the
        # only thing a band can say to a muscle is "turn the way that carries me
        # to the middle", which is the same left/right cell the rest of the
        # gaze wiring already uses.
        # The same five bands, stretched to whatever width the retina is:
        # their edges stay at the same fractions of the picture as they sat at
        # on the 32 wide one (cuts at 6.5 / 13.5 / 16.5 / 23.5 of its 31
        # columns), so raising self.eye_width widens every band instead of changing
        # which part of the picture it is.
        _cuts = [int(round(cut/30.*self.eye_columns)) for cut in (6.5, 13.5, 16.5, 23.5)]
        bands = []
        _start = 0
        for _cut in _cuts + [self.eye_columns]:
            bands.append((_start, _cut - 1))
            _start = _cut
        bands = tuple(bands)
        middle_band = len(bands)//2
        band_left = cells('eye_band_left', len(bands), base=-band_threshold, time=.03)
        band_right = cells('eye_band_right', len(bands), base=-band_threshold, time=.03)
        for index, (first, last) in enumerate(bands):
            share = band_gain/float(16*(last - first + 1))
            for eye, target in ((0, band_left), (1, band_right)):
                for row, col, polarity in np.ndindex(EYE_ROWS, last - first + 1, 2):
                    edge(contrast_copy[eye, row, first + col, polarity], target[index], share)
        # How much picture the two eyes carry together: one cell, the mean over
        # every cell of both pictures. It holds the whole bank down, the way the
        # cell that watches the same thing holds the gaze cells down, so the bank
        # answers where the contrast sits and not how much of it there is. A room
        # of edges lights every band of both eyes, and the bank is then held down
        # to nothing: two pictures that are busy everywhere say nothing about
        # where either eye should look. A ball alone in the same room is not.
        band_common_cell = cells('eye_band_inhibition', 1, -1, time=.05,
                                 budget=1. + float(contrast_copy.size))[0]
        for eye, row, col, polarity in np.ndindex(contrast_copy.shape):
            edge(contrast_copy[eye, row, col, polarity], band_common_cell,
                 1./float(contrast_copy.size))
        band_pairs = [(k, m) for k in range(len(bands)) for m in range(len(bands)) if k != m]
        apart = cells('eye_bands_apart', len(band_pairs), base=-band_threshold, time=.03,
                      budget=1. + 2.*len(bands) + band_common)
        # One command cell for each way an eye can be asked to turn: the left of
        # its own picture, or the right of it. They are read the way the columns
        # read the gaze cells - each pair of bands hands its pull to one of the
        # two cells, not one muscle edge per pair - so the pull a muscle gets is
        # one cell's worth however many bands the picture lights. Two pairs of a
        # kind fill a cell, and the cell's own ceiling does the rest, so the pull
        # on an eye saturates at eye_band_push radians. That is the whole of the
        # gain: a pair that reads close to a whole band out of place moves its
        # eye by about a band's worth of angle, and a pair that reads the bands
        # nearly lined up barely moves it at all.
        share = .5
        steps = float(sum(abs(band - middle_band) for band in range(len(bands))))
        voices = (len(bands) - 1)*share*steps
        pull_left = cells('eye_band_pull_left', 2, time=.2, budget=1. + voices)
        pull_right = cells('eye_band_pull_right', 2, -1, time=.2, budget=1. + voices)
        for index, (left_band, right_band) in enumerate(band_pairs):
            edge(band_left[left_band], apart[index], 1.)
            edge(band_right[right_band], apart[index], 1.)
            edge(band_common_cell, apart[index], band_common)
            for eye, band in ((0, left_band), (1, right_band)):
                if band == middle_band:
                    continue
                # A band one place out of the middle asks for a step; a band two
                # places out asks for two steps, because that is how far its
                # picture has to travel to reach the middle.
                edge(apart[index], pull_right[eye] if band > middle_band
                     else pull_left[eye], share*abs(band - middle_band))
        push(0, pull_left[0], band_push)
        push(0, pull_right[0], band_push)
        push(2, pull_left[1], band_push)
        push(2, pull_right[1], band_push)
        # Both eyes busy in the same band is a pair that has settled on one
        # point: the same thing sits in the same place of both pictures, which
        # is what a pair that is looking at it looks like. That is the reading
        # the fusion cell already carries, so the bands reach it and nothing
        # else.
        agree = cells('eye_bands_agree', len(bands), base=-band_threshold, time=.03)
        for index in range(len(bands)):
            edge(band_left[index], agree[index], 1.)
            edge(band_right[index], agree[index], 1.)
            edge(agree[index], self.groups['eye_fusion'][0], 1.)

        return self.groups['eye_proprioception']

    def step(self, observation, *, eye_pixels=None, ear_waveform=None, **kwargs):
        if eye_pixels is None:
            raw = np.zeros(self.eye_shape, dtype=np.uint8)
        else:
            raw = np.asarray(eye_pixels)
            if raw.shape != self.eye_shape or raw.dtype != np.uint8:
                raise ValueError(f'eye_pixels must be two raw {self.eye_height}x{self.eye_width} uint8 RGB images')
        auditory_activity = None
        auditory_spatial = None
        auditory_pinna = None
        if ear_waveform is not None:
            auditory_output = self.auditory.step(ear_waveform)
            auditory_activity = auditory_output['rates']
            auditory_spatial = auditory_output['spatial_rates']
            auditory_pinna = auditory_output['pinna_rates']
        live = kwargs.get('reflexes', True) and kwargs.get('feedback', True)
        # One division into the vector the network already holds, rather than a
        # fresh float array built from the picture every step. The finally
        # below zeroes the same vector, so no reader sees a stale frame.
        pixels = self.network.pixel_current
        if live:
            np.divide(raw.ravel(), 255., out=pixels)
        else:
            pixels.fill(0.)
        if self.eye_encoder is not None:
            angle = np.asarray(observation['eye_position'], dtype=float).reshape(4)
            normalized = np.clip((angle - self.eye_lower)/self.eye_span, 0, 1)
            self.network.eye_current = (self.eye_encoder.encode(normalized).ravel() if live
                                        else np.zeros(self.eye_encoder.size))
        try:
            return super().step(observation, auditory_activity=auditory_activity,
                                auditory_spatial_activity=auditory_spatial,
                                auditory_pinna_activity=auditory_pinna, **kwargs)
        finally:
            self.network.pixel_current.fill(0.)
            if self.eye_encoder is not None:
                self.network.eye_current.fill(0.)

    def eye_command(self):
        """Angle the eye muscles are currently commanded to, in radians."""
        if self.eye_encoder is None:
            raise RuntimeError('this controller has no eye muscles')
        units = self.network.rates_at(self.groups['eye_motor']).reshape(4, self.eye_motor_units)
        return np.clip(self.eye_lower + units.mean(axis=1)*self.eye_span, self.eye_lower, self.eye_upper)

    def diagnostics(self):
        result = super().diagnostics()
        rates = self.network.activity
        result['binocular_population_activity'] = rates[
            self.groups['binocular']].reshape(
                EYE_ROWS, self.eye_binocular_columns, 2, STEREO_OFFSET_MAX+1).mean(axis=(0, 1, 2)).tolist()
        result['retinal_activity'] = rates[self.groups['retina']].reshape(2,3,3).tolist()
        result['auditory_activity'] = rates[self.groups['cochlea']].reshape(2,3).tolist()
        if self.eye_encoder is not None:
            result['eye_command'] = self.eye_command().tolist()
            result['eye_gaze'] = {name: float(rates[self.groups[name]].mean()) for name in
                                  ('eye_look_left', 'eye_look_right', 'eye_look_up', 'eye_look_down')}
            result['eye_vergence'] = float(rates[self.groups['eye_vergence']].mean())
            result['eye_fusion'] = float(rates[self.groups['eye_fusion']].mean())
            result['eye_change'] = float(rates[self.groups['retinal_change']].mean())
            result['eye_sound'] = rates[self.groups['auditory_spatial']].reshape(2, 3).mean(axis=1).tolist()
            result['eye_memory'] = float(rates[self.groups['retinal_memory']].mean())
            result['auditory_pinna'] = rates[self.groups['auditory_pinna']].reshape(2, 2).tolist()
            result['eye_alignment'] = rates[self.groups['binocular_pool']].tolist()
            result['eye_disparity'] = {name: float(rates[self.groups[name]].mean()) for name in
                                       ('eye_vergence_in', 'eye_vergence_out')}
            result['eye_convergence'] = float(rates[self.groups['eye_convergence_in']].mean()
                                              - rates[self.groups['eye_convergence_out']].mean())
            result['eye_distance'] = rates[self.groups['eye_distance']].tolist()
            result['eye_bands'] = {'left': rates[self.groups['eye_band_left']].tolist(),
                                   'right': rates[self.groups['eye_band_right']].tolist(),
                                   'apart': float(rates[self.groups['eye_bands_apart']].mean()),
                                   'agree': float(rates[self.groups['eye_bands_agree']].mean())}
            result['eye_loom'] = {'rise': float(rates[self.groups['retinal_rise']].mean()),
                                  'fall': float(rates[self.groups['retinal_fall']].mean()),
                                  'outward': float(rates[self.groups['retinal_outward']].mean()),
                                  'inward': float(rates[self.groups['retinal_inward']].mean()),
                                  'growing': rates[self.groups['eye_growing']].tolist(),
                                  'looming': float(rates[self.groups['eye_looming']].mean())}
        return result
