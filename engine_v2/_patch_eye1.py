# -*- coding: utf-8 -*-
import io
p = r"F:\born-wired-cortex\engine_v2\born_wired\embodied.py"
s = io.open(p, encoding="utf-8", newline="").read()
assert s.count("\r\n") > 0
s = s.replace("\r\n", "\n")
def swap(old, new, tag):
    global s
    assert s.count(old) == 1, "MISS " + tag
    s = s.replace(old, new)

# (a) \u8bb0\u4e0b\u6bcf\u4e2a\u7ec6\u80de\u7684\u7b26\u53f7\uff0c\u63a8\u7684\u65f6\u5019\u8981\u7528
swap('''        def cells(name, count, sign=1, base=0., time=.015, adaptation=0., start=None, budget=40.):
            ids = np.arange(old.n_neurons+len(signs), old.n_neurons+len(signs)+count)
            self.groups[name] = ids
            signs.extend(np.broadcast_to(sign, (count,)).tolist())
''', '''        sign_of = {}

        def cells(name, count, sign=1, base=0., time=.015, adaptation=0., start=None, budget=40.):
            ids = np.arange(old.n_neurons+len(signs), old.n_neurons+len(signs)+count)
            self.groups[name] = ids
            signs.extend(np.broadcast_to(sign, (count,)).tolist())
            for offset, value in enumerate(np.broadcast_to(sign, (count,)).tolist()):
                sign_of[int(ids[offset])] = float(value)
''', "a")

# (b) \u65b0\u7684\u773c\u808c\u5f0f\u6837\u7684\u72b6\u6001
swap('''        self.eye_encoder = None
        self.eye_motor_units = 0
        self.eye_gain = eye_gain
''', '''        self._sign_of = sign_of
        self.eye_encoder = None
        self.eye_motor_units = 0
        self.eye_muscle_direct = False
        self.eye_speed = 1.5
        self.eye_rest_time = .6
        self._eye_angle = np.zeros(4)
        self.eye_gain = eye_gain
''', "b")

# (c) \u8ba4\u51fa\u76f4\u63a5\u63a5\u6cd5
swap('''        self.eye_lower, self.eye_upper, self.eye_span = lower, upper, span
        self.eye_motor_units = int(motor_units)
''', '''        self.eye_lower, self.eye_upper, self.eye_span = lower, upper, span
        self.eye_motor_units = int(motor_units)
        direct = bool(red is not None and red.get('direct'))
        self.eye_muscle_direct = direct
''', "c")

# (d) \u808c\u8089\uff1a\u4e24\u79cd\u65b9\u5f0f\u5404\u5efa\u5404\u7684
swap('''        centres = (np.arange(self.eye_motor_units) + .5)/self.eye_motor_units
        muscle_bias = np.tile(.5 - gain*centres, 4)
        resting = gain*operating
        settled = resting + 2.*gaze_gain*gain/span
        settled += vergence_steps*gain/span
        # The band reflex at the end of this method is one more voice on each
        # eye's own muscle, one command cell each way, so the resource a muscle
        # unit can hold has to cover that pair too.
        settled += 2.*band_push*gain/span
        # A red-route push is one more voice on the same muscle, so the resource
        # a muscle unit can hold has to cover that one too.
        if red is not None:
            pushes = red.get('push')
            if pushes is None:
                asked = [float(red['move_push'])]*4
            else:
                yaw = max(float(pushes['left'][0]), float(pushes['right'][0]))
                pitch = max(float(pushes['down'][0]), float(pushes['up'][0]))
                asked = [yaw, pitch, yaw, pitch]
            settled = settled + np.asarray(asked, dtype=float)*gain/span
        # The muscle's own lag is a property of the muscle, not of the wiring
        # above it: slower muscle, smoother following.
        muscle_time = .015 if red is None else float(red.get('motor_time', .015))
        muscle = cells('eye_motor', 4*self.eye_motor_units, base=muscle_bias,
                       time=muscle_time,
                       start=muscle_bias + np.repeat(resting, self.eye_motor_units),
                       budget=1. + float(settled.max()))
''', '''        if direct:
            # \u7528\u6237 2026-09-30 \u6307\u5b9a\u7684\u773c\u808c\uff1a\u4e00\u53ea\u773c 8 \u4e2a\u795e\u7ecf\u5143 \u2014\u2014 4 \u4e2a\u65b9\u5411\u5404 2 \u4e2a\uff0c
            # \u4e00\u4e2a\u5927\u529b\u6c14\uff08\u7ee7\u7eed\u8f6c\uff09\u3001\u4e00\u4e2a\u5c0f\u529b\u6c14\uff08\u4fdd\u6301\uff09\u3002\u7ea2\u8272\u7279\u5f81\u7ec6\u80de\u76f4\u63a5\u63a5\u5b83\u4eec\uff0c\u4e2d\u95f4\u4e0d\u518d\u6709
            # \u6307\u4ee4\u7ec6\u80de\uff0c\u4e5f\u6ca1\u6709\u90a3\u6761 24 \u7ea7\u7684\u9636\u68af\u3002\u600e\u4e48\u8bfb\u5728 _eye_follow\uff1a
            # \u5927\u529b\u6c14 -> \u5f80\u90a3\u4e2a\u65b9\u5411\u7ee7\u7eed\u8f6c\uff1b\u5c0f\u529b\u6c14 -> \u505c\u5728\u539f\u5730\uff1b\u90fd\u4e0d\u4eae -> \u773c\u775b\u81ea\u5df1\u56de\u6b63\u3002
            # \u6bcf\u6839\u8f74\u56db\u4e2a\uff1a\u6b63\u65b9\u5411\u5927\u529b\u6c14\u3001\u6b63\u65b9\u5411\u5c0f\u529b\u6c14\u3001\u8d1f\u65b9\u5411\u5927\u529b\u6c14\u3001\u8d1f\u65b9\u5411\u5c0f\u529b\u6c14\u3002
            self.eye_motor_units = 4
            muscle = cells('eye_motor', 4*4, base=0.,
                           time=float(red.get('motor_time', .2)))
            self.eye_speed = float(red.get('speed', 1.5))
            self.eye_rest_time = float(red.get('rest_time', .6))
        else:
            centres = (np.arange(self.eye_motor_units) + .5)/self.eye_motor_units
            muscle_bias = np.tile(.5 - gain*centres, 4)
            resting = gain*operating
            settled = resting + 2.*gaze_gain*gain/span
            settled += vergence_steps*gain/span
            # The band reflex at the end of this method is one more voice on each
            # eye's own muscle, one command cell each way, so the resource a muscle
            # unit can hold has to cover that pair too.
            settled += 2.*band_push*gain/span
            # A red-route push is one more voice on the same muscle, so the resource
            # a muscle unit can hold has to cover that one too.
            if red is not None:
                pushes = red.get('push')
                if pushes is None:
                    asked = [float(red['move_push'])]*4
                else:
                    yaw = max(float(pushes['left'][0]), float(pushes['right'][0]))
                    pitch = max(float(pushes['down'][0]), float(pushes['up'][0]))
                    asked = [yaw, pitch, yaw, pitch]
                settled = settled + np.asarray(asked, dtype=float)*gain/span
            # The muscle's own lag is a property of the muscle, not of the wiring
            # above it: slower muscle, smoother following.
            muscle_time = .015 if red is None else float(red.get('motor_time', .015))
            muscle = cells('eye_motor', 4*self.eye_motor_units, base=muscle_bias,
                           time=muscle_time,
                           start=muscle_bias + np.repeat(resting, self.eye_motor_units),
                           budget=1. + float(settled.max()))
''', "d")

# (e) \u9759\u606f\u7535\u6d41\uff1a\u53ea\u6709\u65e7\u7684\u9636\u68af\u624d\u6709
swap('''        for joint in range(4):
            for unit in muscle.reshape(4, self.eye_motor_units)[joint]:
                edge(self.groups['tonic'][0], unit, resting[joint])
''', '''        if not direct:
            for joint in range(4):
                for unit in muscle.reshape(4, self.eye_motor_units)[joint]:
                    edge(self.groups['tonic'][0], unit, resting[joint])
''', "e")

# (f) push
swap('''        def push(joint, source, radians):
            """A cell pushes one eye joint; an inhibitory source pushes it back."""
            for unit in muscle.reshape(4, self.eye_motor_units)[joint]:
                edge(source, unit, abs(radians)*gain/span[joint])
''', '''        def push(joint, source, radians):
            """A cell pushes one eye joint; an inhibitory source pushes it back."""
            if direct:
                # \u4e00\u53ea\u773c 8 \u4e2a\u795e\u7ecf\u5143\u65f6\uff0c\u63a8\u7684\u662f\u8fd9\u4e2a\u65b9\u5411\u7684\u300c\u5927\u529b\u6c14\u300d\u795e\u7ecf\u5143\uff08\u7ee7\u7eed\u8f6c\uff09\uff1b
                # \u5174\u594b\u6027\u7684\u63a8\u6b63\u65b9\u5411\uff0c\u6291\u5236\u6027\u7684\u63a8\u8d1f\u65b9\u5411 \u2014\u2014 \u9760\u8fd9\u4e2a\u7ec6\u80de\u81ea\u5df1\u7684\u7b26\u53f7\u8ba4\u3002
                going = 0 if self._sign_of.get(int(source), 1.) > 0 else 2
                edge(source, muscle[joint*4 + going], abs(radians)/span[joint])
                return
            for unit in muscle.reshape(4, self.eye_motor_units)[joint]:
                edge(source, unit, abs(radians)*gain/span[joint])
''', "f")
io.open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", "\r\n"))
print("ok")