# -*- coding: utf-8 -*-
import io
p = r"F:\born-wired-cortex\engine_v2\born_wired\embodied.py"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
def swap(old, new, tag):
    global s
    assert s.count(old) == 1, "MISS " + tag
    s = s.replace(old, new)

head = '''            wanted = {}
            for cell, direction, tier, weight in red['edges']:
                key = (str(tier), str(direction), whose(cell))
                wanted[key] = wanted.get(key, 0.) + float(weight)
'''
old_body = '''            latch = max(0., float(red.get('latch', 0.)))
            cross = max(0., float(red.get('cross', 0.)))
            command, mirrored = {}, {}
            for key, total in wanted.items():
                name = 'eye_red_' + '_'.join(str(part) for part in key)
                command[key] = cells(name, 1, 1, time=float(red.get('cell_time', .03)),
                                     budget=1. + total + latch)[0]
                mirrored[key] = cells(name + '_inhibition', 1, -1, time=.01,
                                      budget=1. + cross)[0]
            # A cell may not reach itself, so the latch is a pair that excites the
            # pair: whatever fires, keeps firing.
            for key, cell in command.items():
                edge(cell, mirrored[key], 1.)
                if latch > 0:
                    relay = cells('eye_red_' + '_'.join(str(part) for part in key) + '_relay',
                                  1, 1, time=.03, budget=1. + latch)[0]
                    edge(cell, relay, latch)
                    edge(relay, cell, latch)
            if cross > 0:
                for key in list(command):
                    tier, side, eye = key
                    for here, other in (('left', 'right'), ('right', 'left'),
                                        ('up', 'down'), ('down', 'up')):
                        if side == here and (tier, other, eye) in mirrored:
                            edge(command[key], mirrored[(tier, other, eye)], cross)
            pushes = red.get('push')
            for tier, rank in (('move', 0), ('hold', 1)):
                for side, axis in (('left', 0), ('right', 0), ('down', 1), ('up', 1)):
                    for eye in ((0, 1) if per_eye else (None,)):
                        key = (tier, side, eye)
                        cell = command.get(key)
                        if cell is None:
                            continue
                        source = mirrored[key] if side in ('right', 'up') else cell
                        if pushes is None:
                            strength = float(red['move_push'] if tier == 'move'
                                             else red['hold_push'])
                        else:
                            strength = float(pushes[side][rank])
                        joints = ((2*eye + axis,) if per_eye
                                  else ((0, 2) if axis == 0 else (1, 3)))
                        for joint in joints:
                            push(joint, source, strength)
            for cell, direction, tier, weight in red['edges']:
                edge(self.groups['retinal_opponent'][cell],
                     command[(str(tier), str(direction), whose(cell))], float(weight))
'''
body = "\n".join("    " + line if line.strip() else line for line in old_body.split("\n"))
direct = '''            if direct:
                # \u7528\u6237 2026-09-30 \u6307\u5b9a\uff1a\u7ea2\u8272\u7279\u5f81\u7ec6\u80de\u76f4\u63a5\u63a5\u5230\u773c\u808c\u795e\u7ecf\u5143\u4e0a\uff0c\u4e2d\u95f4\u4e0d\u518d\u6709\u6307\u4ee4\u7ec6\u80de\u90a3\u4e00\u5c42\u3002
                # \u6bcf\u6839\u8f74\u56db\u4e2a\uff1a\u6b63\u5927\u529b\u6c14\u3001\u6b63\u5c0f\u529b\u6c14\u3001\u8d1f\u5927\u529b\u6c14\u3001\u8d1f\u5c0f\u529b\u6c14\u3002\u4e00\u4e2a\u7ec6\u80de\u644a\u5230\u591a\u5c11
                # \u529b = \u5b83\u5728\u6574\u6863\u91cc\u5360\u591a\u5c11\uff0c\u6240\u4ee5\u7403\u504f\u5f97\u8d8a\u8fdc\u3001\u63a8\u5f97\u8d8a\u4e45\u2014\u2014\u300c\u7ee7\u7eed\u8f6c\u300d\u672c\u6765\u5c31\u4f1a\u8f6c\u5230\u8f6c\u4e0d\u52a8\u4e3a\u6b62\u3002
                rank = {'move': 0, 'hold': 1}
                for cell, direction, tier, weight in red['edges']:
                    side = str(direction)
                    eye = whose(cell)
                    if eye is None:
                        raise ValueError('direct red wiring needs per_eye')
                    total = wanted[(str(tier), side, eye)]
                    if total <= 0:
                        continue
                    axis = 0 if side in ('left', 'right') else 1
                    joint = 2*eye + axis
                    going = 0 if side in ('left', 'down') else 2
                    slot = joint*4 + going + rank[str(tier)]
                    edge(self.groups['retinal_opponent'][int(cell)], muscle[slot],
                         float(weight)/total)
            else:
'''
swap(head + old_body, head + direct + body, "red")

swap('''        finally:
            self.network.pixel_current.fill(0.)
            self.network.opposite_current.fill(0.)
            if self.eye_encoder is not None:
                self.network.eye_current.fill(0.)

    def eye_command(self):
        """Angle the eye muscles are currently commanded to, in radians."""
        if self.eye_encoder is None:
            raise RuntimeError('this controller has no eye muscles')
        units = self.network.rates_at(self.groups['eye_motor']).reshape(4, self.eye_motor_units)
        return np.clip(self.eye_lower + units.mean(axis=1)*self.eye_span, self.eye_lower, self.eye_upper)
''', '''        finally:
            self.network.pixel_current.fill(0.)
            self.network.opposite_current.fill(0.)
            if self.eye_encoder is not None:
                self.network.eye_current.fill(0.)
                if self.eye_muscle_direct:
                    self._eye_follow(float(kwargs.get('dt', .002)))

    def _eye_follow(self, dt):
        """\u4e00\u53ea\u773c 8 \u4e2a\u795e\u7ecf\u5143\u65f6\u773c\u775b\u600e\u4e48\u52a8\u3002

        \u6bcf\u6839\u8f74\u56db\u4e2a\uff1a\u6b63\u65b9\u5411\u5927\u529b\u6c14\u3001\u6b63\u65b9\u5411\u5c0f\u529b\u6c14\u3001\u8d1f\u65b9\u5411\u5927\u529b\u6c14\u3001\u8d1f\u65b9\u5411\u5c0f\u529b\u6c14\u3002
        \u5927\u529b\u6c14\u4eae\u7740\u5c31\u5f80\u90a3\u4e2a\u65b9\u5411\u7ee7\u7eed\u8f6c\uff08\u8f6c\u591a\u5feb\u770b\u5b83\u653e\u591a\u5c11\u7535\uff09\uff0c\u5c0f\u529b\u6c14\u4eae\u7740\u5c31\u628a\u773c\u7403\u505c\u5728
        \u539f\u5730\uff0c\u4e24\u79cd\u90fd\u4e0d\u4eae\u773c\u775b\u81ea\u5df1\u6162\u6162\u56de\u5230\u6b63\u524d\u65b9\u3002
        """
        rates = self.network.rates_at(self.groups['eye_motor']).reshape(4, 4)
        drive = rates[:, 0] - rates[:, 2]
        hold = rates[:, 1] + rates[:, 3]
        angle = self._eye_angle + drive*self.eye_speed*dt
        idle = (np.abs(drive) < 1e-6) & (hold < 1e-6)
        angle = np.where(idle, angle*np.exp(-dt/self.eye_rest_time), angle)
        self._eye_angle = np.clip(angle, self.eye_lower, self.eye_upper)

    def eye_command(self):
        """Angle the eye muscles are currently commanded to, in radians."""
        if self.eye_encoder is None:
            raise RuntimeError('this controller has no eye muscles')
        if self.eye_muscle_direct:
            return self._eye_angle.copy()
        units = self.network.rates_at(self.groups['eye_motor']).reshape(4, self.eye_motor_units)
        return np.clip(self.eye_lower + units.mean(axis=1)*self.eye_span, self.eye_lower, self.eye_upper)
''', "step")
io.open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", "\r\n"))
print("ok")