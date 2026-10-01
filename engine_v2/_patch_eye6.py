# -*- coding: utf-8 -*-
import io
p = r"F:\born-wired-cortex\engine_v2\born_wired\embodied.py"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
def swap(old, new, tag):
    global s
    assert s.count(old) == 1, "MISS " + tag
    s = s.replace(old, new)

swap('''            # \u7528\u6237 2026-09-30 \u6307\u5b9a\u7684\u773c\u808c\uff1a\u4e00\u53ea\u773c 8 \u4e2a\u795e\u7ecf\u5143 \u2014\u2014 4 \u4e2a\u65b9\u5411\u5404 2 \u4e2a\uff0c
            # \u4e00\u4e2a\u5927\u529b\u6c14\uff08\u7ee7\u7eed\u8f6c\uff09\u3001\u4e00\u4e2a\u5c0f\u529b\u6c14\uff08\u4fdd\u6301\uff09\u3002\u7ea2\u8272\u7279\u5f81\u7ec6\u80de\u76f4\u63a5\u63a5\u5b83\u4eec\uff0c\u4e2d\u95f4\u4e0d\u518d\u6709
            # \u6307\u4ee4\u7ec6\u80de\uff0c\u4e5f\u6ca1\u6709\u90a3\u6761 24 \u7ea7\u7684\u9636\u68af\u3002\u600e\u4e48\u8bfb\u5728 _eye_follow\uff1a
            # \u5927\u529b\u6c14 -> \u5f80\u90a3\u4e2a\u65b9\u5411\u7ee7\u7eed\u8f6c\uff1b\u5c0f\u529b\u6c14 -> \u505c\u5728\u539f\u5730\uff1b\u90fd\u4e0d\u4eae -> \u773c\u775b\u81ea\u5df1\u56de\u6b63\u3002
            # \u6bcf\u6839\u8f74\u56db\u4e2a\uff1a\u6b63\u65b9\u5411\u5927\u529b\u6c14\u3001\u6b63\u65b9\u5411\u5c0f\u529b\u6c14\u3001\u8d1f\u65b9\u5411\u5927\u529b\u6c14\u3001\u8d1f\u65b9\u5411\u5c0f\u529b\u6c14\u3002
            self.eye_motor_units = 4
            muscle = cells('eye_motor', 4*4, base=0.,
                           time=float(red.get('motor_time', .2)))
''', '''            # \u7528\u6237 2026-09-30 \u6307\u5b9a\u7684\u773c\u808c\uff1a\u4e00\u53ea\u773c 4 \u4e2a\u795e\u7ecf\u5143 \u2014\u2014 \u56db\u4e2a\u65b9\u5411\u5404\u4e00\u4e2a\uff0c
            # \u5168\u662f\u300c\u5927\u529b\u6c14\u300d\uff08\u7ee7\u7eed\u8f6c\uff09\u3002\u7ea2\u8272\u7279\u5f81\u7ec6\u80de\u76f4\u63a5\u63a5\u5b83\u4eec\uff0c\u4e2d\u95f4\u4e0d\u518d\u6709\u6307\u4ee4\u7ec6\u80de\uff0c
            # \u4e5f\u6ca1\u6709\u90a3\u6761 24 \u7ea7\u7684\u9636\u68af\u3002\u600e\u4e48\u8bfb\u5728 _eye_follow\uff1a\u54ea\u4e2a\u65b9\u5411\u7684\u795e\u7ecf\u5143\u4eae\u7740\uff0c
            # \u773c\u7403\u5c31\u5f80\u90a3\u4e2a\u65b9\u5411\u7ee7\u7eed\u8f6c\uff1b\u6ca1\u6709\u795e\u7ecf\u5143\u4eae\u65f6\uff0c\u773c\u7403\u81ea\u5df1\u5f80\u56de\u98d8\u5230\u6b63\u524d\u65b9\u3002
            # \u6bcf\u6839\u8f74\u4e24\u4e2a\uff1a\u6b63\u65b9\u5411\u4e00\u4e2a\u3001\u8d1f\u65b9\u5411\u4e00\u4e2a\u3002
            self.eye_motor_units = 2
            muscle = cells('eye_motor', 4*2, base=0.,
                           time=float(red.get('motor_time', .2)))
''', "muscle")

swap('''                going = 0 if self._sign_of.get(int(source), 1.) > 0 else 2
                edge(source, muscle[joint*4 + going], abs(radians)/span[joint])
''', '''                going = 0 if self._sign_of.get(int(source), 1.) > 0 else 1
                edge(source, muscle[joint*2 + going], abs(radians)/span[joint])
''', "push")

swap('''                rank = {'move': 0, 'hold': 1}
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
''', '''                for cell, direction, tier, weight in red['edges']:
                    side = str(direction)
                    eye = whose(cell)
                    if eye is None:
                        raise ValueError('direct red wiring needs per_eye')
                    total = wanted[(str(tier), side, eye)]
                    if total <= 0:
                        continue
                    axis = 0 if side in ('left', 'right') else 1
                    joint = 2*eye + axis
                    going = 0 if side in ('left', 'down') else 1
                    edge(self.groups['retinal_opponent'][int(cell)], muscle[joint*2 + going],
                         float(weight)/total)
''', "red")

swap('''    def _eye_follow(self, dt):
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
''', '''    def _eye_follow(self, dt):
        """\u4e00\u53ea\u773c 4 \u4e2a\u795e\u7ecf\u5143\u65f6\u773c\u775b\u600e\u4e48\u52a8\u3002

        \u6bcf\u6839\u8f74\u4e24\u4e2a\uff1a\u6b63\u65b9\u5411\u4e00\u4e2a\u3001\u8d1f\u65b9\u5411\u4e00\u4e2a\uff0c\u90fd\u662f\u5927\u529b\u6c14\u3002\u54ea\u4e2a\u4eae\u7740\uff0c
        \u773c\u7403\u5c31\u5f80\u90a3\u4e2a\u65b9\u5411\u7ee7\u7eed\u8f6c\uff08\u8f6c\u591a\u5feb\u770b\u5b83\u653e\u591a\u5c11\u7535\uff09\uff1b\u540c\u65f6\u773c\u775b\u4e00\u76f4\u88ab\u5f80
        \u6b63\u524d\u65b9\u62c9\uff0c\u6240\u4ee5\u4e00\u65e6\u6ca1\u6709\u795e\u7ecf\u5143\u4eae\u4e86\uff0c\u89c6\u89d2\u5c31\u81ea\u5df1\u5f80\u56de\u98d8\uff1b\u98d8\u51fa\u53bb\u53c8\u4f1a\u78b0\u5230
        \u65c1\u8fb9\u7684\u7ec6\u80de\uff0c\u773c\u775b\u518d\u5f80\u56de\u8f6c \u2014\u2014 \u773c\u775b\u4f1a\u5728\u7403\u9644\u8fd1\u6765\u56de\u6296\uff0c\u8fd9\u662f\u5e94\u8be5\u7684\u3002
        """
        rates = self.network.rates_at(self.groups['eye_motor']).reshape(4, 2)
        drive = rates[:, 0] - rates[:, 1]
        angle = self._eye_angle + (drive*self.eye_speed
                                   - self._eye_angle/self.eye_rest_time)*dt
        self._eye_angle = np.clip(angle, self.eye_lower, self.eye_upper)
''', "follow")
io.open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", "\r\n"))
print("ok")