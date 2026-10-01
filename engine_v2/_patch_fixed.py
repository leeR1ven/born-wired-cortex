# -*- coding: utf-8 -*-
"""按用户 2026-09-30 的说法改：
1) 眼肌神经元被激活就发固定力 —— 每根红线都一样粗（1.0），一根就够推满；
2) 回正力固定、一直都有（不随信号、不做判断）。
"""
import io

p = "tools/wire_red_gaze.py"
text = io.open(p, encoding="utf-8", newline="").read()
edits = [
    ('    rest_time = float(kwargs.pop("rest_time", .6))\n',
     '    rest_time = float(kwargs.pop("rest_time", .6))\n'
     '    rest_speed = float(kwargs.pop("rest_speed", .3))\n'),
    ('''                  dead=dead, per_eye=per_eye, only=only, gain=gain, **kwargs)
    # gain = 一个球团摆在这一档最远处时该给出的总力。归一化的分母要用它，不是这一档
    # 所有线的和：2026-09-30 实测，按整条档带归一，分母是球团的二十几倍（移动档 25.4），
    # 一整个球点亮只推得动 0.06，眼睛几乎不转。
''',
     '''                  dead=dead, per_eye=per_eye, only=only, gain=gain, **kwargs)
    if direct:
        # 用户 2026-09-30：眼肌神经元一被激活就发出**固定的力**，力气不随信号多少变化。
        # 所以每根红线都一样粗（1.0：一根就足够把眼肌神经元推满），不用再算什么参照、
        # 什么分母。球只要偏在那边、有任何一个细胞亮着，眼肌神经元就饱和，眼睛就用
        # 固定速度往那边转。分档这时候只剩「哪些细胞接线」的意思，粗细已经一样了。
        edges = [(cell, direction, tier, 1.0) for cell, direction, tier, weight in edges]
'''),
    ('                direct=direct, speed=speed, rest_time=rest_time,\n',
     '                direct=direct, speed=speed, rest_time=rest_time, rest_speed=rest_speed,\n'),
    ('    ap.add_argument("--rest-time", type=float, default=.6, help="回正的时间常数（秒）")\n',
     '    ap.add_argument("--rest-time", type=float, default=.6,\n'
     '                    help="回正的时间常数（秒）；只有旧的分级模式用，直接接线看 --rest-speed")\n'
     '    ap.add_argument("--rest-speed", type=float, default=.3,\n'
     '                    help="回正力多大：没有信号时眼球每秒回正多少弧度（固定力）")\n'),
    ('                   speed=args.speed, rest_time=args.rest_time)\n',
     '                   speed=args.speed, rest_time=args.rest_time,\n'
     '                   rest_speed=args.rest_speed)\n'),
    ('''                    help="大力气满力时眼球转多快（弧度/秒）")\n''',
     '''                    help="眼肌神经元被激活时眼球每秒转多少弧度（固定力）")\n'''),
    ('''    print("眼肌：%s" % ("一只眼 4 个神经元：四个方向各一个，都是大力气 —— "
                      "亮着就往那边继续转，都不亮就自己回正" if args.direct
                      else "24 级阶梯（旧：静息电流顶在中间）"))''',
     '''    print("眼肌：%s" % ("一只眼 4 个神经元：四个方向各一个，被激活就发出固定的力"
                      "（每秒 %.2f 弧度往那边转）；没有信号时被固定的回正力"
                      "（每秒 %.2f 弧度）拉回正前方" % (args.speed, args.rest_speed)
                      if args.direct else "24 级阶梯（旧：静息电流顶在中间）"))'''),
]
for old, new in edits:
    assert text.count(old) == 1, old[:60]
    text = text.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(text)
print("wire_red_gaze.py 改好")

p = "born_wired/embodied.py"
text = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
edits = [
    ('''            self.eye_motor_units = 2
            muscle = cells('eye_motor', 4*2, base=0.,
                           time=float(red.get('motor_time', .2)))
            self.eye_speed = float(red.get('speed', 1.5))
            self.eye_rest_time = float(red.get('rest_time', .6))
''',
     '''            self.eye_motor_units = 2
            # 预算要给足：每个眼肌神经元会接几百上千根红线，而一根就够推满，所以这些
            # 线不能被预算按比例削细 —— 削细了就变成「信号越少力气越小」，用户
            # 2026-09-30 明确说力气是固定的。
            muscle = cells('eye_motor', 4*2, base=0.,
                           time=float(red.get('motor_time', .2)),
                           budget=1. + float(len(red['edges'])))
            self.eye_speed = float(red.get('speed', 1.5))
            self.eye_rest_time = float(red.get('rest_time', .6))
            self.eye_rest_speed = float(red.get('rest_speed', .3))
'''),
    ('''        """一只眼 4 个神经元时眼睛怎么动。

        每根轴两个：正方向一个、负方向一个，都是大力气。哪个亮着，
        眼球就往那个方向继续转（转多快看它放多少电）；同时眼睛一直被往
        正前方拉，所以一旦没有神经元亮了，视角就自己往回飘；飘出去又会碰到
        旁边的细胞，眼睛再往回转 —— 眼睛会在球附近来回抖，这是应该的。
        """
        rates = self.network.rates_at(self.groups['eye_motor']).reshape(4, 2)
        drive = rates[:, 0] - rates[:, 1]
        angle = self._eye_angle + (drive*self.eye_speed
                                   - self._eye_angle/self.eye_rest_time)*dt
        self._eye_angle = np.clip(angle, self.eye_lower, self.eye_upper)''',
     '''        """一只眼 4 个神经元时眼睛怎么动。

        每根轴两个：四个方向各一个，谁被激活谁就发出**固定的力** —— 眼肌神经元一被推
        就饱和（放电率到 1），所以力气不随信号多少变化：球在视网膜上点亮几个细胞、
        亮了多少，眼睛都只用同一个固定速度往那边转。
        同时眼睛一直被一个固定的回正力往正前方拉。球停在正中间那一小圈时两边都不亮，
        回正把眼睛拉回去，球又偏出去，眼睛再转回来 —— 于是绕着球来回抖，这是应该的。
        """
        rates = self.network.rates_at(self.groups['eye_motor']).reshape(4, 2)
        drive = rates[:, 0] - rates[:, 1]
        pull = -np.sign(self._eye_angle)*self.eye_rest_speed
        angle = self._eye_angle + (drive*self.eye_speed + pull)*dt
        self._eye_angle = np.clip(angle, self.eye_lower, self.eye_upper)'''),
    ('''                # 分母是「一个球团摆在这一档最远处该给的力」（spec 里的 gain），不是这一档
                # 所有线的和。2026-09-30 实测：按整条档带归一，分母是球团的二十几倍
                # （移动档 25.4），一整个球点亮只推得动 0.06，眼睛几乎不转。
                # 按球团归一，球摆在这一档最远处时刚好用满 1.0，越靠中间越轻。
                full = max(1e-9, float(red.get('gain', 1.)))
                for cell, direction, tier, weight in red['edges']:''',
     '''                # 每根线都一样粗（spec 里已经定成 1.0）：一根红线就足以把眼肌神经元推满，
                # 于是它发出的是**固定的力** —— 球点亮几个细胞、亮得多少，都不改变力气。
                # 用户 2026-09-30：力气本来就该是固定的，不是信号越多越大。
                for cell, direction, tier, weight in red['edges']:'''),
    ('''                    edge(self.groups['retinal_opponent'][int(cell)], muscle[joint*2 + going],
                         float(weight)/full)''',
     '''                    edge(self.groups['retinal_opponent'][int(cell)], muscle[joint*2 + going],
                         float(weight))'''),
]
for old, new in edits:
    assert text.count(old) == 1, old[:60]
    text = text.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(text.replace("\n", "\r\n"))
print("embodied.py 改好")