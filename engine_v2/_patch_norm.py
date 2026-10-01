# -*- coding: utf-8 -*-
"""改两处归一化：wires 把「球团该给多少力」写进 spec；眼肌接线按它归一。"""
import io

p = "tools/wire_red_gaze.py"
text = io.open(p, encoding="utf-8", newline="").read()
old = """def spec_of(table, split, move_push, hold_push, dead, **kwargs):
    cell_time = float(kwargs.pop("cell_time", .03))"""
new = """def spec_of(table, split, move_push, hold_push, dead, **kwargs):
    gain = float(kwargs.pop("gain", 1.))
    cell_time = float(kwargs.pop("cell_time", .03))"""
assert text.count(old) == 1
text = text.replace(old, new)

old = """    edges = wires(table, split=split, move_push=move_push, hold_push=hold_push,
                  dead=dead, per_eye=per_eye, only=only, **kwargs)
    return dict(edges=[tuple(edge) for edge in edges], move_push=float(move_push),"""
new = """    edges = wires(table, split=split, move_push=move_push, hold_push=hold_push,
                  dead=dead, per_eye=per_eye, only=only, gain=gain, **kwargs)
    # gain = 一个球团摆在这一档最远处时该给出的总力。归一化的分母要用它，不是这一档
    # 所有线的和：2026-09-30 实测，按整条档带归一，分母是球团的二十几倍（移动档 25.4），
    # 一整个球点亮只推得动 0.06，眼睛几乎不转。
    return dict(gain=gain, edges=[tuple(edge) for edge in edges], move_push=float(move_push),"""
assert text.count(old) == 1
text = text.replace(old, new)

old = """    print("眼肌：%s" % ("一只眼 8 个神经元：4 个方向 × 大/小力气（大力气继续转、"
                      "小力气保持、都不亮回正）" if args.direct
                      else "24 级阶梯（旧：静息电流顶在中间）"))"""
new = """    print("眼肌：%s" % ("一只眼 4 个神经元：四个方向各一个，都是大力气 —— "
                      "亮着就往那边继续转，都不亮就自己回正" if args.direct
                      else "24 级阶梯（旧：静息电流顶在中间）"))"""
assert text.count(old) == 1
text = text.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(text)
print("wire_red_gaze.py 改好")

p = "../born_wired/embodied.py".replace("../", "born_wired/")
p = "born_wired/embodied.py"
raw = io.open(p, encoding="utf-8", newline="").read()
text = raw.replace("\r\n", "\n")
old = """                for cell, direction, tier, weight in red['edges']:
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
                         float(weight)/total)"""
new = """                # 分母是「一个球团摆在这一档最远处该给的力」（spec 里的 gain），不是这一档
                # 所有线的和。2026-09-30 实测：按整条档带归一，分母是球团的二十几倍
                # （移动档 25.4），一整个球点亮只推得动 0.06，眼睛几乎不转。
                # 按球团归一，球摆在这一档最远处时刚好用满 1.0，越靠中间越轻。
                full = max(1e-9, float(red.get('gain', 1.)))
                for cell, direction, tier, weight in red['edges']:
                    side = str(direction)
                    eye = whose(cell)
                    if eye is None:
                        raise ValueError('direct red wiring needs per_eye')
                    axis = 0 if side in ('left', 'right') else 1
                    joint = 2*eye + axis
                    going = 0 if side in ('left', 'down') else 1
                    edge(self.groups['retinal_opponent'][int(cell)], muscle[joint*2 + going],
                         float(weight)/full)"""
assert text.count(old) == 1
text = text.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(text.replace("\n", "\r\n"))
print("embodied.py 改好")