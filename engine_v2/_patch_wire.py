import ast
p = "tools/wire_red_gaze.py"
s = open(p, encoding="utf-8", newline="").read()

# 1) eye_of helper
anchor = "def wires(table, split=.10"
assert s.count(anchor) == 1
helper = '''def eye_of(index, table):
    """这些细胞属于哪只眼：按视网膜层的排布直接算，不用猜。"""
    shape = table["eye_shape"]
    stride = int(shape[1])*int(shape[2])*3
    return np.asarray(index, dtype=int)//stride


'''
s = s.replace(anchor, helper + anchor)

# 2) wires(): per-eye grouping
i = s.find("    red, angles = red_cells(table)\n    bearing")
j = s.find("    return out\n", i)
assert i > 0 and j > i
j += len("    return out\n")
new = '''    red, angles = red_cells(table)
    bearing, elevation = angles[:, 0], angles[:, 1]
    group = eye_of(red, table)
    out = []
    for eye in ((0, 1) if per_eye else (None,)):
        mine = np.ones(len(red), bool) if eye is None else (group == eye)
        for axis, seen, other, positive, negative, top in (
                ("yaw", bearing, elevation, "left", "right", move_top),
                ("pitch", elevation, bearing, "up", "down", pitch_top)):
            cross_axis = "pitch" if axis == "yaw" else "yaw"
            for tier, low, high, band_top, push in (
                    ("hold", dead, split, hold_top, hold_push),
                    ("move", split, np.inf, top, move_push)):
                if not tiers and tier == "hold":
                    continue
                # 球摆在「这一档最远处、正前方那个高度」时点亮的那片细胞。球在视网膜上
                # 亮的是一个团，不是一个竖列，所以两个方向都要卡：只按方位卡，会把所有
                # 高度的细胞都算进分母，接线就细了十倍。per_eye 时分母只算这一只眼的。
                blob = (mine & (np.abs(seen - band_top) <= block[axis])
                        & (np.abs(other) <= block[cross_axis]))
                mass = float(np.abs(seen[blob]).sum())
                if mass <= 0:
                    continue
                thick = float(gain)/mass
                band = (np.abs(seen) >= low) & (np.abs(seen) < high)
                for direction, sign in ((positive, 1.), (negative, -1.)):
                    side = band & mine & (np.sign(seen) == sign)
                    for cell, angle in zip(red[side], seen[side]):
                        out.append([int(cell), direction, tier, thick*abs(float(angle))])
    return out
'''
s = s[:i] + new + s[j:]

# 3) signature gets per_eye
old = ("def wires(table, split=.10, dead=np.radians(3.), move_top=.35, pitch_top=.28,\n"
       "          hold_top=None, move_push=.60, hold_push=.15, gain=1., tiers=True,\n"
       "          block=None):\n")
assert s.count(old) == 1
new = ("def wires(table, split=.10, dead=np.radians(3.), move_top=.35, pitch_top=.28,\n"
       "          hold_top=None, move_push=.60, hold_push=.15, gain=1., tiers=True,\n"
       "          block=None, per_eye=True):\n")
s = s.replace(old, new)

# 4) spec_of carries it
old = ("    motor_time = float(kwargs.pop(\"motor_time\", .015))\n"
       "    edges = wires(table, split=split, move_push=move_push, hold_push=hold_push,\n"
       "                  dead=dead, **kwargs)\n")
assert s.count(old) == 1
new = ("    motor_time = float(kwargs.pop(\"motor_time\", .015))\n"
       "    per_eye = bool(kwargs.pop(\"per_eye\", True))\n"
       "    edges = wires(table, split=split, move_push=move_push, hold_push=hold_push,\n"
       "                  dead=dead, per_eye=per_eye, **kwargs)\n")
s = s.replace(old, new)
old = ("                hold_push=float(hold_push), cell_time=cell_time, latch=latch,\n"
       "                cross=cross, motor_time=motor_time,\n")
assert s.count(old) == 1
new = ("                hold_push=float(hold_push), cell_time=cell_time, latch=latch,\n"
       "                cross=cross, motor_time=motor_time, per_eye=per_eye,\n")
s = s.replace(old, new)

# 5) CLI
old = '    ap.add_argument("--motor-time", type=float, default=.015, help="眼肌自己的时间常数（秒）")\n'
assert s.count(old) == 1
new = (old + '    ap.add_argument("--per-eye", action=argparse.BooleanOptionalAction, default=True,\n'
       '                    help="每只眼只被自己那只眼的红细胞驱动（默认开）")\n')
s = s.replace(old, new)
old = ("                   latch=args.latch, cross=args.cross, motor_time=args.motor_time)\n")
assert s.count(old) == 1
new = ("                   latch=args.latch, cross=args.cross, motor_time=args.motor_time,\n"
       "                   per_eye=args.per_eye)\n")
s = s.replace(old, new)
old = ('          "眼肌时间 %.3f"\n')
assert s.count(old) == 1
new = ('          "眼肌时间 %.3f、每眼独立 %s"\n')
s = s.replace(old, new)
old = "             args.dead, args.gain, args.latch, args.cross, args.motor_time))\n"
assert s.count(old) == 1
new = "             args.dead, args.gain, args.latch, args.cross, args.motor_time,\n             args.per_eye))\n"
s = s.replace(old, new)

open(p, "w", encoding="utf-8", newline="").write(s)
ast.parse(open(p, encoding="utf-8").read())
print("wire_red_gaze per-eye ok")