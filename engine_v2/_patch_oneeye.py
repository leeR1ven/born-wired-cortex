# -*- coding: utf-8 -*-
from pathlib import Path
p = Path("tools/wire_red_gaze.py")
t = p.read_text(encoding="utf-8", newline="")

# 1. wires() 只接一只眼
old = '''def wires(table, split=.10, dead=np.radians(3.), move_top=.35, pitch_top=.28,
          hold_top=None, move_push=.60, hold_push=.15, gain=1., tiers=True,
          block=None, per_eye=True):'''
new = '''def wires(table, split=.10, dead=np.radians(3.), move_top=.35, pitch_top=.28,
          hold_top=None, move_push=.60, hold_push=.15, gain=1., tiers=True,
          block=None, per_eye=True, only=None):'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''    for eye in ((0, 1) if per_eye else (None,)):
        mine = np.ones(len(red), bool) if eye is None else (group == eye)
'''
new = '''    eyes = (None,) if not per_eye else ((0, 1) if only is None else (int(only),))
    for eye in eyes:
        mine = np.ones(len(red), bool) if eye is None else (group == eye)
'''
assert t.count(old) == 1
t = t.replace(old, new)

# 2. spec_of 透传 only，并把「只做哪只眼」记进 spec
old = '''    per_eye = bool(kwargs.pop("per_eye", True))
    edges = wires(table, split=split, move_push=move_push, hold_push=hold_push,
                  dead=dead, per_eye=per_eye, **kwargs)'''
new = '''    per_eye = bool(kwargs.pop("per_eye", True))
    only = kwargs.pop("only", None)
    edges = wires(table, split=split, move_push=move_push, hold_push=hold_push,
                  dead=dead, per_eye=per_eye, only=only, **kwargs)'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''    return dict(edges=[tuple(edge) for edge in edges], move_push=float(move_push),
                hold_push=float(hold_push), cell_time=cell_time, latch=latch,
                cross=cross, motor_time=motor_time, per_eye=per_eye,'''
new = '''    return dict(edges=[tuple(edge) for edge in edges], move_push=float(move_push),
                hold_push=float(hold_push), cell_time=cell_time, latch=latch,
                cross=cross, motor_time=motor_time, per_eye=per_eye, only=only,'''
assert t.count(old) == 1
t = t.replace(old, new)

# 3. measure() 只看指定那只眼
old = '''def measure(brain, body, eyes, aim, target, places, steps, tail):'''
new = '''def measure(brain, body, eyes, aim, target, places, steps, tail, only=None):'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''        asked = [abs(a[0]) <= VIEW and abs(a[1]) <= VIEW for a in ask]'''
new = '''        asked = [abs(a[0]) <= VIEW and abs(a[1]) <= VIEW for a in ask]
        if only is not None:
            asked = [want and which == int(only) for which, want in enumerate(asked)]'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''                 "看" if row["asked"][0] else "盲",
                 row["yaw_error"][1], row["pitch_error"][1],
                 "看" if row["asked"][1] else "盲", row["wander"]))'''
new = '''                 mark(row, 0),
                 row["yaw_error"][1], row["pitch_error"][1],
                 mark(row, 1), row["wander"]))'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''def required(body, target):'''
new = '''def mark(row, which):
    """这一格这只眼算不算数：看/盲（看不见球）/不管（这一轮只做另一只眼）。"""
    if row.get("only") is not None and row["only"] != which:
        return "不管"
    return "看" if row["asked"][which] else "盲"


def required(body, target):'''
assert t.count(old) == 1
t = t.replace(old, new)

# 4. 行里记下是哪只眼在跑
old = '''        rows.append(dict(bearing=true_bearing, elevation=true_elevation, asked=asked,'''
new = '''        rows.append(dict(bearing=true_bearing, elevation=true_elevation, asked=asked,
                         only=None if only is None else int(only),'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''    rows = measure(brain, body, eyes, aim, target, places, args.steps, args.tail)'''
new = '''    rows = measure(brain, body, eyes, aim, target, places, args.steps, args.tail, only)'''
assert t.count(old) == 1
t = t.replace(old, new)

# 5. 命令行：--eye 和 --eye-gain
old = '''    ap.add_argument("--per-eye", action=argparse.BooleanOptionalAction, default=True,
                    help="每只眼只被自己那只眼的红细胞驱动（默认开）")'''
new = '''    ap.add_argument("--per-eye", action=argparse.BooleanOptionalAction, default=True,
                    help="每只眼只被自己那只眼的红细胞驱动（默认开）")
    ap.add_argument("--eye", choices=("both", "left", "right"), default="both",
                    help="这一轮只接、只看哪只眼（默认两只都做）")
    ap.add_argument("--eye-gain", type=float, default=None,
                    help="眼肌自己的增益（不给就用模型里的值；调它能让静止时两端的眼肌神经元不顶死）")'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''    table = load_table(args.table)
    spec = spec_of(table, args.split, args.move_push, args.hold_push, np.radians(args.dead),
                   move_top=args.move_top, pitch_top=args.pitch_top, hold_top=args.hold_top,
                   gain=args.gain, tiers=not args.no_tiers, cell_time=args.cell_time,
                   latch=args.latch, cross=args.cross, motor_time=args.motor_time,
                   per_eye=args.per_eye)'''
new = '''    only = None if args.eye == "both" else (0 if args.eye == "left" else 1)
    table = load_table(args.table)
    spec = spec_of(table, args.split, args.move_push, args.hold_push, np.radians(args.dead),
                   move_top=args.move_top, pitch_top=args.pitch_top, hold_top=args.hold_top,
                   gain=args.gain, tiers=not args.no_tiers, cell_time=args.cell_time,
                   latch=args.latch, cross=args.cross, motor_time=args.motor_time,
                   per_eye=args.per_eye, only=only)'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''          "眼肌时间 %.3f、每眼独立 %s"'''
new = '''          "眼肌时间 %.3f、每眼独立 %s%s"'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''             args.dead, args.gain, args.latch, args.cross, args.motor_time,
             args.per_eye))'''
new = '''             args.dead, args.gain, args.latch, args.cross, args.motor_time,
             args.per_eye, "" if only is None else "、只做%s眼" % args.eye))'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''    body, brain, eyes, target = build(ctx, spec, red=not args.green_ball)'''
new = '''    body, brain, eyes, target = build(ctx, spec, red=not args.green_ball,
                                      eye_gain=args.eye_gain)'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''def build(ctx, spec, red=True, size=.06):'''
new = '''def build(ctx, spec, red=True, size=.06, eye_gain=None):'''
assert t.count(old) == 1
t = t.replace(old, new)

old = '''    parameters = dict(ctx["parameters"])
    for name in MUTED:'''
new = '''    parameters = dict(ctx["parameters"])
    if eye_gain is not None:
        parameters["eye_gain"] = float(eye_gain)
    for name in MUTED:'''
assert t.count(old) == 1
t = t.replace(old, new)

p.write_text(t, encoding="utf-8", newline="")
print("只做一只眼的开关加好了")