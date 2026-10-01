# -*- coding: utf-8 -*-
import io
p = r"F:\born-wired-cortex\engine_v2\tools\wire_red_gaze.py"
s = io.open(p, encoding="utf-8", newline="").read()
def swap(old, new, tag):
    global s
    assert s.count(old) == 1, "MISS " + tag
    s = s.replace(old, new)

swap('''    per_eye = bool(kwargs.pop("per_eye", True))
    only = kwargs.pop("only", None)
''', '''    per_eye = bool(kwargs.pop("per_eye", True))
    only = kwargs.pop("only", None)
    direct = bool(kwargs.pop("direct", True))
    speed = float(kwargs.pop("speed", 1.5))
    rest_time = float(kwargs.pop("rest_time", .6))
''', "spec-head")

swap('''                cross=cross, motor_time=motor_time, per_eye=per_eye, only=only,
''', '''                cross=cross, motor_time=motor_time, per_eye=per_eye, only=only,
                direct=direct, speed=speed, rest_time=rest_time,
''', "spec-tail")

swap('''    ap.add_argument("--eye-gain", type=float, default=None,
''', '''    ap.add_argument("--direct", action=argparse.BooleanOptionalAction, default=True,
                    help="红色特征细胞直接接到眼肌神经元（默认开）")
    ap.add_argument("--speed", type=float, default=1.5,
                    help="大力气满力时眼球转多快（弧度/秒）")
    ap.add_argument("--rest-time", type=float, default=.6, help="回正的时间常数（秒）")
    ap.add_argument("--eye-gain", type=float, default=None,
''', "cli")

swap('''                   per_eye=args.per_eye, only=only)
''', '''                   per_eye=args.per_eye, only=only, direct=args.direct,
                   speed=args.speed, rest_time=args.rest_time)
''', "call")

tail_old = '''             args.per_eye, "" if only is None else "、只做%s眼" % args.eye))
'''
tail_new = '''             args.per_eye, "" if only is None else "、只做%s眼" % args.eye))
    print("眼肌：%s" % ("一只眼 8 个神经元：4 个方向 × 大/小力气（大力气继续转、"
                      "小力气保持、都不亮回正）" if args.direct
                      else "24 级阶梯（旧：静息电流顶在中间）"))
'''
swap(tail_old, tail_new, "header")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")