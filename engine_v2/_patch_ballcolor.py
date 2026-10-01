import io, re, sys
p = r"C:\Users\Administrator\Documents\关联神经元架构\..\..\..\F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
p = r"F:\born-wired-cortex\engine_v2\tools\chase_red_ball.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '''def open_field(body):
    """空旷场地：四面墙挪走，中间那颗球改成红的小球（云端那一趟就是这么跑的）。"""'''
new = '''def open_field(body, rgba=(1., 0., 0., 1.)):
    """空旷场地：四面墙挪走，中间那颗球改成红的小球（云端那一趟就是这么跑的）。"""'''
assert old in s, "open_field 没找到"
s = s.replace(old, new)
old2 = "    body.model.geom_rgba[geom] = [1., 0., 0., 1.]\n    return geom"
new2 = "    body.model.geom_rgba[geom] = list(rgba)\n    return geom"
assert old2 in s, "rgba 行没找到"
s = s.replace(old2, new2)
old3 = "def build(genome, chase, seed=0, eye=True, spec=None):\n    ctx = tb.context(genome, seed)\n    body = tb.clean_body(ctx[\"model_path\"])\n    geom = open_field(body)"
new3 = "def build(genome, chase, seed=0, eye=True, spec=None, ball_rgba=(1., 0., 0., 1.)):\n    ctx = tb.context(genome, seed)\n    body = tb.clean_body(ctx[\"model_path\"])\n    geom = open_field(body, ball_rgba)"
assert old3 in s, "build 头没找到"
s = s.replace(old3, new3)
old4 = '''    ap.add_argument("--seed", type=int, default=0)'''
new4 = '''    ap.add_argument("--ball-color", default="red", choices=("red", "green", "blue", "white"))
    ap.add_argument("--seed", type=int, default=0)'''
assert old4 in s, "seed 参数没找到"
s = s.replace(old4, new4, 1)
old5 = "    body, brain, eyes, geom = build(genome, chase, seed=args.seed, eye=not args.no_eye)"
new5 = ("    colors = {\"red\": (1., 0., 0., 1.), \"green\": (0., 1., 0., 1.),\n"
        "              \"blue\": (0., 0., 1., 1.), \"white\": (1., 1., 1., 1.)}\n"
        "    body, brain, eyes, geom = build(genome, chase, seed=args.seed, eye=not args.no_eye,\n"
        "                                    ball_rgba=colors[args.ball_color])")
assert old5 in s, "build 调用没找到"
s = s.replace(old5, new5)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")