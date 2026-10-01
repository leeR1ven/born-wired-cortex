import io
p = r"F:\born-wired-cortex\engine_v2\tools\_walk_movie.py"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.replace('ap.add_argument("--seconds", type=float, default=12.)',
              'ap.add_argument("--seconds", type=float, default=10.)')
s = s.replace('ap.add_argument("--distance", type=float, default=3.)',
              'ap.add_argument("--distance", type=float, default=3.)\n'
              '    ap.add_argument("--ball", action="store_true",\n'
              '                    help="摆个红球；不给就是空旷场地（走路演化当时就是这么跑的）")')
old = ('    C.G.place_in_front(body.model, body.data, geom, args.bearing, 0., args.distance)\n'
       '    mujoco.mj_forward(body.model, body.data)')
new = ('    if args.ball:\n'
       '        C.G.place_in_front(body.model, body.data, geom, args.bearing, 0., args.distance)\n'
       '    else:\n'
       '        body.model.geom_pos[geom] = list(C.FAR)\n'
       '    mujoco.mj_forward(body.model, body.data)')
assert old in s, "没找到摆球那两行"
s = s.replace(old, new)
old2 = '    from matplotlib import pyplot as plt\n'
new2 = ('    from matplotlib import pyplot as plt\n'
        '    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]\n'
        '    plt.rcParams["axes.unicode_minus"] = False\n')
assert old2 in s
s = s.replace(old2, new2)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched ok")