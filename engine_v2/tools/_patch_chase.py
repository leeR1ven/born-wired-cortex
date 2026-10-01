import io
p = r"F:\born-wired-cortex\engine_v2\tools\_chase_movie.py"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.replace("def run_segment(genome, chase, seed, spec, bearing, distance, seconds, renderer, cam):",
              "def run_segment(genome, chase, seed, spec, bearing, distance, seconds, cam):")
s = s.replace("    body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)\n",
              "    body, brain, eyes, geom = C.build(genome, chase, seed=seed, spec=spec)\n"
              "    renderer = mujoco.Renderer(body.model, height=PANEL[1], width=PANEL[0])\n")
s = s.replace("    eyes.close()\n    return dict(track=",
              "    eyes.close()\n    renderer.close()\n    return dict(track=")
old = ("    body0, _, eyes0, _ = C.build(genome, chase, seed=seed, spec=spec)\n"
       "    eyes0.close()\n"
       "    renderer = mujoco.Renderer(body0.model, height=PANEL[1], width=PANEL[0])\n")
assert old in s, "没找到建渲染器那几行"
s = s.replace(old, "")
s = s.replace("got = run_segment(genome, chase, seed, spec, bearing, distance, args.seconds,\n                          renderer, cam)",
              "got = run_segment(genome, chase, seed, spec, bearing, distance, args.seconds, cam)")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")