import io, re
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()
start = s.index("def place_in_front(")
end = s.index("def wander(")
s = s[:start] + "place_in_front = G.place_in_front    # 摆球的规矩只有一份，在 tools/eye_geometry.py\n\n\n" + s[end:]
s = s.replace("import eye_ball_movie as M",
              "import eye_geometry as G\nimport eye_ball_movie as M")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("eye_live.py 改成用共享的摆球函数")