import io
p = r"F:\born-wired-cortex\engine_v2\tools\merge_cloud_eye.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """    ctx = tb.context(genome, seed)
    body = tb.clean_body(ctx["model_path"])"""
new = """    ctx = tb.context(genome, seed)
    body = tb.clean_body(ctx["model_path"])
    # clean_body 只挪道具，四面墙还立在原地：云端那一趟是「空旷场地」，墙也是挪走的
    # （tools/evolve.py 的 open_props 同样挪这几面墙）。漏了这一步，狗走 3 米就撞墙——
    # 2026-09-30 实测：没挪墙时「不摆球」那一趟只走 3.40 米还翻了。
    for name in WALLS:
        wall = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if wall >= 0:
            body.model.geom_pos[wall] = list(FAR)"""
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("""BALL_EVERY = 20                     # 每 0.2 秒量一次红球在画面里的位置""",
              """BALL_EVERY = 20                     # 每 0.2 秒量一次红球在画面里的位置
WALLS = ("east_wall", "west_wall", "north_wall", "south_wall")
FAR = (60., 60., -8.)""")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("挪墙补上了")