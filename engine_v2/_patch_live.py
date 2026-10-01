import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()

pairs = [
("""        viewer.cam.lookat[:] = [0.32, 0.0, 0.33]
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 28.0
        viewer.cam.elevation = -6.0""",
 """        viewer.cam.lookat[:] = [0.34, 0.0, 0.30]
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 42.0
        viewer.cam.elevation = -10.0"""),

("""LEGEND = ("A 自动走路线   S 停住   0 正前方   1~8 八个位置   方向键 挪球   "
          "Z/X 远近   P 暂停   R 复位")""",
 """LEGEND = ("A 自动走路线   S 停住   0 正前方   1~8 八个位置   方向键 挪球   "
          "Z/X 远近   P 暂停   R 复位     左上角=左眼看到的画面  右上角=右眼看到的画面")

SCALE = 5                       # 眼球画面只有 48x36，放大 5 倍贴到窗口角上"""),

("""    started = time.time()
    next_report = started + 1.0
    frames = 0""",
 """    started = time.time()
    next_report = started + 1.0
    frames = 0
    board = None                # 这一帧两只眼的画面，放大后贴到窗口角上"""),

("""                body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
                frames += 1""",
 """                body.step(np.asarray(body.home_angles), duration=DT, activation=activation)
                frames += 1
                board = [np.repeat(np.repeat(np.asarray(raw[i]), SCALE, 0), SCALE, 1)
                         for i in (0, 1)]"""),

("""            viewer.sync()
            if time.time() > next_report:""",
 """            if board is not None:
                view = viewer.viewport
                high, wide = int(board[0].shape[0]), int(board[0].shape[1])
                viewer.set_images([
                    (mujoco.MjrRect(8, view.height - high - 8, wide, high), board[0]),
                    (mujoco.MjrRect(view.width - wide - 8, view.height - high - 8, wide, high),
                     board[1]),
                ])
            viewer.sync()
            if time.time() > next_report:"""),
]
for old, new in pairs:
    assert s.count(old) == 1, ("没找到唯一位置", s.count(old), old.splitlines()[0][:60])
    s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("eye_live.py 已改：默认视角拉到 1.15 米、并把两只眼画面贴到窗口左上/右上角")