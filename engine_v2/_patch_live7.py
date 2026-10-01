import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()
pairs = [
("""LEGEND = ("A 随机漫游   F 老路线   S 停住   0 正前方   1~8 八个位置   方向键 挪球   "
          "Z/X 远近   P 暂停   R 复位")""",
 """LEGEND = "A随机 F老路 S停 0正前 1~8位置 ←→挪球 Z/X远近 P暂停 R复位\""""),
("""            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 LEGEND, ""),
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                 "球 方位%+.2f 高低%+.2f %.2f米  %s%s"
                 % (state["bearing"], state["elevation"], state["distance"],
                    MODE_NAME[state["mode"]], " 已暂停" if state["paused"] else ""),
                 "左眼 %+5.1f（该到 %+5.1f）  右眼 %+5.1f（该到 %+5.1f）"
                 % (np.degrees(command[0]), np.degrees(ask[0][0]),
                    np.degrees(command[2]), np.degrees(ask[1][0]))),
            ])""",
 """            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "球 方位%+.2f 高低%+.2f %.2f米  %s%s"
                 % (state["bearing"], state["elevation"], state["distance"],
                    MODE_NAME[state["mode"]], " 已暂停" if state["paused"] else ""),
                 "左眼 %+5.1f 度（该到 %+5.1f）"
                 % (np.degrees(command[0]), np.degrees(ask[0][0]))),
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                 "右眼 %+5.1f 度（该到 %+5.1f）"
                 % (np.degrees(command[2]), np.degrees(ask[1][0])),
                 LEGEND),
            ])"""),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:40]
    s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("改好了")