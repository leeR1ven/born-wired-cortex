import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()
pairs = [
# 视角：整个狗都看得见，偏前一点，能同时看到头和球
("""        viewer.cam.lookat[:] = [0.34, 0.0, 0.30]
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 42.0
        viewer.cam.elevation = -10.0""",
 """        viewer.cam.lookat[:] = [0.40, 0.0, 0.22]
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 118.0
        viewer.cam.elevation = -14.0"""),
# 文字挪到底部，把上方的位置留给两只眼的画面
("""            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, LEGEND, ""),
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "球：方位%+.2f 高低%+.2f 距离%.2f 米   模式 %s%s"
                 % (state["bearing"], state["elevation"], state["distance"], state["mode"],
                    "  已暂停" if state["paused"] else ""), ""),
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                 "左眼 %+5.1f 度（该到 %+5.1f）   右眼 %+5.1f 度（该到 %+5.1f）"
                 % (np.degrees(command[0]), np.degrees(ask[0][0]),
                    np.degrees(command[2]), np.degrees(ask[1][0])), ""),
            ])""",
 """            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "球 方位%+.2f 高低%+.2f %.2f米   %s%s"
                 % (state["bearing"], state["elevation"], state["distance"], state["mode"],
                    " 已暂停" if state["paused"] else ""), LEGEND),
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                 "左眼 %+5.1f（该到 %+5.1f）   右眼 %+5.1f（该到 %+5.1f）"
                 % (np.degrees(command[0]), np.degrees(ask[0][0]),
                    np.degrees(command[2]), np.degrees(ask[1][0])),
                 "上排左边 = 左眼看到的   上排右边 = 右眼看到的"),
            ])"""),
# 贴图位置：挂到 3D 视口的上左 / 上右（矩形用客户区像素、左下角原点）
("""            if board is not None:
                view = viewer.viewport
                high, wide = int(board[0].shape[0]), int(board[0].shape[1])
                viewer.set_images([
                    (mujoco.MjrRect(8, view.height - high - 8, wide, high), board[0]),
                    (mujoco.MjrRect(view.width - wide - 8, view.height - high - 8, wide, high),
                     board[1]),
                ])""",
 """            if board is not None:
                view = viewer.viewport
                high, wide = int(board[0].shape[0]), int(board[0].shape[1])
                top = view.bottom + view.height - high - 8
                viewer.set_images([
                    (mujoco.MjrRect(view.left + 8, top, wide, high), board[0]),
                    (mujoco.MjrRect(view.left + view.width - wide - 8, top, wide, high), board[1]),
                ])"""),
]
for old, new in pairs:
    assert s.count(old) == 1, ("没找到唯一位置", s.count(old), old.splitlines()[0][:50])
    s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("改好了")