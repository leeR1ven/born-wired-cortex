import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """            if board is not None:
                view = viewer.viewport
                high, wide = int(board[0].shape[0]), int(board[0].shape[1])
                top = view.bottom + view.height - high - 8"""
new = """            if board is not None:
                view = viewer.viewport
                high, wide = int(board[0].shape[0]), int(board[0].shape[1])
                top = view.bottom + view.height - high - 8
                if not shown:
                    print("贴图：视口 left=%d width=%d height=%d，图 %dx%d，放 x=%d/%d y=%d"
                          % (view.left, view.width, view.height, wide, high,
                             view.left + 8, view.left + view.width - wide - 8, top), flush=True)
                    shown = True"""
assert s.count(old) == 1
s = s.replace(old, new)
old2 = """    board = None                # 这一帧两只眼的画面，放大后贴到窗口角上"""
new2 = """    board = None                # 这一帧两只眼的画面，放大后贴到窗口角上
    shown = False"""
assert s.count(old2) == 1
s = s.replace(old2, new2)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("加好了调试输出")