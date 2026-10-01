import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """                board = [np.repeat(np.repeat(np.asarray(raw[i]), SCALE, 0), SCALE, 1)
                         for i in (0, 1)]"""
new = """                board = []
                for i in (0, 1):
                    tile = np.repeat(np.repeat(np.asarray(raw[i], dtype=np.uint8), SCALE, 0),
                                     SCALE, 1).copy()
                    tile[:2, :] = tile[-2:, :] = (255, 255, 0)   # 黄框：一眼看出哪块是眼睛画面
                    tile[:, :2] = tile[:, -2:] = (255, 255, 0)
                    board.append(tile)"""
assert s.count(old) == 1
io.open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
print("加了黄框")