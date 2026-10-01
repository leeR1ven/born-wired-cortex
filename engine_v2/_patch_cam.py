import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = """        viewer.cam.lookat[:] = [0.32, 0.0, 0.33]
        viewer.cam.distance = 0.95
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -12.0"""
new = """        viewer.cam.lookat[:] = [0.32, 0.0, 0.33]
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 28.0
        viewer.cam.elevation = -6.0"""
assert s.count(old) == 1, s.count(old)
io.open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
print("相机默认视角已改：头部特写（正前方偏 28 度，0.55 米）")