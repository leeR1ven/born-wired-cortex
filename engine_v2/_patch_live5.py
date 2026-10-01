import io
p = r"F:\born-wired-cortex\engine_v2\tools\eye_live.py"
s = io.open(p, encoding="utf-8", newline="").read()
pairs = [("LIMIT_SIDE = .45                # 球最多偏左/偏右这么多弧度（26 度）",
          "LIMIT_SIDE = .42                # 球最多偏左/偏右这么多弧度（24 度）"),
         ("NEAR, FAR = .62, 1.05           # 球最近 / 最远",
          "NEAR, FAR = .70, 1.05           # 球最近 / 最远（太近时眼球会转不过来）"),
         ("    state[\"distance\"] = .84 + .20*np.sin(clock*.6)",
          "    state[\"distance\"] = (NEAR + FAR)/2 + (FAR - NEAR)/2*np.sin(clock*.6)")]
for old, new in pairs:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("改好了")