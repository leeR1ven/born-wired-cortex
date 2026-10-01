import io
p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '"""追球演化：一代生几百上千只、每只换一套随机连接，考完立刻挑爹生下一代。'
new = '"""追球演化：一代只生几十只、每只换一套随机连接，考完立刻挑爹、马上生下一代。'
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")