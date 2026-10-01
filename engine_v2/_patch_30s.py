# -*- coding: utf-8 -*-
"""追球每趟 20 秒 -> 30 秒（用户 2026-10-01：至少让狗能追着球绕个场地一圈）。
场地一圈约 20 米、球 0.6 米/秒，绕一圈要 30 秒出头。"""
import io

a = r"F:\born-wired-cortex\engine_v2\tools\chase_task.py"
s = io.open(a, encoding="utf-8", newline="").read()
old = "CHASE_SECONDS = 20.           # 每趟考多少秒（要够久，才看得出是不是真的追得住）"
new = "CHASE_SECONDS = 30.           # 每趟考多少秒（球 0.6 米/秒绕场地一圈要 30 秒出头）"
assert old in s, "no CHASE_SECONDS"
s = s.replace(old, new, 1)
s = s.replace("    球一开始在两眼正前方 2.5 米，然后一直逃 —— 朝向基本上背着狗，但会平滑地随机摆动，\n"
              "    所以跑出来是一条随机的弯线（用户 2026-10-01）。左、右各一趟，每趟 20 秒\n",
              "    球一开始在两眼正前方 2.5 米，然后一直逃 —— 每一拍都往背对狗的方向挪，\n"
              "    只是叠了一个缓慢摆动的小偏角，所以跑的是一条弯线；快到场地边就贴着场地绕圈\n"
              "    （用户 2026-10-01：始终远离狗、跑的是弯线）。左、右各一趟，每趟 30 秒\n", 1)
io.open(a, "w", encoding="utf-8", newline="").write(s)

b = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(b, encoding="utf-8", newline="").read()
old = '    ap.add_argument("--chase-seconds", type=float, default=20., help="追着跑的球那几趟每趟考多少秒")'
new = '    ap.add_argument("--chase-seconds", type=float, default=30., help="追着跑的球那几趟每趟考多少秒")'
assert old in s, "no chase-seconds arg"
s = s.replace(old, new, 1)
old = '                      chase_seconds=job.get("chase_seconds", 20.),'
new = '                      chase_seconds=job.get("chase_seconds", 30.),'
assert old in s, "no chase_seconds default"
s = s.replace(old, new, 1)
io.open(b, "w", encoding="utf-8", newline="").write(s)
print("patched chase seconds -> 30")