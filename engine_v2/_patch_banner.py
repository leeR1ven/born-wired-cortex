# -*- coding: utf-8 -*-
"""横幅文字跟现在的考试对上：静止球已经从 4 趟改成随机 1 趟。"""
import io, sys
P = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
t = io.open(P, encoding="utf-8").read()
old = '''            print("第 %d 代：生 %d 只、每只考 4 趟不动球（每趟 %.0f 秒）+ 2 趟追着跑的球"'''
new = '''            print("第 %d 代：生 %d 只、每只考 1 趟不动球（位置随机、每趟 %.0f 秒）+ 2 趟追着跑的球"'''
if t.count(old) != 1:
    print("!! \u6ca1\u627e\u5230\u6216\u4e0d\u552f\u4e00 %d" % t.count(old)); sys.exit(1)
io.open(P, "w", encoding="utf-8", newline="\n").write(t.replace(old, new))
print("ok")