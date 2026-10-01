# -*- coding: utf-8 -*-
"""每只一行里也报一下「整趟平均离球几米」。"""
import io
p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '''                    print("%5d  %-14s 追住 %d/%d 趟  落后 %6.2f 米  最近 %5.2f 米  正前方 %3.0f%%  "
                          "静球到了 %d/%d 趟  走了 %5.2f 米%s  路 %s%s  本机 %.0f 秒"
                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             row.get("chase_covered", 0), row.get("chase_trials", 0),
                             row.get("chase_settled", float("nan")),
                             row.get("chase_min", float("nan")),'''
new = '''                    print("%5d  %-14s 追住 %d/%d 趟  平均 %6.2f 米  落后 %6.2f 米  最近 %5.2f 米  "
                          "正前方 %3.0f%%  静球到了 %d/%d 趟  走了 %5.2f 米%s  路 %s%s  本机 %.0f 秒"
                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             row.get("chase_covered", 0), row.get("chase_trials", 0),
                             row.get("chase_mean", float("nan")),
                             row.get("chase_settled", float("nan")),
                             row.get("chase_min", float("nan")),'''
assert old in s, "no row print"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched print")