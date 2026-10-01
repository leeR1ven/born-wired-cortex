import io
p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '''                    print("%5d  %-6s 转 %d/%d 趟  跟住 %d/%d 趟  最后离球 %6.2f 米  "
                          "正前方 %3d%% 时间  走了 %5.2f 米%s  路 %s%s  本机 %.0f 秒"
                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             row.get("covered", 0), row.get("trials", 0),'''
new = '''                    print("%5d  %-6s 走起来 %d/4  转 %d/%d 趟  跟住 %d/%d 趟  最后离球 %6.2f 米  "
                          "正前方 %3d%% 时间  走了 %5.2f 米%s  路 %s%s  本机 %.0f 秒"
                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             row.get("moving", 0),
                             row.get("covered", 0), row.get("trials", 0),'''
assert old in s
s = s.replace(old, new, 1)
old2 = '''    return ("第 %d 代收工：%d 只里建出来 %d 只（没摔 %d）；朝球走的 %d 只，两边都会的 %d 只，"'''
new2 = '''    return ("第 %d 代收工：%d 只里建出来 %d 只（没摔 %d、4 趟都走起来的 %d 只）；朝球走的 %d 只，两边都会的 %d 只，"'''
assert old2 in s
s = s.replace(old2, new2, 1)
old3 = '''            % (gen, len(rows), len(ok), len(upright), len(covering),'''
new3 = '''            % (gen, len(rows), len(ok), len(upright),
               sum(1 for row in ok if row["moving"] >= 4), len(covering),'''
assert old3 in s
s = s.replace(old3, new3, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")