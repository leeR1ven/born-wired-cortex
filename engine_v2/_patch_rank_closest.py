import io
p = r"F:\born-wired-cortex\engine_v2\tools\evolve_chase.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '''    return (0 if row["upright"] else 1,
            -int(row["moving"]),
            -int(row["covered"]),
            -int(bool(row["both"])),
            -int(row["flee_covered"]),
            -int(row["caught"]),
            float(row["score"]),
            -float(row["kept"]),
            -float(row["travelled_m"]),
            -float(row["straightness"]))'''
new = '''    return (0 if row["upright"] else 1,
            -int(row["moving"]),
            -int(row["covered"]),
            -int(bool(row["both"])),
            -int(row["flee_covered"]),
            -int(row["caught"]),
            float(row["score"]),
            float(row["toward_deg"]),
            -float(row["travelled_m"]),
            -float(row["straightness"]))'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''               caught=got["caught"], blind=got["blind"],''',
              '''               caught=got["caught"], blind=got["blind"], best=got["best"],
               toward_deg=got["toward_deg"],''', 1)
old2 = '''                    print("%5d  %-6s 走起来 %d/4  转 %d/%d 趟  跟住 %d/%d 趟  最后离球 %6.2f 米  "
                          "正前方 %3d%% 时间  走了 %5.2f 米%s  路 %s%s  本机 %.0f 秒"
                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             row.get("moving", 0),
                             row.get("covered", 0), row.get("trials", 0),
                             row.get("flee_covered", 0), row.get("flee_trials", 0),
                             row.get("score", float("nan")),
                             round(100*row.get("kept", 0.)), row.get("travelled_m", float("nan")),'''
new2 = '''                    print("%5d  %-6s 走起来 %d/4  到了 %d/%d 趟  最近贴到 %5.2f 米  平均 %5.2f 米  "
                          "位移离球 %5.1f 度  走了 %5.2f 米%s  路 %s%s  本机 %.0f 秒"
                          % (row["index"],
                             "没摔" if row["status"] == "ok" and row["upright"] else "倒了/没建出来",
                             row.get("moving", 0),
                             row.get("covered", 0), row.get("trials", 0),
                             row.get("best", float("nan")), row.get("score", float("nan")),
                             row.get("toward_deg", float("nan")),
                             row.get("travelled_m", float("nan")),'''
assert old2 in s
s = s.replace(old2, new2, 1)
old3 = '''                if row["status"] == "ok" and (champion is None or rank(row) < rank(champion)):
                    champion = row
                    print("  跨代冠军易主：%d 代第 %d 只  转 %d/%d 趟、跟住 %d/%d 趟、"
                          "最后离球 %.2f 米、走了 %.2f 米  %s"
                          % (row["gen"], row["index"], row["covered"], row["trials"],
                             row["flee_covered"], row["flee_trials"], row["score"],
                             row["travelled_m"], "没摔" if row["upright"] else "摔了"),
                          flush=True)'''
new3 = '''                if row["status"] == "ok" and (champion is None or rank(row) < rank(champion)):
                    champion = row
                    print("  跨代冠军易主：%d 代第 %d 只  到了 %d/%d 趟、最近贴到 %.2f 米、"
                          "走了 %.2f 米  %s"
                          % (row["gen"], row["index"], row["covered"], row["trials"],
                             row["best"], row["travelled_m"],
                             "没摔" if row["upright"] else "摔了"), flush=True)'''
assert old3 in s
s = s.replace(old3, new3, 1)
old4 = '''            "最好那只转 %d/%d 趟、跟住 %d/%d 趟、最后离球 %.2f 米%s；本代花了 %.1f 分钟"'''
new4 = '''            "最好那只到了 %d/%d 趟、最近贴到 %.2f 米%s；本代花了 %.1f 分钟"'''
assert old4 in s
s = s.replace(old4, new4, 1)
old5 = '''               best["covered"], best["trials"], best["flee_covered"], best["flee_trials"],
               best["score"], "" if best["upright"] else "(摔了)",'''
new5 = '''               best["covered"], best["trials"], best["best"],
               "" if best["upright"] else "(摔了)",'''
assert old5 in s
s = s.replace(old5, new5, 1)
s = s.replace('''                  % (gen, len(jobs), args.workers), flush=True)''',
              '''                  % (gen, len(jobs), args.workers), flush=True)''', 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")