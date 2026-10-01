import json, subprocess, sys, re
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
print("%5s | %8s %8s %8s %8s" % ("gain", "命中", "中位", "最差", "抖动"))
best = None
for gain in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
    out = "artifacts/_扫增益_%.2f.json" % gain
    job = subprocess.run([py, "tools/wire_red_gaze.py",
                          "--table", r"artifacts\红球位置野_红_大.json",
                          "--split", "0.10", "--move-top", "0.35", "--pitch-top", "0.28",
                          "--move-push", "0.35", "--hold-push", "0.10",
                          "--motor-time", "0.20", "--gain", "%.2f" % gain,
                          "--out", out], capture_output=True, text=True, encoding="utf-8")
    if job.returncode:
        print("gain %.2f 出错：%s" % (gain, job.stderr.strip().splitlines()[-1:]))
        continue
    hits = re.search(r"命中 (\d+)/(\d+)", job.stdout)
    med = re.search(r"误差：中位 ([\d.]+)、平均 [\d.]+、最差 ([\d.]+)", job.stdout)
    shake = re.search(r"后段抖动：yaw 最大 ([\d.]+)", job.stdout)
    row = (gain, "%s/%s" % hits.groups() if hits else "?", med.group(1), med.group(2), shake.group(1))
    print("%5.2f | %8s %8s %8s %8s" % row)
    score = (float(med.group(2)), float(shake.group(1)))
    if best is None or score < best[0]:
        best = (score, row)
print("\n最好：gain %.2f（最差 %s、抖动 %s）" % (best[1][0], best[1][3], best[1][4]))