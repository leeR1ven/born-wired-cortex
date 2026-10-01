import subprocess, re
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
print("%5s | %7s %8s %8s %8s" % ("gain", "命中", "中位", "最差", "抖动"))
best = None
for gain in (1.0, 1.2, 1.4, 1.6, 1.8, 2.1):
    out = "artifacts/_扫增益2_%.2f.json" % gain
    job = subprocess.run([py, "tools/wire_red_gaze.py",
                          "--table", r"artifacts\红球位置野_红_大.json",
                          "--move-push", "0.35", "--hold-push", "0.10",
                          "--motor-time", "0.20", "--gain", "%.2f" % gain,
                          "--steps", "300", "--tail", "60", "--out", out],
                         capture_output=True, text=True, encoding="utf-8")
    if job.returncode:
        print("gain %.2f 出错：%s" % (gain, job.stdout.strip().splitlines()[-1:]))
        continue
    hits = re.search(r"命中 (\d+)/(\d+)", job.stdout)
    med = re.search(r"误差：中位 ([\d.]+)、平均 [\d.]+、最差 ([\d.]+)", job.stdout)
    shake = re.search(r"后段抖动：最大 ([\d.]+)", job.stdout)
    print("%5.2f | %7s %8s %8s %8s" % (gain, "%s/%s" % hits.groups(), med.group(1),
                                       med.group(2), shake.group(1)))
    score = (float(med.group(2)) + 2*float(shake.group(1)),)
    if best is None or score < best[0]:
        best = (score, gain, med.group(1), med.group(2), shake.group(1))
print("\n最平衡：gain %.2f（中位 %s、最差 %s、抖动 %s）" % (best[1], best[2], best[3], best[4]))