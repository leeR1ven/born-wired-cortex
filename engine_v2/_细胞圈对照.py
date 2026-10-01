import subprocess, re
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
runs = {
 "X1 上限0.35 满程0.35/0.28 gain1.4": ["--move-top","0.35","--pitch-top","0.28","--move-push","0.35","--hold-push","0.10","--gain","1.4"],
 "X4 上限0.60 满程0.60/0.48 gain1.4": ["--move-top","0.60","--pitch-top","0.48","--move-push","0.60","--hold-push","0.10","--gain","1.4"],
 "X5 上限0.60 满程0.60/0.48 gain1.7": ["--move-top","0.60","--pitch-top","0.48","--move-push","0.60","--hold-push","0.10","--gain","1.7"],
}
print("%-34s %7s %8s %8s %8s" % ("配置","命中","中位","最差","抖动"))
for name, extra in runs.items():
    tag = name.split()[0]
    job = subprocess.run([py, "tools/wire_red_gaze.py", "--table", r"artifacts\红球位置野_红_大.json",
                          "--dead","1.5","--motor-time","0.20","--steps","300","--tail","60",
                          "--out","artifacts/_对照_%s.json" % tag] + extra,
                         capture_output=True, text=True, encoding="utf-8")
    if job.returncode:
        print("%-34s 出错：%s" % (name, job.stdout.strip().splitlines()[-1:])); continue
    hits = re.search(r"命中 (\d+)/(\d+)", job.stdout)
    med = re.search(r"误差：中位 ([\d.]+)、平均 [\d.]+、最差 ([\d.]+)", job.stdout)
    shake = re.search(r"后段抖动：最大 ([\d.]+)", job.stdout)
    print("%-34s %7s %8s %8s %8s" % (name, "%s/%s"%hits.groups(), med.group(1), med.group(2), shake.group(1)))
    open("logs/眼_细胞圈对照_%s.log" % tag, "w", encoding="utf-8").write(job.stdout + job.stderr)