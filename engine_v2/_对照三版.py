import subprocess, re
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe"
runs = {
 "A 推0.35/0.10 满程0.35/0.28 gain1.2": ["--move-top","0.35","--pitch-top","0.28","--move-push","0.35","--hold-push","0.10","--gain","1.2"],
 "B 推0.60/0.10 满程0.60/0.45 gain1.2": ["--move-top","0.60","--pitch-top","0.45","--move-push","0.60","--hold-push","0.10","--gain","1.2"],
 "C 推0.60/0.10 满程0.60/0.45 gain1.0": ["--move-top","0.60","--pitch-top","0.45","--move-push","0.60","--hold-push","0.10","--gain","1.0"],
}
print("%-38s %7s %8s %8s %8s" % ("配置","命中","中位","最差","抖动"))
for name, extra in runs.items():
    job = subprocess.run([py, "tools/wire_red_gaze.py", "--table", r"artifacts\红球位置野_红_大.json",
                          "--motor-time","0.20","--steps","300","--tail","60","--out","artifacts/_对照_%s.json" % name.split()[0]] + extra,
                         capture_output=True, text=True, encoding="utf-8")
    if job.returncode:
        print("%-38s 出错：%s" % (name, job.stdout.strip().splitlines()[-1:])); continue
    hits = re.search(r"命中 (\d+)/(\d+)", job.stdout)
    med = re.search(r"误差：中位 ([\d.]+)、平均 [\d.]+、最差 ([\d.]+)", job.stdout)
    shake = re.search(r"后段抖动：最大 ([\d.]+)", job.stdout)
    print("%-38s %7s %8s %8s %8s" % (name, "%s/%s"%hits.groups(), med.group(1), med.group(2), shake.group(1)))
    open("logs/眼_对照_%s.log" % name.split()[0], "w", encoding="utf-8").write(job.stdout + job.stderr)