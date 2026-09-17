import pathlib
# -*- coding: utf-8 -*-
"""把 R11 那张表按新代码重测一遍（7 个宽度），每个宽度写一份日志。"""
import os, shutil, subprocess, sys
py = sys.executable
根 = str(pathlib.Path(__file__).resolve().parent)
for 宽 in (16, 32, 48, 64, 96, 128, 192):
    日志 = os.path.join(str(pathlib.Path(__file__).resolve().parent.parent / "logs"), "规模试验_列%d.log" % 宽)
    if os.path.exists(日志) and not os.path.exists(日志 + ".bak_优化前"):
        shutil.copyfile(日志, 日志 + ".bak_优化前")
    env = dict(os.environ)
    if 宽 != 16:
        env["AGI列数"] = str(宽)
        env["AGI数据后缀"] = "_列%d" % 宽
    print("=== 列 %d ===" % 宽, flush=True)
    r = subprocess.run([py, "-X", "utf8", "规模试验_测量.py"], cwd=根, env=env,
                       capture_output=True)
    出 = r.stdout.decode("utf-8", "replace")
    err = r.stderr.decode("utf-8", "replace")
    open(日志, "w", encoding="utf-8", newline="\n").write(出 + (("\nSTDERR:\n" + err) if err.strip() else ""))
    sys.stdout.write(出)
    sys.stdout.flush()
print("七个宽度都量完了")