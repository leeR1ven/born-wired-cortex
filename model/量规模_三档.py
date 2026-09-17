import pathlib
# -*- coding: utf-8 -*-
"""三个宽度各量一遍：建网多久、一拍多久（不带学习 / 带学习）、内存。"""
import os, subprocess, sys
py = sys.executable
根 = str(pathlib.Path(__file__).resolve().parent)
任务 = [("列16", {}),
        ("列32", {"AGI数据后缀": "_列32", "AGI列数": "32"}),
        ("列192", {"AGI数据后缀": "_列192", "AGI列数": "192"})]
for 名, 环境 in 任务:
    env = dict(os.environ); env.update(环境)
    print("=" * 70)
    print("=== %s ===" % 名, flush=True)
    r = subprocess.run([py, "-X", "utf8", "规模试验_测量.py"], cwd=根, env=env,
                       capture_output=True)
    sys.stdout.write(r.stdout.decode("utf-8", "replace"))
    err = r.stderr.decode("utf-8", "replace")
    if err.strip():
        print("STDERR:\n" + err[-2000:])
    sys.stdout.flush()
print("三个都量完了")