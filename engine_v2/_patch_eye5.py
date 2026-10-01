# -*- coding: utf-8 -*-
import io
p = r"F:\born-wired-cortex\engine_v2\tools\wire_red_gaze.py"
s = io.open(p, encoding="utf-8", newline="").read()
def swap(old, new, tag):
    global s
    assert s.count(old) == 1, "MISS " + tag
    s = s.replace(old, new)
swap('''    for name in MUTED:
        if name in parameters:
            parameters[name] = SILENT
''', '''    for name in MUTED:
        # \u65e0\u6761\u4ef6\u8986\u76d6\uff1a\u53ea\u6539\u300c\u53c2\u6570\u8868\u91cc\u5df2\u6709\u7684\u90a3\u51e0\u9879\u300d\u662f\u4e0d\u591f\u7684 \u2014\u2014 2026-09-30 \u5b9e\u6d4b\uff0c
        # \u5148\u5929\u7684\u773c\u52a8\u901a\u8def\uff08\u5f80\u54ea\u770b\u3001\u4f1a\u805a\u3001\u6761\u5e26\u53cd\u5c04\uff09\u56e0\u4e3a\u6ca1\u88ab\u6539\u5230\uff0c\u4e00\u76f4\u5728\u63a8\u773c\u808c\uff0c
        # \u628a\u7ea2\u7403\u90a3\u4e00\u8def\u7684\u8bfb\u6570\u5168\u90e8\u6df9\u6ca1\u4e86\u3002
        parameters[name] = SILENT
''', "mute")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")