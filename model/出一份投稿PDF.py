# -*- coding: utf-8 -*-
"""出一份投稿PDF.py —— 把新版论文渲染成一份可以直接上传投稿系统的 PDF。

为什么要有这个：论文是 Markdown，投稿系统不收 Markdown。这个脚本把正文 + 10 张图
排成 A4 打印版，再调用本机 Chrome 的 headless 模式打成 PDF。
不需要联网，不需要装 LaTeX。

命令：python 出一份投稿PDF.py
产物：投稿_20260917\论文_Born_wired_v20260915.pdf
"""
from __future__ import annotations

import base64
import io
import os
import pathlib
import re
import subprocess
import sys
import tempfile

from markdown_it import MarkdownIt

根 = pathlib.Path(__file__).resolve().parent
论文目录 = 根.parent / "paper"
论文 = 论文目录 / "Born_wired_manuscript.md"
图目录 = 论文目录 / "figures"
出目录 = 论文目录 / "submission"

图们 = [
    "figure1_system_overview.png",
    "figure2_closed_loop.png",
    "figure3_prefrontal_ablation.png",
    "figure4_staged_repertoire.png",
    "figure5_gradient_recognition.png",
    "figure6_compression_cost.png",
    "figure7_emergent_gaze.png",
    "figure8_blurred_wiring.png",
    "figure9_recurrence.png",
    "figure10_development.png",
]

样式 = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Times New Roman", Georgia, "Songti SC", "SimSun", serif;
       font-size: 10pt; line-height: 1.40; color: #000; margin: 0; }
h1 { font-size: 15pt; line-height: 1.28; margin: 0 0 .5em 0; }
h2 { font-size: 12.5pt; margin: 1.3em 0 .45em 0; page-break-after: avoid;
     border-bottom: 1px solid #999; padding-bottom: 2px; }
h3 { font-size: 11pt; margin: 1.1em 0 .35em 0; page-break-after: avoid; color: #000; }
p, li { orphans: 2; widows: 2; }
ul, ol { margin: .4em 0 .6em 0; padding-left: 1.35em; }
table { border-collapse: collapse; width: 100%; margin: .6em 0 .9em 0; font-size: 7.6pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #888; padding: 2px 4px; text-align: left; vertical-align: top; }
th { background: #eee; }
tr { page-break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: .87em; }
pre { background: #f4f4f4; padding: 6px; white-space: pre-wrap; word-break: break-all;
      font-size: 8pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 2px solid #999; margin: .5em 0; padding-left: .7em; color: #222; }
hr { border: none; border-top: 1px solid #bbb; margin: 1.2em 0; }
figure.fig { margin: .9em 0 1.3em 0; text-align: center; page-break-inside: avoid; }
figure.fig img { max-width: 100%; max-height: 165mm; }
figcaption { font-size: 8pt; color: #333; margin-top: 3px; text-align: left; }
"""


def 嵌图(html: str) -> str:
    for i, 名 in enumerate(图们, 1):
        p = 图目录 / 名
        if not p.exists():
            print("  !! 缺图", 名)
            continue
        键 = "Figure %d " % i
        对 = re.compile(r"<li>(?:(?!</li>).)*?" + re.escape(键) + r"(?:(?!</li>).)*?</li>", re.S)
        m = 对.search(html)
        if not m:
            print("  !! 正文里没找到", 键)
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        块 = ('<figure class="fig"><img alt="%s" src="%s%s">'
              '<figcaption>%s</figcaption></figure>'
              % (名, "data:image/png;base64,", b64, 键.strip()))
        html = html[:m.end()] + "\n" + 块 + html[m.end():]
    return html


def 找浏览器():
    候选 = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in 候选:
        if pathlib.Path(c).exists():
            return c
    raise SystemExit("找不到 Chrome 或 Edge，没法打 PDF")


def main():
    if not 论文.exists():
        raise SystemExit("找不到论文：%s" % 论文)
    出目录.mkdir(parents=True, exist_ok=True)
    源 = io.open(论文, encoding="utf-8").read()
    md = MarkdownIt("commonmark", {"html": True, "linkify": False})
    md.enable("table")
    md.enable("strikethrough")
    正文 = 嵌图(md.render(源))
    页 = ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
          '<title>Born wired - submission manuscript</title><style>%s</style>'
          '</head><body>\n%s\n</body></html>\n' % (样式, 正文))

    临时 = pathlib.Path(tempfile.gettempdir()) / "agi_job"
    临时.mkdir(parents=True, exist_ok=True)
    h = 临时 / "paper_print.html"
    io.open(h, "w", encoding="utf-8", newline="\n").write(页)

    出 = 出目录 / "论文_Born_wired_v20260915.pdf"
    档 = 临时 / "chrome_profile"
    浏览 = 找浏览器()
    cmd = [浏览, "--headless=new", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--virtual-time-budget=20000",
           "--user-data-dir=%s" % 档,
           "--print-to-pdf=%s" % 出,
           h.as_uri()]
    print("调用", pathlib.Path(浏览).name, "打印 PDF ...")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if not 出.exists():
        print("STDOUT:", r.stdout[-2000:])
        print("STDERR:", r.stderr[-2000:])
        raise SystemExit("PDF 没生成")
    print("写好 %s （%.2f MB）" % (出, 出.stat().st_size / 1048576.0))


if __name__ == "__main__":
    main()