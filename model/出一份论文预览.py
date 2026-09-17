# -*- coding: utf-8 -*-
"""出一份论文预览.py —— 把新版论文渲染成一个自带图片的 HTML，双击就能看。

为什么要有这个：论文是 Markdown，直接看要装工具；这个脚本把正文 + 10 张图
全部塞进一个 html 文件里（图片用 base64 嵌进去），换台电脑、发给别人都能看。

命令：python 出一份论文预览.py
"""
from __future__ import annotations

import base64
import io
import pathlib
import re

from markdown_it import MarkdownIt

根 = pathlib.Path(__file__).resolve().parent
论文目录 = 根.parent / "paper"
论文 = 论文目录 / "Born_wired_manuscript.md"
图目录 = 论文目录 / "figures"
出 = 论文目录 / "论文预览.html"

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
body { max-width: 940px; margin: 2.5rem auto; padding: 0 1.2rem;
       font-family: Georgia, "Times New Roman", "Songti SC", "SimSun", serif;
       line-height: 1.65; color: #1a1a1a; }
h1 { font-size: 1.8rem; line-height: 1.3; border-bottom: 2px solid #333; padding-bottom: .4rem; }
h2 { font-size: 1.35rem; margin-top: 2.2rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }
h3 { font-size: 1.1rem; margin-top: 1.8rem; color: #14507a; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .86rem; }
th, td { border: 1px solid #bbb; padding: .35rem .5rem; text-align: left; vertical-align: top; }
th { background: #eef3f8; }
code { font-family: Consolas, "Courier New", monospace; font-size: .88em;
       background: #f2f2f2; padding: .08em .3em; border-radius: 3px; }
pre code { display: block; padding: .7rem; overflow-x: auto; background: #f6f6f6; }
blockquote { border-left: 3px solid #9bb8d3; margin-left: 0; padding-left: .9rem; color: #333; }
figure.fig { margin: 1.1rem 0 1.8rem 0; text-align: center; }
figure.fig img { max-width: 100%; border: 1px solid #ddd; }
figcaption { font-size: .8rem; color: #555; margin-top: .35rem; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
"""


def 嵌图(html: str) -> str:
    for i, 名 in enumerate(图们, 1):
        p = 图目录 / 名
        if not p.exists():
            continue
        键 = "Figure %d —" % i
        对 = re.compile(r"<li>(?:(?!</li>).)*?" + re.escape(键) + r"(?:(?!</li>).)*?</li>", re.S)
        m = 对.search(html)
        if not m:
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        块 = ('<figure class="fig"><img alt="%s" src="data:image/png;base64,%s">'
              '<figcaption>%s</figcaption></figure>' % (名, b64, 键[:-2]))
        html = html[:m.end()] + "\n" + 块 + html[m.end():]
    return html


def main():
    if not 论文.exists():
        raise SystemExit("找不到论文：%s" % 论文)
    src = io.open(论文, encoding="utf-8").read()
    md = MarkdownIt("commonmark", {"html": True, "linkify": False})
    md.enable("table")
    md.enable("strikethrough")
    正文 = 嵌图(md.render(src))
    页 = ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
          "<title>Born wired - manuscript</title><style>%s</style></head><body>\n%s\n</body></html>\n"
          % (样式, 正文))
    io.open(出, "w", encoding="utf-8", newline="\n").write(页)
    print("写好", 出, "（%.1f KB）" % (len(页.encode("utf-8")) / 1024.0))


if __name__ == "__main__":
    main()