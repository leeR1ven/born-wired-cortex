# -*- coding: utf-8 -*-
"""出一份PDF.py —— 把论文 Markdown 排成 PDF（不依赖 pandoc / LaTeX）。

Markdown -> HTML（python-markdown + tables 扩展）-> 无头 Chrome 打印成 PDF。
字体用 Times New Roman，正文里的中文（脚本名、日志名）回退到宋体/雅黑。

用法：
    python 出一份PDF.py                                  # 排 paper/Born_wired_manuscript.md
    python 出一份PDF.py 输入.md 输出.pdf                  # 换文件
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

import markdown

根 = pathlib.Path(__file__).resolve().parent
仓库根 = 根.parent if (根.parent / "paper").is_dir() else 根

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
html { -webkit-print-color-adjust: exact; }
body { font-family: "Times New Roman", "Nimbus Roman", "SimSun", serif;
       font-size: 10.4pt; line-height: 1.46; color: #000; margin: 0; }
h1 { font-size: 16pt; line-height: 1.25; margin: 0 0 10px 0; }
h2 { font-size: 12.6pt; margin: 18px 0 6px 0; padding-bottom: 2px;
     border-bottom: 1px solid #999; break-after: avoid; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 13px 0 5px 0; break-after: avoid; page-break-after: avoid; }
p { margin: 6px 0; text-align: justify; }
ul, ol { margin: 6px 0 6px 20px; padding: 0; }
li { margin: 3px 0; text-align: justify; }
li > ul, li > ol { margin: 3px 0 3px 16px; }
strong { font-weight: bold; }
em { font-style: italic; }
hr { border: none; border-top: 1px solid #ccc; margin: 12px 0; }
code { font-family: Consolas, "Courier New", "Microsoft YaHei", monospace; font-size: 9pt; }
pre { font-family: Consolas, "Courier New", "Microsoft YaHei", monospace; font-size: 8.6pt;
      background: #f6f6f6; border: 1px solid #ddd; padding: 6px 8px; white-space: pre-wrap;
      break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.4pt; }
th, td { border: 1px solid #b8b8b8; padding: 2px 5px; vertical-align: top; text-align: left; }
th { background: #f0f0f0; }
tr { break-inside: avoid; page-break-inside: avoid; }
a { color: #000; text-decoration: none; }
blockquote { margin: 6px 0 6px 14px; padding-left: 8px; border-left: 2px solid #ccc; }
"""

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head>
<body>{body}</body></html>
"""


def 找Chrome():
    for 候选 in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if pathlib.Path(候选).exists():
            return 候选
    return shutil.which("chrome") or shutil.which("msedge")


def 排(输入: pathlib.Path, 输出: pathlib.Path):
    # utf-8-sig：论文文件开头有 BOM，不去掉的话第一行的 "# 标题" 不会被识别成标题。
    正文 = 输入.read_text(encoding="utf-8-sig")
    片段 = markdown.markdown(正文, extensions=["tables", "sane_lists", "fenced_code"],
                            output_format="html5")
    标题 = 正文.splitlines()[0].lstrip("# ").strip() if 正文.strip() else 输入.stem
    工作 = 仓库根 / "paper" / "_build"
    工作.mkdir(parents=True, exist_ok=True)
    页 = 工作 / (输出.stem + ".html")
    页.write_text(HTML.format(title=标题, css=CSS, body=片段), encoding="utf-8")

    Chrome = 找Chrome()
    if Chrome is None:
        print("找不到 Chrome/Edge，HTML 已经写好：", 页)
        return 页, None
    with tempfile.TemporaryDirectory() as 临时:
        参 = [Chrome, "--headless=new", "--disable-gpu", "--no-first-run",
              "--user-data-dir=" + 临时, "--no-pdf-header-footer",
              "--virtual-time-budget=20000",
              "--print-to-pdf=" + str(输出), 页.as_uri()]
        r = subprocess.run(参, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=600)
        好 = 输出.exists() and 输出.stat().st_size > 20000
        print(("%s 打印 %s" % ("✓" if 好 else "✗", 输出)) if 好
              else "✗ Chrome 没打出 PDF（退出码 %s）\n%s\n%s"
                   % (r.returncode, (r.stdout or "")[-1500:], (r.stderr or "")[-1500:]))
    return 页, 输出


if __name__ == "__main__":
    输入 = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else 仓库根 / "paper" / "Born_wired_manuscript.md"
    输出 = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else 输入.with_suffix(".pdf")
    if not 输入.is_absolute():
        输入 = 仓库根 / 输入
    if not 输出.is_absolute():
        输出 = 仓库根 / 输出
    页, 好 = 排(输入, 输出)
    print("HTML：", 页)
    print("PDF ：", 好)
