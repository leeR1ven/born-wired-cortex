# -*- coding: utf-8 -*-
"""写一个轻量页面：四段 GIF 放在一起，自动循环播放（不塞 base64，秒开）。"""
import pathlib
根 = pathlib.Path(__file__).resolve().parent
幕 = [("A", "A  趴着，不给画面", "只靠身体感觉觉出自己趴着 → 出生就有的本能把它撑起来；站定之后一步没走。末高 0.260 米。"),
      ("B", "B  站着，正前方一片红", "眼睛看到红 → 视觉那一半 + 前额叶那一半，两股电流加起来过阈值 → 往前走。"),
      ("C", "C  只看一眼红，半秒后就把红撤掉", "红只亮了半秒，眼前全黑了它还在走 —— 前额叶里那点“想”没有跟着刺激一起消失（论文 R8）。5 秒走了 2.38 米。"),
      ("D", "D  什么都不给", "连身体感觉都不给 → 肌肉松掉，瘫下去，末高 0.184 米、歪 41 度。说明动作是皮层下的令，不是物理自己动。")]
块 = []
for 键, 名, 注 in 幕:
    块.append('<div class="格">\n  <h3>' + 名 + '</h3>\n'
              + '  <img src="回放_大脑_' + 键 + '.gif" alt="' + 名 + '">\n'
              + '  <p>' + 注 + '</p>\n</div>')
html = """<!doctype html>
<meta charset="utf-8"><title>大脑自己开机器人 —— 四幕</title>
<style>
body{margin:0;background:#141418;color:#e8e8ee;font:15px/1.7 "Microsoft YaHei",sans-serif}
h1{font-size:20px;text-align:center;margin:18px 0 4px}
.说明{text-align:center;color:#a9a9b6;font-size:13px;margin:0 0 16px}
.网格{display:grid;grid-template-columns:1fr 1fr;gap:14px;max-width:1500px;margin:0 auto 30px;padding:0 14px}
.格{background:#1d1d24;border-radius:10px;padding:10px 12px 14px}
.格 h3{font-size:15px;margin:4px 0 8px;color:#ffd479}
.格 img{width:100%;border-radius:6px;display:block}
.格 p{font-size:13px;color:#b9b9c6;margin:9px 0 0}
</style>
<h1>大脑自己开机器人 —— 四幕</h1>
<p class="说明">每一幕都从<b>刚出生的同一颗脑子</b>开始（同一个种子、同一套出生连线），只换外界信号。<br>
代码里没有任何一行直接摆姿势：身体怎么动，完全看皮层发出什么。</p>
<div class="网格">
__块__
</div>
"""
html = html.replace("__块__", "\n".join(块))
p = 根.parent / "playback" / "回放_大脑_四幕.html"
p.write_text(html, encoding="utf-8", newline="\n")
print("写好", p.name, "%.1f KB" % (p.stat().st_size / 1024))