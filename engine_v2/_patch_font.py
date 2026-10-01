# -*- coding: utf-8 -*-
from pathlib import Path
p = Path("tools/track_red_gaze.py")
t = p.read_text(encoding="utf-8", newline="")
old = 'import matplotlib.pyplot as plt                                    # noqa: E402\n'
new = ('import matplotlib.pyplot as plt                                    # noqa: E402\n'
       '# 图上写的都是中文，matplotlib 自带那套字体没有汉字，会画成一排方框。\n'
       'for _name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):\n'
       '    if any(_name == _font.name for _font in matplotlib.font_manager.fontManager.ttflist):\n'
       '        matplotlib.rcParams["font.sans-serif"] = [_name]\n'
       '        break\n'
       'matplotlib.rcParams["axes.unicode_minus"] = False\n')
assert t.count(old) == 1
p.write_text(t.replace(old, new), encoding="utf-8", newline="")
print("字体改好")