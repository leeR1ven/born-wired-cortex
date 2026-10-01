# -*- coding: utf-8 -*-
import io
p = r"F:\born-wired-cortex\engine_v2\tools\wire_red_gaze.py"
s = io.open(p, encoding="utf-8", newline="").read()
def swap(old, new, tag):
    global s
    assert s.count(old) == 1, "MISS " + tag
    s = s.replace(old, new)
swap('def view_half_angles(body):', 'def view_half_angles(body, brain):', "sig")
swap('    wide = float(np.arctan(np.tan(half)*float(body.eye_width)/float(body.eye_height)))',
     '    wide = float(np.arctan(np.tan(half)*float(brain.eye_width)/float(brain.eye_height)))', "body")
swap('    view = view_half_angles(body)', '    view = view_half_angles(body, brain)', "call")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")