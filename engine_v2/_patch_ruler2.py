# -*- coding: utf-8 -*-
import io
p = r"F:\born-wired-cortex\engine_v2\tools\wire_red_gaze.py"
s = io.open(p, encoding="utf-8", newline="").read()
old = '''    print("\u6bcf\u53ea\u773c\u7684\u753b\u9762\uff1a\u6c34\u5e73\u534a\u89d2 %.3f \u5f27\u5ea6\u3001\u7ad6\u76f4\u534a\u89d2 %.3f \u5f27\u5ea6\uff1b"
          "\u5de6\u773c\u884c\u7a0b yaw %+.2f~%+.2f\u3001pitch %+.2f~%+.2f"
          % (view[0], view[1]) + " \u5f27\u5ea6" % () if False else
          "\u6bcf\u53ea\u773c\u7684\u753b\u9762\uff1a\u6c34\u5e73\u534a\u89d2 %.3f\u3001\u7ad6\u76f4\u534a\u89d2 %.3f \u5f27\u5ea6\uff1b\u5de6\u773c\u884c\u7a0b %s"
          % (view[0], view[1], np.round(np.asarray(eye_reach(body, 0)), 2)))
'''
new = '''    print("\u6bcf\u53ea\u773c\u7684\u753b\u9762\uff1a\u6c34\u5e73\u534a\u89d2 %.3f\u3001\u7ad6\u76f4\u534a\u89d2 %.3f \u5f27\u5ea6\uff1b\u5de6\u773c\u884c\u7a0b %s"
          % (view[0], view[1], np.round(np.asarray(eye_reach(body, 0)), 2)))
'''
assert s.count(old) == 1, "A"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")