import io
for p in (r"F:\born-wired-cortex\engine_v2\born_wired\embodied.py",
          r"F:\born-wired-cortex\engine_v2\tools\wire_red_gaze.py"):
    s = io.open(p, encoding="utf-8", newline="").read()
    print(p, "CRLF", s.count("\r\n"), "LF", s.count("\n") - s.count("\r\n"))