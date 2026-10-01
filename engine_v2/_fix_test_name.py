from pathlib import Path
p = Path("tests/test_eye_geometry.py")
t = p.read_text(encoding="utf-8", newline="")
t = t.replace("    def body(self, name):\n", "    def body_id(self, name):\n")
t = t.replace('self.body("', 'self.body_id("')
p.write_text(t, encoding="utf-8", newline="")
print("ok")