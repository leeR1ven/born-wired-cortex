import json
d = json.load(open("artifacts/眼睛追球.json", encoding="utf-8"))
rows = [r for r in d["trace"] if r["t"] > 1.2]
print("%5s %8s %10s %10s %8s %10s %8s" % ("t","球方位","左眼要","左眼实际","左偏移","右眼要","右偏移"))
for r in rows[::12]:
    print("%5.2f %8.1f %10.1f %10.1f %8.1f %10.1f %8.1f"
          % (r["t"], __import__("math").degrees(r["bearing"]), __import__("math").degrees(r["left_want_yaw"]),
             __import__("math").degrees(r["left_yaw"]), __import__("math").degrees(r["left_error"]),
             __import__("math").degrees(r["right_want_yaw"]), __import__("math").degrees(r["right_error"])))