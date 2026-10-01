# -*- coding: utf-8 -*-
"""从上一批（二批）的台账里挑出「会追球」的当这一批的起点。"""
import io, json, glob, os

d = r"F:\born-wired-cortex\engine_v2\artifacts\云端\二批"
rows = []
for path in sorted(glob.glob(os.path.join(d, "追球_云_二批_g0[0-7].jsonl"))):
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
print("台账里建出来的：", len(rows))
if rows:
    keys = sorted(rows[0].keys())
    print("一行有哪些字段：", keys)
    print("样例：", {k: rows[0][k] for k in ("index", "upright", "flee_covered", "flee_trials",
                                          "flee_settled", "flee_kept", "caught", "covered", "trials")})