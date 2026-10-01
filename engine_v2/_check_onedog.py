import io, json, sys
from pathlib import Path
ROOT = Path(r"F:\born-wired-cortex\engine_v2")
for extra in (ROOT, ROOT/"tools"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))
import tools.chase_task as ct
rows = [json.loads(l) for l in io.open(ROOT/"artifacts"/"追球_批03_g00.jsonl", encoding="utf-8") if l.strip()]
for want in (51, 52):
    row = [r for r in rows if r.get("index") == want and r.get("status") == "ok"]
    if not row:
        print("第 %d 只不在台账里" % want); continue
    row = row[0]
    print("==== 第 %d 只（旧判分：最近贴到 %.2f 米） 追球路 %s%s ====" %
          (want, row.get("best", float("nan")), row["chase"]["route"], " 对调" if row["chase"]["flip"] else ""))
    got = ct.exam(row["genome"], row["seed"], seconds=6, chase=row["chase"], loud=True)
    print("  -> 到了 %d/%d 趟，最近 %.2f 米，走了 %.2f 米，没摔 %s"
          % (got["covered"], got["trials"], got["best"], got["travelled_m"], got["upright"]))