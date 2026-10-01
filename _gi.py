from pathlib import Path
p = Path(".gitignore")
add = """
# engine_v2：手改备份 / 缓存
*.bak*
engine_v2/**/__pycache__/

# engine_v2 的超大台账（单文件几十 MB）：本机留档，不上 GitHub。
# 需要的话用 Git LFS 或发 Release 传。
engine_v2/artifacts/轮6_全库24*.jsonl
engine_v2/artifacts/轮5_筛选.jsonl
engine_v2/artifacts/轮5_筛选_part*.jsonl
engine_v2/artifacts/_轮6_临时8只.jsonl
"""
s = p.read_text(encoding="utf-8")
if "engine_v2/artifacts/轮6_全库24*.jsonl" in s:
    print("已经加过了")
else:
    p.write_text(s.rstrip("\n") + "\n" + add, encoding="utf-8", newline="\n")
    print("已追加")
