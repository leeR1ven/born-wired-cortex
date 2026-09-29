"""Compare two audit files task by task: what moved, and by how much.

    python tools/compare_audits.py --before artifacts/审计_出生动物_无返回线_before_returnline.jsonl \
        --after artifacts/审计_出生动物_返回线_s0.jsonl \
        --json artifacts/_对照_返回线前后.json

Two audits of the *same* animal differ for one of three reasons: a question was
added or taken away, a change to the wiring moved a reading, or the machine was
busier the second time.  Only the second is a result, so this tool prints the
first and third separately from it:

  * ``added``   - in the new file only
  * ``gone``    - in the old file only
  * ``moved``   - the reading the bar looks at changed by more than --tolerance
  * ``flipped`` - pass/fail changed

A reading that moved by less than the tolerance is not a claim that nothing
happened; it is the resolution of this comparison.  5e-4 is the default, which
is below the smallest reading any bar in the bank compares against.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import taskbank                                         # noqa: E402


def read(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["task"]] = row
    return rows


def reading(task, row):
    """The number the bar of ``task`` looks at, if it can be read back."""
    recipe = taskbank.MARGIN.get(task)
    measures = row.get("measures") or {}
    if not recipe or measures.get("status") != "ok":
        return None
    if "path" in recipe:
        value = measures.get(recipe["path"])
        return None if value is None else float(value)
    rule = recipe.get("rule")
    if rule is None:
        return None
    try:
        return float(rule(measures))
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="what moved between two audits")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    before, after = read(args.before), read(args.after)
    added = [name for name in after if name not in before]
    gone = [name for name in before if name not in after]
    moved, flipped, same, unreadable = [], [], [], []
    for name in sorted(set(before) & set(after)):
        old, new = before[name], after[name]
        if bool(old["passed"]) != bool(new["passed"]):
            flipped.append((name, bool(old["passed"]), bool(new["passed"])))
        a, b = reading(name, old), reading(name, new)
        if a is None or b is None:
            unreadable.append(name)
            continue
        if abs(b - a) > args.tolerance:
            moved.append((name, a, b, b - a))
        else:
            same.append(name)

    print("questions: %d before, %d after, %d in both"
          % (len(before), len(after), len(set(before) & set(after))))
    print("added: %d %s" % (len(added), added or ""))
    print("gone:  %d %s" % (len(gone), gone or ""))
    print("pass/fail changed: %d" % len(flipped))
    for name, a, b in flipped:
        print("   FLIP %-46s %s -> %s" % (name, "pass" if a else "FAIL",
                                          "pass" if b else "FAIL"))
    print("reading moved by more than %g: %d" % (args.tolerance, len(moved)))
    for name, a, b, delta in moved:
        print("   %-46s %.6f -> %.6f  (%+.6f)" % (name, a, b, delta))
    print("reading unmoved (or unreadable): %d, of which unreadable %d"
          % (len(same) + len(unreadable), len(unreadable)))
    if unreadable:
        print("   unreadable (no margin recipe, or the run failed):")
        for name in unreadable:
            print("      %s" % name)

    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            before=args.before, after=args.after, tolerance=args.tolerance,
            added=added, gone=gone,
            flipped=[dict(task=name, before_from=a, after_to=b) for name, a, b in flipped],
            moved=[dict(task=name, before=a, after=b, delta=d) for name, a, b, d in moved],
            unmoved=sorted(same), unreadable=sorted(unreadable)),
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())