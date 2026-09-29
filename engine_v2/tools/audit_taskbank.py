"""Run one animal over the whole question bank and write every answer to disk.

    python tools/audit_taskbank.py --label birth --seeds 0 --output artifacts/审计_出生动物.jsonl
    python tools/audit_taskbank.py --label s0-0059 --genome artifacts/基因组_s0_0059.json \
        --seeds 0,1,2 --output artifacts/审计_s0_0059.jsonl

A screening round records only pass/fail over a handful of questions, which is
not enough to see *why* a question failed, how far from the bar it was, or
whether a question is answerable at all.  This tool writes the whole reading of
every question - the ones it passed included - and appends one line the moment
a question ends, so a stopped run resumes where it stopped instead of redoing
the bank.  ``--scan`` reads such a file back and flags the three ways a
question can be broken:

  * dead    the reading the bar looks at is exactly zero, so nothing lights up
  * knife   the reading sits within 10% of the bar
  * error   the question raised instead of answering
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import taskbank                                      # noqa: E402


def readings(measures):
    """Every number the bar could have read, flattened to dotted names."""
    out = {}

    def walk(prefix, value):
        if isinstance(value, (bool, np.bool_)):
            out[prefix] = bool(value)
        elif isinstance(value, (int, float, np.floating, np.integer)):
            out[prefix] = float(value)
        elif isinstance(value, (list, tuple)):
            numbers = [v for v in value
                       if isinstance(v, (int, float, np.floating, np.integer))]
            if numbers and len(numbers) == len(value):
                out[prefix] = float(np.mean(numbers))
            else:
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        walk("%s[%d]" % (prefix, index), item)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk("%s.%s" % (prefix, key) if prefix else str(key), item)

    for key, value in (measures or {}).items():
        walk(str(key), value)
    return out


def write_line(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def read_lines(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def label_of(genome):
    return "-".join("%s%s" % (key.split("_")[0][:4], int(value * 1000))
                    for key, value in sorted(genome.items())[:4])


def run(args):
    genome = {}
    if args.genome:
        genome = json.loads(Path(args.genome).read_text(encoding="utf-8"))
    if isinstance(genome, dict) and isinstance(genome.get("genome"), dict):
        genome = genome["genome"]
    genome = {key: float(value) for key, value in genome.items()}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # the label is part of the key: one file may hold several animals, and a
    # row that belongs to another one must not count as this one's answer
    done = {(row["label"], row["seed"], row["task"]) for row in read_lines(output)}

    entries = taskbank.stage_tasks(args.stage)
    if args.names:
        wanted = {name.strip() for name in args.names.split(",") if name.strip()}
        entries = [entry for entry in entries if entry["name"] in wanted]

    for seed in [int(part) for part in args.seeds.split(",") if part.strip()]:
        for entry in entries:
            if (args.label, seed, entry["name"]) in done:
                continue
            result = taskbank.run_task(entry, genome, seed)
            write_line(output, dict(label=args.label, seed=seed, task=result["task"],
                                    revision=taskbank.revision(),
                                    group=result["group"], passed=result["passed"],
                                    why=result["why"],
                                    wall_seconds=result["wall_seconds"],
                                    readings=readings(result["measures"]),
                                    measures=result["measures"]))
            print("    %-8s %-42s %-4s %6.1fs  %s"
                  % (args.label, result["task"], "pass" if result["passed"] else "FAIL",
                     result["wall_seconds"], (result["why"] or "")[:70]), flush=True)

    rows = [row for row in read_lines(output) if row["label"] == args.label]
    passed = sum(1 for row in rows if row["passed"])
    seconds = sum(row["wall_seconds"] for row in rows)
    print("%s: %d/%d passed, %.1f min of simulation, %s"
          % (args.label, passed, len(rows), seconds / 60., output), flush=True)
    return rows


def scan(args):
    """Read an audit file back and list what looks wrong with the questions."""
    rows = read_lines(args.scan)
    per_task = {}
    for row in rows:
        per_task.setdefault(row["task"], []).append(row)
    dead, knife, errors, free = [], [], [], []
    for name, group in sorted(per_task.items()):
        for row in group:
            if row["measures"].get("status") == "error":
                errors.append((name, row["why"]))
                continue
            numbers = [value for key, value in row["readings"].items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if "gain" in row["readings"] and numbers and max(numbers) == 0. and not row["passed"]:
                dead.append((name, row["why"]))
            if not row["passed"]:
                knife.append((name, row["why"]))
    print("questions run:", len(per_task), " runs:", len(rows))
    print("errors:", len(errors))
    for name, why in errors[:20]:
        print("   ERROR", name, why[:100])
    print("failed and read all-zero:", len(dead))
    for name, why in dead[:40]:
        print("   DEAD ", name, why[:110])
    print("failed (see why):", len(knife))
    for name, why in knife[:80]:
        print("   FAIL ", name, why[:110])


def main(argv=None):
    parser = argparse.ArgumentParser(description="run every question and keep the readings")
    parser.add_argument("--label", default="animal")
    parser.add_argument("--genome", default=None,
                        help="json file of named gains; empty means the animal as built")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--stage", default="full", choices=("screen", "probe", "full"))
    parser.add_argument("--names", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scan", default=None, help="instead of running, read this file back")
    args = parser.parse_args(argv)
    if args.scan:
        scan(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
