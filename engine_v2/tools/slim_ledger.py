"""A ledger small enough to keep in the repository, without losing the answers.

    python tools/slim_ledger.py artifacts/轮5_筛选.jsonl
    python tools/slim_ledger.py artifacts/*.jsonl --out-dir artifacts/slim
    python tools/slim_ledger.py artifacts/轮5_筛选.jsonl --verify

A screening row carries every reading the questions looked at, and the per-cell
dumps under ``parts`` are most of its size: one candidate is 550 KB, of which
537 KB is the feature-layer question's two full readings.  The dumps are worth
having while a round is running and worth nothing afterwards - what a later
reader wants is which questions the animal passed, what the bar said, and how
long it took.

So this writes the same rows with every ``parts`` key removed, compactly.  The
slim file is a different file: **the full ledger stays where it is and stays the
evidence.**  Regenerate the full one from the seeds in the round's own command;
the rows here are the same answer, read without the arrays.

``--verify`` reads both files back and refuses to write unless every row agrees
on label, passed_tasks, n_passed and first_miss - the check that "slim" did not
quietly turn into "different".
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def strip(value):
    """The same object with every ``parts`` key removed, at any depth."""
    if isinstance(value, dict):
        return {name: strip(inside) for name, inside in value.items() if name != "parts"}
    if isinstance(value, list):
        return [strip(inside) for inside in value]
    return value


def rows_of(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


IDENTITY = ("label", "scaffold", "passed_tasks", "n_passed", "n_tasks",
            "n_tasks_run", "first_miss", "complete", "n_errors")


def verify(before, after, path):
    if len(before) != len(after):
        raise SystemExit("%s: %d rows before, %d after" % (path, len(before), len(after)))
    for old, new in zip(before, after):
        for name in IDENTITY:
            if old.get(name) != new.get(name):
                raise SystemExit("%s: row %s disagrees on %s: %r vs %r"
                                 % (path, old.get("label"), name,
                                    old.get(name), new.get(name)))


def slim(path, out_dir=None, verify_=False):
    rows = rows_of(path)
    slim_rows = [strip(row) for row in rows]
    if verify_:
        verify(rows, slim_rows, path)
    source = Path(path)
    target = (Path(out_dir) / (source.stem + "_slim.jsonl")) if out_dir else \
        source.with_name(source.stem + "_slim.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in slim_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print("%-44s %8.2f MB -> %-44s %6.3f MB  (%d rows%s)"
          % (source.name, source.stat().st_size / 1e6, target.name,
             target.stat().st_size / 1e6, len(rows),
             ", verified" if verify_ else ""))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ledgers", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the slim files go; default is beside the original")
    parser.add_argument("--verify", action="store_true",
                        help="refuse to write unless every row answers the same")
    args = parser.parse_args(argv)
    for path in args.ledgers:
        slim(path, args.out_dir, args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
