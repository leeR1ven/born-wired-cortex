"""Validate the freshly downloaded corpus and count usable characters.

The first file was fetched with a Range resume, so every line has to be parsed
to prove the seam did not corrupt anything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw_modelscope'

FILES = (
    ('caoaolong__zhwiki__zhwiki_dataset.jsonl', ('title', 'content')),
    ('AI-ModelScope__wikipedia-cn-20230720-filtered__wikipedia-cn-20230720-filtered.jsonl', ('completion',)),
    ('Illusionna__wiki-zh-521MB__local_train.txt', ('sentences',)),
)


def text_of(doc, keys):
    parts = []
    for key in keys:
        value = doc.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return ''.join(parts)


def main():
    report = {}
    for name, keys in FILES:
        path = RAW / name
        started = time.perf_counter()
        lines = bad = chars = 0
        with path.open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                lines += 1
                try:
                    doc = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                chars += len(text_of(doc, keys))
        entry = dict(file=name, lines=lines, unparsable_lines=bad, characters=chars,
                     seconds=round(time.perf_counter() - started, 1))
        report[name] = entry
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    print(json.dumps(dict(total_characters=sum(e['characters'] for e in report.values())),
                     ensure_ascii=False), flush=True)
    (HERE / 'raw_corpus_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()