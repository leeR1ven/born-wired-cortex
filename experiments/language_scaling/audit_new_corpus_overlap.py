"""How much genuinely new training text do the downloaded files contain?

Same cleaning rule as modern_corpus_v1, same document-hash duplicate rule. For
each new file this reports cleaned characters, characters outside the frozen
19,168-unit vocabulary, and documents already present in the v1 training split.
"""
from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw_modelscope'
V1 = HERE / 'modern_corpus_v1' / 'prepared'

TABLE = {i: None for i in range(0x110000)
         if unicodedata.category(chr(i)).startswith('P')
         or unicodedata.category(chr(i)) in ('Cc', 'Cf') or chr(i).isspace()}

FILES = (
    ('caoaolong__zhwiki__zhwiki_dataset.jsonl', ('content',)),
    ('AI-ModelScope__wikipedia-cn-20230720-filtered__wikipedia-cn-20230720-filtered.jsonl', ('completion',)),
    ('Illusionna__wiki-zh-521MB__local_train.txt', ('sentences',)),
)


def clean(text):
    return text.translate(TABLE)


def turns_of(doc, keys):
    parts = []
    for key in keys:
        value = doc.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return parts


def v1_hashes():
    hashes = set()
    for name in ('dialogue_train.jsonl', 'wiki_train.jsonl'):
        with (V1 / name).open(encoding='utf-8') as handle:
            for line in handle:
                hashes.add(json.loads(line)['sha256'])
    return hashes


def main():
    started = time.perf_counter()
    vocabulary = json.loads((V1 / 'vocabulary.json').read_text(encoding='utf-8'))['vocabulary']
    known = set(vocabulary)
    seen_v1 = v1_hashes()
    seen_new = set()
    report = {}
    added = 0
    for name, keys in FILES:
        t0 = time.perf_counter()
        documents = duplicates = outside = characters = trainable = 0
        with (RAW / name).open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                try:
                    doc = json.loads(line)
                except Exception:
                    continue
                documents += 1
                turns = [t for t in (clean(part) for part in turns_of(doc, keys)) if t]
                text = ''.join(turns)
                if not text:
                    continue
                digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if digest in seen_v1 or digest in seen_new:
                    duplicates += 1
                    continue
                seen_new.add(digest)
                characters += len(text)
                kept = [c for c in text if c in known]
                trainable += len(kept)
                outside += len(text) - len(kept)
        added += trainable
        report[name] = dict(documents=documents, duplicate_documents=duplicates,
                            cleaned_characters=characters, outside_frozen_vocabulary=outside,
                            outside_share=round(outside / characters, 4) if characters else None,
                            trainable_characters=trainable, seconds=round(time.perf_counter() - t0, 1))
        print(json.dumps(report[name], ensure_ascii=False), flush=True)
    report['total_trainable_added'] = added
    report['v1_training_characters'] = 344787049
    report['ratio'] = round((344787049 + added) / 344787049, 3)
    report['seconds'] = round(time.perf_counter() - started, 1)
    print(json.dumps({'total_trainable_added': added, 'v1': 344787049,
                      'ratio': report['ratio']}, ensure_ascii=False), flush=True)
    (HERE / 'new_corpus_overlap_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()