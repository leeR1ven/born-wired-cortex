"""Additional non-synthetic Chinese text for the character network.

Cleaning, duplicate rule and vocabulary are identical to modern_corpus_v1. The
19,168-unit vocabulary is frozen, so a document containing a character outside it
is dropped whole rather than edited in place: no character is ever deleted from
the middle of a passage, and the unit set stays the one the published series
used. Output keeps the same document shape as v1 so the existing stream builder
can read it.
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
OUT = HERE / 'modern_corpus_v2'

TABLE = {i: None for i in range(0x110000)
         if unicodedata.category(chr(i)).startswith('P')
         or unicodedata.category(chr(i)) in ('Cc', 'Cf') or chr(i).isspace()}

SOURCES = (
    ('zhwiki', 'caoaolong__zhwiki__zhwiki_dataset.jsonl', ('content',)),
    ('wikicn', 'AI-ModelScope__wikipedia-cn-20230720-filtered__wikipedia-cn-20230720-filtered.jsonl', ('completion',)),
    ('wikitxt', 'Illusionna__wiki-zh-521MB__local_train.txt', ('sentences',)),
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


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite a prepared corpus')
    started = time.perf_counter()
    vocabulary = json.loads((V1 / 'vocabulary.json').read_text(encoding='utf-8'))['vocabulary']
    known = set(vocabulary)
    seen = set()
    for name in ('dialogue_train.jsonl', 'wiki_train.jsonl'):
        with (V1 / name).open(encoding='utf-8') as handle:
            for line in handle:
                seen.add(json.loads(line)['sha256'])
    print(json.dumps({'v1_training_documents': len(seen), 'vocabulary': len(vocabulary)}), flush=True)
    (OUT / 'extra').mkdir(parents=True)
    stat = {}
    hashes = {}
    for key, filename, keys in SOURCES:
        counts = dict(documents=0, duplicate_documents=0, dropped_for_unknown_character=0,
                      kept_documents=0, characters=0)
        digest = hashlib.sha256()
        with (RAW / filename).open(encoding='utf-8', errors='replace') as handle, \
                (OUT / 'extra' / f'{key}.jsonl').open('w', encoding='utf-8', newline='\n') as out:
            for line in handle:
                try:
                    doc = json.loads(line)
                except Exception:
                    continue
                counts['documents'] += 1
                turns = [t for t in (clean(part) for part in turns_of(doc, keys)) if t]
                text = ''.join(turns)
                if not text:
                    continue
                document_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if document_hash in seen:
                    counts['duplicate_documents'] += 1
                    continue
                if any(character not in known for character in text):
                    counts['dropped_for_unknown_character'] += 1
                    continue
                seen.add(document_hash)
                counts['kept_documents'] += 1
                counts['characters'] += len(text)
                row = dict(source=key, id=doc.get('id', str(counts['documents'])),
                           sha256=document_hash, turns=turns)
                payload = json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                out.write(payload)
                digest.update(payload.encode('utf-8'))
        hashes[key] = digest.hexdigest()
        stat[key] = counts
        print(json.dumps(dict(source=key, **counts), ensure_ascii=False), flush=True)
    total = sum(c['characters'] for c in stat.values())
    manifest = dict(completed=True, sources=stat, files={k: dict(path=f'extra/{k}.jsonl', sha256=v)
                                                         for k, v in hashes.items()},
                    vocabulary=len(vocabulary),
                    clean_rule='Identical to modern_corpus_v1: remove Unicode P punctuation, Cc/Cf control and format codes and whitespace',
                    duplicate_rule='SHA256 of the cleaned complete document, against modern_corpus_v1 training documents and against every earlier extra document',
                    vocabulary_rule='Frozen at the modern_corpus_v1 vocabulary; a document containing any character outside it is dropped whole, never edited',
                    extra_characters=total, v1_training_characters=344787049,
                    combined_characters=344787049 + total,
                    combined_ratio=round((344787049 + total) / 344787049, 3),
                    seconds=round(time.perf_counter() - started, 1))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(extra_characters=total, combined_ratio=manifest['combined_ratio']), ensure_ascii=False), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()