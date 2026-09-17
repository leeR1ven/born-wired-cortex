"""Delete Unicode punctuation only; preserve the existing corpus split."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata
import numpy as np

HERE=Path(__file__).resolve().parent
DEST=HERE/'data_no_punct_v1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():raise SystemExit('Use fresh derived corpus directory')
    DEST.mkdir()
    manifest=dict(version=1,transformation='delete characters whose Unicode category starts with P',
                  whitespace='preserved exactly',unicode_version=unicodedata.unidata_version,
                  original_split_assignment='unchanged',builder_sha256=digest(Path(__file__)),splits={})
    mappings={}
    for split in ('train','validation','test'):
        original=HERE/'data'/f'{split}.txt'
        text=original.read_text(encoding='utf-8')
        mapping=np.full(len(text),-1,np.int64)
        kept=[];removed=Counter()
        for i,ch in enumerate(text):
            if unicodedata.category(ch).startswith('P'):
                removed[ch]+=1
            else:
                mapping[i]=len(kept);kept.append(ch)
        filtered=''.join(kept)
        assert not any(unicodedata.category(ch).startswith('P') for ch in filtered)
        assert Counter(ch for ch in text if ch.isspace())==Counter(ch for ch in filtered if ch.isspace())
        path=DEST/f'{split}.txt';path.write_text(filtered,encoding='utf-8')
        mappings[split]=mapping
        manifest['splits'][split]=dict(source=str(original.relative_to(HERE)),source_sha256=digest(original),
            output=path.name,sha256=digest(path),original_characters=len(text),characters=len(filtered),
            removed_count=sum(removed.values()),removed_characters=removed.most_common(),
            whitespace_count=sum(ch.isspace() for ch in filtered))
    np.savez_compressed(DEST/'old_to_new_positions.npz',**mappings)
    original_protocol=json.loads((HERE/'results_v1/frozen_protocol.json').read_text(encoding='utf-8'))
    original_test=(HERE/'data/test.txt').read_text(encoding='utf-8')
    positions=[p for p in original_protocol['test_positions'] if '\u3400'<=original_test[p]<='\u9fff']
    matched=dict(original_positions=positions,new_positions=[int(mappings['test'][p]) for p in positions],
                 targets=[original_test[p] for p in positions],target_definition='U+3400..U+9FFF',
                 selection='all Han targets in the ORIGINAL 1024-position frozen test sample')
    assert all(i>=0 for i in matched['new_positions'])
    (DEST/'matched_positions.json').write_text(json.dumps(matched,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'splits':{s:{k:v[k] for k in ('original_characters','characters','removed_count','whitespace_count')}
                                  for s,v in manifest['splits'].items()},'matched_han_targets':len(positions)},ensure_ascii=False))


if __name__=='__main__':main()
