"""Append new public-domain book training blocks; preserve all old splits."""
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

HERE = Path(__file__).resolve().parent
RAW = HERE/'corpus_expansion_v1'
OUT = HERE/'data_expanded_v1'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit('Refuse to overwrite derived corpus')
    source_manifest = json.loads((RAW/'manifest.json').read_text(encoding='utf-8'))
    old = {s:(HERE/'data_no_punct_v1'/f'{s}.txt').read_text(encoding='utf-8')
           for s in ('train','validation','test')}
    seen = {re.sub(r'\s+','',p) for text in old.values() for p in text.split('\n\n') if len(p)>=30}
    splits = {s:[] for s in old}
    blocks = []
    books = []
    duplicates = 0
    for book in source_manifest['books']:
        blob = (RAW/book['raw_file']).read_bytes()
        assert sha(blob) == book['raw_sha256']
        text = blob.decode('utf-8-sig')
        start = re.search(r'\*\*\* START OF .*?\*\*\*',text)
        end = re.search(r'\*\*\* END OF .*?\*\*\*',text)
        assert start and end and start.end()<end.start()
        body = text[start.end():end.start()].replace('\r\n','\n').replace('\r','\n').strip()
        saved_body = (RAW/book['body_file']).read_text(encoding='utf-8')
        assert saved_body.strip() == body
        assert sha((RAW/book['body_file']).read_bytes()) == book['body_sha256']
        cleaned_lines = []
        metadata_lines = []
        chapter_boundaries = 0
        for line in body.splitlines():
            stripped = line.strip()
            if (stripped.startswith('Produced by ') or stripped.startswith('End of Project Gutenberg')
                    or stripped in ('/p','p/','<p>','</p>')):
                metadata_lines.append(stripped)
                continue
            # Some editions omit blank lines between whole chapters. Preserve
            # these explicit headings as paragraph boundaries before grouping.
            if re.match(r'^第[一二三四五六七八九十百千零〇0-9]+[回卷章]',stripped):
                cleaned_lines.append('')
                chapter_boundaries += 1
            cleaned_lines.append(line)
        body = '\n'.join(cleaned_lines)
        paragraphs = []
        removed = 0
        local_duplicates = 0
        for chunk in re.split(r'\n\s*\n',body):
            paragraph = ''.join(line.strip() for line in chunk.splitlines()).strip()
            filtered = ''.join(ch for ch in paragraph if not unicodedata.category(ch).startswith('P'))
            removed += len(paragraph)-len(filtered)
            if not filtered:
                continue
            if len(filtered)>=30:
                key = re.sub(r'\s+','',filtered)
                if key in seen:
                    duplicates += 1
                    local_duplicates += 1
                    continue
                seen.add(key)
            paragraphs.append(filtered)
        grouped = []
        current = []
        size = 0
        for paragraph in paragraphs:
            if current and size>=5000:
                grouped.append('\n\n'.join(current))
                current = []
                size = 0
            current.append(paragraph)
            size += len(paragraph)
        if current:
            grouped.append('\n\n'.join(current))
        assert len(grouped)>=10, 'Not enough whole-text blocks for a book split'
        indices = list(range(len(grouped)))
        random.Random(19401+int(book['id'])).shuffle(indices)
        count = max(1,round(len(indices)*.1))
        validation = set(indices[:count])
        test = set(indices[count:2*count])
        for index,chunk in enumerate(grouped):
            split = 'validation' if index in validation else 'test' if index in test else 'train'
            splits[split].append((int(book['id']),index,chunk))
            blocks.append(dict(source=int(book['id']),block=index,split=split,
                characters=len(chunk),sha256=sha(chunk.encode('utf-8'))))
        books.append(dict(id=book['id'],title=book['title'],paragraphs=len(paragraphs),blocks=len(grouped),
            punctuation_removed=removed,duplicate_paragraphs_removed=local_duplicates,
            metadata_lines_removed=metadata_lines,explicit_chapter_boundaries=chapter_boundaries,
            characters=sum(map(len,paragraphs))))
    texts = {}
    for split,records in splits.items():
        records.sort(key=lambda r:(r[1],r[0]))
        texts[split] = '\n\n'.join(r[2] for r in records)+'\n'
    training = old['train']+'\n\n'+texts['train']
    assert training[:len(old['train'])] == old['train']
    all_sets = {f'old_{s}':{re.sub(r'\s+','',p) for p in text.split('\n\n') if len(p)>=30} for s,text in old.items()}
    all_sets.update({f'new_{s}':{re.sub(r'\s+','',p) for p in text.split('\n\n') if len(p)>=30} for s,text in texts.items()})
    overlaps = {a+'__'+b:len(all_sets[a]&all_sets[b]) for i,a in enumerate(all_sets) for b in list(all_sets)[i+1:]}
    assert not any(overlaps.values()), overlaps
    OUT.mkdir()
    files = {'train.txt':training,'new_validation.txt':texts['validation'],'new_test.txt':texts['test']}
    metadata = {}
    for name,text in files.items():
        assert not any(unicodedata.category(ch).startswith('P') for ch in text)
        path = OUT/name
        path.write_text(text,encoding='utf-8')
        metadata[name] = dict(characters=len(text),han_characters=sum('\u3400'<=ch<='\u9fff' for ch in text),
            unique_characters=len(set(text)),sha256=sha(path.read_bytes()))
    manifest = dict(version=1,source_manifest_sha256=sha((RAW/'manifest.json').read_bytes()),
        builder_sha256=sha(Path(__file__).read_bytes()),books=books,files=metadata,blocks=blocks,
        old_training_prefix_characters=len(old['train']),old_training_prefix_sha256=sha(old['train'].encode('utf-8')),
        old_splits_unchanged=True,split='whole consecutive paragraphs in ~5000-character blocks, seeded 80/10/10 per new book',
        new_training_order='interleave source blocks by original block index; retain within-source order',
        duplicate_paragraphs_removed=duplicates,exact_paragraph_overlap_ge30=overlaps,near_duplicates_removed=False,
        script_inputs={f'data_no_punct_v1/{s}.txt':sha((HERE/'data_no_punct_v1'/f'{s}.txt').read_bytes()) for s in old})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=metadata,books=books,duplicates=duplicates,overlaps=overlaps),ensure_ascii=False))


if __name__ == '__main__':
    main()
