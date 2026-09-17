"""Synthetic execution-integrity smoke; no language-performance claims."""
from pathlib import Path
import hashlib,json,tempfile
import numpy as np
import run_modern_scale as r
from hebb_text import CharacterHebb,Config

def main():
    with tempfile.TemporaryDirectory(prefix='modern_hebb_check_') as temp:
        root=Path(temp);r.DATA=root/'prepared';r.OUT=root/'run';r.DATA.mkdir()
        contents={
            'dialogue_train':[dict(source='lccc_base',id='a',turns=['你好你好','为什么不知道'])],
            'wiki_train':[dict(source='wikipedia',id='b',turns=['现代文章介绍世界的知识科学是一种认识世界的方法'])],
            'validation':[dict(source='lccc_base',id='v',turns=['你好吗','很好'])],
            'test':[dict(source='lccc_base',id='t1',turns=['你为什么不知道','请再说一遍']),dict(source='wikipedia',id='t2',turns=['地球在围绕太阳旋转'])]}
        files={};train_text=''
        for kind,docs in contents.items():
            for d in docs:d['sha256']=hashlib.sha256(''.join(d['turns']).encode()).hexdigest()
            p=r.DATA/(kind+'.jsonl');p.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs),encoding='utf-8')
            files[kind]=dict(path=p.name,sha256=r.file_hash(p))
            if kind.endswith('_train'):train_text+=''.join(''.join(d['turns']) for d in docs)
        r.dump(r.DATA/'vocabulary.json',dict(vocabulary=''.join(sorted(set(train_text)))))
        r.dump(r.DATA/'manifest.json',dict(training_characters=len(train_text),files=files,
            training_parts={kind:sum(len(''.join(d['turns'])) for d in docs) for kind,docs in contents.items() if kind.endswith('_train')}))
        r.dump(r.DATA/'common_cues.json',dict(fixed_cues=[dict(prompt='你好')],most_common_short_utterances=[]))
        r.main(milestones=(3,15))
        summary=r.read(r.OUT/'summary.json');assert summary['completed']
        ref=CharacterHebb(''.join(sorted(set(train_text))),Config(trace_decay=.25))
        for count in (3,15,len(train_text)):
            ref.train(train_text[ref.learned_characters:count]);saved=r.load(r.OUT/f'model_{count}')
            assert np.array_equal(ref.weights,saved.weights)
            assert list(ref.activity.items())==list(saved.activity.items())
            assert ref.clock==saved.clock==count and ref.update_events==saved.update_events
            assert int(np.load(r.OUT/f'model_{count}/character_counts.npy').sum())==count
            saved.weights._mmap.close()
        r.dump(Path(__file__).resolve().parent/'modern_pipeline_verification.json',dict(completed=True,
            purpose='Synthetic integrity smoke only',exact_reference_checkpoints=[3,15,len(train_text)],
            verifies=['Final chunk fully consumed','Checkpoint trace and clock continuous','Saved weights exactly equal original core',
                      'Per-character counts sum to training count','Read-only recall does not learn','Unknown heldout characters handled']))
    print('PIPELINE CHECK PASS',flush=True)

if __name__=='__main__':main()
