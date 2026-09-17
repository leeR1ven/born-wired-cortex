"""Independent data/stream/weight/scoring audit of modern_scale_v1.

Reuses only our independent earlier Recall oracle, never the model, fast trainer,
builder or experiment evaluators. Does not change model/data/result evidence.
"""
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import time
import unicodedata

import numpy as np
from verify_expanded_scale import Recall, same, require, period

HERE=Path(__file__).resolve().parent
DATA=HERE/"modern_corpus_v1/prepared"
OUT=HERE/"modern_scale_v1"
CACHE=HERE/"modern_scale_verification_data.json"
CHECKS=Counter()
REMOVE={i:None for i in range(0x110000) if chr(i).isspace() or
        unicodedata.category(chr(i))[0]=="P" or unicodedata.category(chr(i)) in ("Cc","Cf")}
HAN=re.compile(r"[\u3400-\u9fff\U00020000-\U000323af]")
OFFICIAL_LCCC={
    "valid":"5cc27e7ac3447c5a31386178f82ff01cab56e27827445ef8d429809301491759",
    "test":"cf8757587bdb8f360cc94fc38baadf9e185bad65a26155527a8430c048676016",
    "train":"2162e0ed923fba62329cabf7e1493fbe59248afc94a62508e4abdea61e624627"}


def read(path): return json.loads(path.read_text(encoding="utf8"))
def file_hash(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def records(path):
    with path.open(encoding="utf8") as f:
        for line in f:yield json.loads(line)
def hash_checks(mapping,root=HERE):
    for name,expected in mapping.items():
        require(file_hash(root/name)==expected,"Frozen file "+name)
        CHECKS["file_hashes"]+=1


def prepared_audit(manifest):
    for info in manifest["inputs"]:
        path=HERE/info["path"]
        require(path.stat().st_size==info["bytes"] and file_hash(path)==info["sha256"],"Raw input bytes")
        if "lccc_base_" in path.name:
            split=path.name.removeprefix("lccc_base_").split(".")[0]
            require(info["sha256"]==OFFICIAL_LCCC[split],"Official author LFS digest")
        CHECKS["raw_files"]+=1
    wiki_manifest=read(HERE/"modern_corpus_v1/wiki_sources/manifest.json")
    wiki=HERE/"modern_corpus_v1/wiki_sources/train-00001-of-00006.parquet"
    require(file_hash(wiki)==wiki_manifest["files"][1]["lfs"]["oid"],"Official Wikipedia LFS digest")
    total_docs=sum(s.get(k+"_documents",0) for s in manifest["source_stats"].values()
                   for k in ("train","validation","test"))
    doc_hashes=np.empty(total_docs,dtype="S32")
    samples={}; stats={}; frequencies=Counter(); cursor=0
    for kind,info in manifest["files"].items():
        require(file_hash(DATA/info["path"])==info["sha256"],"Prepared file digest")
        split="train" if kind.endswith("_train") else kind
        for doc in records(DATA/info["path"]):
            turns=doc["turns"];text="".join(turns)
            require(turns and all(isinstance(t,str) and t for t in turns),"Nonempty normalized turns")
            require(text.translate(REMOVE)==text,"Normalization is idempotent")
            digest=hashlib.sha256(text.encode("utf8")).digest()
            require(digest.hex()==doc["sha256"],"Document hash")
            source=doc["source"]
            if source=="lccc_base":
                origin,index=doc["id"].split(":")
            else:origin,index="wiki",str(doc["id"])
            forced={"valid":"validation","test":"test"}.get(origin) if source=="lccc_base" else None
            bucket=int.from_bytes(digest[:8],"big")%1000
            expected=forced or ("validation" if bucket<5 else "test" if bucket<10 else "train")
            require(split==expected,"Reserved/hash partition")
            if split=="train":
                require(kind==("wiki_train" if source=="wikipedia" else "dialogue_train"),"Training source file")
                frequencies.update(text)
            st=stats.setdefault(source,Counter())
            st[split+"_documents"]+=1;st[split+"_characters"]+=len(text)
            st[split+"_han_characters"]+=len(HAN.findall(text))
            key=(source,origin,split)
            if len(samples.setdefault(key,[]))<3:samples[key].append(doc)
            doc_hashes[cursor]=digest;cursor+=1
            if cursor%1000000==0:print(json.dumps({"checked_documents":cursor}),flush=True)
    require(cursor==len(doc_hashes),"Complete document count")
    doc_hashes.sort()
    require(not np.any(doc_hashes[1:]==doc_hashes[:-1]),"No complete normalized document duplicate anywhere")
    del doc_hashes
    for source,st in stats.items():
        for key,n in st.items():require(manifest["source_stats"][source][key]==n,"Source count "+source+"/"+key)
    require(sum(frequencies.values())==manifest["training_characters"],"Total training characters")
    vocab=read(DATA/"vocabulary.json")
    require(vocab["vocabulary"]=="".join(sorted(frequencies)),"Vocabulary from complete training only")
    same(vocab["counts"],dict(frequencies),"Training character frequencies")
    CHECKS["prepared_documents"]=cursor
    # Sample original lines by their stored source IDs, covering both official
    # holdouts and additional hash holdouts; no result-based sample selection.
    desired={}
    for (source,origin,split),docs in samples.items():
        if source=="lccc_base":
            for d in docs:desired.setdefault(origin,{})[int(d["id"].split(":")[1])]=d
    for origin,wanted in desired.items():
        path=HERE/f"modern_corpus_v1/dialogue_sources/lccc_base/lccc_base_{origin}.jsonl.gz"
        with gzip.open(path,"rt",encoding="utf8",errors="strict") as f:
            for i,line in enumerate(f):
                if i in wanted:
                    raw=json.loads(line);cleaned=[t.translate(REMOVE) for t in raw]
                    same(wanted[i]["turns"],[t for t in cleaned if t],"Raw LCCC cleaning sample")
                    CHECKS["raw_cleaning_samples"]+=1
                if i>=max(wanted):break
    wanted={str(d["id"]):d for (source,_,_),docs in samples.items() if source=="wikipedia" for d in docs}
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(wiki).iter_batches(batch_size=256,columns=["id","text"]):
        for raw in batch.to_pylist():
            if str(raw["id"]) in wanted:
                expected=wanted.pop(str(raw["id"]))
                same(expected["turns"],[raw["text"].translate(REMOVE)],"Raw article cleaning sample")
                CHECKS["raw_cleaning_samples"]+=1
        if not wanted:break
    require(not wanted,"All selected article samples located")
    return vocab["vocabulary"],frequencies


def check_probe_reservoir(probes):
    rng=random.Random(19460);sample={k:[] for k in ("lccc_base","wikipedia","reply_first")};seen=Counter()
    for doc in records(DATA/"test.jsonl"):
        text="".join(doc["turns"]);choices=[]
        if len(text)>=8:
            p=rng.randrange(1,len(text))
            choices.append((doc["source"],text[max(0,p-64):p],text[p]))
        if doc["source"]=="lccc_base" and len(doc["turns"])>=2:
            p=rng.randrange(1,len(doc["turns"]))
            choices.append(("reply_first","".join(doc["turns"][:p])[-64:],doc["turns"][p][0]))
        for kind,prefix,target in choices:
            seen[kind]+=1
            row=dict(kind=kind,source=doc["source"],id=doc["id"],document_sha256=doc["sha256"],prefix=prefix,target=target)
            if len(sample[kind])<128:sample[kind].append(row)
            else:
                j=rng.randrange(seen[kind])
                if j<128:sample[kind][j]=row
    same(probes["records"],[r for group in sample.values() for r in group],"Frozen deterministic probes")
    same(probes["eligible_records"],dict(seen),"Eligible probe records")
    require(len(probes["records"])==384,"All three 128-record probe groups")


def chunks(path,limit):
    pending=[];length=0
    for doc in records(path):
        text="".join(doc["turns"]);pending.append(text);length+=len(text)
        if length>=limit:
            yield "".join(pending);pending=[];length=0
    if pending:yield "".join(pending)


def interleaved_stream():
    parts=read(DATA/"manifest.json")["training_parts"]
    sizes=(100000,max(1,round(100000*parts["wiki_train"]/parts["dialogue_train"])))
    streams=[iter(chunks(DATA/(k+".jsonl"),n)) for k,n in zip(("dialogue_train","wiki_train"),sizes)]
    alive=[True,True]
    while any(alive):
        for i in range(2):
            if alive[i]:
                text=next(streams[i],None)
                if text is None:alive[i]=False
                else:yield ("lccc_base" if i==0 else "wikipedia"),text


def matrix_audit(stage,vocabulary,expected_counts,oracle=None):
    path=OUT/f"model_{stage}";meta=read(path/"metadata.json")
    require(meta["vocabulary"]==vocabulary,"Checkpoint vocabulary")
    w=np.load(path/"weights.npy",mmap_mode="r",allow_pickle=False)
    require(w.dtype==np.float32 and w.shape==(len(vocabulary),)*2 and w.flags.c_contiguous,"Checkpoint shape/dtype")
    require(np.array_equal(np.load(path/"character_counts.npy",allow_pickle=False),expected_counts),"Checkpoint experience counts")
    digest=hashlib.sha256(vocabulary.encode("utf8"));connections=0;maximum=0.;total=0.;degrees=[]
    for first in range(0,len(vocabulary),128):
        block=w[first:first+128]
        require(np.isfinite(block).all() and np.min(block)>=0,"Finite nonnegative directed weights")
        digest.update(block.tobytes())
        positive=block[block>0];connections+=len(positive)
        if positive.size:maximum=max(maximum,float(positive.max()));total+=float(positive.sum(dtype=np.float64))
        degrees.extend(np.count_nonzero(block,axis=1).tolist())
        if oracle is not None:
            rows,cols=np.nonzero(block)
            for r,c in zip(rows,cols):
                require(oracle.get((first+int(r))*len(vocabulary)+int(c),np.float32(0))==block[r,c],"Exact independent prefix weight")
    if oracle is not None:require(connections==len(oracle),"No missing independently learned edges")
    require(digest.hexdigest()==meta["weight_sha256"],"Vocabulary + full matrix SHA256")
    expected=read(OUT/f"stage_{stage}.json")
    require(expected["weight_sha256"]==digest.hexdigest(),"Stage matrix digest")
    stats=expected["stats"]
    for name,value in dict(connections=connections,weight_max=maximum,weight_mean=total/connections if connections else 0.,
                           matrix_bytes=w.nbytes).items():same(stats[name],value,"Weight summary "+name)
    bins=Counter();tokens=Counter()
    for c,n in zip(vocabulary,expected_counts):
        if "\u3400"<=c<="\u9fff" or 0x20000<=ord(c)<=0x323af:
            key="0" if n==0 else "1-9" if n<10 else "10-99" if n<100 else "100-999" if n<1000 else "1000+"
            bins[key]+=1;tokens[key]+=int(n)
    same(stats["han_frequency_types"],dict(bins),"Stage frequency types")
    same(stats["han_frequency_tokens"],dict(tokens),"Stage frequency tokens")
    selected=np.lexsort((np.arange(len(vocabulary)),-expected_counts))[:30]
    same(stats["top_characters"],[dict(character=vocabulary[i],count=int(expected_counts[i]),out_degree=degrees[i]) for i in selected],
         "Top experienced characters/coverage")
    CHECKS["matrix_checkpoints"]+=1
    return meta


def replay_stream(protocol,vocabulary):
    ids_by_char={c:i for i,c in enumerate(vocabulary)}
    lookup=np.full(0x110000,-1,np.int32)
    for c,i in ids_by_char.items():lookup[ord(c)]=i
    require(protocol["config"]==dict(trace_decay=.25,trace_floor=.02,learning_rate=.25,growth="diminishing",ceiling=4.),"Fixed short graph")
    stages=protocol["stages"];first=stages[0];require(first==3000000,"Independent first three million")
    oracle={};active={};clock=0;events=0;counts=np.zeros(len(vocabulary),np.int64)
    stream_hash=hashlib.sha256();source_counts=Counter();tail=np.empty(0,np.int32);stage_index=0
    for source,text in interleaved_stream():
        offset=0
        while offset<len(text):
            require(stage_index<len(stages),"No stream beyond final stage")
            piece=text[offset:offset+stages[stage_index]-clock];offset+=len(piece)
            ids=lookup[np.frombuffer(piece.encode("utf-32-le"),dtype="<u4")]
            require(np.all(ids>=0),"No train OOV")
            if clock<first:
                for target in ids:
                    target=int(target)
                    for i,a in active.items():
                        key=i*len(vocabulary)+target
                        old=oracle.get(key,np.float32(0.))
                        delta=np.float32(np.float32(.25)*np.float32(a))
                        oracle[key]=np.float32(old+np.float32(delta/np.float32(np.float32(1.)+old)))
                    active={i:a*.25 for i,a in active.items() if a*.25>=.02}
                    active[target]=1.
            combined=np.concatenate((tail,ids));head=max(0,min(len(ids),3-len(tail)))
            for j in range(head):events+=len(set(map(int,combined[:len(tail)+j])))
            start=len(tail)+head;size=len(combined)
            if start<size:
                a=combined[start-1:size-1];b=combined[start-2:size-2];c=combined[start-3:size-3]
                events+=len(a)+int(np.count_nonzero(a!=b))+int(np.count_nonzero((c!=a)&(c!=b)))
            tail=combined[-3:].copy()
            counts+=np.bincount(ids,minlength=len(vocabulary));clock+=len(ids);source_counts[source]+=len(ids)
            stream_hash.update(piece.encode("utf8"))
            if clock==stages[stage_index]:
                meta=matrix_audit(clock,vocabulary,counts,oracle if clock==first else None)
                require(meta["clock"]==meta["learned_characters"]==clock and meta["update_events"]==events,"Stage clock/events")
                if clock==first:
                    same(meta["activity"],list(active.items()),"Exact independent prefix activity order")
                    CHECKS["independent_training_characters"]=clock
                    oracle.clear()
                else:
                    expected_active={}
                    for age,target in enumerate(tail[::-1]):
                        if int(target) not in expected_active:expected_active[int(target)]=.25**age
                    same({int(i):a for i,a in meta["activity"]},expected_active,"Last-three character amplitudes")
                result=read(OUT/f"stage_{clock}.json")
                same(result["source_characters"],dict(source_counts),"Source consumption chronology")
                require(result["stream_sha256"]==stream_hash.hexdigest(),"Actual stage training stream SHA")
                stage_index+=1
                print(json.dumps({"verified_stream_stage":clock}),flush=True)
    require(stage_index==len(stages) and clock==protocol["data_manifest"]["training_characters"],"Exact final stream boundary")
    return counts


def summarize(group):
    known=[r for r in group if r["target_probability"] is not None]
    return dict(n=len(group),known=len(known),top1=sum(r["rank"]==1 for r in group),
                top5=sum(r["rank"] is not None and r["rank"]<=5 for r in group),
                bits_per_known_character=math_fsum_bits(known)/len(known) if known else None)


def math_fsum_bits(rows):
    import math
    return math.fsum(-math.log2(r["target_probability"]) for r in rows)


def results_audit(protocol,vocabulary):
    final=protocol["stages"][-1];frozen_rows=protocol["probes"]["records"];identities={c:i for i,c in enumerate(vocabulary)}
    cfgs=protocol["recall_configs"]
    final_w=np.load(OUT/f"model_{final}/weights.npy",mmap_mode="r",allow_pickle=False)
    selected_prompts=["你好","谢谢","为什么","什么意思","你在干什么",protocol["probes"]["novel_prompts"][0]]
    compact={}
    for stage in protocol["stages"]:
        evidence=read(OUT/f"evaluation_{stage}.json");stage_record=read(OUT/f"stage_{stage}.json")
        counts=np.load(OUT/f"model_{stage}/character_counts.npy",allow_pickle=False)
        require(set(evidence)==set(cfgs),"All fixed recall settings")
        compact[str(stage)]={}
        for name,result in evidence.items():
            rows=result["rows"];require(len(rows)==len(frozen_rows),"All fixed probe predictions")
            for row,original in zip(rows,frozen_rows):
                same({k:row[k] for k in original},original,"Immutable source prefix/target")
                identity=identities.get(row["target"])
                require((row["target_probability"] is None)==(identity is None),"OOV probability")
                require((row["rank"] is None)==(identity is None),"OOV rank")
                require(row["stage_training_target_count"]==(int(counts[identity]) if identity is not None else 0),"Stage target experience")
                if identity is not None:
                    require(0<row["target_probability"]<=1 and 1<=row["rank"]<=len(vocabulary),"Probability/rank bounds")
                require((row["rank"]==1)==(row["predicted"]==row["target"]),"Exact top-one hit")
            computed={kind:summarize([r for r in rows if r["kind"]==kind]) for kind in ("lccc_base","wikipedia","reply_first")}
            same(result["metrics"],computed,"All metric summaries")
            same(stage_record["metrics"][name],computed,"Stage metric mirror")
            CHECKS["metric_prediction_rows"]+=len(rows);CHECKS["metric_groups"]+=3
            require([r["prompt"] for r in result["spoken"]]==protocol["probes"]["prompts"],"All raw prompt outputs retained")
            for row in result["spoken"]:
                require(row["tail_period"]==period(row["emitted"]),"Reported output tail period")
                require(row["clarification_keyword_present"]==any(t in row["emitted"] for t in ("什么意思","为什么","不知道","不懂","再说","说清楚")),"Lexical clarification flag")
            same(stage_record["outputs"][name],result["spoken"],"Stage raw output mirror")
            compact[str(stage)][name]=computed
            if stage==final:
                # Fixed first/last four per source group: 24 predictions per setting.
                for kind in ("lccc_base","wikipedia","reply_first"):
                    group=[r for r in rows if r["kind"]==kind]
                    for row in group[:4]+group[-4:]:
                        probe=Recall(final_w,vocabulary,cfgs[name]);probe.warm(row["prefix"])
                        signal=probe.current().astype(np.float64);total=float(signal.sum())
                        p=.999*signal/total+.001/len(vocabulary) if total else np.full(len(vocabulary),1/len(vocabulary))
                        order=np.lexsort((np.arange(len(vocabulary)),-p));identity=identities.get(row["target"])
                        rank=None if identity is None else 1+int(np.count_nonzero(p>p[identity]))+int(np.count_nonzero(p[:identity]==p[identity]))
                        same(row["rank"],rank,"Independent final rank")
                        same(row["target_probability"],None if identity is None else float(p[identity]),"Independent final probability")
                        require(row["predicted"]==vocabulary[int(order[0])],"Independent final predicted character")
                        same(row["top5"],[vocabulary[int(i)] for i in order[:5]],"Independent final top five")
                        CHECKS["independent_final_predictions"]+=1
                for prompt in selected_prompts:
                    row=next(r for r in result["spoken"] if r["prompt"]==prompt)
                    probe=Recall(final_w,vocabulary,cfgs[name]);probe.warm(prompt);emitted=[]
                    for _ in range(64):
                        raw=probe.current();chosen=vocabulary[int(np.argmax(raw))] if np.any(raw>0) else None
                        probe.step(chosen,raw)
                        if chosen is not None:emitted.append(chosen)
                    require(row["emitted"]=="".join(emitted),"Independent fixed prompt continuation")
                    CHECKS["independent_final_continuations"]+=1
    return compact


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    started=time.perf_counter()
    manifest=read(DATA/"manifest.json")
    if args.prepare_only:
        prepared_audit(manifest)
        paths=[DATA/"manifest.json",DATA/"vocabulary.json"]
        paths += [DATA/v["path"] for v in manifest["files"].values()]
        paths += [HERE/v["path"] for v in manifest["inputs"]]
        cache=dict(status="PASS",verifier_sha256=file_hash(Path(__file__)),checks=dict(CHECKS),
                   file_sha256={str(p.relative_to(HERE)):file_hash(p) for p in paths},
                   seconds=time.perf_counter()-started)
        CACHE.write_text(json.dumps(cache,indent=2),encoding="utf8")
        print(json.dumps(cache),flush=True)
        return
    summary=read(OUT/"summary.json")
    require(summary["completed"] is True,"Await completed modern experiment")
    protocol=read(OUT/"frozen_protocol.json")
    require(file_hash(OUT/"frozen_protocol.json")==summary["protocol_sha256"],"Frozen protocol digest")
    hash_checks(protocol["source_sha256"]);hash_checks(protocol["prepared_metadata_sha256"],DATA)
    same(protocol["data_manifest"],manifest,"Frozen data manifest")
    cache=read(CACHE) if CACHE.exists() else None
    if cache is not None and cache["verifier_sha256"]==file_hash(Path(__file__)):
        require(cache["status"]=="PASS","Prior independent prepared-data audit")
        hash_checks(cache["file_sha256"])
        CHECKS.update(cache["checks"])
        raw=read(DATA/"vocabulary.json");vocabulary=raw["vocabulary"];frequencies=Counter(raw["counts"])
    else:
        vocabulary,frequencies=prepared_audit(manifest)
    check_probe_reservoir(protocol["probes"])
    counts=replay_stream(protocol,vocabulary)
    require(np.array_equal(counts,np.array([frequencies[c] for c in vocabulary],np.int64)),"Final complete training frequencies")
    compact=results_audit(protocol,vocabulary)
    for stage in protocol["stages"]:same(summary["stages"][str(stage)],read(OUT/f"stage_{stage}.json"),"Completed summary stage")
    hash_checks(protocol["source_sha256"]);hash_checks(protocol["prepared_metadata_sha256"],DATA)
    proof=dict(status="PASS",seconds=time.perf_counter()-started,checks=dict(CHECKS),
               protocol_sha256=file_hash(OUT/"frozen_protocol.json"),
               verifier_sha256=file_hash(Path(__file__)),
               independent_recall_helper_sha256=file_hash(HERE/"verify_expanded_scale.py"),
               metrics=compact)
    lines=["# 现代中文规模实验独立核验","","状态：PASS。","",
           "核对原始官方 SHA256、全部整理文件和完整文档哈希；全体完整文档无跨分区重复，保留官方对话留出，另按冻结哈希分区。按原始来源 ID 抽样重建清洗。",
           "独立逐字重放首 3,000,000 字符，与首检查点全部非零权重、活动顺序及计数精确一致；重建整条交替训练流，核对所有阶段流哈希、字符频次、时钟、边更新数和最终边界。所有完整矩阵 SHA256、有限非负权重及覆盖统计通过。",
           "重算所有保存预测的指标汇总；独立复算最终每组首尾固定样本共 72 个预测、18 条固定问法的完整 64 字续写。未重新训练全部数亿字符，也未以大模型判断语义。","",
           "解释边界：完整文档去重不排除近重复、共享对话前缀和短片段背诵。测试按文档均匀抽样后每文档选一个位置，不是按自然字符频率加权；reply_first 单列。clarification_keyword_present 只是关键词出现，不能视为理解或合理澄清。实际训练连续拼接不同对话/文章，会产生跨文档边。","",
           "本核验不声称复算每一项常用短语计数，也不把所有自然文本预测独立重算；核验范围和数量如下。","","~~~json",
           json.dumps(proof,ensure_ascii=False,indent=2),"~~~",""]
    (HERE/"VERIFIED_MODERN_SCALE.md").write_text("\n".join(lines),encoding="utf8")
    print(json.dumps(proof,ensure_ascii=True),flush=True)


if __name__=="__main__":main()
