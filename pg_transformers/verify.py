"""Verify an in-DB model against the exported ground truth ({key}_refs.json).

Four checks:
  1. tokenizer: pgt_tokenize must equal HF ids exactly, every ref
  2. numerics:  pgt_embed_ids on HF ids vs PyTorch embedding (cosine)
  3. end-to-end: pgt_embed(text) vs PyTorch embedding (cosine)
  4. throughput: ms/doc on long refs, ms/query on short refs

Needs only psycopg + numpy + artifacts. refs_emb subsamples the embed loops
for slow/big models (tokenizer parity always runs over ALL refs).
"""
import json
import os

import numpy as np
import psycopg

from . import registry


def verify(key, dsn=None, refs_emb=0, thru_long=20, thru_short=40):
    art = registry.artifacts_dir()
    refs = json.load(open(os.path.join(art, f"{key}_refs.json"), encoding="utf-8"))
    emb_refs = refs if refs_emb <= 0 else refs[:: max(1, len(refs) // refs_emb)][:refs_emb]
    # exact (fp32) ports must hit 0.999; quantized variants declare their own
    # threshold in the registry (min_cosine)
    thr = float(registry.model(key).get("min_cosine", 0.999))

    conn = psycopg.connect(dsn or registry.default_dsn(), autocommit=True)
    cur = conn.cursor()
    cur.execute("select pgt_load(%s)", (key,))
    print(cur.fetchone()[0])
    ok = True

    # 1. tokenizer parity
    mism = 0
    for r in refs:
        cur.execute("select pgt_tokenize(%s,%s)", (key, r["text"]))
        if cur.fetchone()[0] != r["ids"]:
            mism += 1
    ok &= mism == 0
    print(f"tokenizer: {len(refs)-mism}/{len(refs)} exact id match")

    # 2. numeric parity (reference ids -> bypass tokenizer)
    cs = _cosines(cur, key, "pgt_embed_ids", [(r["ids"], r["emb"]) for r in emb_refs])
    ok &= cs.min() > thr
    print(f"numerics: mean_cos={cs.mean():.6f} worst_cos={cs.min():.6f} "
          f">{thr}={100*(cs>thr).mean():.1f}%")

    # 3. end-to-end parity (text -> tokenizer -> embed)
    cs2 = _cosines(cur, key, "pgt_embed", [(r["text"], r["emb"]) for r in emb_refs])
    ok &= cs2.min() > thr
    print(f"end-to-end: mean_cos={cs2.mean():.6f} worst={cs2.min():.6f} "
          f">{thr}={100*(cs2>thr).mean():.1f}%")

    # 4. throughput, reported the way the community reports embedders:
    #    tokens/s on documents at a stated length, latency in ms for queries
    long_ids = [r["ids"] for r in refs if len(r["ids"]) > 120][:thru_long]
    short_ids = [r["ids"] for r in refs if 5 < len(r["ids"]) < 30][:thru_short]
    if long_ids:
        cur.execute("select pgt_bench_ids(%s,%s,1)", (key, json.dumps(long_ids)))
        ms = cur.fetchone()[0]
        ntok = sum(len(x) for x in long_ids)
        mtl = ntok / len(long_ids)
        print(f"throughput: {1000*ntok/ms:.0f} tokens/s on {len(long_ids)} docs "
              f"(mean {mtl:.0f} tok, {ms/len(long_ids):.0f} ms/doc)")
    if short_ids:
        cur.execute("select pgt_bench_ids(%s,%s,3)", (key, json.dumps(short_ids)))
        ms = cur.fetchone()[0]
        print(f"query latency: {ms/(3*len(short_ids)):.1f} ms "
              f"({len(short_ids)} queries of 5-30 tok, x3)")
    conn.close()
    print("verify: " + ("PASS" if ok else "FAIL"))
    return ok


def _cosines(cur, key, fn, pairs):
    cs = []
    for arg, truth in pairs:
        cur.execute(f"select {fn}(%s,%s)", (key, arg))
        e = np.array(cur.fetchone()[0], dtype=np.float32)
        cs.append(float(e @ np.array(truth, dtype=np.float32)))
    return np.array(cs)
