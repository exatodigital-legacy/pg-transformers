"""Diagnose sentencepiece tokenizer divergences against the reference ids.

Runs the Python mirror of the plv8 tokenizer (spm_mirror.py) over the exported
{key}_refs.json and reports the mismatches with the first divergent pieces and
codepoints, so root causes can be grouped and fixed offline (no DB needed).

Usage: python tests/diagnose_spm.py [bge-m3]
"""
import json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pg_transformers import registry
from pg_transformers.spm_mirror import tokenize

ART = registry.artifacts_dir()
KEY = sys.argv[1] if len(sys.argv) > 1 else "bge-m3"
refs = json.load(open(os.path.join(ART, f"{KEY}_refs.json"), encoding="utf-8"))
spm = json.load(open(os.path.join(ART, f"{KEY}_spm.json"), encoding="utf-8"))
meta = json.load(open(os.path.join(ART, f"{KEY}_meta.json")))
nfkc = json.load(open(os.path.join(ART, f"{KEY}_nfkc.json"), encoding="utf-8"))
CLS, SEP = spm["cls_id"], spm["sep_id"]
id2piece = {v + spm.get("fairseq_offset", 1): k for k, v in spm["piece_id"].items()}

mism, cats = [], Counter()
for r in refs:
    if not r["text"].strip():
        continue
    hf = [i for i in r["ids"] if i not in (CLS, SEP)]
    mine = tokenize(r["text"], spm, nfkc)
    if len(r["ids"]) >= meta["maxn"]:
        mine = mine[:len(hf)]  # ref was exported with truncation; mirror doesn't truncate
    if mine != hf:
        mism.append((r["text"], hf, mine))

print(f"mismatches: {len(mism)}/{len(refs)}")
for text, hf, mine in mism[:15]:
    k = 0
    while k < min(len(hf), len(mine)) and hf[k] == mine[k]:
        k += 1
    hseg = [id2piece.get(i, f"<{i}>") for i in hf[max(0, k - 2):k + 4]]
    mseg = [id2piece.get(i, f"<{i}>") for i in mine[max(0, k - 2):k + 4]]
    cats[str(tuple(sorted(set(hseg) ^ set(mseg)))[:3])] += 1
    print(f"\n--- div@{k}  text: {text[:70]!r}")
    print(f"  hf  : {hseg}")
    print(f"  mine: {mseg}")

if cats:
    print("\ncategory counts (symmetric piece diff):")
    for c, v in cats.most_common(15):
        print(f"  {v:3d}  {c}")
