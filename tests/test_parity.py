"""In-DB parity for every exported+loaded model.

Prereqs: kernel/build.sh, `pg-transformers export <key>`, `pg-transformers
load <key>`, and a reachable database ($PGT_DSN). Models without artifacts
are skipped, so CI can run with just all-minilm.

  PGT_MODELS=all-minilm pytest tests/test_parity.py
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pg_transformers import registry
from pg_transformers.verify import verify

KEYS = os.environ.get("PGT_MODELS", "").split(",") if os.environ.get("PGT_MODELS") \
    else list(registry.load_registry())


@pytest.mark.parametrize("key", KEYS)
def test_parity(key):
    refs = os.path.join(registry.artifacts_dir(), f"{key}_refs.json")
    if not os.path.exists(refs):
        pytest.skip(f"no artifacts for {key}; run export+load first")
    refs_emb = int(os.environ.get("PGT_REFS_EMB", "0"))
    assert verify(key, refs_emb=refs_emb), f"{key} failed parity"


def test_spm_mirror_matches_refs():
    """The Python mirror of the plv8 spm tokenizer must match HF on all refs
    (offline; no DB needed). Guards the mirror against drifting from the SQL."""
    from pg_transformers.spm_mirror import tokenize
    art = registry.artifacts_dir()
    keys = [k for k, m in registry.load_registry().items() if m["tokenizer"] == "spm"]
    ran = False
    for key in keys:
        try:
            refs = json.load(open(os.path.join(art, f"{key}_refs.json"), encoding="utf-8"))
            spm = json.load(open(os.path.join(art, f"{key}_spm.json"), encoding="utf-8"))
            nfkc = json.load(open(os.path.join(art, f"{key}_nfkc.json"), encoding="utf-8"))
            meta = json.load(open(os.path.join(art, f"{key}_meta.json")))
        except FileNotFoundError:
            continue
        ran = True
        cls_sep = (spm["cls_id"], spm["sep_id"])
        for r in refs:
            hf = [i for i in r["ids"] if i not in cls_sep]
            mine = tokenize(r["text"], spm, nfkc)
            if len(r["ids"]) >= meta["maxn"]:
                mine = mine[:len(hf)]  # ref exported with truncation
            assert mine == hf, f"{key}: mirror diverges on {r['text'][:60]!r}"
    if not ran:
        pytest.skip("no spm artifacts present")
