"""Adversarial tokenizer inputs: in-DB pgt_tokenize vs HuggingFace, per model.

The reference corpus is realistic text; this covers what it doesn't: emoji,
CJK, astral plane, NFKC ligatures/full-width, >100-char words, whitespace-only,
control and format characters (ZWSP, ZWJ, soft hyphen, bidi, narrow NBSP).
HF output is ground truth. Needs transformers + a reachable DB with the
model loaded; models without artifacts are skipped.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pg_transformers import registry

TEXTS = [
    "",
    "   ",
    "\t\n  \t",
    "a",
    " palavra ",
    "café naïve déjà-vu à propos",
    "AÇÃO RESCISÓRIA Nº 12.345/SP",
    "art. 483, §2º, da CLT c/c art. 5º, XXXV, CF/88",
    "L'exception d'inexécution (art. 1219 du Code civil français)",
    "The plaintiff's motion was denied.",
    "half ½ ligature ﬁle ﬂow №5 Ⅷ",
    "ｆｕｌｌｗｉｄｔｈ　ｔｅｘｔ１２３",
    "①②③ ㊙ ㍿",
    "emoji 😀 no meio 🇧🇷 do texto",
    "中文法律文件 混合 português 텍스트",
    "日本語のテキストです",
    "word" + "x" * 120 + "end",       # >100-char word
    "supercalifragilisticexpialidocious" * 4,
    "hífen-composto auto-aplicável ex-empregado",
    "R$ 1.234.567,89 (um milhão)",
    "e-mail: foo.bar@example.com.br; url: https://ex.com/a?b=1&c=2",
    "quotes “curly” and ‘single’ and «guillemets»",
    "dash — em, – en, − minus",
    "astral \U0001D4D2\U0001D4EA math script",
    "zero​width​space and nbsp",
    "narrow nbsp: 1 234,56 € (French)",
    "soft­hyphen inside",
    "zwj family \U0001F468‍\U0001F469‍\U0001F467 emoji",
    "bidi ‪marks‬ and word⁠joiner",
    "control\x01chars\x02here",
    "Ação:rescisão(art.483,CLT).Sem espaços!",
    "MAIÚSCULAS ACENTUADAS: ÁÉÍÓÚ ÂÊÔ ÃÕ Ç",
]


@pytest.mark.parametrize("key", list(registry.load_registry()))
def test_edge_cases(key):
    import psycopg
    from transformers import AutoTokenizer

    meta_path = os.path.join(registry.artifacts_dir(), f"{key}_meta.json")
    if not os.path.exists(meta_path):
        pytest.skip(f"no artifacts for {key}; run export+load first")
    meta = json.load(open(meta_path))
    tok = AutoTokenizer.from_pretrained(meta["hf_id"])

    conn = psycopg.connect(registry.default_dsn(), autocommit=True)
    cur = conn.cursor()
    mismatches = []
    for t in TEXTS:
        hf = tok(t, truncation=True, max_length=meta["maxn"])["input_ids"]
        cur.execute("select pgt_tokenize(%s,%s)", (key, t))
        mine = cur.fetchone()[0]
        if mine != hf:
            mismatches.append((t[:50], hf[:10], mine[:10]))
    conn.close()
    assert not mismatches, f"{key}: {len(mismatches)}/{len(TEXTS)} diverge: {mismatches[:3]}"
