"""Python mirror of the plv8 sentencepiece tokenizer (pgt_tok_spm in sql/pg_transformers.sql).

Single reference implementation used by the diagnostics so they cannot drift
from each other. Update in lockstep with enc_tok_spm.
"""
MARK = "▁"   # sentencepiece space marker
MAX_PIECE = 32    # Viterbi probe bound; longest XLM-R piece is 16 chars
_WS = {32, 9, 10, 13, 0xA0, 0x3000} | set(range(0x2000, 0x200B))


def normalize(text, nfkc_map=None):
    """nmt_nfkc as done in plv8: per-codepoint NFKC map, drop control/U+FFFD,
    collapse whitespace runs to MARK, dummy prefix, trim trailing whitespace."""
    if nfkc_map:
        text = "".join(nfkc_map.get(str(ord(ch)), ch) for ch in text)
    out = [MARK]
    prev = True
    for ch in text:
        c = ord(ch)
        if c == 0xFFFD or (c < 32 and c not in (9, 10, 13)):
            continue
        if c in _WS:
            if not prev:
                out.append(MARK)
                prev = True
            continue
        out.append(ch)
        prev = False
    s = "".join(out)
    if len(s) > 1 and s.endswith(MARK):
        s = s[:-1]
    return s


def tokenize(text, spm, nfkc_map=None):
    """Middle token ids (no CLS/SEP), HF numbering (fairseq offset applied,
    unknown chars -> unk_id). spm = the exported {key}_spm.json dict."""
    score, piece_id = spm["score"], spm["piece_id"]
    off, unk_id = spm.get("fairseq_offset", 1), spm["unk_id"]
    s = normalize(text, nfkc_map)
    if s == MARK:
        return []
    n = len(s)
    NEG = -1e18
    best = [NEG] * (n + 1)
    best[0] = 0.0
    back = [-1] * (n + 1)
    back_id = [0] * (n + 1)
    unk_score = spm["min_score"] - 10.0
    for i in range(n):
        if best[i] == NEG:
            continue
        for j in range(i + 1, min(n, i + MAX_PIECE) + 1):
            sc = score.get(s[i:j])
            if sc is not None and best[i] + sc > best[j]:
                best[j] = best[i] + sc
                back[j] = i
                back_id[j] = piece_id[s[i:j]]
        if best[i] + unk_score > best[i + 1]:
            best[i + 1] = best[i] + unk_score
            back[i + 1] = i
            back_id[i + 1] = -1
    out, pos = [], n
    while pos > 0:
        out.append(back_id[pos])
        pos = back[pos]
    out.reverse()
    return [(unk_id if x < 0 else x + off) for x in out]
