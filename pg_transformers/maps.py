"""Precomputed per-codepoint normalization maps for the plv8 tokenizers.

plv8's reduced-ICU V8 has no working String.prototype.normalize, so all
Unicode normalization ships as data:

- fold_map(): HF BasicTokenizer._run_strip_accents (NFD, drop combining
  marks), used by uncased WordPiece models after lowercasing.
- nfkc_map(sp): the model's own sentencepiece normalizer (exact nmt_nfkc
  charsmap: NFKC compositions plus NMT rules like ZWSP/narrow-NBSP -> space),
  used by spm models. Per-codepoint by construction: combining marks are
  skipped (real input is NFC-precomposed; parity tests verify).
"""
import unicodedata

MARK = "▁"  # sentencepiece space marker


def fold_map():
    def strip_accents(s):
        return "".join(c for c in unicodedata.normalize("NFD", s)
                       if unicodedata.category(c) != "Mn")
    fold = {}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        f = strip_accents(ch)
        if f != ch:
            fold[str(cp)] = f
    return fold


def nfkc_map(sp):
    """sp: a loaded sentencepiece.SentencePieceProcessor."""
    nfkc = {}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        if unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            continue
        # sandwich between letters so leading/trailing-space trimming can't
        # hide a space mapping; strip the dummy prefix and the context off
        n = sp.normalize("a" + ch + "b")
        if not (n.startswith(MARK + "a") and n.endswith("b")):
            continue  # normalization interacted with the context; leave as-is
        mid = n[2:-1].replace(MARK, " ")
        if mid != ch:
            nfkc[str(cp)] = mid
    return nfkc
