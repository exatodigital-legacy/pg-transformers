"""Export a HuggingFace sentence-transformer to the kernel's weight layout.

Writes to artifacts/:
  {key}.wasm        built separately by kernel/build.sh
  {key}_word.bin    word-embedding table, f32
  {key}_rest.bin    positions[:P] + type0 + emb_LN + per-layer weights, f32,
                    in the exact order kernel/src/lib.rs consumes them
  {key}_meta.json   dims + pooling + tokenizer flags
  {key}_refs.json   reference texts with HF token ids + ground-truth embeddings
  {key}_wordpiece.json / {key}_spm.json + {key}_nfkc.json / foldmap.json
                    tokenizer data for the plv8 side

Registry dims are validated against the real HF config; mismatches fail loudly.
Needs the heavy deps: pip install 'pg-transformers[export]'.
"""
import json
import os

import numpy as np

from . import registry
from .reference_texts import build as build_texts


def export(key):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    m = registry.model(key)
    out_dir = registry.artifacts_dir()
    hf_id, maxn, pos_off = m["hf_id"], m["max_tokens"], m["pos_offset"]
    print(f"=== {key} ({hf_id}) ===", flush=True)

    st = SentenceTransformer(hf_id, device="cpu")
    st.max_seq_length = maxn
    tok = AutoTokenizer.from_pretrained(hf_id)
    encoder = st[0].auto_model
    cfg = encoder.config

    # the registry must match the real model; a silent mismatch would produce
    # garbage embeddings with no error, so fail here instead
    expect = {"vocab_size": cfg.vocab_size, "hidden": cfg.hidden_size,
              "layers": cfg.num_hidden_layers, "intermediate": cfg.intermediate_size,
              "heads": cfg.num_attention_heads,
              "ln_eps": float(cfg.layer_norm_eps)}
    for k, v in expect.items():
        if (abs(m[k] - v) > 1e-15 if k == "ln_eps" else m[k] != v):
            raise SystemExit(f"models.toml mismatch for {key}.{k}: registry {m[k]} != HF {v}")

    sd = {k: v.detach().numpy().astype(np.float32) for k, v in encoder.state_dict().items()}
    V, H, L, I = cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    P = maxn + pos_off
    print(f"  V={V} H={H} L={L} I={I} P={P} pool={m['pooling']}", flush=True)

    sd["embeddings.word_embeddings.weight"].tofile(os.path.join(out_dir, f"{key}_word.bin"))

    parts = []
    def add(name, shape):
        a = sd[name]
        assert a.shape == shape, (name, a.shape, shape)
        parts.append(a.ravel())
    pos = sd["embeddings.position_embeddings.weight"]
    assert pos.shape[0] >= P and pos.shape[1] == H, ("position_embeddings", pos.shape)
    parts.append(pos[:P].ravel())
    parts.append(sd["embeddings.token_type_embeddings.weight"][0].ravel())  # type 0 only
    add("embeddings.LayerNorm.weight", (H,))
    add("embeddings.LayerNorm.bias", (H,))
    for l in range(L):
        p = f"encoder.layer.{l}."
        add(p + "attention.self.query.weight", (H, H)); add(p + "attention.self.query.bias", (H,))
        add(p + "attention.self.key.weight", (H, H));   add(p + "attention.self.key.bias", (H,))
        add(p + "attention.self.value.weight", (H, H)); add(p + "attention.self.value.bias", (H,))
        add(p + "attention.output.dense.weight", (H, H)); add(p + "attention.output.dense.bias", (H,))
        add(p + "attention.output.LayerNorm.weight", (H,)); add(p + "attention.output.LayerNorm.bias", (H,))
        add(p + "intermediate.dense.weight", (I, H)); add(p + "intermediate.dense.bias", (I,))
        add(p + "output.dense.weight", (H, I)); add(p + "output.dense.bias", (H,))
        add(p + "output.LayerNorm.weight", (H,)); add(p + "output.LayerNorm.bias", (H,))
    rest = np.concatenate(parts).astype(np.float32)
    rest.tofile(os.path.join(out_dir, f"{key}_rest.bin"))
    print(f"  word {V*H*4/1e6:.0f}MB + rest {rest.size*4/1e6:.0f}MB", flush=True)

    meta = {"key": key, "hf_id": hf_id, "V": V, "H": H, "L": L, "I": I,
            "maxn": maxn, "pos_offset": pos_off, "pool_cls": m["pooling"] == "cls",
            "tokenizer": m["tokenizer"], "n_word": V * H, "n_rest": int(rest.size),
            "do_lower_case": bool(getattr(tok, "do_lower_case", False))}
    json.dump(meta, open(os.path.join(out_dir, f"{key}_meta.json"), "w"))

    _export_vocab(key, m, tok, out_dir, meta)
    _export_refs(key, st, tok, maxn, out_dir)
    print("  done", flush=True)


def _export_vocab(key, m, tok, out_dir, meta):
    from .maps import fold_map, nfkc_map
    if m["tokenizer"] == "spm":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=tok.vocab_file)
        score, piece_id = {}, {}
        mn = 1e18
        for i in range(sp.get_piece_size()):
            p = sp.id_to_piece(i)
            s = sp.get_score(i)
            score[p] = s
            piece_id[p] = i
            mn = min(mn, s)
        spm_json = {"score": score, "piece_id": piece_id, "min_score": mn,
                    "fairseq_offset": 1, "unk_id": tok.unk_token_id,
                    "cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id}
        json.dump(spm_json, open(os.path.join(out_dir, f"{key}_spm.json"), "w"),
                  ensure_ascii=False)
        print(f"  spm pieces: {len(score)}; generating exact nfkc map "
              f"(1.1M sp.normalize calls, ~1 min)...", flush=True)
        json.dump(nfkc_map(sp), open(os.path.join(out_dir, f"{key}_nfkc.json"), "w"),
                  ensure_ascii=False)
    else:
        v = tok.get_vocab()
        wp = {"vocab": v, "cls": v["[CLS]"], "sep": v["[SEP]"], "unk": v["[UNK]"]}
        json.dump(wp, open(os.path.join(out_dir, f"{key}_wordpiece.json"), "w"),
                  ensure_ascii=False)
        if meta["do_lower_case"]:
            fold_path = os.path.join(out_dir, "foldmap.json")
            if not os.path.exists(fold_path):  # model-independent, generate once
                json.dump(fold_map(), open(fold_path, "w"), ensure_ascii=False)


def _export_refs(key, st, tok, maxn, out_dir):
    texts = build_texts()
    emb = st.encode(texts, batch_size=16, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
    refs = []
    for t, e in zip(texts, emb):
        ids = tok(t, truncation=True, max_length=maxn)["input_ids"]
        refs.append({"text": t, "ids": ids, "emb": [round(float(x), 6) for x in e]})
    json.dump(refs, open(os.path.join(out_dir, f"{key}_refs.json"), "w"),
              ensure_ascii=False)
    print(f"  refs: {len(refs)}, emb dim {emb.shape[1]}", flush=True)
