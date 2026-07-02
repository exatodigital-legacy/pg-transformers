"""Load exported artifacts into a PostgreSQL database.

Creates the pgt_* tables/functions (sql/pg_transformers.sql), upserts the
model's wasm + meta + weight chunks + tokenizer data, then calls pgt_load()
once as a smoke test. Needs only psycopg + the artifacts (no ML deps).
"""
import json
import os

import psycopg

from . import registry

CHUNK = 32 * 1024 * 1024  # bytea chunk size; pgt_load streams these via cursor


def load(key, dsn=None, skip_weights=False):
    art = registry.artifacts_dir()
    meta = json.load(open(os.path.join(art, f"{key}_meta.json")))
    wasm = open(os.path.join(art, f"{key}.wasm"), "rb").read()

    conn = psycopg.connect(dsn or registry.default_dsn(), autocommit=True)
    cur = conn.cursor()
    cur.execute(open(registry.sql_path(), encoding="utf-8").read())

    cur.execute("delete from pgt_model where key=%s", (key,))
    cur.execute("delete from pgt_vocab where key=%s", (key,))
    cur.execute("insert into pgt_model values (%s,%s,%s)", (key, wasm, json.dumps(meta)))

    if not skip_weights:
        cur.execute("delete from pgt_rest where key=%s", (key,))
        cur.execute("delete from pgt_word where key=%s", (key,))
        for table, fname in (("pgt_rest", f"{key}_rest.bin"), ("pgt_word", f"{key}_word.bin")):
            data = open(os.path.join(art, fname), "rb").read()
            for i in range(0, len(data), CHUNK):
                cur.execute(f"insert into {table} values (%s,%s,%s)",
                            (key, i // CHUNK, data[i:i + CHUNK]))
            print(f"  {table}: {len(data)/1e6:.0f}MB in {(len(data)+CHUNK-1)//CHUNK} chunks")

    if meta["tokenizer"] == "spm":
        for kind, fname in (("spm", f"{key}_spm.json"), ("nfkc", f"{key}_nfkc.json")):
            data = open(os.path.join(art, fname), encoding="utf-8").read()
            cur.execute("insert into pgt_vocab values (%s,%s,%s)", (key, kind, data))
    else:
        wp = open(os.path.join(art, f"{key}_wordpiece.json"), encoding="utf-8").read()
        cur.execute("insert into pgt_vocab values (%s,'wordpiece',%s)", (key, wp))
        if meta.get("do_lower_case"):
            fold = open(os.path.join(art, "foldmap.json"), encoding="utf-8").read()
            cur.execute("insert into pgt_vocab values (%s,'fold',%s)", (key, fold))

    cur.execute("select pgt_load(%s)", (key,))
    print(cur.fetchone()[0])
    conn.close()
