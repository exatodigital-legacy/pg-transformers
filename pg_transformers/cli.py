"""pg-transformers command line.

  pg-transformers export <key>...            HF model -> artifacts/
  pg-transformers load <key>... --dsn ...    artifacts -> database
  pg-transformers verify <key>... --dsn ...  in-DB parity vs ground truth
  pg-transformers probe --dsn ...            can this PostgreSQL run us?

DSN defaults to $PGT_DSN or localhost:5432.
"""
import argparse
import os
import sys

from . import registry


def main(argv=None):
    p = argparse.ArgumentParser(prog="pg-transformers", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    keys_help = f"model key(s) from {registry.registry_path()}"
    pe = sub.add_parser("export", help="HF model -> artifacts/")
    pe.add_argument("keys", nargs="+", help=keys_help)

    pl = sub.add_parser("load", help="artifacts -> database")
    pl.add_argument("keys", nargs="+", help=keys_help)
    pl.add_argument("--dsn", default=None)
    pl.add_argument("--skip-weights", action="store_true",
                    help="weights already in DB; refresh wasm/meta/vocab only")

    pv = sub.add_parser("verify", help="in-DB parity vs exported ground truth")
    pv.add_argument("keys", nargs="+", help=keys_help)
    pv.add_argument("--dsn", default=None)
    pv.add_argument("--refs-emb", type=int, default=0,
                    help="subsample embed loops to N refs (0 = all; tokenizer always all)")
    pv.add_argument("--thru-long", type=int, default=20)
    pv.add_argument("--thru-short", type=int, default=40)
    pv.add_argument("--flavor", choices=["baseline", "relaxed"], default=None,
                    help="force a kernel flavor (default: auto-detect)")

    pp = sub.add_parser("probe", help="capability check for a database")
    pp.add_argument("--dsn", default=None)

    a = p.parse_args(argv)

    if a.cmd == "export":
        from .export import export
        for k in a.keys:
            export(k)
    elif a.cmd == "load":
        from .load import load
        for k in a.keys:
            load(k, dsn=a.dsn, skip_weights=a.skip_weights)
    elif a.cmd == "verify":
        from .verify import verify
        ok = all([verify(k, dsn=a.dsn, refs_emb=a.refs_emb, thru_long=a.thru_long,
                         thru_short=a.thru_short, flavor=a.flavor) for k in a.keys])
        sys.exit(0 if ok else 1)
    elif a.cmd == "probe":
        import psycopg
        sql = open(os.path.join(registry.repo_root(), "sql", "probe.sql"),
                   encoding="utf-8").read()
        conn = psycopg.connect(a.dsn or registry.default_dsn(), autocommit=True)
        cur = conn.cursor()
        cur.execute(sql)  # multi-statement; walk every result set
        while True:
            if cur.description:
                for row in cur.fetchall():
                    print("\n".join(str(c) for c in row))
            if not cur.nextset():
                break
        conn.close()


if __name__ == "__main__":
    main()
