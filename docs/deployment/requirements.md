# Requirements (the capability contract)

pg-transformers does not target providers; it targets capabilities. If your
PostgreSQL has these, it runs, whoever hosts it:

1. **plv8 >= 3.1** installed (`CREATE EXTENSION plv8`). On managed offerings
   this means plv8 is on the provider's extension allow-list; no native code
   of ours is ever installed.
2. **WebAssembly with SIMD** exposed by plv8's V8. True for every plv8 3.1+
   build we have seen, but verify with the probe.
3. **Memory** for the model you pick, per session that embeds:
   - all-minilm: ~0.1GB
   - serafim-100m: ~0.45GB
   - bge-m3: ~2.3GB
4. **Long-lived sessions.** Weights load once per connection (sub-second for
   small models, a few seconds for bge-m3) and are cached in `globalThis`.
   Use a connection pooler (RDS Proxy, pgbouncer in session mode) so the
   cache is reused.

Run `sql/probe.sql` (or `pg-transformers probe --dsn ...`) to check all of
this in one shot. It prints PASS/FAIL per capability plus how much wasm
memory is actually allocatable.

Notes that apply everywhere:

- The wasm memory lives outside V8's heap accounting; a `plv8.memory_limit`
  GUC (where it exists) does not block it, and the GUC is session-settable
  anyway.
- bytea may arrive in JS as `Uint8Array` or `ArrayBuffer` depending on the
  plv8 build; the loader accepts both.
- Storage: weights are stored once in `pgt_word`/`pgt_rest` (TOAST-ed bytea
  chunks); size on disk is roughly the f32 weight bytes.
