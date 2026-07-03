# Requirements (the capability contract)

pg-transformers does not target providers; it targets capabilities. If your
PostgreSQL has these, it runs, whoever hosts it:

1. **plv8 >= 3.1** installed (`CREATE EXTENSION plv8`). On managed offerings
   this means plv8 is on the provider's extension allow-list; no native code
   of ours is ever installed.
2. **WebAssembly with SIMD** exposed by plv8's V8. True for every plv8 3.1+
   build we have seen, but verify with the probe.
   Relaxed SIMD (FMA and int8 dot-product instructions) is optional and
   picked up automatically when present: plv8 3.2.x bundles V8 11.5, which
   has it; plv8 3.1.x bundles V8 9.7, which does not. On AWS that maps to
   PostgreSQL 18 (plv8 3.2.4) vs PostgreSQL 14-17 (plv8 3.1.10). What it
   buys depends on the CPU: on Arm (Graviton, Apple) the int8 variants are
   1.4-1.7x faster with it and fp32 about 1.1x; on x86 only the fp32
   models gain (~10%, FMA). `pgt_load` never auto-picks relaxed for int8
   models on x86: V8 runs the relaxed int8 dot slower than the baseline
   kernel there, and V8's first-tier compiler mis-lowers it on VNNI
   hardware (wrong embeddings until tier-up, measured on Sapphire Rapids).
   Do not force `--flavor relaxed` for int8 models on x86.
3. **Memory** for the model you pick, per session that embeds (measured
   backend RSS: weights + tokenizer data + activations):
   - all-minilm: ~0.24GB (int8 variant ~0.11GB)
   - serafim-100m: ~0.60GB (int8 ~0.29GB)
   - bge-m3: ~2.4GB (int8 ~0.79GB)
4. **Long-lived sessions.** Weights load once per connection (sub-second for
   small models, a few seconds for bge-m3) and are cached in `globalThis`.
   Use a connection pooler (RDS Proxy, pgbouncer in session mode) so the
   cache is reused.

Run `sql/probe.sql` (or `pg-transformers probe --dsn ...`) to check all of
this in one shot. It prints PASS/FAIL per capability plus how much wasm
memory is actually allocatable.

Notes that apply everywhere:

- The first embed of a session runs 20-30% slower than steady state: V8
  compiles wasm with its baseline compiler first, tiers hot functions up to
  the optimizing compiler in the background, and cannot switch in the middle
  of a call. `SET plv8.v8_flags = '--no-liftoff'` (a session-settable GUC,
  set it before the first plv8 call, e.g. in a pooler connect hook) compiles
  optimized code upfront and removes the penalty; steady state is unchanged.
- The wasm memory lives outside V8's heap accounting; a `plv8.memory_limit`
  GUC (where it exists) does not block it, and the GUC is session-settable
  anyway.
- bytea may arrive in JS as `Uint8Array` or `ArrayBuffer` depending on the
  plv8 build; the loader accepts both.
- Storage: weights are stored once in `pgt_word`/`pgt_rest` (TOAST-ed bytea
  chunks); size on disk is roughly the f32 weight bytes.
