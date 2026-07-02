# AWS Aurora PostgreSQL

Probe-verified on Aurora PostgreSQL 18.3 (plv8 3.2.4) and 17.9 (plv8 3.1.10),
us-east-1, db.t4g instances (2026-07).

- `CREATE EXTENSION plv8` works with the master user; plv8 is on Aurora's
  supported-extension list. PL/Rust, by contrast, was never on Aurora's list
  (and is dropped from RDS PostgreSQL 18), which is exactly why this project
  uses wasm-in-plv8.
- Kernel flavor by engine version: Aurora PostgreSQL 18 ships plv8 3.2.4
  (V8 11.5) and gets the relaxed-SIMD kernels (FMA; int8 SDOT on Graviton).
  Aurora PostgreSQL 14-17 ship plv8 3.1.10 (V8 9.7): baseline SIMD only,
  `pgt_load` falls back automatically. If throughput matters, prefer PG 18.
- AWS builds set `plv8.memory_limit = 256` (MB). This does not block wasm:
  `WebAssembly.Memory` is allocated outside V8's heap accounting. The GUC is
  also session-settable (`SET plv8.memory_limit = 1024`) with the master user.
- bytea arrives in plv8 as `Uint8Array` (local source builds may give
  `ArrayBuffer`). The loader handles both; custom code should too.
- Sizing: a db.t4g (burstable Graviton2) runs the wasm kernel at roughly a
  quarter of an Apple M-series core. Expect ~4x the throughput numbers in the
  README; r7g/r8g cores land closer to 2x. Budget instance memory for the
  model per concurrently-embedding session.
- Use RDS Proxy or pgbouncer so sessions (and the per-session weight cache)
  are long-lived.
