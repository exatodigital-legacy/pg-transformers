# AWS RDS PostgreSQL

Probe-verified on RDS PostgreSQL 17.10 (plv8 3.1.10), us-east-1, db.t4g
(2026-07). Everything in the [Aurora notes](aws-aurora.md) applies to RDS as
well (memory GUC, Uint8Array bytea, pooling, Graviton sizing).

Kernel flavor by engine version mirrors Aurora: RDS PostgreSQL 18 ships plv8
3.2.4 (V8 11.5, relaxed-SIMD kernels); 14-17 ship plv8 3.1.10 (baseline SIMD
only, automatic fallback).

RDS PostgreSQL 14-17 additionally support PL/Rust as a native-code
alternative, but AWS dropped it from RDS PostgreSQL 18, so it is a dead end
for new work; pg-transformers deploys identically across RDS, Aurora and
anything else with plv8.
