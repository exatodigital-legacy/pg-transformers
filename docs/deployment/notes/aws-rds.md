# AWS RDS PostgreSQL

Probe-verified on RDS PostgreSQL 17.10 (plv8 3.1.10), us-east-1, db.t4g
(2026-07). Everything in the [Aurora notes](aws-aurora.md) applies to RDS as
well (memory GUC, Uint8Array bytea, pooling, Graviton sizing).

RDS additionally supports PL/Rust, so on RDS you have a native-code
alternative; pg-transformers still has the advantage of identical deployment
across RDS, Aurora and anything else with plv8.
