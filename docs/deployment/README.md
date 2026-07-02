# Deployment targets

pg-transformers runs anywhere the [capability contract](requirements.md)
holds: plv8 with wasm+SIMD, and enough memory. `sql/probe.sql` answers "does
my PostgreSQL support this?" in ten seconds, on any provider, including ones
not listed here.

## Support matrix

Status meanings: **verified** = we ran the probe and loaded at least one
model; **probe-verified** = probe passed, models not yet exercised;
**untested** = run the probe and tell us.

| Target | plv8 | wasm+SIMD | Notes | Status |
|---|---|---|---|---|
| Self-hosted PostgreSQL 17 | 3.2.4 | yes | [notes](notes/self-hosted.md) | verified (all 3 models) |
| AWS Aurora PostgreSQL 18.3 | 3.2.4 | yes | [notes](notes/aws-aurora.md) | probe-verified |
| AWS Aurora PostgreSQL 17.9 | 3.1.10 | yes | [notes](notes/aws-aurora.md) | probe-verified |
| AWS RDS PostgreSQL 17.10 | 3.1.10 | yes | [notes](notes/aws-rds.md) | probe-verified |
| GCP Cloud SQL for PostgreSQL | ? | ? | plv8 is on the extension list; unconfirmed by us | untested |
| GCP AlloyDB | ? | ? | | untested |
| Azure Database for PostgreSQL (Flexible) | ? | ? | check the allow-list for plv8 | untested |
| Neon | ? | ? | plv8 documented as available | untested |
| Supabase | ? | ? | plv8 documented as available | untested |
| Crunchy Bridge / Timescale / Heroku | ? | ? | | untested |

## Contributing a row

Run `sql/probe.sql` on your target and open a PR (or issue) with: provider
and PostgreSQL version, `plv8_version()`, the probe output, and, if you went
further, which model you loaded and the `pg-transformers verify` output.
That is the entire process; the matrix is community-maintained evidence, not
marketing.
