# Self-hosted PostgreSQL

Verified with all three models on PostgreSQL 17 + plv8 3.2.4 (arm64 and
x86_64 containers).

- `docker compose -f docker/compose.yml up` builds a PostgreSQL 17 image with
  plv8 from source and starts it on port 5432.
- For an existing server, install plv8 3.1+ from your distro packages or from
  source (https://github.com/plv8/plv8), then run `sql/probe.sql`.
- Memory: the per-session model weights live in the backend process, not
  shared buffers. Size `work_mem`-style thinking accordingly: N sessions
  embedding with bge-m3 = N x 2.3GB of RSS on top of normal PostgreSQL usage.
