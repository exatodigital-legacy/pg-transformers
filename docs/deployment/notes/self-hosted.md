# Self-hosted PostgreSQL

Verified with all three models on PostgreSQL 17 + plv8 3.2.4 (arm64 and
x86_64 containers).

- `docker compose -f docker/compose.yml up` starts PostgreSQL 17 + plv8 on
  port 5432, pulling the prebuilt multi-arch image from
  `ghcr.io/exatodigital-legacy/postgres-plv8` (add `--build` to compile plv8
  from source via docker/Dockerfile instead).
- For an existing server, install plv8 3.1+ from your distro packages or from
  source (https://github.com/plv8/plv8), then run `sql/probe.sql`.
- Memory: the per-session model weights live in the backend process, not
  shared buffers. Size `work_mem`-style thinking accordingly: N sessions
  embedding with bge-m3 = N x 2.3GB of RSS on top of normal PostgreSQL usage.
