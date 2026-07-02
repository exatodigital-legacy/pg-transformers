# PostgreSQL 17 + plv8 built from source.
FROM postgres:17-bookworm

ARG PLV8_VERSION=v3.2.4

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates pkg-config cmake ninja-build \
      postgresql-server-dev-17 libtinfo6 python3 \
    && git clone --depth 1 --branch ${PLV8_VERSION} https://github.com/plv8/plv8.git /tmp/plv8 \
    && cd /tmp/plv8 && make && make install \
    && rm -rf /tmp/plv8 \
    && apt-get purge -y build-essential git cmake ninja-build && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
