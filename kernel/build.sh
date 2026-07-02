#!/bin/sh
# Build the wasm32+simd128 kernel for models -> ../artifacts/<key>.wasm
# Usage: ./build.sh [key ...]     (default: every model in ../models.toml)
set -e
cd "$(dirname "$0")"
KEYS="${*:-$(python3 -c "import tomllib; print(' '.join(tomllib.load(open('../models.toml','rb'))))")}"
mkdir -p ../artifacts
for k in $KEYS; do
  PGT_MODEL="$k" RUSTFLAGS="-C target-feature=+simd128" \
    cargo build --release --target wasm32-unknown-unknown
  cp target/wasm32-unknown-unknown/release/pgt_kernel.wasm "../artifacts/$k.wasm"
  echo "built artifacts/$k.wasm ($(wc -c < "../artifacts/$k.wasm") bytes)"
done
