#!/bin/sh
# Build the wasm32+simd128 kernel for models -> ../artifacts/<key>.wasm,
# plus a +relaxed-simd twin -> <key>.relaxed.wasm (FMA on the fp32 path,
# SDOT/VNNI int8 dot products on int8 models; needs V8 11.4+, e.g. plv8
# 3.2.x). pgt_load() picks the right one at run time.
# Usage: ./build.sh [key ...]     (default: every model in ../models.toml)
set -e
cd "$(dirname "$0")"
KEYS="${*:-$(python3 -c "import tomllib; print(' '.join(tomllib.load(open('../models.toml','rb'))))")}"
mkdir -p ../artifacts
for k in $KEYS; do
  PGT_MODEL="$k" RUSTFLAGS="-C target-feature=+simd128" \
    cargo build --release --target wasm32-unknown-unknown
  cp target/wasm32-unknown-unknown/release/pgt_kernel.wasm "../artifacts/$k.wasm"
  PGT_MODEL="$k" RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" \
    cargo build --release --target wasm32-unknown-unknown
  cp target/wasm32-unknown-unknown/release/pgt_kernel.wasm "../artifacts/$k.relaxed.wasm"
  echo "built artifacts/$k.wasm ($(wc -c < "../artifacts/$k.wasm") bytes)" \
       "+ $k.relaxed.wasm ($(wc -c < "../artifacts/$k.relaxed.wasm") bytes)"
done
