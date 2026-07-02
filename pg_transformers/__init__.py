"""pg-transformers: transformer sentence embeddings inside PostgreSQL.

A Rust kernel compiled to wasm32+simd128 runs in plv8's V8; weights live in
regular tables; tokenizers are plain plv8 JavaScript. No native extensions
beyond plv8, so it works on managed PostgreSQL (Aurora, RDS, and friends).
"""
__version__ = "0.1.0"
