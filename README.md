# pg-transformers

Transformer sentence embeddings inside PostgreSQL. A Rust kernel compiled to
WebAssembly runs in plv8's V8 engine, model weights live in regular tables,
and tokenizers are plain plv8 JavaScript. No native extensions beyond plv8,
which means it works on managed PostgreSQL where you cannot install anything:
AWS Aurora and RDS (verified), and any other offering whose extension
allow-list includes plv8.

```sql
select pgt_embed('bge-m3', 'The contract was terminated for cause.');
-- float4[1024], unit-normalized, identical to sentence-transformers output
```

Embeddings are faithful to the original models: every registered fp32 model
reproduces its HuggingFace/PyTorch output at cosine 1.000000 with exact
tokenizer id match, verified on a multilingual reference corpus plus an
adversarial Unicode suite (emoji, CJK, zero-width and bidi characters,
ligatures). `pg-transformers verify` reruns that proof against your database.
Each model also has an int8 variant that trades exact parity (cosine still
at or above 0.997) for a 3x smaller memory footprint and, where the V8 in
plv8 has relaxed SIMD (plv8 3.2.x; PostgreSQL 18 on AWS), about 1.5x the
fp32 throughput.

## Models

All three registered models are ported and verified end to end in-DB:

| Key | Base model | Params | Dim | Languages | License | Tokenizer parity | Cosine vs PyTorch |
|---|---|---|---|---|---|---|---|
| `all-minilm` | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | 384 | English | Apache-2.0 | exact (76/76) | 1.000000 |
| `serafim-100m` | [serafim-100m-ir](https://huggingface.co/PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir) | 100M | 768 | Portuguese | MIT | exact (76/76) | 1.000000 |
| `bge-m3` | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 568M | 1024 | 100+ | MIT | exact (76/76) | 1.000000 |

Measured on PostgreSQL 17 + plv8 3.2.4 with the full test suite (`pytest
tests/`): exact token-id match on every reference text, end-to-end cosine
against sentence-transformers output (worst case 0.999999), and the
adversarial Unicode edge cases. Reproduce with `pg-transformers verify <key>`.

Each model also has a weight-only int8 variant (`all-minilm-int8`,
`serafim-100m-int8`, `bge-m3-int8`): linear weights and the word table are
stored as int8 with one f32 scale per row. The kernel quantizes activations
per token (block-wise, one scale per 128 columns) and runs the GEMMs in the
integer domain with SIMD dot-product instructions, so the int8 variants are
the fastest as well as the smallest. Same tokenizer, same verify pipeline,
but parity is no longer exact; the measured numbers are in the benchmarks
below.

Weights are converted locally from HuggingFace by you (bring-your-own-weights);
this repository contains no model weights.

## Benchmarks

In-DB, single core, Apple M-series (server-grade Arm is typically 2-4x
slower per core), measured by `pg-transformers verify` on its reference
corpus, after warmup. Throughput is tokens embedded per second on
full-length documents (512 tokens; 256 for all-minilm, its maximum). Query
latency is the end-to-end time to embed one short query (10-30 tokens), the
interactive path. Cosine is the worst case against the PyTorch original
over the corpus. RAM is the measured PostgreSQL backend RSS after loading
the model and embedding (weights + tokenizer data + activations); every
session that embeds holds its own copy.

Two numbers per model because there are two kernel flavors: plv8 3.2.x
(PostgreSQL 18 on AWS) has relaxed SIMD (FMA, int8 dot products) and runs
the fast kernels; plv8 3.1.x (PostgreSQL 14-17 on AWS) runs the baseline
SIMD kernels. `pgt_load` detects and picks automatically; the load message
says which one you got.

| Key | Cosine (worst) | RAM | plv8 3.2: tok/s | query | plv8 3.1: tok/s | query |
|---|---|---|---|---|---|---|
| `all-minilm` | 0.999999 | 0.24GB | 2237 | 8.6 ms | 1945 | 9.9 ms |
| `all-minilm-int8` | 0.9973 | 0.11GB | 2924 | 6.3 ms | 1786 | 10.6 ms |
| `serafim-100m` | 0.999999 | 0.60GB | 284 | 60 ms | 250 | 69 ms |
| `serafim-100m-int8` | 0.9990 | 0.29GB | 363 | 43 ms | 231 | 74 ms |
| `bge-m3` | 0.999999 | 2.4GB | 80 | 216 ms | 68 | 249 ms |
| `bge-m3-int8` | 0.9973 | 0.79GB | 102 | 144 ms | 67 | 243 ms |

In document terms, on relaxed SIMD: serafim-100m-int8 embeds a full
512-token document in about 1.4s (fp32: 1.8s), bge-m3-int8 in about 5s
(fp32: 6.4s), and all-minilm-int8 a 256-token document in 88ms. On modern
plv8 the int8 variants are the fastest and smallest option; on plv8 3.1
they match fp32 speed on Arm (the baseline integer path pays off on x86,
where `i32x4.dot_i16x8_s` lowers to a single instruction) and still keep
the 3x memory win.

Two one-time costs to know about: the first embed of a session runs 20-30%
slower while V8 warms the wasm up to its optimizing compiler (it cannot
switch mid-call; `SET plv8.v8_flags = '--no-liftoff'` before the first
plv8 call removes the penalty entirely), and int8 models spend a moment in
`pgt_load` precomputing weight row sums.

For calibration: fp32 is roughly 6-9x slower than native PyTorch on the
same CPU, the cost of wasm's 128-bit SIMD, and the price of needing no
native extensions. In practice it matters less than it looks: queries (the
latency-critical path) are milliseconds, documents embed once at write
time, and bulk backfill scales linearly with parallel sessions since each
PostgreSQL backend runs on its own core (budget RAM per the table above).

## Prerequisites

- Python 3.11+
- Rust with the wasm target: `rustup target add wasm32-unknown-unknown`
- A PostgreSQL with plv8 3.1+. No server nearby? `docker compose -f
  docker/compose.yml up -d` builds and starts one on localhost:5432.

## Quickstart

```sh
pip install -e '.[export]'
export PGT_DSN="host=... port=5432 user=... dbname=..."   # or --dsn per command

# 0. can your PostgreSQL run this? (ten seconds, works on any provider)
pg-transformers probe

# 1. build the wasm kernels (baseline + relaxed-simd twin) and convert the model from HuggingFace
kernel/build.sh all-minilm
pg-transformers export all-minilm

# 2. load into the database (weights + tokenizer data) and prove parity
pg-transformers load all-minilm
pg-transformers verify all-minilm
```

`export` downloads the model from HuggingFace and writes `artifacts/`
(~90MB for all-minilm, ~2.3GB for bge-m3); `load` pushes them into the
`pgt_*` tables, so budget a few minutes for the big models. Then, in SQL:

```sql
select pgt_load('all-minilm');                  -- once per session, <1s warm
select pgt_embed('all-minilm', 'some text');    -- float4[384]
select pgt_tokenize('all-minilm', 'some text'); -- HF-identical token ids
```

Pair with [pgvector](https://github.com/pgvector/pgvector) for indexing and
similarity search. Weights load once per session and are cached in
`globalThis`; use a connection pooler (RDS Proxy, pgbouncer in session mode)
so sessions live long.

## How it works

- `kernel/` is a small Rust encoder forward pass (embeddings, multi-head
  attention, GELU FFN, LayerNorm, pooling, L2 normalize) compiled to
  wasm32+simd128, about 10-25KB. Model dims are compile-time constants that
  `build.rs` generates from `models.toml`; `quant = "int8"` entries compile
  the integer-GEMM kernel instead. Every model builds twice: a baseline
  SIMD blob that any plv8 3.1+ runs, and a relaxed-SIMD twin (FMA, int8
  dot products) that `pgt_load` picks automatically when the session's V8
  validates the feature (plv8 3.2.x / V8 11.4+).
- `sql/pg_transformers.sql` holds the loader and tokenizers. The loader
  streams weight chunks from tables into wasm memory via a cursor; tokenizers
  (WordPiece and sentencepiece-unigram Viterbi) run in plv8 JavaScript with
  Unicode normalization shipped as precomputed data, since plv8's reduced ICU
  has no working `String.normalize`.
- `pg_transformers/` (Python) converts HF models to the weight layout, loads
  artifacts into a database, and verifies parity against ground truth.
- `tests/` re-checks all of it: exact tokenizer parity over the reference
  corpus, kernel numerics, end-to-end cosine, and the adversarial Unicode
  edge cases.

## Adding a model

Any post-LN BERT or XLM-RoBERTa sentence encoder ports with one registry
entry in `models.toml` and no code changes; the exporter validates the entry
against the real HF config. See [docs/adding-a-model.md](docs/adding-a-model.md).

## Deployment targets

The requirement is a capability, not a provider: plv8 whose V8 has
WebAssembly+SIMD, and enough memory for your model. `sql/probe.sql` answers
it in seconds. Verified so far: self-hosted PostgreSQL 17, AWS Aurora
PostgreSQL 18.3 and 17.9, AWS RDS PostgreSQL 17.10. The support matrix and
provider notes live in [docs/deployment](docs/deployment/README.md), and
matrix rows are community-contributed: run the probe on your provider and
open a PR.

## Acknowledgments

This project was designed, implemented and verified in collaboration with
Claude Fable 5 (Anthropic), working in Claude Code. Every model port is
machine-checked against its PyTorch original by the parity suite in tests/,
so correctness claims rest on the tests, not on who or what wrote the code.

## License

Apache-2.0 for everything in this repository. Converted model weights keep
the license of their source model (see the table above).
