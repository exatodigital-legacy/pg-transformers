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

The goal is convenience, not throughput: text is embedded where it already
lives, in SQL, with no ETL pipeline, no embedding service to deploy, and no
GPU to operate. A native runtime on a dedicated machine will always be
faster; the benchmarks below measure by exactly how much, so you can decide
whether the simpler architecture is fast enough for your workload.

Embeddings are faithful to the original models: every registered fp32 model
reproduces its HuggingFace/PyTorch output at cosine 1.000000 with exact
tokenizer id match, verified on a multilingual reference corpus plus an
adversarial Unicode suite (emoji, CJK, zero-width and bidi characters,
ligatures). `pg-transformers verify` reruns that proof against your database.
Each model also has an int8 variant that trades exact parity (cosine still
at or above 0.997) for a 3x smaller memory footprint and the highest
throughput of the set; on Arm cores with relaxed SIMD (plv8 3.2.x;
PostgreSQL 18 on AWS) the gap over fp32 reaches 1.3-1.5x.

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

In-DB, single core (one PostgreSQL backend), measured by `pg-transformers
verify` on its reference corpus, after warmup. Throughput is tokens
embedded per second on full-length documents (512 tokens; 256 for
all-minilm, its maximum). Query latency is the end-to-end time to embed
one short query (10-30 tokens), the interactive path. Cosine is the worst
case against the PyTorch original over the corpus. RAM is the measured
PostgreSQL backend RSS after loading the model and embedding (weights +
tokenizer data + activations); every session that embeds holds its own copy.

There are two kernel flavors: plv8 3.2.x (PostgreSQL 18 on AWS) has relaxed
SIMD (FMA, int8 dot products); plv8 3.1.x (PostgreSQL 14-17 on AWS) runs
the baseline SIMD kernels. `pgt_load` picks per session: relaxed when the
V8 supports it, except for int8 models on x86, which run the baseline
kernel everywhere (see below). The load message says which one you got.

On a laptop (Apple M5 Max performance core, plv8 3.2.4):

| Key | Cosine (worst) | RAM | relaxed: tok/s | query | baseline: tok/s | query |
|---|---|---|---|---|---|---|
| `all-minilm` | 0.999999 | 0.24GB | 2237 | 8.6 ms | 1945 | 9.9 ms |
| `all-minilm-int8` | 0.9973 | 0.11GB | 2924 | 6.3 ms | 1786 | 10.6 ms |
| `serafim-100m` | 0.999999 | 0.60GB | 284 | 60 ms | 250 | 69 ms |
| `serafim-100m-int8` | 0.9990 | 0.29GB | 363 | 43 ms | 231 | 74 ms |
| `bge-m3` | 0.999999 | 2.4GB | 80 | 216 ms | 68 | 249 ms |
| `bge-m3-int8` | 0.9973 | 0.79GB | 102 | 144 ms | 67 | 243 ms |

On server cores (EC2, PostgreSQL + plv8 in Docker, single session; cells
are tokens/s and ms per query). "PG 18" is plv8 3.2.4 with the flavor
`pgt_load` auto-picks; "PG 14-17" is a real plv8 3.1.10 (V8 9.7):

| Key | Graviton4 PG 18 | Graviton4 PG 14-17 | Xeon SPR PG 18 | Xeon SPR PG 14-17 |
|---|---|---|---|---|
| `all-minilm` | 962 · 19 ms | 875 · 21 ms | 873 · 18 ms | 758 · 21 ms |
| `all-minilm-int8` | 1286 · 14 ms | 745 · 25 ms | 1001 · 13 ms | 983 · 14 ms |
| `serafim-100m` | 137 · 131 ms | 123 · 151 ms | 122 · 141 ms | 109 · 163 ms |
| `serafim-100m-int8` | 177 · 94 ms | 104 · 170 ms | 149 · 93 ms | 148 · 93 ms |
| `bge-m3` | 40 · 461 ms | 36 · 507 ms | 33 · 454 ms | 29 · 526 ms |
| `bge-m3-int8` | 51 · 310 ms | 31 · 550 ms | 44 · 298 ms | 44 · 301 ms |

(Graviton4 = c8g.2xlarge, Neoverse V2; Xeon SPR = c7i.2xlarge, Sapphire
Rapids Platinum 8488C.)

What the two tables say: a server core runs these kernels 2.0-2.4x slower
than an M5 Max core, with Graviton4 and Sapphire Rapids within about 15% of
each other. The int8 variants are the fastest option everywhere. On Arm,
relaxed SIMD is where the int8 speed lives (SDOT; 1.4-1.7x the baseline
kernel), so PostgreSQL 18 is a real upgrade on Graviton. On x86 the int8
models run the baseline kernel on every PostgreSQL version, so PG 14-17
give up almost nothing there; PG 18 buys x86 only the fp32 FMA gain
(~10%). plv8 3.1.10's older V8 costs about 5% versus the same baseline
kernel on plv8 3.2.4.

Why int8 ignores relaxed SIMD on x86: V8's optimizing compiler lowers the
relaxed int8 dot product slower than the baseline `i32x4.dot_i16x8_s` path
there (measured 2x slower), and its first-tier compiler mis-lowers the
instruction's operand signedness on VNNI hardware, corrupting the first
embeds of a session until tier-up (verified on Sapphire Rapids; caught by
`verify`). `pgt_load` therefore never auto-picks it for quantized kernels
on x86, and you should not force `--flavor relaxed` on one.

### What the wasm layer costs

Same models, same protocol, same machines, outside the database
(`bench/node_wasm.js` runs the identical wasm blobs in Node 22;
`bench/native_cpu.py` runs the fastest CPU-only native paths, single
thread). serafim-100m shown; tokens/s:

| Runtime (single core) | Graviton4 fp32 | Graviton4 int8 | Xeon SPR fp32 | Xeon SPR int8 |
|---|---|---|---|---|
| in-database (plv8 3.2.4) | 137 | 177 | 122 | 149 |
| same wasm, Node 22 | 151 | 185 | 151 | 181 |
| native, PyTorch fp32 | 927 | - | 841 | - |
| native, ONNX Runtime int8 | - | 894 | - | 2504 |

Two conclusions. The database layer is nearly free: plv8 runs the wasm
within 5% of Node on Arm (the larger x86 gap is Node's newer V8, 12.4 vs
11.5, not PostgreSQL). The real cost is wasm itself: 128-bit SIMD and no
native int8 GEMM put it at 5-7x slower than native fp32, and further from
native int8 where VNNI applies (SPR). That is the price of the convenience
this project exists for: running on a managed database with no native
extensions and no infrastructure beside it. In practice it matters less
than it looks: queries (the latency-critical path) are still tens of
milliseconds, documents embed once at write time, and bulk backfill scales
linearly with parallel sessions since each PostgreSQL backend runs on its
own core (budget RAM per the table above; native multi-thread numbers for
comparison are in `bench/`).

Two one-time costs to know about: the first embed of a session runs 20-30%
slower while V8 warms the wasm up to its optimizing compiler (it cannot
switch mid-call; `SET plv8.v8_flags = '--no-liftoff'` before the first
plv8 call removes the penalty entirely), and int8 models spend a moment in
`pgt_load` precomputing weight row sums.

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
