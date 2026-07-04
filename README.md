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
at or above 0.996) for a 3x smaller memory footprint and the highest
throughput of the set; on Arm cores with relaxed SIMD (plv8 3.2.x;
PostgreSQL 18 on AWS) the gap over fp32 reaches 1.8-2x.

## Models

All six registered models are ported and verified end to end in-DB:

| Key | Base model | Params | Dim | Languages | License | Tokenizer parity | Cosine vs PyTorch |
|---|---|---|---|---|---|---|---|
| `all-minilm` | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | 384 | English | Apache-2.0 | exact (76/76) | 1.000000 |
| `serafim-100m` | [serafim-100m-ir](https://huggingface.co/PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir) | 100M | 768 | Portuguese | MIT | exact (76/76) | 1.000000 |
| `multilingual-minilm` | [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 118M | 384 | 50+ | Apache-2.0 | exact (76/76) | 1.000000 |
| `multilingual-e5-small` | [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 118M | 384 | 94 | MIT | exact (76/76) | 1.000000 |
| `serafim-335m` | [serafim-335m-ir](https://huggingface.co/PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir) | 335M | 1024 | Portuguese | MIT | exact (76/76) | 1.000000 |
| `bge-m3` | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 568M | 1024 | 100+ | MIT | exact (76/76) | 1.000000 |

Measured on PostgreSQL 17 + plv8 3.2.4 with the full test suite (`pytest
tests/`): exact token-id match on every reference text, end-to-end cosine
against sentence-transformers output (worst case 0.999999), and the
adversarial Unicode edge cases. Reproduce with `pg-transformers verify <key>`.

The two Serafim models are PORTULAN's sentence-encoder fine-tunes of
BERTimbau base and large (same weights lineage and vocabulary), so this is
BERTimbau in the form that produces useful sentence embeddings. Note that
`multilingual-e5-small` was trained with input prefixes: prepend
`"query: "` to search queries and `"passage: "` to documents (or
`"query: "` to both for symmetric similarity); embedding bare text degrades
its ranking quality.

Each model also has a weight-only int8 variant (same key plus `-int8`,
e.g. `serafim-335m-int8`): linear weights and the word table are
stored as int8 with one f32 scale per row. The kernel quantizes activations
per token (block-wise; exact u8 on the baseline kernel, u7 on the relaxed
one, which is what lets it use a single relaxed dot per 16 columns) and
runs the GEMMs in the integer domain with SIMD dot-product instructions,
so the int8 variants are the fastest as well as the smallest. Same tokenizer, same verify pipeline,
but parity is no longer exact; the measured numbers are in the benchmarks
below.

### Mixing with other runtimes

The embeddings are compatible with vectors produced by any other runtime
of the same model. The port is a faithful reimplementation, not a
retrained variant, so its vectors live in the model's original embedding
space. A common setup is to bulk-embed documents outside the database
(PyTorch or ONNX Runtime on a GPU, an embedding API serving the same
model) and use pg-transformers only at query time, or the other way
around. Three rules make it safe:

- Same base model on both sides, same weights. Never mix vectors across
  models.
- The outside pipeline must use the standard sentence-transformers
  conventions: the HF tokenizer, the model's own pooling (CLS for bge-m3,
  mean for the others), L2 normalization. This port reproduces those
  exactly, so if the spaces diverge, the difference is on the other side.
- Pick the wasm variant by how sensitive your ranking is. The fp32
  kernels deviate from the PyTorch vectors by about 0.1 degree (cosine
  0.999999), less than typical GPU fp16 noise; mix them freely. The int8
  kernels add up to a few degrees of noise per vector (cosine 0.996+),
  which most retrieval workloads never notice, but if results hinge on
  fine distinctions or on fixed similarity thresholds, compare top-k
  overlap against the fp32 model on your own data first.

Weights are converted locally from HuggingFace by you (bring-your-own-weights);
this repository contains no model weights.

## Benchmarks

In-DB, single core (one PostgreSQL backend), measured by `pg-transformers
verify` on its reference corpus, after warmup. Throughput is tokens
embedded per second on full-length documents (512 tokens; 256 for
all-minilm and 128 for multilingual-minilm, their maxima). Query latency
is the end-to-end time to embed
one short query (10-30 tokens), the interactive path. Cosine is the worst
case against the PyTorch original over the corpus. RAM is the measured
PostgreSQL backend RSS after loading the model and embedding (weights +
tokenizer data + activations); every session that embeds holds its own copy.

There are two kernel flavors: plv8 3.2.x (PostgreSQL 18 on AWS) has relaxed
SIMD (FMA, int8 dot products); plv8 3.1.x (PostgreSQL 14-17 on AWS) runs
the baseline SIMD kernels. `pgt_load` picks per session: relaxed when the
V8 supports it, except for int8 models on x86, which run the baseline
kernel everywhere (see below). The load message says which one you got.

On a laptop (Apple M5 Max performance core, plv8 3.2.4, v0.3.3 kernels):

| Key | Cosine (worst) | RAM | relaxed: tok/s | query | baseline: tok/s | query |
|---|---|---|---|---|---|---|
| `all-minilm` | 0.999999 | 0.24GB | 2778 | 7.1 ms | 2302 | 8.6 ms |
| `all-minilm-int8` | 0.9964 | 0.11GB | 4982 | 3.6 ms | 2017 | 9.6 ms |
| `serafim-100m` | 0.999999 | 0.60GB | 330 | 52 ms | 284 | 61 ms |
| `serafim-100m-int8` | 0.9989 | 0.29GB | 645 | 21 ms | 258 | 66 ms |
| `multilingual-minilm` | 0.999999 | 0.72GB | 1478 | 12.7 ms | 1233 | 14.9 ms |
| `multilingual-minilm-int8` | 0.9991 | 0.37GB | 2869 | 6.2 ms | 1061 | 16.7 ms |
| `multilingual-e5-small` | 0.999999 | 0.74GB | 1044 | 13.6 ms | 962 | 15.0 ms |
| `multilingual-e5-small-int8` | 0.9995 | 0.38GB | 1702 | 6.3 ms | 843 | 16.7 ms |
| `serafim-335m` | 0.999999 | 1.5GB | 93 | 190 ms | 77 | 233 ms |
| `serafim-335m-int8` | 0.9992 | 0.56GB | 190 | 73 ms | 71 | 232 ms |
| `bge-m3` | 0.999999 | 2.4GB | 94 | 181 ms | 81 | 210 ms |
| `bge-m3-int8` | 0.9965 | 0.79GB | 179 | 75 ms | 72 | 218 ms |

On server cores (EC2, PostgreSQL + plv8 in Docker, single session, v0.3.3
kernels; cells are tokens/s and ms per query). "PG 18" is plv8 3.2.4 with
the flavor `pgt_load` auto-picks; "PG 14-17" is a real plv8 3.1.10 (V8 9.7):

| Key | Graviton4 PG 18 | Graviton4 PG 14-17 | Xeon SPR PG 18 | Xeon SPR PG 14-17 |
|---|---|---|---|---|
| `all-minilm` | 1308 · 17 ms | 1126 · 19 ms | 1192 · 18 ms | 988 · 21 ms |
| `all-minilm-int8` | 2112 · 8 ms | 875 · 23 ms | 1377 · 13 ms | 1357 · 13 ms |
| `serafim-100m` | 159 · 134 ms | 138 · 152 ms | 150 · 152 ms | 127 · 167 ms |
| `serafim-100m-int8` | 314 · 48 ms | 117 · 152 ms | 196 · 82 ms | 194 · 83 ms |
| `bge-m3` | 43 · 448 ms | 38 · 502 ms | 41 · 506 ms | 34 · 555 ms |
| `bge-m3-int8` | 97 · 164 ms | 35 · 496 ms | 56 · 274 ms | 56 · 274 ms |

(Graviton4 = c8g.2xlarge, Neoverse V2; Xeon SPR = c7i.2xlarge, Sapphire
Rapids Platinum 8488C.)

What the two tables say: on same-flavor kernels a server core runs
2.0-2.4x slower than an M5 Max core, except the baseline int8 kernels on
Sapphire Rapids, which come in at 1.3-1.5x (their `i32x4.dot_i16x8_s` inner
loop lowers to pmaddwd, which x86 does relatively better than NEON; the
same kernels run 1.5x faster on SPR than on Graviton). The int8 variants
are the fastest option everywhere. On Arm, relaxed SIMD is where the int8
speed lives (one SDOT per 16 columns; 2.4-2.8x the baseline kernel), so
PostgreSQL 18 is a real upgrade on Graviton. On x86 the int8 models run
the baseline kernel on every PostgreSQL version, so PG 14-17 give up
almost nothing there; PG 18 buys x86 only the fp32 FMA gain (~20%). plv8
3.1.10's older V8 costs 1-2% versus the same baseline kernel on plv8
3.2.4 (the x86 int8 columns).

Why int8 ignores relaxed SIMD on x86: the V8 in plv8 3.2.x (11.5) lowers
the relaxed int8 dot to a 5-instruction sequence on all x86 (VNNI support
only arrived in V8 12.6, which no plv8 carries), so it cannot beat the
baseline `i32x4.dot_i16x8_s` kernel there; measured, it ties at best. On
top of that, V8 11.5's baseline compiler has a register-allocation bug in
that instruction (crbug.com/1484978, fixed in V8 11.9) that corrupts the
first embeds of a session until tier-up (verified on Sapphire Rapids;
caught by `verify`). `pgt_load` therefore never auto-picks relaxed for
quantized kernels on x86, and you should not force `--flavor relaxed` on
one. On Arm neither problem exists, and the relaxed kernel is both correct
and much faster.

### What the wasm layer costs

Same models, same protocol, same machines, outside the database
(`bench/node_wasm.js` runs the identical wasm blobs in Node 22;
`bench/native_cpu.py` runs the fastest CPU-only native paths, single
thread). serafim-100m shown; tokens/s:

| Runtime (single core) | Graviton4 fp32 | Graviton4 int8 | Xeon SPR fp32 | Xeon SPR int8 |
|---|---|---|---|---|
| in-database (plv8 3.2.4) | 159 | 314 | 150 | 196 |
| same wasm, Node 22 | 196 | 329 | 189 | 203 |
| native, PyTorch fp32 | 893 | - | 728 | - |
| native, ONNX Runtime int8 | - | 944 | - | 2239 |

Two conclusions. The database layer is nearly free on the int8 kernels:
plv8 runs them within 5% of Node. The fp32 kernels run about 20% slower
under plv8 than Node, which is V8 codegen on the FMA-heavy relaxed path
(11.5 vs 12.4), not PostgreSQL overhead. The real cost is wasm itself:
128-bit SIMD and no native int8 GEMM put it at 4-5x slower than native
fp32, and 3x (Arm) to 11x (SPR, where VNNI applies) slower than native
int8. That is the price of the convenience
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
