# Adding a model

Any post-LN BERT or XLM-RoBERTa sentence encoder (the sentence-transformers
kind: token embeddings, N encoder layers, mean or CLS pooling, L2 normalize)
ports without code changes.

## Requirements

- Architecture: BERT or XLM-R shaped, post-LayerNorm, GELU FFN. RoPE models,
  pre-LN models, and decoder LLMs need kernel changes.
- Dims: `hidden`, `intermediate` and `hidden/heads` must be multiples of 16
  (the SIMD dot product's unroll). Checked at build time.
- Tokenizer: WordPiece (cased or uncased) or sentencepiece unigram. A new
  tokenizer family needs a new plv8 function.

## Steps

1. Add an entry to `models.toml`. Copy the dims from the model's
   `config.json` on HuggingFace (`vocab_size`, `hidden_size` -> `hidden`,
   `num_hidden_layers` -> `layers`, `intermediate_size` -> `intermediate`,
   `num_attention_heads` -> `heads`, `layer_norm_eps` -> `ln_eps`).
   `pos_offset` is 0 for BERT, 2 for RoBERTa-family. `max_tokens` is the
   sentence-transformers `max_seq_length`. Don't worry about typos: the
   exporter validates every dim against the real config and fails loudly.

2. Build and convert:

   ```sh
   kernel/build.sh my-model
   pg-transformers export my-model     # downloads from HF, writes artifacts/
   ```

3. Load and verify:

   ```sh
   pg-transformers load my-model --dsn ...
   pg-transformers verify my-model --dsn ...
   pytest tests/ -k my-model
   ```

   `verify` must show exact tokenizer match and end-to-end cosine ~1.0
   against the PyTorch ground truth. If the tokenizer diverges, `tests/
   diagnose_spm.py` (for spm models) pinpoints the first divergent pieces.

4. Open a PR with the registry entry and the verify output. Do not commit
   weights; artifacts stay local (bring-your-own-weights).

## int8 variants

Any model gets a weight-only int8 variant with a three-line registry entry:

```toml
[my-model-int8]
base = "my-model"       # inherits every field
quant = "int8"
min_cosine = 0.99       # verify threshold vs the fp32 original
```

Linear weights and the word table are stored as int8 with one f32 scale per
output row; biases, LayerNorm and positions stay f32. Weights shrink 4x. At
run time the kernel quantizes each linear's input activations to u8 (one
scale/offset per 128-column block per token) and computes the GEMM in the
integer domain - `i32x4.dot_i16x8_s` on baseline SIMD, single dot-product
instructions (SDOT/VNNI) on the relaxed-SIMD build - which also makes the
int8 variants the fastest ones. The port is no longer bit-exact, so `verify`
checks cosine against the fp32 PyTorch ground truth using `min_cosine`
instead of the 0.999 exact-port default; state the measured worst-case
cosine in your PR. Export reuses the base model's tokenizer and reference
artifacts when they are present.

## Sizing

Per-session memory is roughly the f32 weight bytes plus activations and
tokenizer data: `4 * (vocab_size * hidden + layers * (12 * hidden^2 + ...))`.
Measured RSS reference points: 22M params = ~0.24GB, 100M = ~0.60GB,
568M = ~2.4GB; int8 variants ~0.11/0.29/0.79GB. Check the target's memory
with `sql/probe.sql` before porting something big.
