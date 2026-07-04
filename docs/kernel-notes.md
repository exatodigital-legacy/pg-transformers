# Kernel notes: int8 quantization and SIMD flavors

The mechanics behind the int8 variants and the per-platform kernel
choices, moved here from the README for readers who want the details.

## How the int8 variants work

Linear weights and the word-embedding table are stored as int8 with one
f32 scale per output row (symmetric per-row quantization; biases,
LayerNorm parameters and position embeddings stay f32). At run time the
kernel quantizes each linear layer's input activations per token and per
column block, and computes the GEMMs in the integer domain with SIMD
dot-product instructions:

- the baseline SIMD kernel uses exact u8 activations in 128-column blocks
  with `i32x4.dot_i16x8_s`;
- the relaxed-SIMD kernel quantizes activations to u7 in 64-column
  blocks, which is the deterministic envelope of the relaxed int8 dot
  product (`i16x8.relaxed_dot_i8x16_i7x16_s`), so each 16 columns cost a
  single SDOT instruction on Arm.

This makes the int8 variants the fastest as well as the smallest models.
The port is no longer bit-exact against the fp32 original, so `verify`
checks worst-case cosine against the PyTorch ground truth instead
(`min_cosine` in `models.toml`, 0.995 for the shipped models; measured
worst cases are in the README benchmark tables). int8 models also spend a
moment in `pgt_load` precomputing weight row sums, part of moving the
activation zero-point out of the inner loop.

## Why int8 ignores relaxed SIMD on x86

The V8 in plv8 3.2.x (11.5) lowers the relaxed int8 dot to a
5-instruction sequence on all x86 (VNNI support only arrived in V8 12.6,
which no plv8 carries), so it cannot beat the baseline
`i32x4.dot_i16x8_s` kernel there; measured, it ties at best. On top of
that, V8 11.5's baseline compiler has a register-allocation bug in that
instruction ([crbug.com/1484978](https://crbug.com/1484978), fixed in V8
11.9) that corrupts the first embeds of a session until tier-up (verified
on Sapphire Rapids; caught by `verify`). `pgt_load` therefore never
auto-picks relaxed for quantized kernels on x86, and you should not force
`--flavor relaxed` on one. On Arm neither problem exists, and the relaxed
kernel is both correct and much faster: one SDOT per 16 columns gives the
int8 models 2.4-2.8x their baseline-kernel throughput on Graviton4.

On x86 the baseline int8 kernel is nothing to apologize for: its
`i32x4.dot_i16x8_s` inner loop lowers to `pmaddwd`, which x86 executes
relatively better than NEON does, and the same kernel runs about 1.5x
faster on Sapphire Rapids than on Graviton4.
