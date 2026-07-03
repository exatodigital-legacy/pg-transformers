// Standalone-wasm reference bench: runs the exact same kernel blobs outside
// PostgreSQL, in Node's V8, so the plv8/in-database overhead can be measured
// separately from the wasm-vs-native gap. Mirrors the loader in
// sql/pg_transformers.sql (file reads instead of pgt_* tables) and the
// throughput protocol in pg_transformers/verify.py (warmup, then tokens/s on
// long refs and ms/query on short refs).
//
//   node bench/node_wasm.js <key> [--flavor baseline|relaxed] [--artifacts dir]
'use strict';
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const key = args.find(a => !a.startsWith('--'));
const flag = n => { const i = args.indexOf('--' + n); return i >= 0 ? args[i + 1] : null; };
if (!key) { console.error('usage: node bench/node_wasm.js <key> [--flavor baseline|relaxed] [--artifacts dir]'); process.exit(2); }
const art = flag('artifacts') || process.env.PGT_ARTIFACTS || path.join(__dirname, '..', 'artifacts');

// same 60-byte relaxed-SIMD probe as pgt_load
const probe = new Uint8Array([0x00,0x61,0x73,0x6d,1,0,0,0, 1,4,1,0x60,0,0, 3,2,1,0,
  0x0a,0x3e,1,0x3c,0,
  0xfd,0x0c,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0xfd,0x0c,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0xfd,0x0c,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0xfd,0x85,0x02, 0x1a, 0x0b]);
const relaxedOk = WebAssembly.validate(probe);
const use = flag('flavor') || (relaxedOk && fs.existsSync(path.join(art, key + '.relaxed.wasm')) ? 'relaxed' : 'baseline');
if (use === 'relaxed' && !relaxedOk) { console.error("this V8 has no relaxed SIMD; use --flavor baseline"); process.exit(1); }

const wasm = fs.readFileSync(path.join(art, use === 'relaxed' ? key + '.relaxed.wasm' : key + '.wasm'));
const ex = new WebAssembly.Instance(new WebAssembly.Module(wasm)).exports;

const restBytes = ex.n_rest_bytes ? ex.n_rest_bytes() : ex.n_rest() * 4;
const qwBytes = ex.n_qw_bytes ? ex.n_qw_bytes() : 0;
const wordBytes = ex.n_word_bytes ? ex.n_word_bytes() : ex.n_word() * 4;

const heapBase = ex.__heap_base ? ex.__heap_base.value : ex.__heap_base;
const wordBase = (heapBase + 15) & ~15;
const need = wordBase + wordBytes, have = ex.memory.buffer.byteLength;
if (need > have) ex.memory.grow(Math.ceil((need - have) / 65536));

const rest = fs.readFileSync(path.join(art, key + '_rest.bin'));
if (rest.length !== restBytes + qwBytes) { console.error(`rest.bin is ${rest.length} bytes, kernel expects ${restBytes + qwBytes}`); process.exit(1); }
new Uint8Array(ex.memory.buffer, ex.rest_ptr(), restBytes).set(rest.subarray(0, restBytes));
if (qwBytes > 0) new Uint8Array(ex.memory.buffer, ex.qw_ptr(), qwBytes).set(rest.subarray(restBytes));
const word = fs.readFileSync(path.join(art, key + '_word.bin'));
if (word.length !== wordBytes) { console.error(`word.bin is ${word.length} bytes, kernel expects ${wordBytes}`); process.exit(1); }
new Uint8Array(ex.memory.buffer, wordBase, wordBytes).set(word);
ex.set_word_base(wordBase);
if (ex.prep) ex.prep();

const H = ex.hidden(), MAXN = ex.max_tokens();
function embed(ids) {
  const n = Math.min(ids.length, MAXN);
  const view = new Uint32Array(ex.memory.buffer, ex.ids_ptr(), n);
  for (let i = 0; i < n; i++) view[i] = ids[i];
  ex.forward(n);
  return new Float32Array(ex.memory.buffer, ex.out_ptr(), H);
}

const refs = JSON.parse(fs.readFileSync(path.join(art, key + '_refs.json'), 'utf8'));

// sanity: cosine vs the PyTorch ground truth on a few refs
let worst = 1;
for (const r of refs.filter((_, i) => i % 10 === 0).slice(0, 8)) {
  const e = embed(r.ids);
  let dot = 0;
  for (let i = 0; i < H; i++) dot += e[i] * r.emb[i];
  worst = Math.min(worst, dot);
}
console.log(`${key} [${use === 'relaxed' ? 'relaxed-simd' : 'simd128'}] node ${process.version}, worst_cos=${worst.toFixed(6)}`);

// throughput, same selection as verify.py
const longIds = refs.filter(r => r.ids.length > 120).slice(0, 20).map(r => r.ids);
const shortIds = refs.filter(r => r.ids.length > 5 && r.ids.length < 30).slice(0, 40).map(r => r.ids);
const now = () => Number(process.hrtime.bigint()) / 1e6;

if (longIds.length) {
  for (const ids of longIds.slice(0, 2)) embed(ids);            // warmup (V8 tier-up)
  const t0 = now();
  for (const ids of longIds) embed(ids);
  const ms = now() - t0;
  const ntok = longIds.reduce((s, x) => s + x.length, 0);
  console.log(`throughput: ${(1000 * ntok / ms).toFixed(0)} tokens/s on ${longIds.length} docs ` +
              `(mean ${(ntok / longIds.length).toFixed(0)} tok, ${(ms / longIds.length).toFixed(0)} ms/doc)`);
}
if (shortIds.length) {
  const t0 = now();
  for (let r = 0; r < 3; r++) for (const ids of shortIds) embed(ids);
  const ms = now() - t0;
  console.log(`query latency: ${(ms / (3 * shortIds.length)).toFixed(1)} ms (${shortIds.length} queries of 5-30 tok, x3)`);
}
