-- pg-transformers capability probe. Answers: can THIS PostgreSQL run
-- pg-transformers? Works in any SQL client (no psql meta-commands).
-- Run the whole file; the SELECTs print a pass/fail report.
--
-- Requires: plv8 installed (CREATE EXTENSION may need rds_superuser or
-- equivalent on managed offerings).

CREATE EXTENSION IF NOT EXISTS plv8;

CREATE OR REPLACE FUNCTION pgt_probe() RETURNS text LANGUAGE plv8 AS $js$
  var out = [];
  function line(ok, name, detail) {
    out.push((ok ? 'PASS ' : 'FAIL ') + name + (detail ? ': ' + detail : ''));
    return ok;
  }

  // 1. WebAssembly runtime
  var hasWasm = line(typeof WebAssembly !== 'undefined', 'WebAssembly runtime',
                     typeof WebAssembly);
  if (!hasWasm) return out.join('\n');

  // 2. instantiate + run a minimal module: exports add(i32,i32)->i32
  try {
    var bytes = new Uint8Array([0,97,115,109,1,0,0,0,1,7,1,96,2,127,127,1,127,
      3,2,1,0,7,7,1,3,97,100,100,0,0,10,9,1,7,0,32,0,32,1,106,11]);
    var inst = new WebAssembly.Instance(new WebAssembly.Module(bytes));
    line(inst.exports.add(2, 3) === 5, 'wasm execute', 'add(2,3)=' + inst.exports.add(2, 3));
  } catch (e) { line(false, 'wasm execute', '' + e); }

  // 3. SIMD: validate a module using v128 (i32x4.splat / extract_lane)
  try {
    var simd = new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,127,3,2,1,0,
      10,14,1,12,0,65,0,253,17,253,27,0,26,65,42,11]);
    new WebAssembly.Module(simd);
    line(true, 'wasm SIMD (v128)');
  } catch (e) { line(false, 'wasm SIMD (v128)', '' + e); }

  // 4. relaxed SIMD (optional: FMA + int8 dot products, V8 11.4+, e.g.
  //    plv8 3.2.x). pg-transformers falls back to baseline SIMD without it,
  //    at lower throughput.
  try {
    var relaxed = new Uint8Array([0,97,115,109,1,0,0,0, 1,4,1,96,0,0, 3,2,1,0,
      10,62,1,60,0,
      253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      253,133,2, 26, 11]);
    var rOk = false;
    try { rOk = WebAssembly.validate(relaxed); } catch (e) {}
    line(true, 'wasm relaxed SIMD (optional)',
         rOk ? 'available (FMA + int8 dot kernels)'
             : 'not available; baseline SIMD kernels used');
  } catch (e) { line(true, 'wasm relaxed SIMD (optional)', 'not available: ' + e); }

  // 5. memory headroom (bge-m3 needs ~2.3GB in-wasm, int8 variant ~0.6GB;
  //    serafim ~0.45GB / 0.11GB; minilm ~0.1GB / 0.03GB)
  var mb = 0;
  try {
    for (var pages of [2048, 8192, 16384, 40960]) {  // 128MB..2.5GB
      var mem = new WebAssembly.Memory({ initial: pages });
      var v = new Float32Array(mem.buffer); v[v.length - 1] = 1.5;
      mb = pages / 16; mem = null; v = null;
    }
  } catch (e) {}
  line(mb >= 128, 'wasm memory', mb + 'MB allocatable' +
       (mb >= 2560 ? ' (all models incl. bge-m3)' : mb >= 512 ? ' (small/medium models)' : ''));

  // 6. how bytea arrives from SQL (informational; the loader accepts both)
  try {
    var r = plv8.execute("select '\\x0102'::bytea as b")[0].b;
    line(true, 'bytea to JS', Object.prototype.toString.call(r));
  } catch (e) { line(false, 'bytea to JS', '' + e); }

  globalThis.__pgt_probe = 41;   // checked by pgt_probe_session() below
  return out.join('\n');
$js$;

-- separate function + separate call: proves globalThis persists across
-- statements in a session (so model weights load once per connection)
CREATE OR REPLACE FUNCTION pgt_probe_session() RETURNS text LANGUAGE plv8 AS $js$
  return (globalThis.__pgt_probe === 41)
    ? 'PASS globalThis persists across calls (weights load once per session)'
    : 'FAIL globalThis did not persist (weights would reload per call)';
$js$;

SELECT current_setting('server_version') AS postgres, plv8_version() AS plv8;
SELECT pgt_probe() AS report;
SELECT pgt_probe_session() AS session_check;

DROP FUNCTION pgt_probe();
DROP FUNCTION pgt_probe_session();
