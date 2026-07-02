-- pg-transformers: transformer sentence embeddings inside PostgreSQL.
-- Requires only plv8 (V8 with WebAssembly+SIMD). Multiple models coexist,
-- keyed by name:
--   pgt_model(key, wasm bytea, meta jsonb)
--   pgt_rest(key, idx, chunk bytea)   -- positions+layers, ordered
--   pgt_word(key, idx, chunk bytea)   -- word-embedding table, ordered
--   pgt_vocab(key, kind text, data text)  -- kind: 'wordpiece'|'spm'|'fold'|'nfkc'
-- Load a model into the session with pgt_load(key); embed with pgt_embed(key, text).

CREATE EXTENSION IF NOT EXISTS plv8;

CREATE TABLE IF NOT EXISTS pgt_model (key text PRIMARY KEY, wasm bytea, meta jsonb);
CREATE TABLE IF NOT EXISTS pgt_rest (key text, idx int, chunk bytea, PRIMARY KEY (key, idx));
CREATE TABLE IF NOT EXISTS pgt_word (key text, idx int, chunk bytea, PRIMARY KEY (key, idx));
CREATE TABLE IF NOT EXISTS pgt_vocab (key text, kind text, data text, PRIMARY KEY (key, kind));

CREATE OR REPLACE FUNCTION pgt_load(mkey text) RETURNS text LANGUAGE plv8 AS $js$
  globalThis.__pgt = globalThis.__pgt || {};
  if (globalThis.__pgt[mkey]) return 'already loaded';
  var t0 = Date.now();
  var u8 = function (b) { return (b instanceof Uint8Array) ? b : new Uint8Array(b); };
  var meta = plv8.execute("select meta from pgt_model where key=$1", [mkey])[0].meta;
  var wasm = u8(plv8.execute("select wasm from pgt_model where key=$1", [mkey])[0].wasm);
  var inst = new WebAssembly.Instance(new WebAssembly.Module(wasm));
  var ex = inst.exports;

  // grow memory to fit static rest region + dynamically-placed word table
  var heapBase = ex.__heap_base ? ex.__heap_base.value : ex.__heap_base;
  var wordBase = (heapBase + 15) & ~15;
  var wordBytes = ex.n_word() * 4;
  var need = wordBase + wordBytes;
  var have = ex.memory.buffer.byteLength;
  if (need > have) ex.memory.grow(Math.ceil((need - have) / 65536));

  // Stream chunks ONE AT A TIME via a cursor. plv8.execute() would materialize
  // every chunk simultaneously (2.2GB for bge-m3) on top of the wasm memory and
  // OOM V8; a cursor holds only the current chunk.
  function loadInto(tbl, dst, want) {
    var plan = plv8.prepare("select chunk from " + tbl + " where key=$1 order by idx", ["text"]);
    var cur = plan.cursor([mkey]);
    var off = 0, row;
    while ((row = cur.fetch())) { var c = u8(row.chunk); dst.set(c, off); off += c.length; }
    cur.close(); plan.free();
    if (off !== want) plv8.elog(ERROR, tbl + ' bytes ' + off + ' != ' + want);
  }
  loadInto("pgt_rest", new Uint8Array(ex.memory.buffer, ex.rest_ptr(), ex.n_rest() * 4), ex.n_rest() * 4);
  loadInto("pgt_word", new Uint8Array(ex.memory.buffer, wordBase, wordBytes), wordBytes);
  ex.set_word_base(wordBase);

  var vrows = plv8.execute("select kind, data from pgt_vocab where key=$1", [mkey]);
  var vocab = {}, kind = null;
  for (var i = 0; i < vrows.length; i++) {
    if (vrows[i].kind === 'wordpiece') { vocab.wp = JSON.parse(vrows[i].data); kind = 'wordpiece'; }
    else if (vrows[i].kind === 'spm') { vocab.spm = JSON.parse(vrows[i].data); kind = 'spm'; }
    else if (vrows[i].kind === 'fold') { vocab.fold = JSON.parse(vrows[i].data); }
    else if (vrows[i].kind === 'nfkc') { vocab.nfkc = JSON.parse(vrows[i].data); }
  }
  globalThis.__pgt[mkey] = { ex: ex, meta: meta, vocab: vocab, kind: kind };
  return 'loaded ' + mkey + ' (' + kind + ', ' + (ex.n_word()*4 + ex.n_rest()*4) +
         ' weight bytes) in ' + (Date.now() - t0) + 'ms';
$js$;

-- WordPiece (BERT), cased or uncased per meta. Returns int[] ids.
CREATE OR REPLACE FUNCTION pgt_tok_wordpiece(mkey text, input text) RETURNS int[] LANGUAGE plv8 AS $js$
  var m = globalThis.__pgt[mkey], V = m.vocab.wp, meta = m.meta;
  var vocab = V.vocab, fold = m.vocab.fold || {};
  var lower = !!meta.do_lower_case;

  function cp(c){return c.codePointAt(0);}
  // HF _is_whitespace: \t\n\r, space, and Unicode category Zs (full list)
  function isWs(c){return c===32||c===9||c===10||c===13||c===0xA0||c===0x1680||(c>=0x2000&&c<=0x200A)||c===0x202F||c===0x205F||c===0x3000;}
  // HF _is_control: categories Cc and Cf (except \t\n\r); Cf ranges = Unicode 15.1
  function isCf(c){
    return c===0xAD||(c>=0x600&&c<=0x605)||c===0x61C||c===0x6DD||c===0x70F
      ||(c>=0x890&&c<=0x891)||c===0x8E2||c===0x180E||(c>=0x200B&&c<=0x200F)
      ||(c>=0x202A&&c<=0x202E)||(c>=0x2060&&c<=0x2064)||(c>=0x2066&&c<=0x206F)
      ||c===0xFEFF||(c>=0xFFF9&&c<=0xFFFB)||c===0x110BD||c===0x110CD
      ||(c>=0x13430&&c<=0x1343F)||(c>=0x1BCA0&&c<=0x1BCA3)||(c>=0x1D173&&c<=0x1D17A)
      ||c===0xE0001||(c>=0xE0020&&c<=0xE007F);
  }
  function isCtl(c){if(c===9||c===10||c===13)return false;return c<32||(c>=0x7F&&c<=0x9F)||isCf(c);}
  function isPunct(c){
    if((c>=33&&c<=47)||(c>=58&&c<=64)||(c>=91&&c<=96)||(c>=123&&c<=126))return true;
    if(c===0xA1||c===0xA7||c===0xAB||c===0xB6||c===0xB7||c===0xBB||c===0xBF)return true;
    if(c>=0x2010&&c<=0x2027)return true; if(c>=0x2030&&c<=0x205E)return true;
    if((c>=0x3001&&c<=0x3011)||(c>=0x3014&&c<=0x301F)||c===0x30FB)return true;
    if((c>=0xFF01&&c<=0xFF0F)||(c>=0xFF1A&&c<=0xFF20))return true;
    return false;
  }
  function isCJK(c){return (c>=0x4E00&&c<=0x9FFF)||(c>=0x3400&&c<=0x4DBF)||(c>=0xF900&&c<=0xFAFF);}

  var buf='';
  for (var ch of input){var c=cp(ch);
    if(c===0||c===0xFFFD||isCtl(c))continue;
    if(isWs(c)){buf+=' ';continue;}
    buf += isCJK(c) ? (' '+ch+' ') : ch;
  }
  if (lower) {
    var lo=buf.toLowerCase(), f='';
    for (var ch of lo){var c=cp(ch),r=fold[c];f+=(r!==undefined)?r:ch;}
    buf=f;
  }
  var words=buf.split(/ +/).filter(Boolean), toks=[];
  for (var w=0; w<words.length; w++){var cur='';
    for (var ch of words[w]){ if(isPunct(cp(ch))){if(cur)toks.push(cur);toks.push(ch);cur='';} else cur+=ch; }
    if(cur)toks.push(cur);
  }
  var MAX=meta.maxn, ids=[V.cls];
  for (var t=0; t<toks.length && ids.length<MAX-1; t++){
    var word=toks[t]; if(word.length>100){ids.push(V.unk);continue;}
    var start=0, sub=[], bad=false;
    while(start<word.length){
      var end=word.length, id=-1;
      while(start<end){var s=(start>0?'##':'')+word.slice(start,end);
        if(Object.prototype.hasOwnProperty.call(vocab,s)){id=vocab[s];break;} end--;}
      if(id<0){bad=true;break;} sub.push(id); start=end;
    }
    if(bad)ids.push(V.unk); else for(var j=0;j<sub.length&&ids.length<MAX-1;j++)ids.push(sub[j]);
  }
  ids.push(V.sep);
  return ids;
$js$;

-- sentencepiece unigram (XLM-R). Viterbi best segmentation over normalized text.
CREATE OR REPLACE FUNCTION pgt_tok_spm(mkey text, input text) RETURNS int[] LANGUAGE plv8 AS $js$
  var m = globalThis.__pgt[mkey], S = m.vocab.spm, meta = m.meta;
  var score = S.score, off = S.fairseq_offset;   // piece -> unigram score; hf_id = spm_id+off
  var pieceId = S.piece_id;                       // piece -> spm_id
  var MARK = String.fromCharCode(0x2581);         // sentencepiece space marker (▁)
  var MAX_PIECE = 32;                             // probe bound; longest XLM-R piece is 16 chars
  var nfkc = m.vocab.nfkc || {};
  // nmt_nfkc normalization WITHOUT regex/String.normalize (both broken in
  // plv8's reduced ICU): per-codepoint map precomputed from the model's own
  // sentencepiece normalizer, then drop control/U+FFFD, collapse whitespace
  // runs, add dummy prefix, space->U+2581.
  function isWsCp(c){return c===32||c===9||c===10||c===13||c===0xA0||(c>=0x2000&&c<=0x200A)||c===0x3000;}
  var s = MARK, prevSpace = true;
  function push(ch) {
    var c = ch.codePointAt(0);
    if (c === 0xFFFD || (c < 32 && c !== 9 && c !== 10 && c !== 13)) return;
    if (isWsCp(c)) { if (!prevSpace) { s += MARK; prevSpace = true; } return; }
    s += ch; prevSpace = false;
  }
  for (var ch of input) {
    var rep = nfkc[ch.codePointAt(0)];         // 1->many possible (½ -> "1/2")
    if (rep === undefined) push(ch);
    else for (var rc of rep) push(rc);
  }
  // nmt_nfkc trims trailing whitespace: drop the trailing space marker
  if (s.length > 1 && s.charCodeAt(s.length - 1) === 0x2581) s = s.slice(0, -1);
  if (s === MARK) return [S.cls_id, S.sep_id];   // empty/whitespace-only input
  var n = s.length;
  var NEG = -1e18;
  var best = new Float64Array(n + 1).fill(NEG); best[0] = 0;
  var back = new Int32Array(n + 1).fill(-1);     // best segmentation: start of last piece
  var backId = new Int32Array(n + 1).fill(0);    // ...and that piece's spm_id (-1 = unk char)
  var UNKSC = S.min_score - 10.0;                // below every real piece: unk is last resort
  for (var i = 0; i < n; i++) {
    if (best[i] === NEG) continue;
    var maxsub = Math.min(n, i + MAX_PIECE);
    for (var j = i + 1; j <= maxsub; j++) {
      var piece = s.slice(i, j);
      var sc = score[piece];
      if (sc !== undefined) {
        var v = best[i] + sc;
        if (v > best[j]) { best[j] = v; back[j] = i; backId[j] = pieceId[piece]; }
      }
    }
    // single-char unknown fallback so we never dead-end
    var v1 = best[i] + UNKSC;
    if (v1 > best[i + 1]) { best[i + 1] = v1; back[i + 1] = i; backId[i + 1] = -1; }
  }
  // backtrack
  var out = [], pos = n;
  while (pos > 0) { out.push(backId[pos]); pos = back[pos]; }
  out.reverse();
  var ids = [S.cls_id];
  var MAX = meta.maxn;
  for (var k = 0; k < out.length && ids.length < MAX - 1; k++) {
    ids.push(out[k] < 0 ? S.unk_id : out[k] + off);
  }
  ids.push(S.sep_id);
  return ids;
$js$;

CREATE OR REPLACE FUNCTION pgt_tokenize(mkey text, input text) RETURNS int[] LANGUAGE plv8 AS $js$
  if (!globalThis.__pgt || !globalThis.__pgt[mkey]) plv8.execute("select pgt_load($1)", [mkey]);
  var kind = globalThis.__pgt[mkey].kind;
  var fn = kind === 'spm' ? 'pgt_tok_spm' : 'pgt_tok_wordpiece';
  return plv8.find_function(fn)(mkey, input);
$js$;

CREATE OR REPLACE FUNCTION pgt_embed_ids(mkey text, ids int[]) RETURNS float4[] LANGUAGE plv8 AS $js$
  if (!globalThis.__pgt || !globalThis.__pgt[mkey]) plv8.execute("select pgt_load($1)", [mkey]);
  var ex = globalThis.__pgt[mkey].ex, H = ex.hidden();
  var n = Math.min(ids.length, ex.max_tokens());
  var view = new Uint32Array(ex.memory.buffer, ex.ids_ptr(), n);
  for (var i = 0; i < n; i++) view[i] = ids[i];
  ex.forward(n);
  return Array.from(new Float32Array(ex.memory.buffer, ex.out_ptr(), H));
$js$;

CREATE OR REPLACE FUNCTION pgt_embed(mkey text, input text) RETURNS float4[] LANGUAGE plv8 AS $js$
  var ids = plv8.find_function("pgt_tokenize")(mkey, input);
  return plv8.find_function("pgt_embed_ids")(mkey, ids);
$js$;

-- throughput helper: embed each id-array `reps` times (skips tokenization)
CREATE OR REPLACE FUNCTION pgt_bench_ids(mkey text, id_arrays text, reps int) RETURNS float8 LANGUAGE plv8 AS $js$
  if (!globalThis.__pgt || !globalThis.__pgt[mkey]) plv8.execute("select pgt_load($1)", [mkey]);
  var arrs = JSON.parse(id_arrays), emb = plv8.find_function("pgt_embed_ids");
  var t0 = Date.now();
  for (var r = 0; r < reps; r++) for (var i = 0; i < arrs.length; i++) emb(mkey, arrs[i]);
  return Date.now() - t0;
$js$;
