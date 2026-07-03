"""Native-CPU reference bench: the same models outside the database, in
PyTorch and ONNX Runtime (fp32 and dynamic-int8), so users can see what the
wasm layer costs vs the fastest CPU-only path. Uses the exported refs'
token ids (identical inputs to the in-DB and Node benches) and the same
protocol as verify.py: warmup, tokens/s on long refs, ms/query on short refs.

  python bench/native_cpu.py <key>... [--threads 1] [--backends torch,onnx,onnx-int8]

Keys are base (fp32) registry keys; onnx-int8 is the native analog of the
{key}-int8 variants. --threads 1 matches the one-core-per-session reality of
plv8; pass more to see multi-threaded native numbers.

Needs: pip install torch onnxruntime optimum[exporters]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pg_transformers import registry  # noqa: E402


def bench(run, refs, label):
    long_ids = [r["ids"] for r in refs if len(r["ids"]) > 120][:20]
    short_ids = [r["ids"] for r in refs if 5 < len(r["ids"]) < 30][:40]
    for ids in long_ids[:2]:
        run(ids)
    t0 = time.perf_counter()
    for ids in long_ids:
        run(ids)
    ms = 1000 * (time.perf_counter() - t0)
    ntok = sum(len(x) for x in long_ids)
    print(f"{label}: {1000*ntok/ms:.0f} tokens/s on {len(long_ids)} docs "
          f"(mean {ntok/len(long_ids):.0f} tok, {ms/len(long_ids):.0f} ms/doc)", flush=True)
    t0 = time.perf_counter()
    for _ in range(3):
        for ids in short_ids:
            run(ids)
    ms = 1000 * (time.perf_counter() - t0)
    print(f"{label}: query latency {ms/(3*len(short_ids)):.1f} ms "
          f"({len(short_ids)} queries of 5-30 tok, x3)", flush=True)


def torch_runner(hf_id, threads):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(threads)
    model = AutoModel.from_pretrained(hf_id).eval()

    def run(ids):
        with torch.inference_mode():
            t = torch.tensor([ids])
            model(input_ids=t, attention_mask=torch.ones_like(t))
    return run


def onnx_path(key, hf_id):
    d = os.path.join(registry.artifacts_dir(), "onnx", key)
    if not os.path.exists(os.path.join(d, "model.onnx")):
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        ORTModelForFeatureExtraction.from_pretrained(hf_id, export=True).save_pretrained(d)
    return os.path.join(d, "model.onnx")


def onnx_runner(path, threads):
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}

    def run(ids):
        a = np.array([ids], dtype=np.int64)
        feed = {"input_ids": a, "attention_mask": np.ones_like(a)}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(a)
        sess.run(None, feed)
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("keys", nargs="+")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--backends", default="torch,onnx,onnx-int8")
    a = p.parse_args()
    backends = a.backends.split(",")

    for key in a.keys:
        m = registry.model(key)
        hf_id = m["hf_id"]
        refs = json.load(open(os.path.join(registry.artifacts_dir(), f"{key}_refs.json"),
                              encoding="utf-8"))
        print(f"== {key} ({hf_id}), {a.threads} thread(s)", flush=True)
        if "torch" in backends:
            bench(torch_runner(hf_id, a.threads), refs, "torch fp32")
        if "onnx" in backends or "onnx-int8" in backends:
            fp32 = onnx_path(key, hf_id)
            if "onnx" in backends:
                bench(onnx_runner(fp32, a.threads), refs, "onnx fp32")
            if "onnx-int8" in backends:
                q = fp32.replace("model.onnx", "model.int8.onnx")
                if not os.path.exists(q):
                    from onnxruntime.quantization import QuantType, quantize_dynamic
                    quantize_dynamic(fp32, q, weight_type=QuantType.QInt8)
                bench(onnx_runner(q, a.threads), refs, "onnx int8-dynamic")


if __name__ == "__main__":
    main()
