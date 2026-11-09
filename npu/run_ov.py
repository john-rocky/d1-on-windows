"""Answer questions on a static OpenVINO IR from export_ir.py; same output lines as run_genai.py.

    python run_ov.py --upstream <dir> --ir <dir>/model.xml --device NPU --ids ids.json --out out.jsonl [--passes 2] [--cache <dir>] [--config K=V] [--head lm_head.npy]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import openvino as ov


def prop(compiled, name):
    try:
        return str(compiled.get_property(name))
    except Exception:  # not every device reports every property
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--ir", required=True)
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--cache", default="")
    parser.add_argument("--config", action="append", default=[], help="OpenVINO property K=V, repeatable")
    parser.add_argument("--pad", type=int, default=124893)
    parser.add_argument("--head", default="", help="lm_head.npy, for a graph that returns the hidden state")
    args = parser.parse_args()

    from run_genai import load_upstream, probs_of

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from head import load_head, read_out

    up = load_upstream(args.upstream)
    head = load_head(args.head)
    core = ov.Core()
    if args.cache:
        core.set_property({"CACHE_DIR": args.cache})
    t = time.perf_counter()
    ir = core.read_model(args.ir)
    read_s = time.perf_counter() - t
    length = ir.input("input_ids").get_shape()[1]
    config = dict(item.split("=", 1) for item in args.config)
    t = time.perf_counter()
    compiled = core.compile_model(ir, args.device, config)
    compile_s = time.perf_counter() - t
    print(f"read {read_s:.1f} s, compile on {args.device} {compile_s:.1f} s, length {length}", flush=True)
    request = compiled.create_infer_request()
    with open(args.ids, encoding="utf-8") as f:
        rows = json.load(f)

    def forward(ids):
        padded = np.full((1, length), args.pad, dtype=np.int64)
        padded[0, : len(ids)] = ids
        request.infer({"input_ids": padded, "last": np.array(len(ids) - 1, dtype=np.int64)})
        return request.get_output_tensor(0).data.copy().reshape(-1)  # a quantized graph returns [1, vocab] on the CPU

    for _ in range(args.passes - 1):
        for row in rows:
            if "ids" in row and len(row["ids"]) <= length:
                forward(row["ids"])
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps({"_meta": True, "model": args.ir, "ep": f"OpenVINO {args.device} static {length}", "ep_options": config,
                              "load_s": read_s + compile_s, "openvino": ov.__version__,
                              "device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
                              "execution_devices": prop(compiled, "EXECUTION_DEVICES"),
                              "inference_precision": prop(compiled, "INFERENCE_PRECISION_HINT")}) + "\n")
        done = 0
        for row in rows:
            if "refused" in row:
                out.write(json.dumps({"q": row["q"], "refused": row["refused"]}, ensure_ascii=False) + "\n")
                continue
            if len(row["ids"]) > length:
                continue
            t = time.perf_counter()
            probs = read_out(up, forward(row["ids"]), row["group"], head)
            ms = (time.perf_counter() - t) * 1000
            out.write(json.dumps({"q": row["q"], "ms": ms, "tokens": len(row["ids"]),
                                  "probs": probs_of(up.answer(row["question"], probs))}, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
    print(f"done {done} questions", flush=True)


if __name__ == "__main__":
    main()
