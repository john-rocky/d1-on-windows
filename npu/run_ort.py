"""Answer questions on a static IR through ONNX Runtime and the pip OpenVINO EP plugin; same output lines as run_genai.py.

    python run_ort.py --upstream <dir> --model <dir>/model_ctx.onnx --device NPU --ids ids.json --out out.jsonl [--passes 2] [--head lm_head.npy]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import onnxruntime as ort
import onnxruntime_ep_openvino as ov_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--pad", type=int, default=124893)
    parser.add_argument("--head", default="", help="lm_head.npy, for a graph that returns the hidden state")
    args = parser.parse_args()

    from run_genai import load_upstream, probs_of

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from head import load_head, read_out

    up = load_upstream(args.upstream)
    head = load_head(args.head)
    ort.register_execution_provider_library(ov_ep.get_ep_name(), ov_ep.get_library_path())
    devices = [d for d in ort.get_ep_devices() if d.ep_name == ov_ep.get_ep_name() and d.ep_metadata.get("ov_device") == args.device]
    if not devices:
        raise SystemExit(f"no OpenVINO EP device {args.device}: {[d.ep_metadata.get('ov_device') for d in ort.get_ep_devices()]}")
    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    t = time.perf_counter()
    session = ort.InferenceSession(args.model, sess_options=options)
    load_s = time.perf_counter() - t
    length = session.get_inputs()[0].shape[1]
    names = [i.name for i in session.get_inputs()]
    print(f"load on {args.device} {load_s:.1f} s, providers {session.get_providers()}, inputs {names}, length {length}", flush=True)
    with open(args.ids, encoding="utf-8") as f:
        rows = json.load(f)

    def forward(ids):
        padded = np.full((1, length), args.pad, dtype=np.int64)
        padded[0, : len(ids)] = ids
        return session.run(None, {"input_ids": padded, "last": np.array(len(ids) - 1, dtype=np.int64)})[0].reshape(-1)

    for _ in range(args.passes - 1):
        for row in rows:
            if "ids" in row and len(row["ids"]) <= length:
                forward(row["ids"])
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps({"_meta": True, "model": args.model, "ep": f"ONNX Runtime OpenVINO EP {args.device} static {length}",
                              "ep_options": {}, "load_s": load_s, "ort": ort.__version__, "providers": session.get_providers()}) + "\n")
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
