"""Where a static IR spends its time on a device: OpenVINO's per-op profile for a few forwards.

    python profile_ov.py --ir <dir>/model.xml --device NPU --ids ids.json [--top 25] [--cache <dir>]
"""

import argparse
import collections
import json
import time

import numpy as np
import openvino as ov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", required=True)
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--cache", default="")
    parser.add_argument("--pad", type=int, default=124893)
    args = parser.parse_args()

    core = ov.Core()
    if args.cache:
        core.set_property({"CACHE_DIR": args.cache})
    ir = core.read_model(args.ir)
    length = ir.input("input_ids").get_shape()[1]
    t = time.perf_counter()
    compiled = core.compile_model(ir, args.device, {"PERF_COUNT": "YES"})
    print(f"compile on {args.device} {time.perf_counter() - t:.1f} s, length {length}", flush=True)
    request = compiled.create_infer_request()
    with open(args.ids, encoding="utf-8") as f:
        row = next(r for r in json.load(f) if "ids" in r and len(r["ids"]) <= length)
    padded = np.full((1, length), args.pad, dtype=np.int64)
    padded[0, : len(row["ids"])] = row["ids"]
    inputs = {"input_ids": padded, "last": np.array(len(row["ids"]) - 1, dtype=np.int64)}
    walls = []
    for _ in range(6):
        t = time.perf_counter()
        request.infer(inputs)
        walls.append((time.perf_counter() - t) * 1000)
    print("forward ms", [round(w, 1) for w in walls], flush=True)

    rows = request.get_profiling_info()
    ran = [r for r in rows if r.real_time.total_seconds() > 0 or r.cpu_time.total_seconds() > 0]
    print(f"profile rows {len(rows)}, with time {len(ran)}", flush=True)
    by_type = collections.Counter()
    by_exec = collections.Counter()
    for r in ran:
        ms = r.real_time.total_seconds() * 1000
        by_type[r.node_type] += ms
        by_exec[r.exec_type] += ms
    total = sum(by_type.values())
    print(f"sum of real time {total:.1f} ms", flush=True)
    print("by exec type", [(k, round(v, 2)) for k, v in by_exec.most_common(12)], flush=True)
    print("by op type", [(k, round(v, 2)) for k, v in by_type.most_common(25)], flush=True)
    for r in sorted(ran, key=lambda r: -r.real_time.total_seconds())[: args.top]:
        print(f"{r.real_time.total_seconds() * 1000:8.3f} ms  {r.exec_type:<16} {r.node_type:<20} {r.node_name[:110]}", flush=True)


if __name__ == "__main__":
    main()
