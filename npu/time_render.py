"""How long does upstream's decide() take without the model? The NPU rows are timed from ready token ids, the GenAI rows from the request.

    python time_render.py --upstream <dir> -m <any GenAI export of d1-3B> --requests speed_requests.json
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--requests", required=True)
    args = parser.parse_args()

    from run_genai import load_upstream

    up = load_upstream(args.upstream)
    import onnxruntime_genai as og

    model = og.Model(og.Config(args.model_path))

    class NoModel(up.D1Decisions):
        def prefill(self, head_ids, inputs, tail, group):
            self.tokens = len(head_ids) + len(tail)
            return [1.0 / len(group)] * len(group)

    runner = NoModel(model, up.model_context_length(args.model_path))
    with open(args.requests, encoding="utf-8") as f:
        fixture = json.load(f)
    by = {}
    for _ in range(3):
        for record in fixture["records"]:
            request = record["request"]
            for qid, question in request["questions"].items():
                t = time.perf_counter()
                runner.decide({"state": request["state"], "questions": {qid: question}})
                by.setdefault(runner.tokens, []).append((time.perf_counter() - t) * 1000)
    for tokens, values in sorted(by.items()):
        print(f"{tokens} tokens: {statistics.median(values):.2f} ms without the model (n={len(values)})")


if __name__ == "__main__":
    main()
