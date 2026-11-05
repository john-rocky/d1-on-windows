"""Token ids and readout groups of every question of a fixture, from upstream's own code, so the OpenVINO runner needs no GenAI model.

    python make_ids.py --upstream <dir> -m <any GenAI export of d1-3B> --requests requests.json --out ids.json [--max_tokens N]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_tokens", type=int, default=0)
    args = parser.parse_args()

    from run_genai import load_upstream

    up = load_upstream(args.upstream)
    import onnxruntime_genai as og

    model = og.Model(og.Config(args.model_path))

    class Capture(up.D1Decisions):
        def prefill(self, head_ids, inputs, tail, group):
            self.captured = (head_ids + tail, group)
            return [1.0 / len(group)] * len(group)

    runner = Capture(model, up.model_context_length(args.model_path))
    with open(args.requests, encoding="utf-8") as f:
        fixture = json.load(f)
    rows, skipped = [], 0
    for record in fixture["records"]:
        request = record["request"]
        if "images" in request:
            continue
        for qid, question in request["questions"].items():
            single = {"state": request["state"], "questions": {qid: question}}
            try:
                runner.decide(single)
            except ValueError as error:
                rows.append({"q": f"{record['id']}/{qid}", "refused": str(error)})
                continue
            ids, group = runner.captured
            if args.max_tokens and len(ids) > args.max_tokens:
                skipped += 1
                continue
            _, checked = up.check_request(single)
            rows.append({"q": f"{record['id']}/{qid}", "ids": ids, "group": group, "question": checked[qid]})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"{len(rows)} questions, {skipped} longer than {args.max_tokens} tokens left out")


if __name__ == "__main__":
    main()
