"""Re-run the reference on this machine: LiquidAI/d1-3B's own code in transformers, float32 on the CPU, one question per
request, and write the same JSON lines as run_genai.py so score.py can compare them.

    python hf_reference.py --model <d1-3B folder> --requests requests.json --ids id1,id2 --out ref.jsonl
"""

import argparse
import json
import sys
import time

import torch
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).to(args.device).eval()
    load_s = time.perf_counter() - t0
    wanted = {i for i in args.ids.split(",") if i}
    fixture = json.load(open(args.requests, encoding="utf-8"))
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps({"_meta": True, "model": args.model, "ep": f"transformers-{args.device}", "load_s": load_s,
                              "torch": torch.__version__}) + "\n")
        for record in fixture["records"]:
            request = record["request"]
            if "images" in request or (wanted and record["id"] not in wanted):
                continue
            for qid, question in request["questions"].items():
                row = {"q": f"{record['id']}/{qid}"}
                t = time.perf_counter()
                try:
                    response = model.system_one(state=request["state"], questions={qid: question})
                except KeyError as error:
                    row["refused"] = f"KeyError {error}"
                    out.write(json.dumps(row) + "\n")
                    continue
                row["ms"] = (time.perf_counter() - t) * 1000
                row["tokens"] = response["usage"]["input_tokens"]
                answer = response["answers"][qid]
                if answer["type"] == "noul":
                    row["probs"] = {"true": answer["noul"], "false": 1.0 - answer["noul"]}
                else:
                    row["probs"] = answer["probabilities"]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
