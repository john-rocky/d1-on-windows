"""Answer the picture requests with GenAI (one resize per picture) or with LiquidAI/d1-3B's own code (tiles).

    python run_pictures.py genai --upstream <dir> -m <lfm2_vl model dir> -e cpu --pictures pictures --out genai.jsonl
    python run_pictures.py hf --model <d1-3B folder> [--device cuda] [--no_tiling] --pictures pictures --out hf.jsonl
"""

import argparse
import importlib.util
import json
import os
import sys
import time


def load_records(folder):
    with open(os.path.join(folder, "requests.json"), encoding="utf-8") as f:
        return json.load(f)


def run_genai(args):
    sys.path.insert(0, args.upstream)
    spec = importlib.util.spec_from_file_location("d1_decisions", os.path.join(args.upstream, "d1-decisions.py"))
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    import common
    import onnxruntime_genai as og

    common.register_ep(args.execution_provider, None, False)
    model = og.Model(common.get_config(args.model_path, args.execution_provider, None))
    runner = up.D1Decisions(model, up.model_context_length(args.model_path))
    with open(args.out, "w", encoding="utf-8") as out:
        for record in load_records(args.pictures):
            path = os.path.join(args.pictures, record["picture"])
            t = time.perf_counter()
            response = runner.decide(record["request"], [path])
            ms = (time.perf_counter() - t) * 1000
            answer = response["answers"]["total"]
            out.write(json.dumps({"id": record["id"], "ms": ms, "tokens": response["usage"]["input_tokens"],
                                  "probs": answer["probabilities"], "top": answer["choice"]}) + "\n")
            out.flush()


def run_hf(args):
    import torch
    from PIL import Image
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).to(args.device).eval()
    if args.no_tiling:  # the control: the same code and weights, one resize per picture
        engine = model.engine
        engine.processor = engine._load_processor()
        engine.processor.image_processor.do_image_splitting = False
    with open(args.out, "w", encoding="utf-8") as out:
        for record in load_records(args.pictures):
            image = Image.open(os.path.join(args.pictures, record["picture"])).convert("RGB")
            t = time.perf_counter()
            request = record["request"]
            response = model.system_one(request["state"], request["questions"], images=[image])
            ms = (time.perf_counter() - t) * 1000
            answer = response["answers"]["total"]
            out.write(json.dumps({"id": record["id"], "ms": ms, "tokens": response["usage"]["input_tokens"],
                                  "probs": answer["probabilities"], "top": answer["choice"]}) + "\n")
            out.flush()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="arm", required=True)
    g = sub.add_parser("genai")
    g.add_argument("--upstream", required=True)
    g.add_argument("-m", "--model_path", required=True)
    g.add_argument("-e", "--execution_provider", default="cpu")
    g.add_argument("--pictures", required=True)
    g.add_argument("--out", required=True)
    h = sub.add_parser("hf")
    h.add_argument("--model", required=True)
    h.add_argument("--device", default="cpu")
    h.add_argument("--no_tiling", action="store_true", help="control: set do_image_splitting to False")
    h.add_argument("--pictures", required=True)
    h.add_argument("--out", required=True)
    args = parser.parse_args()
    (run_genai if args.arm == "genai" else run_hf)(args)


if __name__ == "__main__":
    main()
