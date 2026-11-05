"""Write speed_requests.json: one noul question over states padded to exact prompt lengths, each repeated a few times.

The prompt length is counted by upstream's own D1Decisions, so this needs the model and any execution provider.

    python make_speed_fixture.py --upstream <dir> -m <model dir> -e cpu --out speed_requests.json
"""

import argparse
import json

from run_genai import load_upstream

BASE = "I was charged twice this month, please refund one of them."
PADDING = " Please check the order history and the card statement before you reply to the customer."
QUESTION = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
TARGETS = [39, 101, 467, 1024, 3470]
REPEATS = 9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-e", "--execution_provider", default="follow_config")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    up = load_upstream(args.upstream)
    import common
    import onnxruntime_genai as og

    common.register_ep(args.execution_provider, None, False)
    config = common.get_config(args.model_path, args.execution_provider, None, {})
    model = og.Model(config)
    runner = up.D1Decisions(model, up.model_context_length(args.model_path))

    def count(state):
        return runner.decide({"state": state, "questions": {"refund": QUESTION}})["usage"]["input_tokens"]

    records = []
    for target in TARGETS:
        words = PADDING.split(" ")
        state = BASE
        index = 0
        while count(state) < target:
            # add whole sentences while far away, single words near the end
            step = PADDING if target - count(state) > 40 else " " + words[index % len(words)]
            state += step
            index += 1
        tokens = count(state)
        print("target", target, "tokens", tokens, "chars", len(state))
        for repeat in range(REPEATS):
            records.append({"id": f"speed_{target}_{repeat}", "request": {"state": state, "questions": {"refund": QUESTION}}})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f)


if __name__ == "__main__":
    main()
