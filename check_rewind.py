"""Does a GenAI generator on d1-3B rewind? (the "state" lever: reuse the state's prefill across questions)

    python check_rewind.py -m <model dir> -e cuda
"""
import argparse

import onnxruntime_genai as og

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", required=True)
parser.add_argument("-e", "--execution_provider", default="cuda")
parser.add_argument("--upstream", default=None, help="unused; accepted so every script here takes the same arguments")
args = parser.parse_args()
config = og.Config(args.model_path)
config.clear_providers()
if args.execution_provider != "cpu":
    config.append_provider(args.execution_provider)
model = og.Model(config)
tokenizer = og.Tokenizer(model)
ids = tokenizer.encode("I was charged twice this month, please refund one of them.")
params = og.GeneratorParams(model)
params.set_search_options(max_length=256, do_sample=False)
generator = og.Generator(model, params)
generator.append_tokens(ids)
first = generator.get_logits()
print("logits after prefill", first.shape, "tokens", len(ids))
try:
    generator.rewind_to(len(ids) - 4)
    print("rewind_to ok, sequence length", len(generator.get_sequence(0)))
    generator.append_tokens(ids[-4:])
    second = generator.get_logits()
    print("logits after append", second.shape, "max abs diff from the first prefill", float(abs(second - first).max()))
except Exception as error:
    print("rewind failed:", str(error)[:800])
