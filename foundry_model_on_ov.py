"""Load a model folder from the Foundry Local cache with onnxruntime-genai and the pip OpenVINO plugin, outside Foundry Local."""
import json
import os
import sys
import time

import onnxruntime_ep_openvino as ov_ep
import onnxruntime_genai as og

folder, device = sys.argv[1], sys.argv[2]
print("model", folder, "device", device, "og", og.__version__)
print(json.dumps(json.load(open(os.path.join(folder, "genai_config.json"), encoding="utf-8"))["model"]["decoder"].get("session_options"), indent=0)[:800])
og.register_execution_provider_library(ov_ep.get_ep_name(), ov_ep.get_library_path())
config = og.Config(folder)
config.clear_providers()
config.append_provider("OpenVINO")
config.set_provider_option("OpenVINO", "device_type", device)
t = time.perf_counter()
model = og.Model(config)
print("load", round(time.perf_counter() - t, 2), "s device_type", model.device_type)
tokenizer = og.Tokenizer(model)
prompt = tokenizer.apply_chat_template(json.dumps([{"role": "user", "content": "Name the capital of Japan in one word."}]), add_generation_prompt=True)
params = og.GeneratorParams(model)
params.set_search_options(max_length=128, do_sample=False)
generator = og.Generator(model, params)
generator.append_tokens(tokenizer.encode(prompt))
stream = tokenizer.create_stream()
out = ""
n = 0
t = time.perf_counter()
while not generator.is_done() and n < 24:
    generator.generate_next_token()
    out += stream.decode(generator.get_next_tokens()[0])
    n += 1
print("text:", out)
print("tokens", n, "s", round(time.perf_counter() - t, 2))
