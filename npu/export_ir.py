"""d1-3B's text forward as an OpenVINO IR with one static prompt length, for a device that needs static shapes.

Inputs: input_ids [1, N], right-padded; last, the index of the last real token. Output: the logits at that token.
The stack is causal, so a pad never reaches a real token and there is no attention mask (the provider's hybrid.py says
the same). No cache: the decision step reads one forward per question.

    python export_ir.py --model <d1-3B folder> --length 128 --out ir_128 [--weights fp16|int8|int4] [--head full|hidden]
"""

import argparse
import os
import sys
import time

import numpy as np
import openvino as ov
import torch
from transformers import AutoModel


class TextForward(torch.nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.language = model.model.language_model
        self.lm_head = model.lm_head if head == "full" else torch.nn.Identity()

    def forward(self, input_ids, last):
        hidden = self.language.run(self.language.embed_tokens(input_ids))
        return self.lm_head(hidden[0, last])


def plain_attention(length):
    """Causal group-query attention as explicit products: the same numbers, with no enable_gqa argument to convert."""
    mask = torch.full((length, length), -1e4).triu(1)

    def attention(q, k, v, tree, window, scale):
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = (q @ k.transpose(-1, -2)) * scale + mask
        return torch.softmax(scores, dim=-1) @ v

    return attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default="fp16", choices=["fp16", "int8", "int4"])
    parser.add_argument("--head", default="full", choices=["full", "hidden"],
                        help="hidden: the graph returns the hidden state, and the head matrix goes to lm_head.npy for head.py")
    args = parser.parse_args()

    t = time.perf_counter()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()
    print(f"loaded in {time.perf_counter() - t:.1f} s", flush=True)
    module = sys.modules[type(model.model.language_model.layers[2].self_attn).__module__]
    module.attention = plain_attention(args.length)
    os.makedirs(args.out, exist_ok=True)
    if args.head == "hidden":
        if model.lm_head.bias is not None:
            raise SystemExit("the head has a bias: head.py reads a matrix only")
        np.save(os.path.join(args.out, "lm_head.npy"), model.lm_head.weight.detach().numpy())

    wrapped = TextForward(model, args.head).eval()
    ids = torch.zeros((1, args.length), dtype=torch.long)
    last = torch.tensor(args.length - 1, dtype=torch.long)
    probe = torch.randint(1000, 5000, (1, args.length))
    with torch.no_grad():
        a = wrapped(probe, torch.tensor(20)).numpy()
        padded = probe.clone()
        padded[0, 21:] = 124893
        b = wrapped(padded, torch.tensor(20)).numpy()
    print("guard: a pad after the last token moves the logits by at most", float(np.abs(a - b).max()), flush=True)

    t = time.perf_counter()
    with torch.no_grad():
        ir = ov.convert_model(wrapped, example_input={"input_ids": ids, "last": last},
                              input=[("input_ids", [1, args.length], ov.Type.i64), ("last", [], ov.Type.i64)])
    ir.output(0).get_tensor().set_names({"logits" if args.head == "full" else "hidden"})
    print(f"converted in {time.perf_counter() - t:.1f} s; inputs {[str(p) for p in ir.inputs]}", flush=True)
    del model, wrapped
    ops = {}
    for op in ir.get_ops():
        ops[op.get_type_name()] = ops.get(op.get_type_name(), 0) + 1
    print("ops", dict(sorted(ops.items(), key=lambda kv: -kv[1])), flush=True)

    path = os.path.join(args.out, "model.xml")
    t = time.perf_counter()
    if args.weights == "fp16":
        ov.save_model(ir, path, compress_to_fp16=True)
    else:
        import nncf

        mode = nncf.CompressWeightsMode.INT8_ASYM if args.weights == "int8" else nncf.CompressWeightsMode.INT4_SYM
        kwargs = {} if args.weights == "int8" else {"ratio": 1.0, "group_size": 128}
        ov.save_model(nncf.compress_weights(ir, mode=mode, **kwargs), path, compress_to_fp16=False)
    size = os.path.getsize(path.replace(".xml", ".bin")) / 2**30
    print(f"saved {path} in {time.perf_counter() - t:.1f} s; weights {size:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
