"""Quantize the activations and weights of a static IR from export_ir.py with NNCF, calibrated on real questions.

    python quant_ir.py --ir <dir>/model.xml --ids ids.json --out <dir> [--subset 100] [--range default|minmax]
                       [--smooth_quant 0.95] [--smooth_quant_conv 0.95] [--ignore <node name>,<node name>] [--ignore_re <regex>] [--bias] [--bits 8]
"""

import argparse
import json
import os
import random
import time

import nncf
import numpy as np
import openvino as ov
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.range_estimator import RangeEstimatorParametersSet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", required=True)
    parser.add_argument("--ids", required=True, help="token ids from make_ids.py, used as the calibration set")
    parser.add_argument("--out", required=True)
    parser.add_argument("--subset", type=int, default=100)
    parser.add_argument("--range", choices=["default", "minmax"], default="default")
    parser.add_argument("--smooth_quant", type=float, default=-1, help="the MatMul alpha; -1 leaves SmoothQuant off")
    parser.add_argument("--smooth_quant_conv", type=float, default=-1, help="the convolution alpha; -1 leaves it off")
    parser.add_argument("--ignore", default="", help="node names to leave unquantized, comma separated")
    parser.add_argument("--ignore_re", default="", help="a regular expression over node names to leave unquantized")
    parser.add_argument("--bias", action="store_true", help="run fast bias correction")
    parser.add_argument("--bits", type=int, default=8, help="activation bits")
    parser.add_argument("--pad", type=int, default=124893)
    args = parser.parse_args()

    ir = ov.Core().read_model(args.ir)
    length = ir.input("input_ids").get_shape()[1]
    with open(args.ids, encoding="utf-8") as f:
        rows = [row for row in json.load(f) if "ids" in row and len(row["ids"]) <= length]
    if not rows:
        raise SystemExit(f"no question in {args.ids} fits {length} tokens")
    random.Random(0).shuffle(rows)

    def to_inputs(row):
        padded = np.full((1, length), args.pad, dtype=np.int64)
        padded[0, : len(row["ids"])] = row["ids"]
        return {"input_ids": padded, "last": np.array(len(row["ids"]) - 1, dtype=np.int64)}

    advanced = {
        "disable_bias_correction": not args.bias,
        "smooth_quant_alphas": nncf.AdvancedSmoothQuantParameters(matmul=args.smooth_quant, convolution=args.smooth_quant_conv),
    }
    if args.bits != 8:
        advanced["activations_quantization_params"] = QuantizationParameters(num_bits=args.bits)
    if args.range == "minmax":
        advanced["activations_range_estimator_params"] = RangeEstimatorParametersSet.MINMAX
    names = [name for name in args.ignore.split(",") if name]
    patterns = [args.ignore_re] if args.ignore_re else []
    ignored = nncf.IgnoredScope(names=names, patterns=patterns, validate=False) if names or patterns else None
    size = min(args.subset, len(rows))
    print(f"nncf {nncf.__version__}, openvino {ov.__version__}, {size} questions, length {length}, {vars(args)}", flush=True)
    t = time.perf_counter()
    options = {"ignored_scope": ignored} if ignored else {}
    quantized = nncf.quantize(
        ir,
        nncf.Dataset(rows, to_inputs),
        preset=nncf.QuantizationPreset.MIXED,
        target_device=nncf.TargetDevice.NPU,
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=size,
        fast_bias_correction=True,
        advanced_parameters=nncf.AdvancedQuantizationParameters(**advanced),
        **options,
    )
    count = sum(op.get_type_name() == "FakeQuantize" for op in quantized.get_ops())
    print(f"quantized in {time.perf_counter() - t:.1f} s, {count} FakeQuantize nodes", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "model.xml")
    ov.save_model(quantized, path, compress_to_fp16=False)
    size = os.path.getsize(os.path.join(args.out, "model.bin")) / 2**30
    print(f"saved {path}; weights {size:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
