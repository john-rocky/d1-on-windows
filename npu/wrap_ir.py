"""Write a one-node EPContext model that points at an IR from export_ir.py, so ONNX Runtime's OpenVINO EP can load it.

This is the layout of Foundry Local's own NPU model folders (a small .onnx beside openvino_model_dy.xml).

    python wrap_ir.py <dir>/model.xml      # writes <dir>/model_ctx.onnx
"""

import os
import sys

import onnx
import openvino as ov
from onnx import TensorProto, helper

path = sys.argv[1]
named = os.path.join(os.path.dirname(path), "model_named.xml")
if os.path.exists(named):
    path = named
ir = ov.Core().read_model(path)
if not ir.output(0).get_names():  # the context node has to name its output, so the IR's output needs a name too
    ir.output(0).get_tensor().set_names({"output"})
    path = named
    ov.save_model(ir, path, compress_to_fp16=False)


def name_of(port):
    """A port can carry several names, such as {"last", "101"}: use the readable one."""
    return next((n for n in sorted(port.get_names()) if not n.isdigit()), port.get_any_name())


ort_type = {"i64": TensorProto.INT64, "f32": TensorProto.FLOAT, "f16": TensorProto.FLOAT16}
inputs = [helper.make_tensor_value_info(name_of(p), ort_type[p.get_element_type().get_type_name()], list(p.get_shape())) for p in ir.inputs]
outputs = [helper.make_tensor_value_info(name_of(p), TensorProto.FLOAT, list(p.get_shape())) for p in ir.outputs]
node = helper.make_node(
    "EPContext",
    [i.name for i in inputs],
    [o.name for o in outputs],
    name="ContextNode",
    domain="com.microsoft",
    embed_mode=0,
    ep_cache_context=os.path.basename(path),
    ep_sdk_version=ov.__version__.split("-")[0],
    main_context=1,
    max_size=0,
    source="OpenVINOExecutionProvider",
    DeviceClass="NPU",
)
graph = helper.make_graph([node], "d1_3b_static", inputs, outputs)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
out = os.path.join(os.path.dirname(path), "model_ctx.onnx")
onnx.save(model, out)
print("wrote", out, "inputs", [i.name for i in inputs], "outputs", [o.name for o in outputs])
