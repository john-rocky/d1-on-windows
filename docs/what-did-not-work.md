# What did not work

```
# A GenAI export on the NPU through the OpenVINO EP
Compilation failed. vclAllocatedExecutableCreate4 result: 0x78000004 - [NPU_VCL] Compiler returned msg:
Got negative shape dim bound: '-1'

# The same with enable_causallm=True
'enable_causallm' is enabled, but the model graph has been partitioned into 31 subgraphs. CausalLM mode requires the entire model to be compiled as a single subgraph.

# Builder: -p fp32 -e cuda exits 0, then the model does not load
This is an invalid model. Graph output (present.2.key) does not exist in the graph.

# Builder: --help on the 0.19.0 wheel and on main
ValueError: unsupported format character ' ' (0x20) at index 10597

# The static graph on the Intel GPU with 1-D short convolutions (export_ir.py's default --conv 1d), with fp16 weights and with 16-bit activations; with fp16 weights and INFERENCE_PRECISION_HINT=f32, the second line
[GPU] Unexpected layout format f16:b_fs_yx_fsv16:1x2048x128:nopad
Weights feature maps number(=1) is not equal to: input feature maps number(=2048)

# The static graph on the Intel GPU with 2-D short convolutions and int8 weights, without --fold, with ACTIVATIONS_SCALE_FACTOR=8
[MultiplyPartialTransformation] END: node: opset1::Multiply __module.language.layers.0.conv/aten::mul/Multiply ... CALLBACK HAS THROWN: ...
While validating node 'opset1::VariadicSplit __module.language.layers.0.conv/prim::ListUnpack/VariadicSplit ...':
Default output not supported

# DirectML: PyPI has no onnxruntime-genai-directml 0.19.0; its latest, 0.14.1, does not install
ERROR: No matching distribution found for onnxruntime-directml>=1.26.0
```

The `ACTIVATIONS_SCALE_FACTOR=8` failure on the Intel GPU is filed as [openvinotoolkit/openvino#39635](https://github.com/openvinotoolkit/openvino/issues/39635).
