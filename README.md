# d1-3B on a Windows laptop: CUDA, WebGPU, OpenVINO and the Intel NPU

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) answers typed questions about a state with probabilities from one forward pass. This repo runs its decision step on every processor of one laptop and compares each result with d1-3B's own code in fp32.

The GPU, CPU and OpenVINO EP rows use [`d1-decisions.py`](https://github.com/microsoft/onnxruntime-genai/blob/v0.19.0/examples/python/d1-decisions.py) from ONNX Runtime GenAI v0.19.0, imported unchanged. The NPU rows use the same read-out code on a static graph made here (`npu/`), because the GenAI exports do not compile on this NPU. That graph has int8 weights and 16-bit activations, and no head: the options' rows of the head run on the CPU.

Machine: MSI Stealth 16 AI+, Windows 11 build 26200, Core Ultra 9 386H, 31.5 GB RAM, RTX 5080 Laptop 16 GB (driver 591.74), Intel Graphics (driver 32.0.101.8724), Intel AI Boost NPU (driver 32.0.100.4841). d1-3B at revision `051bcc464b`.

## Results

| Processor, runtime, weights | same top answer | largest probability difference | 39 / 101 / 467 tokens |
| --- | --- | --- | --- |
| RTX 5080 Laptop, GenAI CUDA, fp16 | 415/415 | 0.0049 | 15.7 / 18.4 / 64.8 |
| RTX 5080 Laptop, GenAI WebGPU plugin, fp32 | 415/415 | 1.1e-05 | 64.7 / 113 / 381 |
| RTX 5080 Laptop, d1-3B's own code, fp32 | 415/415 | 8.4e-06 | 46.2 / 60.4 / 201 |
| Intel NPU, static graph of 64 tokens | 18/18 | 0.018 | 44.5 / — / — |
| Intel NPU, static graph of 128 tokens | 311/311 | 0.047 | 54.8 / 54.9 / — |
| Intel NPU, the same graph in ONNX Runtime + OpenVINO EP | 311/311 | 0.047 | 54.2 / 54.2 / — |
| Intel NPU, static graph of 192 tokens | 370/370 | 0.047 | 66.7 / 66.9 / — |
| Intel NPU, static graph of 256 tokens | 388/388 | 0.047 | 105 / 104 / — |
| Intel NPU, static graph of 512 tokens | 395/395 | 0.047 | 211 / 211 / 211 |
| Intel GPU, GenAI OpenVINO EP, fp32 | 415/415 | 0.0022 | 108 / 140 / 397 |
| Intel GPU, GenAI WebGPU plugin, fp32 | 415/415 | 1.3e-05 | 497 / 706 / 2376 |
| CPU, GenAI OpenVINO EP, int8 | 414/415 | 0.051 | 211 / 594 / 2305 |
| CPU, GenAI CPU EP, int8, 16 threads | 415/415 | 0.049 | 258 / 639 / 2960 |
| CPU, GenAI CPU EP, int8, default threads | 415/415 | 0.049 | 353 / 864 / 3950 |
| CPU, GenAI CPU EP, fp32, first 40 questions | 40/40 | 3e-06 | 716 / 1697 / 6578 |
| CPU, d1-3B's own code, fp32, first 40 questions | 40/40 | 1.9e-06 | 1237 / 2754 / 11046 |

Times are in ms: the median of 9 runs after one warm pass, one question per request. GenAI and own-code rows time the whole call: tokenizer, forward and read-out. NPU rows start from ready token ids; without the model, the rest of the call takes 0.24 ms at 467 tokens (`npu/time_render.py`).

The set is the 415 text questions of [these fixtures](https://github.com/john-rocky/hf-to-litertlm/tree/main/d1_3b_work/fixtures), 39 to 3,470 tokens. The reference in those fixtures is d1-3B's own code in fp32 on the CPU. The one changed answer, on the OpenVINO EP on the CPU, is `tv4_013/answer`, which the fixtures mark as a near tie.

A static graph takes the same time for every prompt it holds, and it can only answer the prompts that fit: 18, 311, 370, 388 and 395 of the 415.

With CUDA on the RTX, longer prompts take 120 ms at 1,024 tokens and 376 ms at 3,470.

Power: on AC, in Windows' Balanced mode. While deciding at 467 tokens, the RTX ran at 1600 to 1800 MHz of its 3090 MHz maximum, with its memory at 9001 of 14001 MHz, against a software power cap of 95 W (`nvidia-smi` reports a 125 W maximum; `nvidia-smi -pl` is refused on this laptop). Windows' Best performance mode left the cap and the CUDA times unchanged, and made the OpenVINO EP on the CPU slower: 770 ms at 101 tokens, against 601 and 614 ms in Balanced mode before and after.

System CPU while deciding, as a 15-second average (idle 1.4 %): NPU 1.2 %, RTX with CUDA 9.5 %, Intel GPU with the OpenVINO EP 51 %, CPU with the OpenVINO EP 67 %.

The NPU takes a few forwards to settle. After loading, the 128-token graph took 406, 71, 62, 58, 55 and 54 ms in one run (`npu/profile_ov.py`).

### What did not make it faster or broke the answers

| Try | 101 tokens, ms | same top answer |
| --- | --- | --- |
| CUDA fp16, the form above | 18.4 | 415/415 |
| CUDA bf16 | 18.4 | 414/415, largest difference 0.04 |
| CUDA fp16, full head | 18.9 | 415/415 |
| CUDA fp16, `enable_cuda_graph=1` | 19.1 | 415/415 |
| CUDA fp16, `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0` / `1` | 19.4 / 21.3 | same export |
| CUDA fp16, `cudnn_conv_algo_search`, `cudnn_conv1d_pad_to_nc1d`, `prefer_nhwc`, `do_copy_in_default_stream`, one CPU thread | 20.7 to 23.6 | same export; 20.7 and 21.4 without them in the same window |
| NPU 128, the form above | 54.9 | 311/311 |
| NPU 128, the form above, `NPU_TURBO=YES` | 54.8 | same graph |
| NPU 128, head in the graph | 59.9 | 311/311 |
| NPU 128, head in the graph, SmoothQuant 0.95 | 61.7 | 300/311 |
| NPU 128, int8 activations | 61.2 | 274/311 |
| NPU 128, int8 activations, last norm and head not quantized | 66.2 | 276/311 |
| The same, MinMax ranges on 311 questions | 66.4 | 275/311 |
| The same, SmoothQuant 0.95 | 67.7 | 299/311 |
| NPU 128, fp16 weights, no quantizers | 103 | 311/311 |
| NPU 128, fp16 weights, `NPU_TURBO=YES` | 102 | 311/311 |
| NPU 128, fp16 weights, `PERFORMANCE_HINT=LATENCY` / `THROUGHPUT` | 104 / 120 | same graph |
| NPU 128, int8 weights only | 99.7 | 114/311 |
| NPU 128, int4 weights only | 96.0 | 115/311 |
| CPU EP int8, 4 / 8 / 12 / 16 threads (default 864) | 1456 / 859 / 717 / 639 | same export |
| CPU EP int4 | 872 | 38/40 |
| OpenVINO EP on the CPU, `num_of_threads=16` (default 594) | 798 | same export |

On the CPU, the graphs with int8 and int4 weights only give the same top answer on 40 and 39 of the first 40 questions. On the NPU, the same graphs do on 10 and 12 of them. The graph with int8 activations gives 33 of 40 on the CPU and 34 of 40 on the NPU.

## Pictures: what the missing tiles cost

ONNX Runtime GenAI resizes a picture once. d1-3B's own code cuts a large picture into tiles. On 20 generated invoices (1600×1200, the total printed at 8 to 14 px, one `choice` question out of 4 options per picture):

| Run | right answers | prompt tokens |
| --- | --- | --- |
| d1-3B's own code, default tiles | 20/20 | 1832 |
| d1-3B's own code, `do_image_splitting=False` | 13/20 | 289 |
| GenAI 0.19.0, fp16, CUDA | 13/20 | 289 |

The last two rows give the same top answer on all 20 pictures.

## Run it

Run these blocks in order in one PowerShell window, from this folder. Each package brings its own ONNX Runtime, so each block has its own virtual environment. They need Python 3.12, about 30 GB of disk next to the model folder, and a machine like this one with 32 GB of RAM: the int4 export reached 21.5 GB. On this machine the CUDA block took 35 minutes and the NPU block 11.

```powershell
# The fixtures, upstream's decision script and the model folder
foreach ($f in "rebuild_requests.py","requests_public.json","reference_probs.json") { curl.exe -sSLO "https://raw.githubusercontent.com/john-rocky/hf-to-litertlm/main/d1_3b_work/fixtures/$f" }
python rebuild_requests.py --out requests.json
New-Item -ItemType Directory -Force upstream | Out-Null
foreach ($f in "d1-decisions.py","common.py") { curl.exe -sSL -o "upstream\$f" "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v0.19.0/examples/python/$f" }
$model = "<folder with LiquidAI/d1-3B>"
```

`rebuild_requests.py` downloads one public file and checks the SHA-256 of each result. See [its README](https://github.com/john-rocky/hf-to-litertlm/tree/main/d1_3b_work/fixtures).

```powershell
# ONNX Runtime GenAI on CUDA
py -3.12 -m venv .venv-cuda
$py = ".venv-cuda\Scripts\python.exe"
& $py -m pip install onnxruntime-genai-cuda==0.19.0 torch transformers==5.22.1 onnx==1.23.2 onnx-ir==1.1.0 nvidia-cuda-runtime==13.5.37 nvidia-cuda-nvrtc==13.5.37 nvidia-cublas==13.10.0.20 nvidia-cudnn-cu13==9.28.0.42
$sp = "$PWD\.venv-cuda\Lib\site-packages\nvidia"
$env:Path = "$sp\cu13\bin\x86_64;$sp\cudnn\bin;$env:Path"
& $py -m onnxruntime_genai.models.builder -i $model -o cuda-fp16 -p fp16 -e cuda -c cache --extra_options prune_lm_head=true
& $py fix_tokenizer.py cuda-fp16
& $py run_genai.py --upstream upstream -m cuda-fp16 -e cuda --requests requests.json --out cuda-fp16.jsonl
& $py score.py cuda-fp16.jsonl reference_probs.json
# The times: one warm pass, then 9 timed runs per prompt length
& $py make_speed_fixture.py --upstream upstream -m cuda-fp16 -e cuda --out speed_requests.json
& $py run_genai.py --upstream upstream -m cuda-fp16 -e cuda --requests speed_requests.json --passes 2 --out speed_cuda-fp16.jsonl
& $py speed_lines.py speed_cuda-fp16.jsonl
```

`fix_tokenizer.py` is [step 3 of the earlier tutorial](https://github.com/microsoft/onnxruntime-genai/blob/7f090db1731887b6a1b3f2715642df4ffe0e7863/examples/python/d1-tutorial.md#3-fix-the-tokenizer). The PyPI 0.19.0 wheels need it: without it, `og.Tokenizer` fails with `Invalid regex`.

```powershell
# OpenVINO Runtime on the Intel NPU: a static graph of 128 tokens
py -3.12 -m venv .venv-npu
$py = ".venv-npu\Scripts\python.exe"
& $py -m pip install onnxruntime-genai==0.19.0 torch transformers==5.22.1 onnx==1.23.2 onnx-ir==1.1.0 openvino==2026.4.1 nncf==3.4.0
& $py -m onnxruntime_genai.models.builder -i $model -o cpu-int4 -p int4 -e cpu -c cache --extra_options prune_lm_head=true
& $py fix_tokenizer.py cpu-int4
& $py npu\export_ir.py --model $model --length 128 --head hidden --out ir-128
& $py npu\make_ids.py --upstream upstream -m cpu-int4 --requests requests.json --out ids-128.json --max_tokens 128
& $py npu\quant_ir.py --ir ir-128\model.xml --ids ids-128.json --out ir-128-q --subset 100 --bits 16
& $py npu\run_ov.py --upstream upstream --ir ir-128-q\model.xml --head ir-128\lm_head.npy --device NPU --ids ids-128.json --out npu-128.jsonl
& $py score.py npu-128.jsonl reference_probs.json
# The times
& $py npu\make_ids.py --upstream upstream -m cpu-int4 --requests speed_requests.json --out speed-ids-128.json --max_tokens 128
& $py npu\run_ov.py --upstream upstream --ir ir-128-q\model.xml --head ir-128\lm_head.npy --device NPU --ids speed-ids-128.json --passes 2 --out speed_npu-128.jsonl
& $py speed_lines.py speed_npu-128.jsonl
```

`export_ir.py` writes one graph for one prompt length, with no cache. A shorter prompt is padded on the right, and its answer is read at its last token. d1-3B's stack is causal, so a pad does not reach an earlier token; the script prints the largest logit change a pad causes on one probe. `make_ids.py` needs a CPU export only for the tokenizer and upstream's request checks.

With `--head hidden`, the graph returns the hidden state, and `npu/head.py` multiplies it by the options' rows of the head on the CPU. Upstream's softmax over the whole vocabulary has the same denominator for every option, so it cancels. On the CPU, the fp16 graph read this way gives the same top answer on the first 40 questions, with a largest difference of 7.4e-06.

`quant_ir.py` runs NNCF on 100 of the questions, chosen with a fixed seed. The other 211 of the 128-token set do no worse: same top answer on all 211, largest difference 0.033 (0.047 on the 100).

```powershell
# The same graph through ONNX Runtime and its OpenVINO EP: one EPContext node that points at the IR
& ".venv-npu\Scripts\python.exe" npu\wrap_ir.py ir-128-q\model.xml
py -3.12 -m venv .venv-ort
$py = ".venv-ort\Scripts\python.exe"
& $py -m pip install onnxruntime-genai==0.19.0 onnxruntime==1.31.1 onnxruntime-ep-openvino==1.8.1
& $py npu\run_ort.py --upstream upstream --model ir-128-q\model_ctx.onnx --head ir-128\lm_head.npy --device NPU --ids ids-128.json --out ort-npu-128.jsonl
& $py score.py ort-npu-128.jsonl reference_probs.json
```

```powershell
# Pictures: GenAI against d1-3B's own code on the GPU, with and without tiles
py -3.12 -m venv .venv-hf
$hf = ".venv-hf\Scripts\python.exe"
& $hf -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
& $hf -m pip install transformers==5.22.1 pillow
& $hf tiling\make_pictures.py --out fine --sizes 1600x1200 --fonts 8,10,12,14 --per 5 --seed 2000
& $hf tiling\run_pictures.py hf --model $model --device cuda --pictures fine --out fine_own.jsonl
& $hf tiling\run_pictures.py hf --model $model --device cuda --no_tiling --pictures fine --out fine_own_notile.jsonl
$py = ".venv-cuda\Scripts\python.exe"
$sp = "$PWD\.venv-cuda\Lib\site-packages\nvidia"
$env:Path = "$sp\cu13\bin\x86_64;$sp\cudnn\bin;$env:Path"
& $py -m onnxruntime_genai.models.builder -i $model -o cuda-fp16-vl -p fp16 -e cuda -c cache --extra_options exclude_embeds=true prune_lm_head=true
& $py fix_tokenizer.py cuda-fp16-vl
& $py tiling\run_pictures.py genai --upstream upstream -m cuda-fp16-vl -e cuda --pictures fine --out fine_genai.jsonl
& $hf tiling\score_pictures.py fine\requests.json fine_own.jsonl fine_own_notile.jsonl fine_genai.jsonl
```

```powershell
# d1_vision.py against transformers on 38 pictures: pixels, masks and grids, bit for bit
New-Item -ItemType Directory -Force d1_vision | Out-Null
curl.exe -sSL -o d1_vision\d1_vision.py "https://raw.githubusercontent.com/john-rocky/hf-to-litertlm/main/d1_3b_work/host/d1_vision.py"
$hf = ".venv-hf\Scripts\python.exe"
& $hf tiling\make_pictures.py --out pictures
& $hf tiling\check_tiles.py --d1_vision d1_vision --model $model --pictures fine --pictures pictures --out tiles.json
# The same check with one bit of one pixel flipped: all 38 must now read as different
& $hf tiling\check_tiles.py --d1_vision d1_vision --model $model --pictures fine --pictures pictures --guard --out tiles_guard.json
```

## What did not work

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

# The static graph on the Intel GPU, with fp16 weights and with 16-bit activations; with fp16 weights and INFERENCE_PRECISION_HINT=f32, the second line
[GPU] Unexpected layout format f16:b_fs_yx_fsv16:1x2048x128:nopad
Weights feature maps number(=1) is not equal to: input feature maps number(=2048)

# DirectML: PyPI has no onnxruntime-genai-directml 0.19.0; its latest, 0.14.1, does not install
ERROR: No matching distribution found for onnxruntime-directml>=1.26.0
```

## License

The code here is under the Apache License 2.0. This repo holds no weights: d1-3B is under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/main/LICENSE). `results/` holds question ids, times and probabilities only; `results/fine_requests.json` and `results/pictures_requests.json` hold the generated invoice questions.
