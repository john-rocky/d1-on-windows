# Run it

These commands run from the root of this repo.

Run these blocks in order in one PowerShell window, from this folder. Each package brings its own ONNX Runtime, so each block has its own virtual environment. They need Python 3.12, about 30 GB of disk next to the model folder, and a machine like this one with 32 GB of RAM. On this machine the CUDA block took 35 minutes, the NPU block 11, and the Intel GPU blocks 2.3 and 9.6.

```powershell
# The fixtures, upstream's decision script and the model folder
foreach ($f in "rebuild_requests.py","requests_public.json","reference_probs.json") { curl.exe -sSLO "https://raw.githubusercontent.com/john-rocky/hf-to-litertlm/main/d1_3b_work/fixtures/$f" }
python rebuild_requests.py --out requests.json
New-Item -ItemType Directory -Force upstream | Out-Null
foreach ($f in "d1-decisions.py","common.py") { curl.exe -sSL -o "upstream\$f" "https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v0.19.0/examples/python/$f" }
# A folder with LiquidAI/d1-3B from https://huggingface.co/LiquidAI/d1-3B (revision 051bcc464b here)
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
# The times. speed_requests.json comes from the CUDA block; without that block, use the copy in results (the same bytes)
if (-not (Test-Path speed_requests.json)) { Copy-Item results\speed_requests.json . }
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
# OpenVINO Runtime on the Intel GPU: the same graph of 128 tokens with int8 weights, its 22 short convolutions as 2-D convolutions
$py = ".venv-npu\Scripts\python.exe"
& $py npu\export_ir.py --model $model --length 128 --head hidden --conv 2d --fold --weights int8 --out ir-128-gpu
& $py npu\run_ov.py --upstream upstream --ir ir-128-gpu\model.xml --head ir-128-gpu\lm_head.npy --device GPU --ids ids-128.json --out gpu-128.jsonl
& $py score.py gpu-128.jsonl reference_probs.json
& $py npu\run_ov.py --upstream upstream --ir ir-128-gpu\model.xml --head ir-128-gpu\lm_head.npy --device GPU --ids speed-ids-128.json --passes 2 --out speed_gpu-128.jsonl
& $py speed_lines.py speed_gpu-128.jsonl
```

`--conv 2d` writes each of the 22 short convolutions as a 2-D convolution on `[1, C, N + 2, 1]`: the same numbers in another shape. The script prints the largest output change this makes on one probe: 0.0 here. OpenVINO's Intel GPU plugin does not compile the 1-D form (see What did not work).

The Intel GPU plugin computes in f16 by default. `--fold` has OpenVINO compute the rotary tables (the cos and sin values for each position) once, in fp32, during the export, and store them as constants. Without it, the plugin computes those tables itself, and the same top answer holds on 300 of the 311 questions, with fp16 weights and with int8 weights. Without `--fold`, `--config INFERENCE_PRECISION_HINT=f32` on `run_ov.py` also gives 311/311 with either weights, at 287 to 293 ms for 101 tokens. With `--fold`, the graph still runs in f16 and gives 311/311 at 50.3 ms.

```powershell
# OpenVINO Runtime on the Intel GPU: the same form with 192, 256 and 512 tokens
$py = ".venv-npu\Scripts\python.exe"
& $py npu\make_ids.py --upstream upstream -m cpu-int4 --requests requests.json --out ids-512.json --max_tokens 512
& $py npu\make_ids.py --upstream upstream -m cpu-int4 --requests speed_requests.json --out speed-ids-512.json --max_tokens 512
foreach ($n in 192, 256, 512) {
  & $py npu\export_ir.py --model $model --length $n --head hidden --conv 2d --fold --weights int8 --out ir-$n-gpu
  & $py npu\run_ov.py --upstream upstream --ir ir-$n-gpu\model.xml --head ir-$n-gpu\lm_head.npy --device GPU --ids ids-512.json --out gpu-$n.jsonl
  & $py score.py gpu-$n.jsonl reference_probs.json
  & $py npu\run_ov.py --upstream upstream --ir ir-$n-gpu\model.xml --head ir-$n-gpu\lm_head.npy --device GPU --ids speed-ids-512.json --passes 2 --out speed_gpu-$n.jsonl
  & $py speed_lines.py speed_gpu-$n.jsonl
}
```

The id lists of up to 512 tokens serve all three graphs: `run_ov.py` skips the prompts longer than its graph, so the graphs of 192, 256 and 512 tokens answer the same 370, 388 and 395 questions as the NPU rows.

```powershell
# ONNX Runtime GenAI with the OpenVINO EP on the CPU, int8, with OpenVINO's CPU pinning
py -3.12 -m venv .venv-ov
$py = ".venv-ov\Scripts\python.exe"
& $py -m pip install onnxruntime-genai==0.19.0 onnxruntime==1.31.1 onnxruntime-ep-openvino==1.8.1 torch transformers==5.22.1 onnx==1.23.2 onnx-ir==1.1.0
& $py -m onnxruntime_genai.models.builder -i $model -o cpu-int8 -p int8 -e cpu -c cache --extra_options prune_lm_head=true
& $py fix_tokenizer.py cpu-int8
'{"CPU": {"ENABLE_CPU_PINNING": "YES"}}' | Set-Content -Encoding ascii ov_cpu_pinning.json
& $py run_genai.py --upstream upstream -m cpu-int8 --openvino_device CPU --ep_option "load_config=$PWD\ov_cpu_pinning.json" --requests requests.json --out ov-cpu-int8.jsonl
& $py score.py ov-cpu-int8.jsonl reference_probs.json
# The times at 39, 101 and 467 tokens. speed_requests.json comes from the CUDA block; without that block, use the copy in results (the same bytes)
if (-not (Test-Path speed_requests.json)) { Copy-Item results\speed_requests.json . }
$ids = (39, 101, 467 | ForEach-Object { $t = $_; 0..8 | ForEach-Object { "speed_${t}_$_" } }) -join ","
& $py run_genai.py --upstream upstream -m cpu-int8 --openvino_device CPU --ep_option "load_config=$PWD\ov_cpu_pinning.json" --requests speed_requests.json --passes 2 --ids $ids --out speed_ov-cpu-int8.jsonl
& $py speed_lines.py speed_ov-cpu-int8.jsonl
```

`load_config` hands the OpenVINO EP a JSON file of OpenVINO properties per device; `ENABLE_CPU_PINNING` is one of the CPU's. Without `--ep_option`, the same commands give the same probabilities on all 415 questions and take longer (see the results).

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
