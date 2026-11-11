# Results

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) answers typed questions about a state with probabilities from one forward pass. This repo runs its decision step on every processor of one laptop and compares each result with d1-3B's own code in fp32.

The GPU, CPU and OpenVINO EP rows use [`d1-decisions.py`](https://github.com/microsoft/onnxruntime-genai/blob/v0.19.0/examples/python/d1-decisions.py) from ONNX Runtime GenAI v0.19.0, imported unchanged. The NPU rows use the same read-out code on a static graph made here (`npu/`), because the GenAI exports do not compile on this NPU. That graph has int8 weights and 16-bit activations, and no head: the options' rows of the head run on the CPU. The static Intel GPU rows run that graph through OpenVINO Runtime with int8 weights only, its short convolutions written as 2-D convolutions and its rotary tables stored as constants (see the Intel GPU blocks under Run it).

Machine: MSI Stealth 16 AI+, Windows 11 build 26200, Core Ultra 9 386H, 31.5 GB RAM, RTX 5080 Laptop 16 GB (driver 591.74), Intel Graphics (driver 32.0.101.8724), Intel AI Boost NPU (driver 32.0.100.4841). d1-3B at revision `051bcc464b`.

The commands are in [Run it](run-it.md), and the errors in [What did not work](what-did-not-work.md).

## Results

| Processor, runtime, weights | same top answer | largest probability difference | 39 / 101 / 467 tokens |
| --- | --- | --- | --- |
| RTX 5080 Laptop, GenAI CUDA, fp16 | 415/415 | 0.0049 | 15.7 / 18.4 / 64.8 |
| RTX 5080 Laptop, GenAI WebGPU plugin, fp32 | 415/415 | 1.1e-05 | 64.7 / 113 / 381 † |
| RTX 5080 Laptop, d1-3B's own code, fp32 | 415/415 | 8.4e-06 | 46.2 / 60.4 / 201 † |
| Intel NPU, static graph of 64 tokens | 18/18 | 0.018 | 44.5 / — / — |
| Intel NPU, static graph of 128 tokens | 311/311 | 0.047 | 54.8 / 54.9 / — |
| Intel NPU, the same graph in ONNX Runtime + OpenVINO EP | 311/311 | 0.047 | 54.2 / 54.2 / — |
| Intel NPU, static graph of 192 tokens | 370/370 | 0.047 | 66.7 / 66.9 / — |
| Intel NPU, static graph of 256 tokens | 388/388 | 0.047 | 105 / 104 / — |
| Intel NPU, static graph of 512 tokens | 395/395 | 0.047 | 211 / 211 / 211 |
| Intel GPU, static graph of 128 tokens, int8 weights | 311/311 | 0.11 | 50.2 / 50.3 / — |
| Intel GPU, static graph of 192 tokens, int8 weights | 369/370 | 0.11 | 66.0 / 66.1 / — |
| Intel GPU, static graph of 256 tokens, int8 weights | 387/388 | 0.11 | 73.9 / 73.4 / — |
| Intel GPU, static graph of 512 tokens, int8 weights | 394/395 | 0.11 | 148 / 148 / 148 |
| Intel GPU, GenAI OpenVINO EP, fp32 | 415/415 | 0.0022 | 108 / 140 / 417 |
| Intel GPU, GenAI WebGPU plugin, fp32 | 415/415 | 1.3e-05 | 497 / 706 / 2376 † |
| CPU, GenAI OpenVINO EP, int8 | 414/415 | 0.051 | 211 / 594 / 2305 |
| CPU, GenAI CPU EP, int8, 16 threads | 415/415 | 0.049 | 258 / 639 / 2960 |
| CPU, GenAI CPU EP, int8, default threads | 415/415 | 0.049 | 322 / 794 / 3680 |
| CPU, GenAI CPU EP, fp32, first 40 questions | 40/40 | 3e-06 | 716 / 1697 / 7173 |
| CPU, d1-3B's own code, fp32, first 40 questions | 40/40 | 1.9e-06 | 1026 / 2427 / 10101 |

Times are in ms: the median of 9 runs after one warm pass, one question per request. GenAI and own-code rows time the whole call: tokenizer, forward and read-out. NPU rows and the static Intel GPU rows start from ready token ids; without the model, the rest of the call takes 0.24 ms at 467 tokens (`npu/time_render.py`).

The speed was measured again on 11-09 on Windows build 26200.9711 and, for the four GenAI CPU rows, on 11-11 on the same build (the rows measured again were timed on 11-05 and 11-06, on build 26200.6899, before an update), with the same scripts, models and arguments: per row, two processes that opt out of Windows power throttling at their start, before the script and its imports run (High QoS: `SetProcessInformation` with `ProcessPowerThrottling`), and two that do not. Where the mean of the two High QoS processes moved more than 5 % from the published value, the table shows it: the OpenVINO EP on the Intel GPU at 467 tokens (397 ms on 11-05), d1-3B's own code on the CPU (1237 / 2754 / 11046 ms on 11-05), the GenAI CPU EP with the default threads (353 / 864 / 3950 ms on 11-05) and the GenAI CPU EP in fp32 at 467 tokens (6578 ms on 11-05). The other rows measured again in a session that counts stayed within 5 % (`results/speed_high_qos_1109.txt`, `results/speed_high_qos_1111.txt`; for the OpenVINO EP on the CPU, `results/speed_high_qos_1111_ov.txt`). For the OpenVINO EP on the CPU, that session is the last of four on 11-11: in two of the others, one of the two High QoS processes ran 15 to 19 % slower than the other at 39 and 467 tokens (cause not found), and the table keeps the times without that slowdown. Rows marked † were not measured again. On 11-11, MSI Center's User Scenario was Balanced (registry value `Mode` 2) and Windows' power mode was Balanced, its default; both were read before and after each session. For the 11-09 sessions, that registry value was 2 on 11-07 and still 2 at 20:12 on 11-09, after them. For the 11-05 and 11-06 numbers it is not recorded; MSI Center's log names the same scenario at the sign-in before them and at the sign-in after them. The NPU and static Intel GPU rows were not measured again either, after one A/B each: at 101 tokens, the 128-token NPU graph took 54.1 ms with High QoS and 56.0 ms without, and a static fp16 graph of 128 tokens on Windows ML's OpenVINO EP (not one of the table's graphs) 104.3 and 99.1 ms.

Started from an ssh session while the desktop is locked, without that call, the same runs take: CUDA 20.9 / 22.3 / 64.8 ms and the OpenVINO EP on the Intel GPU 128 / 180 / 667 ms (`results/speed_high_qos_1109.txt`), the GenAI CPU EP with 16 threads 843 / 2144 / 8804 ms, with the default threads 777 / 1876 / 8346 ms and in fp32 835 / 1898 / 8558 ms (`results/speed_high_qos_1111.txt`), and the OpenVINO EP on the CPU 453 / 1353 / 5242 ms (`results/speed_high_qos_1111_ov.txt`).

The set is the 415 text questions of [these fixtures](https://github.com/john-rocky/hf-to-litertlm/tree/main/d1_3b_work/fixtures), 39 to 3,470 tokens. The request file holds one more text question, `own_email_03/next_step`, which has no instructions; upstream refuses it and `score.py` lists it under `refused`. The reference in those fixtures is d1-3B's own code in fp32 on the CPU. The one changed answer, on the OpenVINO EP on the CPU and on the static Intel GPU graphs of 192, 256 and 512 tokens, is `tv4_013/answer`, which the fixtures mark as a near tie. It has 183 tokens, so the graphs of 128 tokens do not hold it. On the Intel GPU it stays changed with `INFERENCE_PRECISION_HINT=f32`, so it does not come from the GPU's f16.

A static graph takes the same time for every prompt it holds, and it can only answer the prompts that fit: 18, 311, 370, 388 and 395 of the 415.

With CUDA on the RTX, longer prompts take 120 ms at 1,024 tokens and 376 ms at 3,470.

Power: on AC, in Windows' Balanced mode. While deciding at 467 tokens, the RTX ran at 1600 to 1800 MHz of its 3090 MHz maximum, with its memory at 9001 of 14001 MHz, against a software power cap of 95 W (`nvidia-smi` reports a 125 W maximum; `nvidia-smi -pl` is refused on this laptop: `results/r3_power_limits.txt`). Windows' Best performance mode left the cap and the CUDA times unchanged, and made the OpenVINO EP on the CPU slower: 770 ms at 101 tokens, against 614 and 601 ms in Balanced mode before and after.

System CPU while deciding, as a 15-second average (idle 1.4 %): NPU 1.2 %, RTX with CUDA 9.5 %, Intel GPU with the OpenVINO EP 51 %, CPU with the OpenVINO EP 67 %.

The NPU takes a few forwards to settle. After loading, the 128-token graph took 406, 71, 62, 58, 55 and 54 ms in one run (`npu/profile_ov.py`).

### What did not make it faster or broke the answers

These tries were timed on 11-05 and 11-06 (the Intel GPU 128 rows on 11-09), without the High QoS call; compare them within this table.

| Try | 101 tokens, ms | same top answer |
| --- | --- | --- |
| CUDA fp16, the form above | 18.4 | 415/415 |
| CUDA bf16 | 18.4 | 414/415, largest difference 0.04 |
| CUDA fp16, full head | 18.9 | 415/415 |
| CUDA fp16, `enable_cuda_graph=1` | 19.1 | 415/415 |
| CUDA fp16, `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0` / `1` | 19.4 / 21.3 | same export |
| CUDA fp16, `cudnn_conv_algo_search`, `cudnn_conv1d_pad_to_nc1d`, `prefer_nhwc`, `do_copy_in_default_stream`, one CPU thread | 20.7 to 23.6 | same export; without them, 21.4 before and 20.7 after in the same session |
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
| Intel GPU 128, the form above | 50.3 | 311/311 |
| Intel GPU 128, without `--fold` | 49.6 | 300/311 |
| Intel GPU 128, fp16 weights, without `--fold` | 92.2 | 300/311 |
| Intel GPU 128, without `--fold`, `INFERENCE_PRECISION_HINT=f32` | 293 | 311/311 |
| Intel GPU 128, fp16 weights, without `--fold`, `INFERENCE_PRECISION_HINT=f32` | 287 | 311/311 |
| CPU EP int8, 4 / 8 / 12 / 16 threads (default 864) | 1456 / 859 / 717 / 639 | same export |
| CPU EP int4 | 872 | 38/40 |
| OpenVINO EP on the CPU, `num_of_threads=16` (default 594) | 798 | same export |

On the CPU, the graphs with int8 and int4 weights only give the same top answer on 40 and 39 of the first 40 questions. On the NPU, the same graphs do on 10 and 12 of them. On the Intel GPU, the graph with int8 weights only, the form above, gives the same top answer on all 311. The graph with int8 activations gives 33 of 40 on the CPU and 34 of 40 on the NPU.

## Pictures: what the missing tiles cost

ONNX Runtime GenAI resizes a picture once. d1-3B's own code cuts a large picture into tiles. On 20 generated invoices (1600×1200, the total printed at 8 to 14 px, one `choice` question out of 4 options per picture):

| Run | right answers | prompt tokens |
| --- | --- | --- |
| d1-3B's own code, default tiles | 20/20 | 1832 |
| d1-3B's own code, `do_image_splitting=False` | 13/20 | 289 |
| GenAI 0.19.0, fp16, CUDA | 13/20 | 289 |

The last two rows give the same top answer on all 20 pictures.

## What results/ holds

`results/` holds question ids, times and probabilities, and two NPU op profiles (`r3_profile_*.txt`); `results/fine_requests.json` and `results/pictures_requests.json` hold the generated invoice questions.
