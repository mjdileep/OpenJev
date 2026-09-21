# Validation

Recorded on 2026-09-21. Local checks used an Apple Silicon Mac with Python
3.12.10; the Colab checks below used a Tesla T4.

## MLX cache benchmark

Command (three measured iterations after warming both execution paths):

```bash
openjev benchmark examples/triage.json --backend mlx --iterations 3
```

| Measurement | Cached | Uncached |
|---|---:|---:|
| Median latency | 185.11 ms | 305.32 ms |
| Evaluated input tokens | 371 | 959 |
| Reused input tokens | 588 | 0 |
| Generated tokens | 0 | 0 |

Seven candidates across three questions, using the default `yes`/`no` verdicts.
Observed speedup: **1.65×**. The maximum absolute candidate-support difference was
**0.0593**; discrete choices agreed.
The benchmark excludes model loading and image file preprocessing. These are
local observations, not a throughput guarantee or a comparison against Jev.

The default model was `mlx-community/Qwen3.5-0.8B-4bit`, snapshot
`da28692b5f139cb0ec58a356b437486b7dac7462`, with MLX 0.32.2 and MLX-LM 0.31.3.

## Correctness checks

- Unit tests validate exact prompt reconstruction, both cache levels, candidate
  order independence, changed content, stable normalization, schema validation,
  tokenizer-boundary checks, context limits, and cleanup.
- MLX and GGUF cached scores matched full-prompt scores exactly in the diagnostic
  using singleton batches and one-token prefill chunks. These settings keep the
  numerical operations consistent so the check isolates cache behavior.
- Optimized MLX two-row projection was compared against the full output head.
- MLX image tests matched cached and uncached evaluation and correctly changed
  the selected color between red and blue fixtures. MLX-VLM version: 0.7.1.
- GGUF used llama-cpp-python 0.3.35 on CPU with
  `unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`, snapshot
  `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.
- Transformers passed the text cache/order and red/blue image tests on CPU with
  PyTorch 2.14.0 and Transformers 5.17.0. Model: `Qwen/Qwen3.5-0.8B`, snapshot
  `2fc06364715b967f1860aea9cf38778875588b17`. The image comparison uses a 0.002
  absolute support tolerance because visual prefixes must stay intact. This CPU
  run used the earlier `1`/`0` verdicts.

Optimized batching and larger prefill chunks can produce different floating-point
results, particularly with quantized hybrid models. Observed differences on the
small smoke fixtures were a few percentage points. Use the benchmark command to
measure this on your own workload; matching argmax decisions is not proof of
calibration or general accuracy.

## Colab CUDA notebook

### Current 4B NF4 default

The [published quickstart](https://colab.research.google.com/github/mjdileep/OpenJev/blob/main/notebooks/OpenJev_Quickstart.ipynb)
completed **Run all** with `Qwen/Qwen3.5-4B`, `USE_4BIT=True`, and images enabled
on a Colab Tesla T4. This used PyTorch 2.11.0+cu128, Transformers 5.17.0, and the
default `yes`/`no` verdicts. Model loading, text and image scoring, benchmarking,
JSON export, and cleanup all completed.

The support message selected `billing`; the sample image selected `red`.
Text scoring reused 582 input tokens and generated zero answer tokens.

| Text benchmark | 4B NF4 |
|---|---:|
| Cached median | 4071.4 ms |
| Uncached median | 5677.3 ms |
| Observed speedup | 1.39× |
| Maximum absolute support difference | 0.0279 |
| Same cached/uncached choices | Yes |

Three measured iterations followed warmup. These results use the reference
PyTorch convolution and DeltaNet implementations, without optional optimized
kernels. They verify this small example runs on a T4; they are not a model-quality
benchmark or a guarantee for other inputs.

### Earlier 0.8B configuration

The quickstart's earlier 0.8B configuration completed **Run all** on a Colab
Tesla T4 with both `USE_4BIT=False` and
`USE_4BIT=True` (bitsandbytes NF4). Both runs used the `yes`/`no` verdicts,
PyTorch 2.11.0+cu128, Transformers 5.17.0, and `Qwen/Qwen3.5-0.8B` with images
enabled. Installation, model loading, text scoring, image scoring, benchmarking,
JSON export, and model cleanup all completed.

Both runs chose `billing` for the support message and `red` for the sample image.
Text scoring reused 582 input tokens; image scoring reused 643. No answer tokens
were generated.

| Text benchmark | Default weights | NF4 weights |
|---|---:|---:|
| Cached median | 739.9 ms | 1005.7 ms |
| Uncached median | 770.2 ms | 860.9 ms |
| Observed speedup | 1.04× | 0.86× |
| Maximum absolute support difference | 0.0136 | 0.0308 |
| Same cached/uncached choices | Yes | Yes |

Three measured iterations followed warmup. Transformers used its reference
PyTorch convolution and DeltaNet implementations; optional optimized kernels
were not installed. NF4 caching was slower on this short example, so reducing
weight memory does not imply lower latency. These are smoke checks, not an
accuracy evaluation or a general performance claim.

## Bonsai 2 27B GGUF

The [Bonsai notebook](https://colab.research.google.com/github/mjdileep/OpenJev/blob/main/notebooks/OpenJev_Bonsai.ipynb)
is published with these pinned components:

- GGUF: `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision
  `6ed5e12bf84b7a63069882c91dd9e9218647d17b`, file
  `Ternary-Bonsai-2-27B-PTQ1_0.gguf`.
- Tokenizer: `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, revision
  `3f926b415992eaa2ae9dd7b573706494d6bbf787`.
- Runtime: Prism release `prism-b10709-9a9394a`; CUDA 12.4 archive for Colab.
- Python bindings: llama-cpp-python 0.3.35, with the pinned fork's
  `dspark_head_source` and `path_kv_mean_center` struct fields added.

Local verification passed: the real Bonsai tokenizer maps `yes` to 9405 and
`no` to 2083, and disables thinking through its chat template. The adapted
bindings loaded the same Prism release's macOS library and passed the existing
real-model cache and candidate-order integration test with the previously
downloaded Qwen3.5-0.8B GGUF on CPU. This verifies the bindings and state handling
on that fixture; it does not verify inference with Bonsai's 27B weights.

The published notebook was opened in signed-in Colab and **Run all** was
requested, but the Chrome connection was lost before execution output could be
read. **Bonsai T4 inference, memory usage, and latency are not yet verified.**
The notebook records these measurements, raw candidate evidence, and a cache
comparison when run. No Bonsai output or performance result is claimed here.

CUDA GGUF execution remains unverified. GGUF image inference is unsupported.
No calibrated-accuracy or Jev-comparison benchmark is claimed.
