# Local validation

Recorded on 2026-09-21 on an Apple Silicon Mac, Python 3.12.10.

## MLX cache benchmark

Command (three measured iterations after warming both execution paths):

```bash
openjev benchmark examples/triage.json --backend mlx --iterations 3
```

| Measurement | Cached | Uncached |
|---|---:|---:|
| Median latency | 190.93 ms | 333.68 ms |
| Evaluated input tokens | 373 | 973 |
| Reused input tokens | 600 | 0 |
| Generated tokens | 0 | 0 |

Seven candidates across three questions. Observed speedup: **1.75×**. The maximum
absolute candidate-support difference was **0.0152**; discrete choices agreed.
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
  absolute support tolerance because visual prefixes must stay intact.

Optimized batching and larger prefill chunks can produce different floating-point
results, particularly with quantized hybrid models. Observed differences on the
small smoke fixtures were a few percentage points. Use the benchmark command to
measure this on your own workload; matching argmax decisions is not proof of
calibration or general accuracy.

**NVIDIA CUDA execution and bitsandbytes NF4 have not been tested on this Mac.**
GGUF image inference is deliberately unsupported. No calibrated-accuracy or
Jev-comparison benchmark is claimed.
