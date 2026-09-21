# Validation

Recorded on 2026-09-21. Local checks used an Apple Silicon Mac with Python
3.12.10; the Colab checks below used a Tesla T4.

## Shared-prefix batching on MLX

Current implementation, Qwen3.5-0.8B 4-bit, seven candidates across three questions.
Five measured iterations followed warmup; each strategy alternated with its own
independent full-prompt baseline. Mac M3 Pro, MLX 0.32.2, MLX-LM 0.31.3.

| Mode | Median latency | Candidate batches | Real tokens evaluated | Padding tokens |
|---|---:|---|---:|---:|
| Shared prefix, batch size 8 (default) | 144.66 ms | `[7]` | 431 | 49 |
| Question-level tree, batch size 8 | 158.80 ms | `[1, 3, 3]` | 371 | 7 |
| Shared prefix, batch size 1 | 190.19 ms | seven singleton batches | 431 | 0 |
| Independent full prompts (default comparison) | 309.29 ms | seven singleton batches | 959 | 0 |

The default was **2.14×** faster than its independent baseline on this short
example. It reused 528 input tokens and generated zero tokens. Discrete choices
agreed; maximum absolute candidate-support difference was **0.0318**. The tree
and batch-size-1 comparisons differed from their baselines by 0.0617 and 0.0587.
Reducing model calls can be faster even when it processes more real tokens.
These are local measurements, not guaranteed improvements on other hardware.

Mixed-length text and image checks used one four-candidate batch, with no score
change when question/candidate order was reversed. Single-question text and image
checks used one three-candidate batch with no prefix prefill. Against independent
scoring, the largest observed support difference was 0.0708 on the quantized image
fixture. The selected color remained correct. Scores are not calibrated and
should be evaluated on the intended task.

Reproduce using the commands in [development](development.md). Model snapshot:
`da28692b5f139cb0ec58a356b437486b7dac7462`. MLX-VLM 0.7.1 supplied image support.

## Earlier MLX tree-cache benchmark

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

- Unit tests validate exact prompt reconstruction, both cache strategies, single-question
  prefill bypass, cross-question batching, candidate
  order independence, changed content, stable normalization, schema validation,
  tokenizer-boundary checks, context limits, and cleanup.
- MLX and GGUF cached scores matched full-prompt scores exactly in the diagnostic
  using singleton batches and one-token prefill chunks. These settings keep the
  numerical operations consistent so the check isolates cache behavior.
- Optimized MLX two-row projection was compared against the full output head.
- Tiny FP32 Qwen3.5 text and vision models check real PyTorch padded batches
  against independent scoring at 1e-5 tolerance, including cache isolation,
  multimodal positions, candidate ordering, and one forward pass for a single
  question. A separate CI job runs these without model downloads.
- The current shared and single-question paths also passed mixed-length text
  and image tests with the full Qwen3.5-0.8B weights on CPU, using PyTorch 2.14.0
  and Transformers 5.17.0, at 0.002 absolute support tolerance.
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

### Current 0.8B NF4 shared batching

The updated quickstart completed **Run all** on a Tesla T4 with
`Qwen/Qwen3.5-0.8B`, `USE_4BIT=True`, `ENABLE_IMAGES=True`, and `BATCH_SIZE=8`.
It installed core commit `c4cf18d6bae7197810d57181d442c141d4428767`, using
PyTorch 2.11.0+cu128 and Transformers 5.17.0. Installation, loading, all examples,
benchmarking, JSON export, and cleanup completed.

| Text benchmark | Shared batching | Independent full prompts |
|---|---:|---:|
| Median latency | 338.8 ms | 825.2 ms |
| Candidate batches | `[7]` | seven singleton batches |
| Reused input tokens | 522 | 0 |
| Padding tokens | 49 | 0 |
| Generated tokens | 0 | 0 |

Three measured iterations followed warmup. Observed speedup was **2.44×**;
maximum absolute candidate-support difference was **0.0249**. Both paths chose
`billing`. These compare the new implementation's two execution modes, not an
isolated before/after measurement against an older commit.

The single-question example reported `cache_strategy="single"`, batch `[1]`,
and zero separately prepared prefix tokens. Its urgency support matched the
multi-question run at the displayed precision (0.5595). The image example chose
`red`, scored five candidates in batch `[5]`, and reused 592 input tokens.

Transformers used reference PyTorch convolution and DeltaNet kernels; optional
optimized kernels were not installed. This is a smoke and performance check on
one small example, not an accuracy evaluation or a general speed guarantee.

### Earlier 0.8B runs (before shared batching)

The [published quickstart](https://colab.research.google.com/github/mjdileep/OpenJev/blob/main/notebooks/OpenJev_Quickstart.ipynb)
uses `Qwen/Qwen3.5-0.8B` with `USE_4BIT=True` and images enabled. This configuration
was validated on a Colab Tesla T4 before restoring it as the default. The timings
below use the earlier sequential Transformers implementation and do not validate
the new shared batching path. Both
`USE_4BIT=False` and `USE_4BIT=True` (bitsandbytes NF4) completed **Run all**.
Both runs used the `yes`/`no` verdicts, PyTorch 2.11.0+cu128, Transformers 5.17.0,
and `Qwen/Qwen3.5-0.8B` with images enabled. Installation, model loading, text
scoring, image scoring, benchmarking, JSON export, and model cleanup all completed.

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

### Earlier 4B NF4 configuration

Before the default was restored to 0.8B, the quickstart completed **Run all** with
`Qwen/Qwen3.5-4B`, `USE_4BIT=True`, and images enabled on a Colab Tesla T4. This
used PyTorch 2.11.0+cu128, Transformers 5.17.0, and the default `yes`/`no` verdicts.
Model loading, text and image scoring, benchmarking, JSON export, and cleanup
all completed.

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

CUDA GGUF execution remains untested. GGUF image inference is deliberately
unsupported. No calibrated-accuracy or Jev-comparison benchmark is claimed.
