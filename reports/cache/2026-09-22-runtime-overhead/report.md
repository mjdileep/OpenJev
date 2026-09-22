# OpenJev tokenization and cache-branching optimizations

Measured on 22 September 2026 on an Apple M3 Pro with 18 GiB memory. Same pinned Qwen3.5-0.8B MLX 4-bit model, adaptive caching, original full prompts, full-vocabulary scoring and candidate batches of four on both sides.

## Results

Median end-to-end milliseconds, five repeats per workload and mode. The last column is the reduction in latency relative to the previous implementation.

| Workload | Previous | Tokenization only | Cache only | Both | Reduction |
|---|---:|---:|---:|---:|---:|
| 2048 board 0 (4 candidates) | 136.50 | 132.39 | 136.73 | 131.75 | 3.5% |
| 2048 board 12 (4 candidates) | 134.37 | 131.82 | 132.22 | 128.06 | 4.7% |
| Single yes/no question | 29.42 | 29.11 | 29.55 | 28.78 | 2.2% |
| 3 mixed questions (7 candidates) | 136.12 | 130.40 | 130.23 | 125.47 | 7.8% |
| 12 mixed questions (28 candidates) | 421.12 | 398.50 | 399.99 | 378.77 | 10.1% |
| 24 mixed questions (56 candidates) | 802.67 | 748.69 | 754.31 | 709.32 | 11.6% |
| 12 mixed questions, long context | 651.74 | 585.09 | 638.33 | 554.74 | 14.9% |

## What changed

- Prompt preparation deduplicates complete strings and encodes them in native batches of at most 128. Each yes/no marker is encoded once per request. Every full prompt and both exact answer continuations are still checked; token fragments are never joined.
- The tokenizer batch API is obtained from the owner of its bound encode method. This supports MLX's tokenizer wrapper while preserving its chat template and disabled thinking. Image and GGUF tokenizer overrides retain their own encode behavior.
- MLX branches known KV and recurrent caches directly into separate candidate rows, avoiding redundant deep copies and repeated zero-fill/slice assignments. Unknown cache classes retain the defensive copy/merge path. Single-candidate calls keep their existing execution path.
- Question normalization, prompts, model weights, precision, cache planning and batch sizes are unchanged. The improvements require no new user-facing options.

## Isolated prompt preparation

Measured separately after inference validation, using the actual loaded MLX tokenizer. Two warmups and five measured repetitions per builder; order alternates.

| Workload | Previous ms | Current ms | Speedup |
|---|---:|---:|---:|
| 2048 board 0 (4 candidates) | 4.93 | 1.34 | 3.67× |
| 2048 board 12 (4 candidates) | 4.86 | 1.32 | 3.67× |
| Single yes/no question | 1.03 | 0.52 | 1.99× |
| 3 mixed questions (7 candidates) | 7.03 | 1.92 | 3.67× |
| 12 mixed questions (28 candidates) | 27.89 | 6.33 | 4.41× |
| 24 mixed questions (56 candidates) | 54.41 | 12.50 | 4.35× |
| 12 mixed questions, long context | 93.16 | 16.42 | 5.67× |

## Correctness and measurement limits

- Exact prompt equality for 164 workload/style combinations (82 workloads with both full and short prompts).
- Exact answer equality for all 140 timed requests and 82 additional previous/current request pairs. Maximum observed score difference: **0**. Warmup outputs also matched exactly.
- Expanded checks include all 24 original fixed 2048 boards, mixed question types, long questions, uneven candidates, wide option sets and changed contexts. Candidate batch sizes match; the permanent instruction cache remains unchanged.
- Execution order rotates across four modes; the model object is shared. End-to-end measurements synchronize at request boundaries, with no extra barriers inside inference. Loading, warmup, static instruction-cache initialization and logging are excluded; per-request prompt work, prefix computation and branching are included.
- This is a small local latency experiment, not a new full-game or SemIf benchmark. Individual small differences can reflect timing noise. No new gameplay quality or cross-hardware speed claim is made.

## Reproduction and artifacts

Run `benchmarks/runtime_overhead/run.py --output <new-directory> --repeats 5` in the existing MLX benchmark environment, then `python benchmarks/runtime_overhead/report.py <new-directory>`.

[Protocol and versions](protocol.json) · [Raw timed requests](requests.jsonl) · [Summary](summary.json) · [Prompt timing](prompt-timing.json) · [Validation](validation.json) · [Executed source](source)
