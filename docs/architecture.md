# Architecture and scores

## Score semantics

The model is instructed to emit `yes` for a correct candidate and `no` otherwise.
Neither token is sampled. No output grammar, logit bias, temperature, or top-k
filter is applied.

- **`full` (default):** support is `P("yes")` under the complete vocabulary. This
  implements the proposed single-positive-token method.
- **`binary`:** support is `P("yes") / (P("yes") + P("no"))`. This is conditional on a
  binary verdict and can differ substantially when the model prefers other tokens.

Candidate records include the raw log probability, positive and negative
probabilities, binary conditional probability, and binary token mass where
available. With an optimized two-row MLX projection, full-vocabulary probabilities
are unavailable and are explicitly `null`.

`choice.probabilities` and `score.probabilities` normalize independent candidate
supports to sum to one. Choice takes the largest value; score is the expected
zero-based level. Noul directly uses the positive candidate's support. **These
are not calibrated outcome probabilities.** Every response says `calibrated: false`.
A confident-looking normalized choice can arise even if all candidates have very
low support; inspect `candidates`, not just the normalized distribution. Equal
scores use the first option as a deterministic tie break.

## What is trimmed or optimized

- No autoregressive generation, reasoning stream, JSON generation, or sampling.
- Multiple questions share the instruction/content prefix within each request.
  Their candidate statements are pooled into batches across question boundaries.
- A single question skips prefix preparation. Its full candidate prompts are
  scored directly, in one batch when they fit the configured batch size.
- MLX text-only loading excludes the vision tower and MTP components through
  the inference library's supported model loader. Image-capable loading retains
  the vision tower. GGUF text loading does not download or load an `mmproj` file.
- Supported MLX Qwen and Transformers Qwen3.5 models skip the vocabulary projection during prefix
  prefill and project only the final position when scoring a candidate.
- `--score-mode binary` additionally projects just the two verdict rows on
  supported MLX Qwen models. `--no-head-optimization` runs the reference head.
- Other Transformers models use the smallest supported logits tail covering the
  last real token in each batch row.

No decoder layers, input vocabulary, or attention heads are deleted. Those are
part of interpreting the input, including when the answer is one token. The
full vocabulary projection is necessary for exact `P("yes")` normalization. With
tied embeddings, the embedding matrix is still needed to read arbitrary input
even in binary mode. Quantization can alter scores and should be evaluated on
your task; removing more model capacity requires retraining or distillation.


## Implementation details

The default `cache_strategy="shared"` uses this shape:

```text
instruction + content (prefill once, including images)
  ├─ question A + candidate 1 ─┐
  ├─ question A + candidate 2  │ score in batches → P(yes) per row
  └─ question B + candidate 1 ─┘
```

Each statement retains its question and candidate criteria, so the meaning and
scoring prompt are unchanged. `cache_strategy="tree"` additionally prefills each
question's prefix and batches only within that question. This saves more tokens
but adds model calls; it can help with long question instructions. Both strategies
automatically use direct full-prompt scoring for a single question. `use_cache=False`
is the diagnostic baseline: independent full prompts, one candidate at a time.

Transformers and supported MLX Qwen decoders sort candidates by length and right-pad
each batch, then gather the last real token from every row. The padded states are
discarded. Branches copy or merge the complete cache, including Qwen3.5's recurrent
states, attention KV tensors, and multimodal position offsets. This reuses prefix
computation but duplicates cache memory; it is not a paged, zero-copy allocator.
Image features enter the shared prefix once. A single-question image batch repeats
the image inputs for its full candidate prompts.

GGUF still scores candidates serially. Generic MLX models retain equal-length
groups; unsupported cache merge/reorder implementations fall back to singleton
branches. `usage.candidate_batches` reports the actual batch sizes, and
`usage.padding_tokens` counts extra padded positions separately from real
`evaluated_input_tokens`. Reused tokens measure avoided input computation, not
saved cache storage. Quantization and changes in batch shape can alter scores.

Complete chat prompts are tokenized before computing shared prefixes, avoiding
BPE boundary errors. Caches are scoped to a single request; model calls from
different requests are serialized. Only the candidate rows within a request run
together. Short, general instructions improve reuse, but sharing also requires
an identical token prefix, model, and image input.
