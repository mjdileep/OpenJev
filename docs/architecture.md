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
- Both levels of prefix computation are reused within each request.
- MLX text-only loading excludes the vision tower and MTP components through
  the inference library's supported model loader. Image-capable loading retains
  the vision tower. GGUF text loading does not download or load an `mmproj` file.
- Supported MLX Qwen models skip the vocabulary projection during prefix
  prefill and project only the final position when scoring a candidate.
- `--score-mode binary` additionally projects just the two verdict rows on
  supported MLX Qwen models. `--no-head-optimization` runs the reference head.
- Transformers asks for only the last position's logits when the model supports it.

No decoder layers, input vocabulary, or attention heads are deleted. Those are
part of interpreting the input, including when the answer is one token. The
full vocabulary projection is necessary for exact `P("yes")` normalization. With
tied embeddings, the embedding matrix is still needed to read arbitrary input
even in binary mode. Quantization can alter scores and should be evaluated on
your task; removing more model capacity requires retraining or distillation.


## Implementation details

All backends reuse prefix computation. They copy or merge cache snapshots; this is not a paged, zero-copy cache allocator. MLX batches equal-length candidate suffixes when every layer supports cache merging. GGUF and Transformers evaluate branches sequentially.

Qwen3.5 uses recurrent state as well as attention KV tensors. Branches preserve both. Visual branches also preserve image embeddings and multimodal position offsets. Complete chat prompts are tokenized before computing shared prefixes, avoiding BPE boundary errors. Caches are scoped to a single request.
