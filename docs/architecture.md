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

## Prompt styles

`prompt_style="full"` is the original default. `prompt_style="short"` removes the
long system instruction and uses a user message containing the state, complete
question, candidate, and a short yes/no judging instruction. Explicit negative
criteria for `Noul` are retained. State and candidate data still escape embedded
chat delimiters. Both styles disable thinking and use all decoder layers.

Combine `prompt_style="short"` with `score_mode="binary"` to try the short-instruction,
existing yes/no head variant. Neither setting trains a model or fits a new head.
Prompt style and scoring mode are separate options, and responses record both.
Changing them can change choices; evaluate the combination on your own task.

## What is trimmed or optimized

- No autoregressive generation, reasoning stream, JSON generation, or sampling.
- Multiple questions share the instruction/content prefix within each request.
  Their candidate statements are pooled into batches across question boundaries.
- A single question skips prefix preparation. Its full candidate prompts are
  scored directly, in one batch when they fit the configured batch size.
- MLX text-only loading excludes the vision tower and MTP components through
  the inference library's supported model loader. Image-capable loading retains
  the vision tower. GGUF text loading does not download or load an `mmproj` file.
- Supported MLX Qwen/LFM2 and Transformers Qwen3.5 models skip the vocabulary projection during prefix
  prefill and project only the final position when scoring a candidate.
- `--score-mode binary` additionally projects just the two verdict rows on
  supported MLX Qwen/LFM2 models. `--no-head-optimization` runs the reference head.
- Other Transformers models use the smallest supported logits tail covering the
  last real token in each batch row.

No decoder layers, input vocabulary, or attention heads are deleted. Those are
part of interpreting the input, including when the answer is one token. The
full vocabulary projection is necessary for exact `P("yes")` normalization. With
tied embeddings, the embedding matrix is still needed to read arbitrary input
even in binary mode. Quantization can alter scores and should be evaluated on
your task. Removing more model capacity changes behavior and needs separate
quality evaluation; the production scorer keeps every decoder layer.


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

Transformers and supported MLX Qwen/LFM2 decoders sort candidates by length and right-pad
each batch, then gather the last real token from every row. The padded states are
discarded. Branches copy or merge the complete cache, including Qwen3.5's recurrent
states, LFM2's convolution states, attention KV tensors, and multimodal position offsets. This reuses prefix
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
BPE boundary errors. With `shared` and `tree`, caches are scoped to a single request; model calls from
different requests are serialized. Only the candidate rows within a request run
together. Short, general instructions improve reuse, but sharing also requires
an identical token prefix, model, and image input.

## Adaptive text caching (opt-in)

```python
with DecisionEngine.from_pretrained(cache_strategy="adaptive") as engine:
    result = engine.decide(state=context, questions=questions)
```

The engine computes the constant instruction/chat-template prefix through `State:`
once during initialization. This immutable snapshot remains with the loaded engine.
Each request still tokenizes complete prompts and verifies their exact prefix before
using the snapshot; an unusual tokenizer/template boundary falls back to computing
that request from scratch. No user context is retained between requests.

Before inference, a compressed token-prefix tree compares three execution shapes:
flat suffix batches, a longer common prefill followed by a batch, and recursively
split subgroups. Short branches that do not justify another cache are pooled at
their parent. Shared text can include the question, criteria, or any identical
candidate-prefix tokens. The planner uses token IDs only, without model scores or
semantic matching. Candidate attention and scores remain independent; normalization
still happens separately for each question.

The cost heuristic accounts for launches, padded batch lengths, configured batch
size, and cache copying. Its constants are estimates, not a calibrated hardware
profile or a guarantee of the globally fastest plan. Selected subgroups run in
sequence, with candidate rows parallel within each supported backend batch. A
single candidate uses one scoring pass from the permanent instruction cache,
without a separate request prefill. Complete KV and recurrent/conv states are
copied, and only caches along the active branch need to remain live.

`usage.instruction_prefix_tokens` reports the permanent prefix reused by the request.
`cache_prefill_tokens` lists newly computed shared segments. `scoring_prefix_tokens`
lists each candidate's retained prefix length, in question/candidate input order.
`planning_seconds` measures the Python planning step. Initialization cost is exposed
as `engine.instruction_cache_seconds` and is excluded from per-request timing/token
counts. `use_cache=False` bypasses both permanent and dynamic caches.

The existing `shared` default remains available for comparison. With `vision=True`,
`adaptive` uses the existing image-aware `shared`/`single` path; splitting the
multimodal prefill before image encoding is not enabled. GGUF retains serial
candidate execution. The benchmark in [adaptive_cache](../benchmarks/adaptive_cache/README.md)
measures this planner on Qwen 0.8B with MLX.
