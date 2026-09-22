# Qwen 0.8B: shared context, batched candidates

This small test compares two execution paths on the same loaded Qwen3.5-0.8B
4-bit MLX model, keeping the original full prompt and full-vocabulary `P(yes)`:

```text
Current single-question OpenJev:
  [instructions + context + candidate 1]
  [instructions + context + candidate 2]  → one full-prompt batch
  [instructions + context + candidate 3]

Shared context:
  instructions + context → prefill once → saved cache
    [cache copy + candidate 1]
    [cache copy + candidate 2]            → one suffix batch
    [cache copy + candidate 3]
```

Question text remains in each candidate suffix, preserving the exact prompts.
The saved cache includes attention KV and Qwen's recurrent state. Each candidate
receives an independent copy. Cache creation and copying count toward latency.

The corpus is eight synthetic customer-routing contexts: four short messages and
four longer documents. Each has four candidates. Three repetitions alternate
method order after warming both paths on every context. There is no gameplay,
generation, training, or instruction-only cache across requests.

Use the isolated MLX environment from the
[benchmark setup](../semif_2048/README.md), then run from the repository root:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .cache/semif-benchmark/venv/bin/python benchmarks/cache_reuse_qwen/run.py \
  --output .cache/cache-reuse-qwen/my-run
```

Use a new output directory. Remove `HF_HUB_OFFLINE=1` if the pinned model is not
already cached. The runner saves the corpus, source snapshots, every measured
call, a summary, and `report.md`. The alternative is an experiment-only engine;
production defaults are unchanged. Choice agreement measures consistency with
the current path, not decision accuracy.
