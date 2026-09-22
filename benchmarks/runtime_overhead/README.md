# Tokenization and cache-branching comparison

Compare the current implementation with the source saved in the published
adaptive-cache 2048 report. Both use the same loaded Qwen3.5-0.8B MLX 4-bit model,
full prompts, full-vocabulary probabilities, adaptive caching and batches of four.

[Measured results](../../reports/cache/2026-09-22-runtime-overhead/report.md)

```sh
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .cache/semif-benchmark/venv/bin/python benchmarks/runtime_overhead/run.py \
  --output .cache/runtime-comparison --repeats 5
```

The environment and model are those installed by the
[2048 benchmark setup](../semif_2048/README.md). The offline flag requires the
model to have been downloaded already. Choose a new output directory for each run.

Seven timed workloads cover two 2048 boards, one yes/no question, and mixed
Noul/Choice/Score requests with 3, 12 and 24 questions and short/long context.
Four modes isolate the previous implementation, tokenization changes alone,
cache changes alone, and both together. Execution order rotates each repeat.

The script checks exact prompt equality across 82 workloads in both prompt styles,
exact answers across all timed modes, and 82 additional previous/current scoring
pairs. Candidate batch sizes remain identical, and the permanent cache must remain
unchanged. A mismatch fails the run rather than being accepted as a speedup.

Model loading, warmup, instruction-cache initialization and result logging are
outside the timed API calls. Results include raw requests, source snapshots,
runtime versions, summary timings and correctness checks. This is a latency and
equivalence experiment; it does not replay full games or measure new game scores.

To verify the raw records and regenerate the report without an inference backend:

```sh
python benchmarks/runtime_overhead/report.py .cache/runtime-comparison
```
