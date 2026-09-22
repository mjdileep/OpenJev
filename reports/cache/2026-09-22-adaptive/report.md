# Qwen 0.8B: adaptive caching versus previous OpenJev

The adaptive planner reduces median latency in every tested category. It retains the same full prompts and full-vocabulary P(yes) scoring. Batched quantized scores are not numerically identical: 21/24 fixed 2048 boards select the same move. This is an execution benchmark, not a game-quality test.

## Timing

| Request category | Calls per path | Previous median | Adaptive median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 2048 boards | 72 | 225.81 ms | 134.74 ms | 1.68× |
| short context | 12 | 99.71 ms | 64.39 ms | 1.55× |
| long context | 12 | 311.98 ms | 272.62 ms | 1.14× |
| mixed questions | 6 | 142.54 ms | 122.91 ms | 1.16× |
| long question groups | 6 | 848.79 ms | 417.13 ms | 2.03× |
| single candidate | 6 | 139.62 ms | 121.38 ms | 1.15× |

| Request category | Previous p95 | Adaptive p95 | Previous evaluated tokens | Adaptive evaluated tokens |
| --- | ---: | ---: | ---: | ---: |
| 2048 boards | 239.80 ms | 151.72 ms | 754.62 | 426.46 |
| short context | 112.36 ms | 76.44 ms | 321.25 | 169.25 |
| long context | 340.26 ms | 294.17 ms | 1065.75 | 913.75 |
| mixed questions | 154.70 ms | 136.10 ms | 425.00 | 321.00 |
| long question groups | 998.21 ms | 538.86 ms | 2835.00 | 1299.00 |
| single candidate | 242.77 ms | 222.53 ms | 525.00 | 451.00 |

Evaluated token counts are means and exclude padding. The pooled single-candidate category contains one short and one long context; raw per-case times are in calls.jsonl.

Initialization computed **74 permanent tokens** in **70.40 ms** (once, excluding model load). This cost is excluded from steady-state request timings and token counts. The prefix includes the chat template and ends exactly at `State:`.

## Selected execution plans

All 2048 requests use one dynamic prefill through the shared game question and candidate header, followed by one batch of the 2–4 legal candidates. The previous version cached only through the context and repeated the question.

Mixed short questions pool all seven independent candidates into batches of 4 and 3. Long policies select three scoring batches: one standalone candidate and two groups of three sharing their respective question prefixes. Those batches run sequentially; rows within each batch run together. Single-candidate requests use one suffix pass, without a request-specific prefill.

| Example | New prefill segment lengths | Retained prefix per candidate | Batches |
| --- | --- | --- | --- |
| board-0 | [168] | [242, 242, 242, 242] | [4] |
| mixed-0 | [24] | [98, 98, 98, 98, 98, 98, 98] | [4, 3] |
| long-policies | [24, 378, 338] | [476, 476, 476, 436, 436, 436, 98] | [1, 3, 3] |
| single-0 | [] | [74] | [1] |

The cost heuristic considers launches, padded batch lengths, batch size, and cache copying. It can bypass tiny intermediate prefixes. Planning medians range from 0.018 to 0.519 ms. Its fixed cost coefficients are estimates; no global optimality or hardware calibration is claimed.

## Score differences

| Request category | Choice agreement (includes repetitions) | Maximum normalized probability difference | Maximum raw P(yes) difference |
| --- | ---: | ---: | ---: |
| 2048 boards | 63/72 | 0.026251 | 0.059846 |
| short context | 12/12 | 0.020557 | 0.031596 |
| long context | 12/12 | 0.011342 | 0.031071 |
| mixed questions | 6/6 | 0.036149 | 0.031609 |
| long question groups | 12/12 | 0.011885 | 0.037096 |
| single candidate | n/a | 0.000000 | 0.049315 |

The three changed boards below changed consistently in all repetitions. Every other tested Choice question retained its selected answer. All repeated calls within a path produced identical scores.

| Board index | Previous move | Adaptive move | Previous top-two probability gap |
| --- | --- | --- | ---: |
| board-3 | right | up | 0.011462 |
| board-14 | up | right | 0.000190 |
| board-21 | up | right | 0.010274 |

The strict real-model diagnostic uses batch size 1 and token-by-token arithmetic for both cached and independent inference. It passed with raw support differences below 1e-5 on two contexts and three candidates per context. CPU tests also check nested Qwen KV/recurrent caches against independent FP32 inference. These checks support execution-shape numerical differences as the explanation; they do not make the normal quantized paths equivalent or establish which changed move is better.

## Protocol and validation

Model: `mlx-community/Qwen3.5-0.8B-4bit` at `da28692b5f139cb0ec58a356b437486b7dac7462`. Both paths reference the same loaded model object. All 24 layers are retained, thinking is disabled, and no tokens are generated. Runtime: `{'mlx': '0.32.2', 'mlx-lm': '0.32.0', 'transformers': '5.17.0'}`; platform: `macOS-27.0-arm64-arm-64bit`.

38 fixed requests × 3 repeats × 2 paths = 228 measured calls. Both paths are warmed on all requests; timed order alternates. Synchronization, prompt compilation, planning, prefill, cache copies, scoring and result construction are included. Audit hashing, warmup, model load and instruction initialization are excluded. No games are replayed.

For single Choice requests, previous means the published shared-prefix helper. For multiple questions or a single Noul, it means the previous shared API path. Complete prompts are compared against the published original prompt builder.

The untimed audit verified 76 requests and 118 parent-cache uses, reconstructing exact full token streams and hashing all KV/recurrent state. The permanent cache remained unchanged after every warmup and all timed calls. 64 non-integration tests passed (1 skipped), and the strict real-model test passed. The standard-library verifier checks all 228 calls and source snapshots.

## Use and reproduce

```python
engine = DecisionEngine.from_pretrained(cache_strategy="adaptive")
```

This option is enabled explicitly; the shared default remains unchanged. The performance measurements cover MLX text inference on this Mac. Vision retains the existing image-aware path, and GGUF retains serial candidate scoring.

[Reproduction instructions](../../../benchmarks/adaptive_cache/README.md) · [Raw calls](calls.jsonl) · [Inputs](cases.json) · [Summary](summary.json) · [Protocol](protocol.json) · [Initialization](initialization.json) · [Verification](verification.json) · [Validation](validation.json)
