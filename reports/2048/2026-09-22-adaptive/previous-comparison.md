# OpenJev rerun versus the original comparison

Same Qwen3.5-0.8B 4-bit weights, patched MLX runtime, game instructions,
paired seeds, and frozen timing boards. All 24 layers are executed. No training.

| Metric | OpenJev previous | OpenJev current | SemIf current |
| --- | ---: | ---: | ---: |
| Mean game score | 3,402.4 | 2,789.2 | 1,816.8 |
| Median game score | 3,024.0 | 2,526.0 | 1,522.0 |
| Best tile | 512.0 | 512.0 | 256.0 |
| 2048 wins | 0.0 | 0.0 | 0.0 |
| Identical-board median | 226.8 ms | 136.6 ms | 162.4 ms |

Previous OpenJev: `full` prompt / `full` scoring / `shared_prefix_batch` execution. Current: `full` prompt / `full` scoring / `adaptive` execution.

OpenJev's median latency decreased by 39.8%.
Mean paired score change from previous OpenJev: -613.2.
Bootstrap 95% interval: -1,918.4 to 904.0.
SemIf reproduced the same action sequences: True. Identical native choice distributions: True.
On the frozen boards, old/new OpenJev move agreement: 87.5%; maximum normalized probability difference: 0.026251. SemIf fixed-board distributions were identical: True.

The historical timings come from separate runs. Current OpenJev and SemIf timings
were interleaved on identical boards. This small sample does not establish general
decision quality. Configuration differences are recorded above. Quantized kernels can
give different scores when the same prompt is split into a prefill and candidate batch.

| Seed | OpenJev previous | OpenJev current | SemIf current |
| --- | ---: | ---: | ---: |
| 42 | 6508 | 3936 | 1616 |
| 43 | 4348 | 1580 | 1064 |
| 44 | 3144 | 3316 | 1336 |
| 45 | 2904 | 3396 | 1228 |
| 46 | 2308 | 3204 | 1464 |
| 47 | 3576 | 588 | 2628 |
| 48 | 4916 | 1848 | 1356 |
| 49 | 1708 | 6364 | 2564 |
| 50 | 1816 | 1844 | 3332 |
| 51 | 2796 | 1816 | 1580 |

[Current report and replays](report.html) · [Machine-readable comparison](previous-comparison.json)
