# Comparison with the previous run

Same Qwen3.5-0.8B 4-bit weights, patched MLX runtime, game instructions,
paired seeds, and frozen timing boards. All 24 layers are executed. No training.

| Metric | OpenJev previous | OpenJev current | SemIf current |
| --- | ---: | ---: | ---: |
| Mean game score | 2,789.2 | 2,789.2 | 1,816.8 |
| Median game score | 2,526.0 | 2,526.0 | 1,522.0 |
| Best tile | 512.0 | 512.0 | 256.0 |
| 2048 wins | 0.0 | 0.0 | 0.0 |
| Identical-board median | 136.6 ms | 136.6 ms | 171.5 ms |

Previous OpenJev: `full` prompt / `full` scoring / `adaptive` execution. Current: `full` prompt / `full` scoring / `adaptive` execution.

OpenJev's median latency decreased by 0.1%.
Mean paired score change from previous OpenJev: +0.0.
Bootstrap 95% interval: 0.0 to 0.0.
OpenJev reproduced the same action sequences: True. Identical native choice distributions: True.
SemIf reproduced the same action sequences: True. Identical native choice distributions: True.
On the frozen boards, old/new OpenJev move agreement: 100.0%; maximum normalized probability difference: 0.000000. SemIf fixed-board distributions were identical: True.

The historical timings come from separate runs. Current OpenJev and SemIf timings
were interleaved on identical boards. This small sample does not establish general
decision quality. Configuration differences are recorded above. Quantized kernels can
give different scores when the same prompt is split into a prefill and candidate batch.

| Seed | OpenJev previous | OpenJev current | SemIf current |
| --- | ---: | ---: | ---: |
| 42 | 3936 | 3936 | 1616 |
| 43 | 1580 | 1580 | 1064 |
| 44 | 3316 | 3316 | 1336 |
| 45 | 3396 | 3396 | 1228 |
| 46 | 3204 | 3204 | 1464 |
| 47 | 588 | 588 | 2628 |
| 48 | 1848 | 1848 | 1356 |
| 49 | 6364 | 6364 | 2564 |
| 50 | 1844 | 1844 | 3332 |
| 51 | 1816 | 1816 | 1580 |

[Current report and replays](report.html) · [Machine-readable comparison](previous-comparison.json)
