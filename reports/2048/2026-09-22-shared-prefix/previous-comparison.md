# OpenJev rerun versus the original comparison

Same Qwen3.5-0.8B 4-bit weights, patched MLX runtime, game instructions,
paired seeds, and frozen timing boards. All 24 layers are executed. No training.

| Metric | OpenJev previous | OpenJev current | SemIf current |
| --- | ---: | ---: | ---: |
| Mean game score | 2,940.0 | 3,402.4 | 1,816.8 |
| Median game score | 3,372.0 | 3,024.0 | 1,522.0 |
| Best tile | 512.0 | 512.0 | 256.0 |
| 2048 wins | 0.0 | 0.0 | 0.0 |
| Identical-board median | 340.2 ms | 226.8 ms | 162.9 ms |

Previous OpenJev: `full` prompt / `full` scoring / `single` execution. Current: `full` prompt / `full` scoring / `shared_prefix_batch` execution.

OpenJev's median latency decreased by 33.3%.
Mean paired score change from previous OpenJev: +462.4.
Bootstrap 95% interval: -650.8 to 1,673.2.
SemIf reproduced the same action sequences: True. Identical native choice distributions: True.
On the frozen boards, old/new OpenJev move agreement: 95.8%; maximum normalized probability difference: 0.025747. SemIf fixed-board distributions were identical: True.

The historical timings come from separate runs. Current OpenJev and SemIf timings
were interleaved on identical boards. This small sample does not establish general
decision quality. Configuration differences are recorded above. Quantized kernels can
give different scores when the same prompt is split into a prefill and candidate batch.

| Seed | OpenJev previous | OpenJev current | SemIf current |
| --- | ---: | ---: | ---: |
| 42 | 4144 | 6508 | 1616 |
| 43 | 3896 | 4348 | 1064 |
| 44 | 5416 | 3144 | 1336 |
| 45 | 3188 | 2904 | 1228 |
| 46 | 3556 | 2308 | 1464 |
| 47 | 916 | 3576 | 2628 |
| 48 | 936 | 4916 | 1356 |
| 49 | 1672 | 1708 | 2564 |
| 50 | 2008 | 1816 | 3332 |
| 51 | 3668 | 2796 | 1580 |

[Current report and replays](report.html) · [Machine-readable comparison](previous-comparison.json)
