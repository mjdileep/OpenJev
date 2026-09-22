# OpenJev versus SemIf: 2048

Model: `mlx-community/Qwen3.5-0.8B-4bit`. Thinking disabled; no generated tokens.
Same loaded model parameters, shared MLX runtime, paired seeds, and game information.
Each method keeps its own prompt and readout.

| Metric | OpenJev | SemIf |
| --- | ---: | ---: |
| Games | 10 | 10 |
| Mean score | 2,940.0 | 1,816.8 |
| Median score | 3,372.0 | 1,522.0 |
| Largest tile | 512 | 256 |
| Mean moves | 251.6 | 175.1 |
| Games reaching 2048 | 0 | 0 |
| Matched-board median latency | 340.2 ms | 163.1 ms |
| Matched-board p95 latency | 381.0 ms | 183.6 ms |
| Games reaching the move cap | 0 | 0 |

OpenJev scored higher on **6** seeds; SemIf on **4**; ties **0**.
Mean paired score difference (OpenJev minus SemIf): **1,123.2**.
Descriptive bootstrap 95% interval: **-65.6 to 2,287.2**.

Latency is measured on the same frozen boards, excluding loading and warm-up.
The games diverge after different choices. A small game benchmark does not establish
general decision accuracy. This single-question task does not test shared-state caching.

Both use SemIf's pinned MLX-LM 0.32.0 revision, including its Qwen3.5 normalization fix.
OpenJev's unchanged source runs outside its usual `<0.32` dependency range for this test.

[Interactive report and replays](report.html) · [All games (CSV)](games.csv) ·
[Raw summary](summary.json) · [Frozen protocol](protocol.json)
