# OpenJev versus SemIf: 2048

Model: `mlx-community/Qwen3.5-0.8B-4bit`.
Same loaded model parameters, shared MLX runtime, paired seeds, and game information.
No generated reasoning or answer tokens.
Each method keeps its own prompt and readout.
OpenJev prompt: **full**; score mode: **full**; all 24 decoder layers.
OpenJev prefills instructions + user context once per decision, then scores all legal-move suffixes in one batch from independent cache copies. Both attention KV and recurrent state are copied; the saved prefix was checked for mutations on every warm-up board.

| Metric | OpenJev | SemIf |
| --- | ---: | ---: |
| Games | 10 | 10 |
| Mean score | 3,402.4 | 1,816.8 |
| Median score | 3,024.0 | 1,522.0 |
| Largest tile | 512 | 256 |
| Mean moves | 281.0 | 175.1 |
| Games reaching 2048 | 0 | 0 |
| Matched-board median latency | 226.8 ms | 162.9 ms |
| Matched-board p95 latency | 240.2 ms | 171.1 ms |
| Games reaching the move cap | 0 | 0 |

OpenJev scored higher on **8** seeds; SemIf on **2**; ties **0**.
Mean paired score difference (OpenJev minus SemIf): **1,585.6**.
Descriptive bootstrap 95% interval: **437.2 to 2,745.6**.

Latency is measured on the same frozen boards, excluding loading and warm-up.
The games diverge after different choices. A small game benchmark does not establish
general decision accuracy. This single-question task does not test caching across questions or successive decisions.
Cache copies and prefix prefill are included in the measured API latency.

Both use SemIf's pinned MLX-LM 0.32.0 revision, including its Qwen3.5 normalization fix.
OpenJev runs outside its usual `<0.32` dependency range for this test.
The protocol records the prompt, score mode, source hashes, and local modifications.

[Interactive report and replays](report.html) · [All games (CSV)](games.csv) ·
[Raw summary](summary.json) · [Frozen protocol](protocol.json)
