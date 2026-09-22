# OpenJev versus SemIf: 2048

Model: `mlx-community/Qwen3.5-0.8B-4bit`.
Same loaded model parameters, shared MLX runtime, paired seeds, and game information.
No generated reasoning or answer tokens.
Each method keeps its own prompt and readout.
OpenJev prompt: **full**; score mode: **full**; all 24 decoder layers.
OpenJev computes the generic instruction prefix once during initialization. Each decision prefills the board and shared game question, then scores all legal candidates in one batch from independent complete KV/recurrent cache copies. The planner chooses shared token prefixes without changing the full prompt. Retained caches and exact prompt reconstruction were checked on every warm-up board.

| Metric | OpenJev | SemIf |
| --- | ---: | ---: |
| Games | 10 | 10 |
| Mean score | 2,789.2 | 1,816.8 |
| Median score | 2,526.0 | 1,522.0 |
| Largest tile | 512 | 256 |
| Mean moves | 240.0 | 175.1 |
| Games reaching 2048 | 0 | 0 |
| Matched-board median latency | 136.6 ms | 162.4 ms |
| Matched-board p95 latency | 158.2 ms | 183.1 ms |
| Games reaching the move cap | 0 | 0 |

OpenJev scored higher on **8** seeds; SemIf on **2**; ties **0**.
Mean paired score difference (OpenJev minus SemIf): **972.4**.
Descriptive bootstrap 95% interval: **-86.4 to 1,993.6**.

Latency is measured on the same frozen boards, excluding loading and warm-up.
The games diverge after different choices. A small game benchmark does not establish
general decision accuracy. This single-question task does not test caching across questions. The board context changes on every move.
Cache copies and prefix prefill are included in the measured API latency.
The permanent 74-token prefix took 54.4 ms to initialize once after model loading. This is excluded from request timings; see [initialization.json](initialization.json).

Both use SemIf's pinned MLX-LM 0.32.0 revision, including its Qwen3.5 normalization fix.
OpenJev runs outside its usual `<0.32` dependency range for this test.
The protocol records the prompt, score mode, source hashes, and local modifications.

[Interactive report and replays](report.html) · [All games (CSV)](games.csv) ·
[Raw summary](summary.json) · [Frozen protocol](protocol.json)

Compared with the previous shared-prefix run, median latency fell 39.8% while mean game score fell from 3,402.4 to 2,789.2. [Full historical comparison](previous-comparison.md).
