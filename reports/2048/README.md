# OpenJev versus SemIf: 2048 results

The current featured run uses **adaptive caching** on an Apple M3 Pro with 18 GiB
memory. Both methods use the same pinned Qwen3.5-0.8B 4-bit weights and shared MLX
runtime. Generic instructions stay cached with the OpenJev model; each request
prefills the shared board/question prefix and independently scores legal moves.

[Full report](2026-09-22-adaptive/report.md) ·
[Recorded replay](2026-09-22-adaptive/report.html) ·
[Per-game scores](2026-09-22-adaptive/games.csv) ·
[Raw summary](2026-09-22-adaptive/summary.json)

| Metric | OpenJev adaptive | SemIf |
| --- | ---: | ---: |
| Mean game score | 2,789.2 | 1,816.8 |
| Median game score | 2,526 | 1,522 |
| Largest tile | 512 | 256 |
| Identical-board median latency | 136.6 ms | 162.4 ms |
| Games reaching 2048 | 0/10 | 0/10 |

Each method played ten games, seeds 42–51. Timing used 24 frozen boards with three
repetitions, alternating method order. OpenJev scored higher on 8 seeds and SemIf
on 2. This is a small local game experiment, not a general accuracy ranking.
The one-time 54.4 ms instruction-cache initialization is reported separately;
per-request prefill, planning, copying and scoring are included in timing.

## What changed from the previous run

The [comparison with the previous shared-prefix run](2026-09-22-adaptive/previous-comparison.md)
shows **39.8% lower OpenJev median latency**, while average game score decreased
from **3,402.4 to 2,789.2**. Old/new OpenJev moves agree on **21/24** frozen boards.
Quantized execution changes can alter scores and later game trajectories. SemIf
reproduced every previous game action and probability distribution exactly.
Historical timing comparisons come from separate runs.

## Watch the recorded games

After cloning the repository, open the standalone HTML replay. On macOS:

```bash
open reports/2048/2026-09-22-adaptive/report.html
```

On other systems, double-click that file. It works offline without model weights
or Python packages. GitHub shows HTML source; clone or download the file to play
it. Pick a seed, then play, pause, step, or scrub. Playback aligns move numbers,
not elapsed time. This is a recorded replay, not live inference.

## Verify and reproduce

From the repository root, with Python 3.11 or newer:

```bash
python3 -S benchmarks/semif_2048/verify.py reports/2048/2026-09-22-adaptive
```

Verification uses only the standard library and this checkout. It loads no models
and requires no GPU. The current run verifies **20 games, 4,151 moves and 144 timed
decisions**. Warmup also checked full prompt reconstruction and immutable retained
KV/recurrent caches; the permanent instruction cache stayed unchanged after all games.

Each run contains full Markdown/HTML reports, CSV scores, raw game transitions,
fixed-board decisions, protocol, source snapshots, model artifact hashes and
verification results. The new run also records `initialization.json` and
`cache-validation.json`. Model weights are not committed.

Follow the [benchmark setup](../../benchmarks/semif_2048/README.md) and use
`--openjev-execution adaptive` to reproduce this execution path. Both methods use
SemIf's pinned MLX-LM 0.32.0 revision, outside OpenJev's normal `<0.32` dependency
range. The API's default remains unchanged; opt in with `cache_strategy="adaptive"`.

For requests containing several question types, see the
[mixed Noul/Choice/Score report](../cache/2026-09-22-mixed-types/report.md).

## Historical runs

These retain their original evidence and are no longer the featured results:

- [Previous shared-prefix run](2026-09-22-shared-prefix/report.md)
- [Original full-prompt batch baseline](2026-09-22-full-prompt-baseline/report.md)
