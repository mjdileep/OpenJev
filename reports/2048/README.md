# OpenJev versus SemIf: published 2048 results

These are recorded local experiments on an Apple M3 Pro with 18 GiB memory.
Both methods use the same pinned Qwen3.5-0.8B 4-bit weights and shared MLX runtime.
The reports retain the original measurements and protocols.

| Run | OpenJev execution | Mean score: OpenJev / SemIf | Median latency: OpenJev / SemIf |
| --- | --- | ---: | ---: |
| [Shared prefix, September 22, 2026](2026-09-22-shared-prefix/report.md) | One context prefill, then independent candidate suffixes in one batch | 3,402.4 / 1,816.8 | 226.8 / 162.9 ms |
| [Earlier full-prompt baseline](2026-09-22-full-prompt-baseline/report.md) | Complete candidate prompts in one batch | 2,940.0 / 1,816.8 | 340.2 / 163.1 ms |

Each run has ten games per method, seeds 42–51, and 24 frozen timing boards with
three repetitions. Neither method reached 2048. OpenJev scored higher on eight
seeds in the shared-prefix run. This is a small game experiment, not a general
accuracy ranking or a calibrated probability of winning.

The [comparison with the earlier run](2026-09-22-shared-prefix/previous-comparison.md)
shows a 33.3% reduction in OpenJev's median latency. Historical timings come from
separate runs. Old/new OpenJev choices agree on 23 of 24 frozen boards; quantized
execution differences can change scores and later game trajectories. SemIf
reproduced every earlier game action and choice distribution.

## Watch the recorded games

After cloning this repository, open
[the standalone replay](2026-09-22-shared-prefix/report.html) in your browser.
On macOS, from the repository root:

```bash
open reports/2048/2026-09-22-shared-prefix/report.html
```

On other systems, double-click that HTML file. It works offline without model
weights or Python packages. GitHub displays HTML as source; download the file or
clone the repository to play it. Pick a seed, then play, pause, step, or scrub.
This is a recorded replay; move numbers are aligned, rather than elapsed time.

## Verify the evidence

From the repository root, using Python 3.11 or newer:

```bash
python3 benchmarks/semif_2048/verify.py reports/2048/2026-09-22-shared-prefix
python3 benchmarks/semif_2048/verify.py reports/2048/2026-09-22-full-prompt-baseline
```

Verification uses the standard library and this checkout's source. It loads no
models and requires no GPU. The shared-prefix run verifies 20 games, 4,561 moves,
and 144 fixed-board decisions. The baseline verifies 20 games, 4,267 moves, and
144 fixed-board decisions.

Each run contains:

- `report.md`, `report.html`, `games.csv`, and `summary.json`: readable reports,
  replay, every seed's result, and aggregate statistics.
- `game-*-*.jsonl` and `game-*-*-summary.json`: every board, move, tile spawn,
  native score, measured latency, and game outcome.
- `fixed-boards.json` and `fixed-board-decisions.jsonl`: the frozen timing corpus
  and all timed decisions.
- `protocol.json`, `model-artifacts.json`, and `environment-validation.json`:
  model revisions, parameter-identity checks, model artifact hashes, runtime,
  hardware, and measurement settings.
- `source/`: the source files identified by the frozen protocol's SHA-256 hashes.
  The baseline snapshots were recovered from its recorded Git commit and saved
  runner, and checked against those original hashes.
- `verification.json`: the result of verifying the published evidence.

The shared-prefix run also includes `cache-validation.json`, recording the
unchanged saved prefix on all 24 warm-up boards. Model weights are downloaded
separately and are not part of these results. The original `.cache/...` names in
`previous-comparison.json` identify the local runs; the folders above are their
published copies.

## Run a new comparison

Follow the [benchmark setup and commands](../../benchmarks/semif_2048/README.md).
Use `--openjev-execution shared-prefix` for the new path or
`--openjev-execution current` for the original full-prompt batch. Always write to
a new output directory. Both use full prompts and full-vocabulary scoring by default.

The benchmark uses SemIf's pinned MLX-LM 0.32.0 commit, outside OpenJev's normal
MLX dependency range. This is documented in both reports. Shared-prefix scoring
is an experiment-only path for this single `Choice` question; production defaults
are unchanged. Candidate rows score independently; they do not attend to each other.
