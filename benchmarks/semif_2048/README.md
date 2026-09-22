# OpenJev versus SemIf on 2048

Compare the two decision methods with **the same Qwen3.5-0.8B 4-bit weights**
on Apple Silicon. This benchmark uses upstream SemIf's native MLX scorer.
It does not translate SemIf's method into OpenJev or generate answer text.

[Published results, raw evidence, and offline replays](../../reports/2048/README.md)
feature a fresh adaptive-cache run with batched tokenization and faster MLX cache
branching. Earlier runs remain available as historical evidence.

## Setup

Run from the OpenJev repository root. Python 3.12 and Git are required.
Use the separate environment below: SemIf pins a newer MLX-LM revision containing
a Qwen3.5 normalization fix. Both methods use that revision for this comparison.
Your normal OpenJev environment is not changed.

```bash
git clone https://github.com/TheoLeeCJ/SemIf.git .cache/semif-benchmark/upstream
git -C .cache/semif-benchmark/upstream checkout 1f2dea3e25379f9dfc98cb83c324f00ab5deda37
python3.12 -m venv .cache/semif-benchmark/venv
.cache/semif-benchmark/venv/bin/pip install \
  -c benchmarks/semif_2048/constraints.txt \
  -e '.cache/semif-benchmark/upstream[test,mlx]'
```

The runner imports this checkout of OpenJev directly. Its current MLX extra
specifies `<0.32`; this controlled comparison explicitly runs the same source
on SemIf's patched MLX-LM **0.32.0** commit
`a63e24c389382619eb6d9af656e3b46024be217a`. It is not a timing comparison between
each project's different default dependency environments.

## Run

```bash
HF_HOME="$PWD/.cache/huggingface" \
  .cache/semif-benchmark/venv/bin/python benchmarks/semif_2048/run.py \
  --openjev-execution adaptive \
  --output .cache/semif-benchmark/runs/my-comparison
```

The model downloads on first use. Add `HF_HUB_OFFLINE=1` after it is cached.
Use a **new output directory** for each run. The runner preserves partial
evidence if interrupted and never overwrites an earlier run. Avoid running
other GPU workloads during the comparison.

The featured `adaptive` path computes generic instructions through `State:` once
at model initialization. For each board, the planner caches the complete shared
board/question prefix and scores legal-move suffixes together from independent
copies of attention KV and recurrent state. The complete candidate prompts and
full-vocabulary `P(yes)` readout are unchanged. Warmup verifies exact prompt
reconstruction and retained cache immutability. The permanent cache is checked
again after every game has finished.

Per-request prefill, planning and copying are included in latency. The one-time
instruction-cache cost is recorded separately in `initialization.json`, outside
request timings. The normal API exposes this path with
`DecisionEngine.from_pretrained(cache_strategy="adaptive")`; its default remains
`shared`. For mixed Noul/Choice/Score timing, see the
[mixed-question benchmark](../adaptive_cache/README.md#mixed-question-types).

To prefill instructions + user context once, then score all legal moves together
from independent cache copies, use the experimental shared-prefix path:

```bash
HF_HOME="$PWD/.cache/huggingface" \
  .cache/semif-benchmark/venv/bin/python benchmarks/semif_2048/run.py \
  --openjev-execution shared-prefix \
  --output .cache/semif-benchmark/runs/my-shared-prefix-comparison
```

This retains the full prompt, full-vocabulary `P(yes)`, and all layers. Both the
attention KV cache and recurrent state are copied for each candidate. All legal
moves are scored in one batch; the retained prefix is checked for mutation on
every warm-up board. Prefix computation and cache copying are included in timing.
The cache is rebuilt for each new board. Production defaults remain unchanged.
Batching is only a compute operation: each row gets its own `P(yes)` without
attending to other rows. This game supplies one `Choice` question per board, so
only that question's legal-move scores are normalized together. It does not
benchmark a batch containing several unrelated questions.

For the short-instruction, existing yes/no head variant, keep all 24 layers and run:

```bash
HF_HOME="$PWD/.cache/huggingface" \
  .cache/semif-benchmark/venv/bin/python benchmarks/semif_2048/run.py \
  --prompt-style short --score-mode binary \
  --output .cache/semif-benchmark/runs/my-short-comparison
```

This changes only OpenJev's prompt and score normalization. SemIf, the game inputs,
model weights, random seeds, and board corpus follow the same protocol. The shorter
prompt retains the complete game question and each move's preview. It adapts the
earlier state/interpretation pilot to typed questions; it is not the literal pilot
prompt with the game question removed. No weights are trained or layers skipped.

Defaults are 10 games per method (seeds 42–51), 24 fixed boards, three timing
repetitions, and a 2,000-move safety cap. A two-board smoke check is:

```bash
HF_HOME="$PWD/.cache/huggingface" \
  .cache/semif-benchmark/venv/bin/python benchmarks/semif_2048/run.py \
  --games 0 --fixed-boards 2 --repeats 1 \
  --output .cache/semif-benchmark/runs/my-smoke-check
```

## What is controlled

- **Weights:** `mlx-community/Qwen3.5-0.8B-4bit`, revision
  `da28692b5f139cb0ec58a356b437486b7dac7462`. Both complete loaded parameter trees
  must compare equal before the benchmark starts. Source artifact hashes are saved.
- **Runtime:** the same MLX, MLX-LM, tokenizer, hardware, and 256 MiB inactive
  allocation-cache limit. Model loading and initial warm-up are excluded from timing.
- **Inputs:** the same current board, exact legal-move previews, merge points,
  empty-cell counts, and strategy instructions. Option order is always
  up, right, down, left, with illegal moves removed. Instructions are frozen
  before the run and are not tuned after observing results.
- **Gameplay:** the tested game rules from the original 2048 example. Each paired
  seed starts from the same two tiles and resets the same random-number stream.
  Tile values and positions are sampled in the original game's order. Once
  actions differ, the board trajectories and available spawn positions differ.
- **Execution:** GPU requests run sequentially. The method playing first alternates
  across seeds. A sole legal move is applied directly for both methods, marked
  forced, and excluded from inference timing.

## What differs

OpenJev scores independent `yes` support for each legal move, then normalizes
those scores. By default it uses the full prompt and full-vocabulary scoring mode.
`--prompt-style short --score-mode binary` uses a user-only short judging prompt
and support normalized over the existing `yes` and `no` logits. MLX projects only
those vocabulary rows in binary mode. Both configurations use at most four
candidate rows per batch. The default `--openjev-execution current` uses the
single-question path without prefix prefill; `shared-prefix` enables per-board
context caching, and `adaptive` uses permanent instructions and prefix planning.

SemIf receives one choice question listing all legal moves. Its unmodified
`mlx_backend.score()` reads the logits for the declared answer-letter tokens
and normalizes across them. This is its direct mode: every new board is a new
state, and there is only one question per board.

The semantic information is the same; rendered prompts, token counts, readouts,
and execution paths differ. Those differences are the methods being compared.
Neither generates text, uses reasoning generation, or applies temperature calibration.
This task does not test either system's shared-state, multi-question caching.

## Measurements

**Identical-board latency:** 24 boards come from an independent uniform-random
policy. The protocol and corpus are saved before loading either model. Both
methods warm every board once, then score each board three times with alternating
method order. Timings include prompt rendering, tokenization, inference, output
readout, and GPU synchronization. They exclude common game-preview construction,
evidence hashing, logging, model loading, HTTP, and rendering.

**Playing outcomes:** report each seed's score, largest tile, move count, whether
2048 was reached, and any cap or error. Game latency is reported separately because
the methods encounter different boards. The aggregate includes a paired score
difference and a deterministic bootstrap interval across seeds. Ten seeds are
still a small exploratory sample; this is not general decision-accuracy evidence.

Outputs include `protocol.json`, `model-artifacts.json`, `fixed-boards.json`,
every timed fixed-board decision, every game transition, per-game summaries, and
`summary.json`. Native scores are retained. Choice agreement on fixed boards
measures agreement, not correctness; neither distribution is a win probability.
The protocol records prompt style and scoring mode. Modified source files are
hashed and copied into `source/`; the base Git commit alone does not identify
uncommitted changes.

## View and verify the results

After the run finishes, check its recorded evidence and build the standalone
report. These commands do not load a model or use the GPU:

```bash
.cache/semif-benchmark/venv/bin/python benchmarks/semif_2048/verify.py \
  .cache/semif-benchmark/runs/my-comparison
.cache/semif-benchmark/venv/bin/python benchmarks/semif_2048/report.py \
  .cache/semif-benchmark/runs/my-comparison
open .cache/semif-benchmark/runs/my-comparison/report.html
```

The verifier replays every move and seeded tile spawn, checks native decisions
and scores, and recomputes the aggregate. The report shows all paired outcomes
and lets you replay either seed side by side, pause, step, or scrub through moves.
It is a **recorded replay**, not a live inference demonstration. Playback aligns
move numbers rather than elapsed time. The report requires at least one paired
game; the verifier also accepts the zero-game smoke check.
It also writes `report.md` for a short text summary and `games.csv` for the
per-game scores. Existing reports are never overwritten.

To compare a completed short-prompt rerun with the earlier full-prompt run:

```bash
.cache/semif-benchmark/venv/bin/python benchmarks/semif_2048/compare_runs.py \
  .cache/semif-benchmark/runs/my-comparison \
  .cache/semif-benchmark/runs/my-short-comparison
```

This verifies both runs, checks that the model and game protocol match, compares
each paired seed, and reports whether SemIf reproduced its earlier actions and
native choice distributions. The output is `previous-comparison.md` in the newer
run directory. Historical timing comparisons are labeled as separate runs.

## Sources

- [SemIf](https://github.com/TheoLeeCJ/SemIf/tree/1f2dea3e25379f9dfc98cb83c324f00ab5deda37)
- [SemIf MLX methodology](https://github.com/TheoLeeCJ/SemIf/blob/1f2dea3e25379f9dfc98cb83c324f00ab5deda37/docs/MLX.md)
- [MLX-LM normalization fix](https://github.com/ml-explore/mlx-lm/commit/a63e24c389382619eb6d9af656e3b46024be217a)
- [Original game attribution](../../examples/game_2048/README.md#attribution)
