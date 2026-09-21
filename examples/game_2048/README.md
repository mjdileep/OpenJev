# Watch OpenJev play 2048

A live local demo using Gabriele Cirulli's original, MIT-licensed 2048 game logic.
Qwen3.5-0.8B chooses each swipe through OpenJev. The page shows the board, action
scores, decision time, batch size, and generated token count.

From the OpenJev repository, after the normal installation:

```bash
source .venv/bin/activate
python examples/game_2048/server.py
```

Open **http://127.0.0.1:8765**, wait for **Ready**, then press **Play**.
Use **Pause**, **One move**, and **New game** while recording. **Export run**
downloads the actual states, decisions, scores, and timings as JSON.
Stop the server with Ctrl+C when finished.

The default is Qwen3.5-0.8B: MLX on Apple Silicon, otherwise Transformers on
CUDA or CPU. Install `pip install -e '.[mlx]'` on Apple Silicon, or
`pip install -e '.[transformers]'` elsewhere. No additional web dependencies,
API keys, or internet connection are needed after the model is cached.

Choose another model supported by your backend:

```bash
python examples/game_2048/server.py --model mlx-community/Qwen3.5-0.8B-4bit --backend mlx
python examples/game_2048/server.py --model Qwen/Qwen3.5-0.8B --backend transformers --device cpu
```

Use `--revision COMMIT` to pin weights, `--batch-size 2` to reduce batch memory,
or `--port 8766` if the default port is occupied.

## What the model sees

The browser sends a 4×4 board to `POST /api/decide`. The adapter removes swipes
that would leave it unchanged and describes the resulting board, merge points,
and empty cells for each remaining swipe, before the random tile appears.
These are exact one-move rule calculations. There is no search tree, heuristic
ranking, fallback player, or generated reasoning. The question includes basic
2048 strategy; the model's highest OpenJev score selects the move.

All legal candidates belong to one `Choice` question, so OpenJev evaluates their
complete prompts in a single batch by default. It does not build a separate
shared-prefix cache for this single-question request. With only one legal move,
it evaluates one `Noul` question and takes that forced move.

The game then applies the move and spawns a random tile using the original
engine. Tile placement is random, so results vary. This is a demonstration of
local decisions, not a claim that 0.8B can solve 2048 reliably. Action percentages
are normalized independent `yes` support, not calibrated success probabilities.

Decision latency is OpenJev's measured call time: prompt preparation and model
scoring. It excludes initial loading, HTTP transport, rendering, and the selected
pause between moves. The first move includes first-use overhead. Peak MLX memory
is the process allocator peak, not the Mac's total memory use. Server-side
decision records are also written under `.cache/game_2048/`.

## First Mac run

On an Apple M3 Pro with 18 GiB unified memory, the default 4-bit MLX model
completed **162 moves**, scored **1,584**, and reached a **128 tile** before losing.
Mean decision time was **350 ms**, median **380 ms**, and 95th percentile **396 ms**;
the first decision took **513 ms**. Peak MLX allocation was **1.83 GB** and no tokens
were generated. This is one unseeded game, not a statistical playing-strength benchmark.
[Recorded states, actions, model revision, and timings](../../docs/game-2048-run.json).

## Attribution

`vendor/game_manager.js`, `vendor/grid.js`, and `vendor/tile.js` are unmodified
from [gabrielecirulli/2048](https://github.com/gabrielecirulli/2048), revision
`478b6ec346e3787f589e4af751378d06ded4cbbc`.
[Original MIT license](vendor/LICENSE.txt). OpenJev supplies the UI and model adapter.
