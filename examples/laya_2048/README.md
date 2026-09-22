# Watch Laya play 2048

Run [Laya](https://huggingface.co/convaiinnovations/laya), a 421M decision model,
against the same original 2048 game used by the OpenJev demo. The page shows
the board, move scores, decision latency, and forward-pass count. This example
calls Laya's native choice head through its SDK; it does not use OpenJev's
next-token `yes` scoring or change the default Qwen model.

## Start the game

From the repository root, with your Python 3.11+ virtual environment active:

```bash
pip install -e '.[laya]'
python examples/laya_2048/server.py
```

Open **http://127.0.0.1:8766**, wait for **Ready**, and click **Play**.
Use **Pause**, **One move**, and **New game** while recording. **Export run**
downloads the boards, scores, full model inputs, and timings as JSON.
Stop the server with Ctrl+C.

The first run downloads approximately 845 MB. Later runs use the Hugging Face
cache. No API key is needed. The SDK automatically selects CUDA when available,
then Apple Silicon MPS, then CPU. This example was validated on MPS and CPU;
the CUDA path uses the SDK's device support.

```bash
# Force CPU, or choose a different port.
python examples/laya_2048/server.py --device cpu
python examples/laya_2048/server.py --port 8767

# Once downloaded, run without network access.
HF_HUB_OFFLINE=1 python examples/laya_2048/server.py
```

The SDK is pinned to **0.3.5**. The example downloads only the English root
checkpoint at revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
Model loading and a warm-up finish before the page reports Ready.

## Choose a move in Python

```bash
python examples/laya_2048/decide.py
```

The [script](decide.py) contains this small example. To use it from your own
script inside this example directory:

```python
from laya_player import Player

player = Player(device="auto")  # Or device="cpu", "mps", or "cuda".
result = player.decide(
    [
        [2, 2, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)
print(result["action"])
print(result["probabilities"])
```

Keep `player` loaded for subsequent boards. Rows run from top to bottom; zero
means an empty cell. The adapter accepts a 4×4 board and returns a legal swipe.

## How the experiment works

The model sees the current board and exact one-move previews for every legal
swipe, before random tile spawning. Each option includes its merge points and
empty-cell count. The question describes basic strategy: merge equal tiles,
preserve space, and keep large tiles together at an edge. Code applies game
rules but does not rank moves, search future boards, or substitute a fallback
player. Laya selects the highest-scoring legal action. A single legal action is
marked as forced.

Laya's bidirectional encoder scores all options in one question in one forward
pass. It does not generate reasoning or answer tokens and does not use an
autoregressive KV cache. Short option descriptions avoid the SDK's per-option
token limit; the full board previews live in the state. The adapter checks
that instructions, options, and state reach the model without truncation.

The displayed scores are the SDK's choice distribution. They have not been
validated as calibrated probabilities for 2048 or winning. Laya's escalation
head is retained in the raw output but does not control this game.

## Recorded Mac run

One unseeded game on an **Apple M3 Pro with 18 GiB unified memory**, using
PyTorch MPS and float32 weights:

| Measurement | Result |
| --- | --- |
| Moves | 238 |
| Score | 2,792 |
| Largest tile | 256; game ended without reaching 2048 |
| Mean decision time | 118.7 ms |
| Median / 95th percentile | 121.7 / 189.0 ms |
| MPS tensor allocation at the last move | 1.69 GB |
| Generated tokens | 0 |

[Recorded boards, actions, scores, model revision, and timings](../../docs/laya-2048-run.json).
Every transition was checked against the game rules, including exactly one
random tile spawn. This is one exploratory game with random tile placement,
not a controlled playing-strength or speed comparison with Qwen.

Decision timing includes input preparation, truncation checks, tokenization,
inference, and score conversion. It excludes initial loading, warm-up, HTTP
transport, browser rendering, and the pause between moves. MPS memory is the
current tensor allocation, not peak process memory. Fresh decision logs are
saved under `.cache/laya_2048/`.

## Attribution

The original MIT-licensed game engine, license, styles, and browser controller
are reused from [the OpenJev game example](../game_2048/README.md#attribution).
Laya's [weights](https://huggingface.co/convaiinnovations/laya) and
[SDK](https://github.com/NandhaKishorM/laya) retain their Apache-2.0 licenses.
