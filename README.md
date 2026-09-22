# OpenJev

Turn text and images into typed decisions using a small local model.

OpenJev scores each possible answer using the model's probability of the token
`yes`. Multiple questions share one content cache, then their candidates are scored
in batches. A single question goes straight to scoring, without a separate cache
prefill. No generated text, JSON parsing, or API key is needed.

The local default is **Qwen3.5-0.8B**, with 4-bit weights for MLX and GGUF.
Device selection is automatic: **MLX on Apple Silicon, CUDA when available,
otherwise CPU**. You can swap in another supported Hugging Face model.

## Try it in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjdileep/OpenJev/blob/main/notebooks/OpenJev_Quickstart.ipynb)

Choose **Run all**. The notebook defaults to Qwen3.5-0.8B. A GPU is optional;
select one under **Runtime → Change runtime type** for faster inference.
The notebook installs everything and includes text decisions, an image example,
and a cache benchmark. No API key is needed.
See the [validation results](docs/validation.md), including the current Colab T4 run.

## Get started

Requires Python 3.11+; Python 3.12 is recommended.

```bash
git clone https://github.com/mjdileep/OpenJev.git
cd OpenJev
python3.12 -m venv .venv
source .venv/bin/activate
```

**Mac with Apple Silicon:**

```bash
pip install -e '.[mlx,vision]'
openjev run examples/triage.json
```

**CPU or NVIDIA GPU:**

```bash
pip install -e '.[transformers]'
openjev run examples/triage.json
```

No GPU is required. If your PyTorch installation supports CUDA and a GPU is
available, OpenJev uses it automatically; otherwise it uses CPU. For CUDA setup,
see the [PyTorch installer](https://pytorch.org/get-started/locally/).
The model downloads on first use. Later runs use the local cache.

## Python example

```python
from openjev import Choice, DecisionEngine, Noul

state = "Help! My payouts have been failing for three days."
questions = {
    "urgent": Noul("Does the message convey urgency?"),
    "team": Choice(
        "Which team should handle this?",
        {
            "billing": "Payments, invoices, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing and upgrades",
        },
    ),
}

with DecisionEngine.from_pretrained(device="auto") as engine:
    result = engine.decide(state, questions)
    print(result.answers["team"]["choice"])
    print(result.answers["team"]["probabilities"])
    print(result.answers["urgent"]["noul"])
```

`device="auto"` is already the default, so you can omit it. To force CPU, install
the CPU/NVIDIA dependencies above and set `device="cpu"`.
`Noul` asks a yes/no question; `Choice` picks from your options. Import `Score`
from `openjev` for ratings such as
`Score("How frustrated?", ["Calm", "Frustrated", "Very angry"])`.
That returns an average from 0 to 2 and the full distribution.

## Choose a model in Python

Run the example above first. Then pick **one** of the blocks below to use the
same `state` and `questions` with a model you choose. The repository name is the
first argument to `from_pretrained()`.

**Apple Silicon — a model converted for MLX:**

```python
from openjev import DecisionEngine

with DecisionEngine.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-4bit",  # Replace with another supported MLX repo.
    backend="mlx",
) as engine:
    result = engine.decide(state, questions)
    print(result.answers["team"]["choice"])
```

**Hugging Face model — automatically use CUDA or CPU:**

```python
from openjev import DecisionEngine

with DecisionEngine.from_pretrained(
    "Qwen/Qwen3.5-0.8B",  # Change to Qwen/Qwen3.5-4B to try a larger model.
    backend="transformers",
    device="auto",
) as engine:
    result = engine.decide(state, questions)
    print(result.answers["team"]["choice"])
```

Change the repository name to another instruction model supported by your backend.
MLX needs an MLX-converted model; Transformers uses the original Hugging Face model.
Standard weights work on CPU and CUDA; 4-bit CUDA loading is an optional setting
covered in [backend setup](docs/backends.md#cuda-4-bit-safetensors).

Keep the model loaded while calling `engine.decide(...)` for more inputs. The
`with` block releases it when finished. Caching and batching work automatically;
add `batch_size=4` to the model settings if GPU memory is tight.
See [backend setup](docs/backends.md) for a GGUF Python example, local model files,
version pinning, and other settings.

## Images in Python

Load with `vision=True` and pass local image paths to `decide()`. This example
uses the default 0.8B model and selects MLX on Apple Silicon or Transformers elsewhere:

```python
from openjev import Choice, DecisionEngine

with DecisionEngine.from_pretrained(device="auto", vision=True) as engine:
    result = engine.decide(
        "Inspect the supplied image.",
        {
            "color": Choice(
                "Which color covers most of the image?",
                {"red": "Red", "blue": "Blue", "green": "Green"},
            ),
        },
        images=["examples/images/red-square.png"],  # Replace with your image path.
    )
    print(result.answers["color"]["choice"])
```

To choose a different Qwen3.5 vision model, add `vision=True` to its MLX or Transformers
configuration above. Image scoring currently supports Qwen3.5 through those two
backends; GGUF is text-only.

## Watch it play 2048

After installation, run:

```bash
python examples/game_2048/server.py
```

Open **http://127.0.0.1:8765** and click **Play**. The local 0.8B model chooses
each move, with live action scores and decision timings. You can pause, step,
record the screen, or export a run. [Game demo details](examples/game_2048/README.md).

### Also try Laya

[Laya](https://huggingface.co/convaiinnovations/laya) is a 421M decision model.
This separate example uses Laya's native choice head to score all legal moves in
one forward pass. The OpenJev example above uses Qwen's next-token `yes` scores.

From this repository, with your virtual environment active:

```bash
pip install -e '.[laya]'
python examples/laya_2048/server.py
```

Open **http://127.0.0.1:8766** and click **Play**. The pinned model downloads on
first use. Device selection is automatic: CUDA, Apple Silicon MPS, then CPU.
Add `--device cpu` to run explicitly on CPU.

One recorded M3 Pro game reached a **256 tile**, scored **2,792** over **238 moves**,
and averaged **119 ms per decision**, with zero generated tokens. It did not reach
2048. This is one unseeded game, not a controlled comparison with Qwen.
[Laya setup, Python example, and recorded results](examples/laya_2048/README.md).

## Check the cache benefit

```bash
openjev benchmark examples/triage.json
```

This compares shared caching plus batching with independent full-prompt scoring.
It reports latency, actual batch sizes, padding, reused tokens, and score differences.
[Local validation results](docs/validation.md) include a seven-candidate MLX run
at approximately **145 ms batched vs 309 ms independent**.
Results depend on the model, hardware, and input.

Scores are **not calibrated probabilities of correctness**. Choice distributions
are normalized independent candidate scores; the raw scores are also returned.
This project is an independent experiment, not an official Jev implementation.

[How scoring and trimming work](docs/architecture.md) ·
[Development and tests](docs/development.md) · [MIT license](LICENSE)

## OpenJev vs SemIf: 2048 results

Tested on **Apple M3 Pro, 18 GiB**, using identical **Qwen3.5-0.8B 4-bit** weights
and the same MLX runtime. Each method played 10 games with seeds 42–51.
Thinking was disabled; neither method generated answer tokens.

OpenJev used the experimental **shared-prefix** path: compute instructions +
context once, then score all legal-move candidates in one batch from independent
cache copies. It keeps the full prompt and full-vocabulary `P(yes)` scoring.
This experiment is enabled in the benchmark; the normal API defaults are unchanged.

| Metric | OpenJev | SemIf |
| --- | ---: | ---: |
| Average score | **3,402.4** | 1,816.8 |
| Median score | **3,024** | 1,522 |
| Highest tile | **512** | 256 |
| Median decision latency | 226.8 ms | **162.9 ms** |
| Games reaching 2048 | 0/10 | 0/10 |

OpenJev scored higher on **8/10 seeds**. Latency was measured on **24 identical
boards, three repetitions each**, including cache creation and copying, excluding
model loading and warm-up. All **4,561 game moves and 144 timed decisions** were
verified. Ten seeds are a small gameplay benchmark, not general decision-accuracy evidence.

[Full report](reports/2048/2026-09-22-shared-prefix/report.md) ·
[Per-game scores](reports/2048/2026-09-22-shared-prefix/games.csv) ·
[Recorded replay and raw evidence](reports/2048/README.md) ·
[Run the benchmark](benchmarks/semif_2048/README.md)
