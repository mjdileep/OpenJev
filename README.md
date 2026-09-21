# OpenJev

Turn text and images into typed decisions using a small local model.

OpenJev scores each possible answer using the model's probability of the token
`yes`. Multiple questions share one content cache, then their candidates are scored
in batches. A single question goes straight to scoring, without a separate cache
prefill. No generated text, JSON parsing, or API key is needed.

The local default is **Qwen3.5-0.8B**, with 4-bit weights for MLX and GGUF.
The Colab notebook also uses **Qwen3.5-0.8B with 4-bit weights**. You can swap in
another supported Hugging Face model.

## Try it in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjdileep/OpenJev/blob/main/notebooks/OpenJev_Quickstart.ipynb)

Choose a GPU under **Runtime → Change runtime type**, then **Run all**. The
notebook installs everything and includes text decisions, an image example,
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
openjev run examples/triage.json --backend mlx
```

**NVIDIA GPU with CUDA:**

Install a [CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/), then:

```bash
pip install -e '.[cuda]'
openjev run examples/triage.json --backend transformers --device cuda
```

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

with DecisionEngine.from_pretrained() as engine:
    result = engine.decide(state, questions)
    print(result.answers["team"]["choice"])
    print(result.answers["team"]["probabilities"])
    print(result.answers["urgent"]["noul"])
```

The automatic backend uses MLX on Apple Silicon and Transformers elsewhere.
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

**NVIDIA GPU — switch to a larger model with 4-bit weights:**

Install `pip install -e '.[cuda,cuda-4bit]'` first, alongside CUDA-enabled PyTorch.

```python
from openjev import DecisionEngine

with DecisionEngine.from_pretrained(
    "Qwen/Qwen3.5-4B",  # Use Qwen/Qwen3.5-0.8B to keep the smaller model.
    backend="transformers",
    device="cuda",
    load_in_4bit=True,
) as engine:
    result = engine.decide(state, questions)
    print(result.answers["team"]["choice"])
```

Change the repository name to another instruction model supported by your backend.
MLX needs an MLX-converted model; Transformers uses the original Hugging Face model.
For CPU, use the smaller model with `device="cpu"` and `load_in_4bit=False`.

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

with DecisionEngine.from_pretrained(vision=True) as engine:
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

To choose a different Qwen3.5 vision model, add `vision=True` to its MLX or CUDA
configuration above. Image scoring currently supports Qwen3.5 through those two
backends; GGUF is text-only.

## Check the cache benefit

```bash
openjev benchmark examples/triage.json --backend mlx
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
