# OpenJev

Turn text and images into typed decisions using a small local model.

OpenJev scores each possible answer using the model's probability of the token
`yes`. It shares the content cache across questions and each question's cache across
answers. No generated text, JSON parsing, or API key is needed.

The local default is **Qwen3.5-0.8B**, with 4-bit weights for MLX and GGUF.
The Colab notebook uses **Qwen3.5-4B with 4-bit weights**. You can swap in another
supported Hugging Face model.

## Try it in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjdileep/OpenJev/blob/main/notebooks/OpenJev_Quickstart.ipynb)

Choose a GPU under **Runtime → Change runtime type**, then **Run all**. The
notebook installs everything and includes text decisions, an image example,
and a cache benchmark. No API key is needed.
See the [Colab T4 validation results](docs/validation.md#colab-cuda-notebook).

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

with DecisionEngine.from_pretrained() as engine:
    result = engine.decide(
        state="Help! My payouts have been failing for three days.",
        questions={
            "urgent": Noul("Does the message convey urgency?"),
            "team": Choice(
                "Which team should handle this?",
                {
                    "billing": "Payments, invoices, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing and upgrades",
                },
            ),
        },
    )
    print(result.answers["team"]["choice"])
    print(result.answers["team"]["probabilities"])
    print(result.answers["urgent"]["noul"])
```

The automatic backend uses MLX on Apple Silicon and Transformers elsewhere.
Use `device="cuda"` to require a GPU instead of permitting a CPU fallback.
`Score("How frustrated?", ["Calm", "Frustrated", "Very angry"])` adds an ordered
rating: its score is a weighted average from 0 to 2, with the full distribution
included. Questions can also be plain dictionaries; see [the triage request](examples/triage.json).

## Images

Try the included sample, then replace its path with your own image:

```bash
openjev run examples/image.json --backend mlx --image examples/images/red-square.png

# With the CUDA installation:
openjev run examples/image.json --backend transformers --device cuda \
  --image examples/images/red-square.png
```

In Python, load with `vision=True` and pass `images=["picture.jpg"]` to `decide()`.
Image scoring supports Qwen3.5 through MLX or Transformers. GGUF is
text-only in this version.

## Use another model

```bash
# MLX model repository or local model directory
openjev run examples/triage.json --backend mlx --model YOUR_ORG/YOUR_MLX_MODEL

# Original Hugging Face model on CUDA
openjev run examples/triage.json --backend transformers --device cuda \
  --model YOUR_ORG/YOUR_MODEL
```

Choose an instruction model supported by that backend. Use `--revision COMMIT_SHA`
to pin its version. See [backend setup](docs/backends.md) for GGUF, CPU, and CUDA
4-bit installation.

## Check the cache benefit

```bash
openjev benchmark examples/triage.json --backend mlx
```

This compares cached and uncached runs and reports latency, reused tokens, and
score differences. [Local validation results](docs/validation.md) include a
seven-candidate MLX run at approximately **185 ms cached vs 305 ms uncached**.
Results depend on the model, hardware, and input.

Scores are **not calibrated probabilities of correctness**. Choice distributions
are normalized independent candidate scores; the raw scores are also returned.
This project is an independent experiment, not an official Jev implementation.

[How scoring and trimming work](docs/architecture.md) ·
[Development and tests](docs/development.md) · [MIT license](LICENSE)
