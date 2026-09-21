# Backend setup

| Backend | Hardware | Format | Images | Default model |
|---|---|---|---|---|
| `mlx` | Apple Silicon | 4-bit MLX safetensors | Optional MLX-VLM | `mlx-community/Qwen3.5-0.8B-4bit` |
| `gguf` | CUDA, Metal, CPU | GGUF | No | `unsloth/Qwen3.5-0.8B-GGUF` |
| `transformers` | CUDA, CPU | HF safetensors / optional NF4 | Yes | `Qwen/Qwen3.5-0.8B` |

As checked on 2026-09-21, Qwen3.5-0.8B was the newest sub-billion Qwen instruction
model found. Qwen3-0.6B and Qwen2.5-0.5B are smaller, older alternatives. A model
must be supported by the selected inference library. Remote model code is disabled.

## Automatic device selection and CPU

`backend="auto"` and `device="auto"` are the defaults. OpenJev selects MLX on
Apple Silicon and Transformers elsewhere. Transformers uses CUDA when available,
otherwise CPU. Standard weights are the default on CPU and CUDA; no bitsandbytes
installation is needed.

```bash
pip install -e '.[transformers]'
openjev run examples/triage.json

# Force CPU, even on a machine with a GPU:
openjev run examples/triage.json --device cpu
```

On Apple Silicon, install `.[mlx,vision]` for automatic MLX selection. Install
`.[transformers]` as well if you want to force CPU on that Mac. The older
`.[cuda]` installation name remains an alias for `.[transformers]`.

In Python, `DecisionEngine.from_pretrained(device="cpu")` forces the CPU path.
An explicit `device="cuda"` still fails if CUDA is unavailable, and
`backend="mlx"` requires Apple Silicon; use the automatic backend to switch
between hardware types. Model files must match the selected backend.

## GGUF on CUDA

Requires the CUDA toolkit, compatible driver, and a C++ build toolchain:

```bash
CMAKE_ARGS='-DGGML_CUDA=ON' pip install --no-cache-dir --force-reinstall \
  --no-binary llama-cpp-python 'llama-cpp-python>=0.3.35,<0.4'
pip install -e '.[gguf]'
openjev run examples/triage.json --backend gguf --device cuda
```

## GGUF on Metal or CPU

For Metal, build with `CMAKE_ARGS='-DGGML_METAL=ON'` instead and select `--device metal`.
For CPU:

```bash
pip install -e '.[gguf]'
openjev run examples/triage.json --backend gguf --device cpu
```

The default GGUF is `Qwen3.5-0.8B-Q4_K_M.gguf`. An explicit `cuda` or `metal`
request fails if the installed library lacks that backend. `auto` permits a CPU fallback.

## Bonsai 2 27B on Apple Silicon

Use Prism's [MLX 2-bit pack](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit).
OpenJev recognizes its `prism_hadamard_qwen35` format and applies the required
Hadamard transforms, including the inverse embedding transform and the output
head. The loader is included in OpenJev; no remote Python is executed. It loads
only the language model, leaving the bundled vision tower unused. Bonsai image
scoring is not supported by this adapter yet.

```python
from openjev import DecisionEngine, Choice

with DecisionEngine.from_pretrained(
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
    revision="3f926b415992eaa2ae9dd7b573706494d6bbf787",
    backend="mlx",
    batch_size=1,
) as engine:
    result = engine.decide(
        "The apple is red.",
        {"color": Choice("What color is the apple?", {"red": "Red", "blue": "Blue"})},
    )
    print(result.answers["color"]["choice"])
```

Install the usual `.[mlx]` dependencies. The complete download is approximately
8.60 GB; the loaded language weights use about 7.68 GB. Inference needs additional
memory. This revision was tested on an M3 Pro with 18 GiB unified memory. Begin
with `batch_size=1` on that machine; larger batches use more memory. The default
OpenJev model remains Qwen3.5-0.8B. Thinking is disabled: the published chat
template emits an empty, closed `<think>` block before OpenJev reads verdict
logits. No reasoning or answer tokens are generated.

On one recorded four-action game board, batch sizes 1, 2, and 4 took **17.13 s**,
**16.28 s**, and **19.33 s** respectively. Peak MLX allocation was **9.87 GB**,
**10.93 GB**, and **12.68 GB**. These are exploratory single measurements, not
repeated benchmarks. Disabling thinking does not remove prompt-processing cost.
The loader's logits matched Prism's reference loader within 0.000021 across two
fixed prompts; cache, tokenizer, and binary-head tests also passed.
[Raw validation results](bonsai-validation.json).

The [GGUF version](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
uses custom PQ2_0/PTQ1_0 packing and needs Prism's llama.cpp fork. It cannot be
substituted into the stock `llama-cpp-python` installation above. The MLX path
is the Bonsai integration validated here. Weights retain Prism's Apache-2.0
license; adapted runtime code carries its [MIT notice](../src/openjev/backends/bonsai-LICENSE.txt).

## CUDA 4-bit safetensors

After installing a CUDA-enabled PyTorch build:

```bash
pip install -e '.[transformers,cuda-4bit]'
openjev run examples/triage.json --backend transformers --device cuda --load-in-4bit
openjev run examples/image.json --backend transformers --device cuda --load-in-4bit \
  --image examples/images/red-square.png
```

This uses bitsandbytes NF4. Qwen3.5-0.8B with NF4 weights was
validated for batched text and image scoring and the single-question path on a
Tesla T4. Earlier runs also used default weights; see [validation](validation.md)
for the scope of each run.
Other GPU/model combinations require their own validation.
The Colab notebook enables NF4 only when CUDA is available and `USE_4BIT=True`;
on CPU it automatically uses standard weights. In the library API,
`load_in_4bit=True` explicitly requests CUDA quantization and requires CUDA.

## Custom GGUF

The tokenizer must match the original model's chat template:

```bash
openjev run examples/triage.json --backend gguf \
  --model YOUR_ORG/YOUR_GGUF_REPO --filename YOUR_FILE.gguf \
  --tokenizer ORIGINAL_ORG/ORIGINAL_MODEL
```

`--model /path/to/model.gguf` accepts a local file. Use `--revision` and
`--tokenizer-revision` for independently pinned model/tokenizer commits.

The same setup in Python, with a complete example:

```python
from openjev import DecisionEngine, Noul

with DecisionEngine.from_pretrained(
    "unsloth/Qwen3.5-0.8B-GGUF",
    backend="gguf",
    filename="Qwen3.5-0.8B-Q4_K_M.gguf",
    tokenizer="Qwen/Qwen3.5-0.8B",
    device="cpu",
) as engine:
    result = engine.decide(
        "Please help me today!",
        {"urgent": Noul("Does the message convey urgency?")},
    )
    print(result.answers["urgent"]["noul"])
```

To change the model, change the repository, filename, and matching tokenizer
together. For a local GGUF, pass its path as the first argument and omit
`filename`. Quantization is part of the GGUF file or MLX weights;
`load_in_4bit=True` is only for CUDA Transformers loading.

For Transformers and MLX, the first argument can also be a local model directory
in the backend's format. In Python, use `revision="COMMIT_SHA"` to pin a Hugging
Face model and `tokenizer_revision="COMMIT_SHA"` to pin a separate tokenizer.

## Runtime controls

- `--device auto`: select an available device automatically (the default).
- `--device cpu`: force CPU execution through the automatic or Transformers backend.
- `--n-ctx 8192`: context budget, including image tokens; excess input is rejected.
- `--batch-size 8`: maximum candidates per batch on MLX and Transformers. Lower
  it to reduce branch-cache and activation memory. GGUF remains sequential.
- `--cache-strategy shared`: one content prefix, candidates batched across questions.
- `--cache-strategy tree`: also cache each question prefix; batch within questions.
  A single question skips separate prefills with either strategy.
- `--prefill-chunk-size 128`: text prefix processing chunk size. MLX's explicit
  value `1` also enables token-by-token singleton scoring for diagnostics.
- `--score-mode full`: exact full-vocabulary positive-token probability.
- `--score-mode binary`: conditional probability over the two verdict tokens.
- `--no-head-optimization`: reference output projection.
- `--no-cache`: independently evaluate complete prompts with singleton batches.

Thinking is disabled when the model's chat template supports `enable_thinking=False`.
Models that ignore this may score poorly at the first answer position. Verdict
markers default to `yes` and `no` and must each be one token; this is checked at the
actual prompt boundary. Python callers can override `positive_token` and
`negative_token` when a model needs different single-token labels.

Set `HF_HOME="$PWD/.cache/huggingface"` for a project-local cache, or use Hugging
Face's default. Set `HF_HUB_OFFLINE=1` after download for offline execution. Weights
are excluded from Git. Different weights retain their original licenses.
