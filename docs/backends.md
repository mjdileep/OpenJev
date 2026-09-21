# Backend setup

| Backend | Hardware | Format | Images | Default model |
|---|---|---|---|---|
| `mlx` | Apple Silicon | 4-bit MLX safetensors | Optional MLX-VLM | `mlx-community/Qwen3.5-0.8B-4bit` |
| `gguf` | CUDA, Metal, CPU | GGUF | No | `unsloth/Qwen3.5-0.8B-GGUF` |
| `transformers` | CUDA, CPU | HF safetensors / optional NF4 | Yes | `Qwen/Qwen3.5-0.8B` |

As checked on 2026-09-21, Qwen3.5-0.8B was the newest sub-billion Qwen instruction
model found. Qwen3-0.6B and Qwen2.5-0.5B are smaller, older alternatives. A model
must be supported by the selected inference library. Remote model code is disabled.

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

## CUDA 4-bit safetensors

After installing a CUDA-enabled PyTorch build:

```bash
pip install -e '.[cuda,cuda-4bit]'
openjev run examples/triage.json --backend transformers --device cuda --load-in-4bit
openjev run examples/image.json --backend transformers --device cuda --load-in-4bit \
  --image examples/images/red-square.png
```

This uses bitsandbytes NF4. The Colab default is Qwen3.5-4B with NF4 weights,
validated for text and images on a Tesla T4. The earlier 0.8B configuration also
passed with both default and NF4 weights; see [validation](validation.md).
Other GPU/model combinations require their own validation.

## Custom GGUF

The tokenizer must match the original model's chat template:

```bash
openjev run examples/triage.json --backend gguf \
  --model YOUR_ORG/YOUR_GGUF_REPO --filename YOUR_FILE.gguf \
  --tokenizer ORIGINAL_ORG/ORIGINAL_MODEL
```

`--model /path/to/model.gguf` accepts a local file. Use `--revision` and
`--tokenizer-revision` for independently pinned model/tokenizer commits.

### Ternary Bonsai 2 27B

Use the [Bonsai Colab notebook](https://colab.research.google.com/github/mjdileep/OpenJev/blob/main/notebooks/OpenJev_Bonsai.ipynb)
for `prism-ml/Ternary-Bonsai-2-27B-gguf` and
`Ternary-Bonsai-2-27B-PTQ1_0.gguf`. This format needs
[Prism's runtime](https://github.com/PrismML-Eng/llama.cpp/tree/9a9394a895b96003ca842a6041cb28ac49a108f7);
installing stock `llama-cpp-python` is insufficient.

The notebook downloads the pinned CUDA 12.4 release, verifies its checksum, and
uses an isolated copy of llama-cpp-python 0.3.35 with the two struct fields added
by that Prism release. The [setup helper](../examples/bonsai_runtime.py) does not
modify system packages. Do not change the runtime or wrapper revision without
rechecking their native ABI. Start with a fresh Colab session.

The GGUF and tokenizer revisions are pinned separately. The matching tokenizer
comes from `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`; this does not load MLX weights.
The example uses a 2,048-token context and full GPU offload. OpenJev currently
supports text only through this backend, although Bonsai has a separate vision
projector. See [validation status](validation.md#bonsai-2-27b-gguf).

## Runtime controls

- `--n-ctx 8192`: context budget, including image tokens; excess input is rejected.
- `--batch-size 8`: equal-length candidate batching on MLX. Other backends are sequential.
- `--prefill-chunk-size 128`: prefix processing chunk size.
- `--score-mode full`: exact full-vocabulary positive-token probability.
- `--score-mode binary`: conditional probability over the two verdict tokens.
- `--no-head-optimization`: reference MLX output projection.
- `--no-cache`: independently evaluate complete prompts.

Thinking is disabled when the model's chat template supports `enable_thinking=False`.
Models that ignore this may score poorly at the first answer position. Verdict
markers default to `yes` and `no` and must each be one token; this is checked at the
actual prompt boundary. Python callers can override `positive_token` and
`negative_token` when a model needs different single-token labels.

Set `HF_HOME="$PWD/.cache/huggingface"` for a project-local cache, or use Hugging
Face's default. Set `HF_HUB_OFFLINE=1` after download for offline execution. Weights
are excluded from Git. Different weights retain their original licenses.
