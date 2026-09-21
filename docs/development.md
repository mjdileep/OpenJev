# Development

## Validate and benchmark

```bash
pip install -e '.[dev]'
pytest -m 'not integration'
ruff check .
python -m build

# Download the chosen default model first, or allow its first test to download it.
OPENJEV_TEST_BACKEND=mlx pytest -m integration
OPENJEV_TEST_BACKEND=gguf OPENJEV_TEST_DEVICE=cpu pytest -m integration
OPENJEV_TEST_BACKEND=transformers OPENJEV_TEST_DEVICE=cuda pytest -m integration

openjev benchmark examples/triage.json --backend mlx --iterations 5
openjev benchmark examples/image.json --backend mlx --image /path/to/picture.jpg
```

Benchmarks warm both paths, alternate their execution order, exclude model load
and image file preprocessing, and report median latency, evaluated/reused token counts, and the maximum
candidate-score difference. They measure this implementation's cache behavior;
they do not compare against Jev or establish classification accuracy.

Quantized prefill, singleton decode, and batched decode may select different
numerical kernels. The small model can show several percentage points of score
variation between execution shapes. Strict cache tests use
`--prefill-chunk-size 1 --batch-size 1` to keep arithmetic consistent; these are
verification settings, not recommended throughput settings. Always benchmark
your intended model, quantization, hardware, and candidate count.

The engine serializes calls to one loaded model. Caches are ephemeral and are
not persisted to disk or shared across requests. Inputs exceeding `--n-ctx` are
rejected rather than truncated. The default context budget is 8,192 tokens,
including expanded image tokens.

## Sources

- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [MLX conversion](https://huggingface.co/mlx-community/Qwen3.5-0.8B-4bit)
- [GGUF conversion](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)

The code is MIT licensed. Model weights and dependencies retain their respective
licenses; review the selected model's license when distributing a deployment.
