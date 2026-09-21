"""Text-only support for Prism's schema-2 Hadamard MLX packs.

Packed transforms are adapted from the model's MIT-licensed runtime (see
bonsai-LICENSE.txt). Model repositories supply data, never executable Python.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def transform(x, block, signs, *, inverse=False):
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block must divide the activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    def __init__(self, arrays, block, signs, embedding):
        super().__init__()
        self.weight, self.scales, self.biases = arrays
        self.block, self.signs, self.embedding = block, signs, embedding

    def __call__(self, x):
        if self.embedding:
            shape = x.shape
            ids = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[ids], self.scales[ids], self.biases[ids], group_size=128, bits=2
                )
                .reshape(*shape, -1)
                .astype(mx.float16)
            )
            return transform(out, self.block, self.signs, inverse=True) if self.block else out
        return self.project(x)

    def project(self, x, ids=None):
        if self.block:
            x = transform(x, self.block, self.signs)
        arrays = (self.weight, self.scales, self.biases)
        if ids is not None:
            arrays = tuple(a[ids] for a in arrays)
        return mx.quantized_matmul(x, *arrays, transpose=True, group_size=128, bits=2)

    def project_selected(self, x, ids):
        # Binary scoring must apply the same transform as the full vocabulary head.
        return self.project(x, ids)


def load_bonsai(path):
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from transformers import Qwen2TokenizerFast

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("model_type") != "prism_hadamard_qwen35"
        or config.get("schema_version") != 2
        or config.get("base_model_type") != "qwen3_5"
        or config.get("tensor_namespace") != "mlx-vlm-qwen3_5"
        or config.get("gdn_activation_layout") != "grouped"
        or config.get("quantization") != {"bits": 2, "group_size": 128, "mode": "affine"}
        or config.get("text_config", {}).get("tie_word_embeddings", True)
    ):
        raise ValueError("Unsupported Bonsai pack: expected schema-2 grouped Qwen3.5 MLX weights")
    model = TextModel(TextModelArgs.from_dict(copy.deepcopy(config["text_config"])))
    # Keep only language tensors; do not evaluate or retain the bundled vision tower.
    weights = {
        k.removeprefix("language_model."): v
        for k, v in mx.load(str(path / "model.safetensors")).items()
        if k.startswith("language_model.")
    }
    seen = set()
    for record in config["modules"]:
        name = record["path"]
        if name in seen or record["dtype"] != "float16":
            raise ValueError("Duplicate or unsupported Bonsai packed module")
        seen.add(name)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        original = getattr(parent, parts[-1])
        if not isinstance(original, (nn.Linear, nn.Embedding)):
            raise ValueError("Unsupported Bonsai packed target")
        if record["embedding"] != isinstance(original, nn.Embedding):
            raise ValueError("Bonsai embedding kind mismatch")
        rows, width = original.weight.shape
        arrays = [weights[name + "." + suffix] for suffix in ("weight", "scales", "biases")]
        expected = [(rows, width // 16), (rows, width // 128), (rows, width // 128)]
        if width % 128 or [a.shape for a in arrays] != expected or arrays[0].dtype != mx.uint32:
            raise ValueError("Invalid Bonsai packed tensor shapes")
        if any(a.dtype != mx.float16 for a in arrays[1:]):
            raise ValueError("Bonsai affine parameters must use float16")
        block, signs = record["block"], weights.get(name + ".signs")
        if block:
            if block not in (512, 1024, 2048, 4096) or width % block:
                raise ValueError("Invalid Bonsai Hadamard block")
            if signs is None or signs.shape != (width,):
                raise ValueError("Missing Bonsai Hadamard signs")
            if not mx.all((signs == 1) | (signs == -1)).item():
                raise ValueError("Invalid Bonsai Hadamard signs")
        elif signs is not None:
            raise ValueError("Unexpected Bonsai Hadamard signs")
        setattr(parent, parts[-1], Packed(arrays, block, signs, record["embedding"]))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    # Preserve the pack's Qwen pre-tokenizer. Transformers' generic Mistral
    # heuristic misidentifies local configs without a transformers_version field.
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(path), fix_mistral_regex=False)
    return model, tokenizer
