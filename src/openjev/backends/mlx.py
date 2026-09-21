from __future__ import annotations

import copy
import json
import platform
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import MLX_MODEL, ModelConfig, TokenScore
from .base import BatchStats, HFTokenizer, verdict_ids
from .vision import QwenVisionTokenizer, open_images


@dataclass
class MLXSnapshot:
    cache: list[Any]
    length: int
    rope_delta: Any = None


class MLXBackend:
    name = "mlx"

    def __init__(self, config: ModelConfig):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("The MLX backend requires Apple Silicon macOS")
        if config.device not in {"auto", "metal"}:
            raise ValueError("Use device='metal' or 'auto' with MLX")
        if config.load_in_4bit:
            raise ValueError("MLX loads prequantized models; choose a 4-bit MLX model repository")
        try:
            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
        except ModuleNotFoundError as exc:
            raise ImportError("Install MLX support with pip install 'openjev[mlx]'") from exc
        except ImportError as exc:
            raise RuntimeError(f"MLX could not initialize the Metal device: {exc}") from exc
        self.mx = mx
        self.config = config
        self.model_id = config.model or MLX_MODEL
        self.make_cache = make_prompt_cache
        self.processor = None
        self.image_data = None
        from huggingface_hub import hf_hub_download, snapshot_download

        config_path = Path(self.model_id) / "config.json"
        if not config_path.is_file():
            config_path = Path(
                hf_hub_download(self.model_id, "config.json", revision=config.revision)
            )
        is_bonsai = json.loads(config_path.read_text()).get("model_type") == "prism_hadamard_qwen35"
        if is_bonsai and config.vision:
            raise ValueError("Bonsai Hadamard MLX support currently scores text only")
        if config.vision:
            try:
                from mlx_vlm import load
            except ImportError as exc:
                raise ImportError(
                    "Install image support with pip install 'openjev[mlx,vision]'"
                ) from exc
            path = self.model_id
            if config.revision:
                path = snapshot_download(path, revision=config.revision)
            self.model, self.processor = load(path, trust_remote_code=False)
            if self.model.config.model_type not in {"qwen3_5", "qwen3_5_moe"}:
                raise ValueError("Image scoring currently supports Qwen3.5 models")
            self.tokenizer = QwenVisionTokenizer(self.processor.tokenizer)
            self.language_model = self.model.language_model
            self.decoder = self.language_model.model
        else:
            from mlx_lm import load

            if is_bonsai:
                from .bonsai import load_bonsai

                path = self.model_id
                if not Path(path).is_dir():
                    path = snapshot_download(
                        path,
                        revision=config.revision,
                        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
                    )
                self.model, tokenizer = load_bonsai(path)
            else:
                self.model, tokenizer = load(
                    self.model_id,
                    revision=config.revision,
                    tokenizer_config={"trust_remote_code": False},
                )
            self.tokenizer = HFTokenizer(tokenizer)
            self.language_model = getattr(self.model, "language_model", self.model)
            safe_types = {"qwen2", "qwen3", "qwen3_5", "qwen3_5_moe"}
            self.decoder = (
                self.language_model.model
                if config.optimize_head
                and (is_bonsai or getattr(self.model, "model_type", None) in safe_types)
                else None
            )
        if config.tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer,
                revision=config.tokenizer_revision,
                trust_remote_code=False,
            )
            if config.vision:
                self.processor.tokenizer = tokenizer
                self.tokenizer = QwenVisionTokenizer(tokenizer)
            else:
                self.tokenizer = HFTokenizer(tokenizer)
        self.positive_id, self.negative_id = verdict_ids(self.tokenizer, config)
        self.head = None
        self.tied = False
        if self.decoder is not None:
            self.tied = self.language_model.args.tie_word_embeddings
            self.head = self.decoder.embed_tokens if self.tied else self.language_model.lm_head
        mx.eval(self.model.parameters())

    def prepare_request(self, images):
        self.stats = BatchStats()
        self.image_data = None
        if self.config.vision:
            self.tokenizer.image_counts = []
        if not images:
            return
        if not self.config.vision:
            raise ValueError("Reload the model with vision=True to score images")
        from mlx_vlm.utils import prepare_inputs

        prompt = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        prepared = prepare_inputs(
            self.processor,
            images=open_images(images),
            prompts=prompt,
            image_token_index=self.model.config.image_token_index,
        )
        self.image_data = {
            k: v for k, v in prepared.items() if k in {"pixel_values", "image_grid_thw"}
        }
        grid = prepared["image_grid_thw"].tolist()
        merge = self.model.config.vision_config.spatial_merge_size
        self.tokenizer.image_counts = [int(t * h * w // (merge * merge)) for t, h, w in grid]
        if self.tokenizer.encode(prompt) != prepared["input_ids"][0].tolist():
            raise ValueError("Image placeholder expansion differs from the model processor")

    def finish_request(self):
        self.image_data = None
        if self.config.vision:
            self.tokenizer.image_counts = []

    def _empty(self):
        return MLXSnapshot(self.make_cache(self.language_model), 0)

    def _head(self, hidden):
        mx = self.mx
        if self.config.score_mode == "binary" and self.config.optimize_head:
            ids = mx.array([self.positive_id, self.negative_id])
            if hasattr(self.head, "project_selected"):
                return self.head.project_selected(hidden, ids), True
            if hasattr(self.head, "scales"):
                biases = self.head.get("biases")
                logits = mx.quantized_matmul(
                    hidden,
                    self.head.weight[ids],
                    self.head.scales[ids],
                    biases[ids] if biases is not None else None,
                    transpose=True,
                    group_size=self.head.group_size,
                    bits=self.head.bits,
                    mode=self.head.mode,
                )
            else:
                logits = hidden @ self.head.weight[ids].T
            bias = self.head.get("bias")
            if bias is not None:
                logits = logits + bias[ids]
            return logits, True
        return (self.head.as_linear(hidden) if self.tied else self.head(hidden)), False

    def _forward(self, rows, snapshot, *, want_scores=False, lengths=None):
        mx = self.mx
        x = mx.array(rows)
        embeds = positions = None
        if self.config.vision:
            if snapshot.length == 0:
                image_data = self.image_data or {}
                if x.shape[0] > 1:
                    image_data = {
                        key: mx.concatenate([value] * x.shape[0], axis=0)
                        for key, value in image_data.items()
                    }
                features = self.model.get_input_embeddings(x, **image_data)
                embeds = features.inputs_embeds
                positions = features.position_ids
                snapshot.rope_delta = features.rope_deltas
            else:
                delta = snapshot.rope_delta if snapshot.rope_delta is not None else 0
                positions = mx.arange(x.shape[1])[None, :] + snapshot.length + delta
                positions = mx.broadcast_to(positions, (x.shape[0], x.shape[1]))
                positions = mx.broadcast_to(positions[None], (3, *positions.shape))
        # Right padding appears only after each verdict. These causal branches
        # are terminal: their padded recurrent states are never reused. Evaluate
        # a padded scoring batch intact so every row's last real hidden state is
        # available for the one-position vocabulary projection.
        # Score complete branches in one pass. The explicit chunk-size=1
        # diagnostic retains token-by-token arithmetic for singleton checks.
        single_pass = lengths is not None or (want_scores and self.config.prefill_chunk_size != 1)
        chunk = x.shape[1] if single_pass else self.config.prefill_chunk_size
        logits = None
        selected = False
        for start in range(0, x.shape[1], chunk):
            end = min(start + chunk, x.shape[1])
            tokens = x[:, start:end]
            if self.decoder is not None:
                kwargs = {"cache": snapshot.cache}
                if self.config.vision:
                    kwargs["position_ids"] = positions[..., start:end]
                    if embeds is not None:
                        kwargs["inputs_embeds"] = embeds[:, start:end]
                hidden = self.decoder(tokens, **kwargs)
                if want_scores and end == x.shape[1]:
                    last = (
                        hidden[:, -1, :]
                        if lengths is None
                        else hidden[mx.arange(x.shape[0]), mx.array(lengths) - 1, :]
                    )
                    logits, selected = self._head(last)
            else:
                all_logits = self.model(tokens, cache=snapshot.cache)
                if want_scores and end == x.shape[1]:
                    logits = all_logits[:, -1, :]
            mx.eval([c.state for c in snapshot.cache])
        snapshot.length += x.shape[1]
        if not want_scores:
            return None
        if logits is None:
            raise ValueError("A candidate must have at least one uncached token")
        logits = logits.astype(mx.float32)
        pair = logits if selected else logits[:, [self.positive_id, self.negative_id]]
        binary = pair[:, 0] - mx.logsumexp(pair, axis=-1)
        if selected:
            mx.eval(binary)
            return [TokenScore(None, None, float(value)) for value in binary.tolist()]
        z = mx.logsumexp(logits, axis=-1)
        pos, neg = pair[:, 0] - z, pair[:, 1] - z
        mx.eval(pos, neg, binary)
        return [
            TokenScore(float(a), float(b), float(c))
            for a, b, c in zip(pos.tolist(), neg.tolist(), binary.tolist(), strict=True)
        ]

    def prefill(self, tokens, parent=None):
        snapshot = self._empty() if parent is None else copy.deepcopy(parent)
        if tokens:
            self._forward([tokens], snapshot)
        return snapshot

    def score(self, parent, suffixes):
        # Qwen decoder scoring can gather each row's real verdict before right
        # padding. Generic models retain equal-length groups.
        groups = defaultdict(list)
        for i, suffix in enumerate(suffixes):
            groups[0 if self.decoder is not None else len(suffix)].append((i, suffix))
        result = [None] * len(suffixes)
        for group in groups.values():
            group.sort(key=lambda item: len(item[1]))
            size = self.config.batch_size
            seed = parent or self._empty()
            if not all(hasattr(c, "merge") for c in seed.cache):
                size = 1
            for start in range(0, len(group), size):
                batch = group[start : start + size]
                snapshot = copy.deepcopy(seed)
                if len(batch) > 1:
                    snapshot.cache = [
                        type(c).merge([copy.deepcopy(c) for _ in batch]) for c in seed.cache
                    ]
                rows = [row for _, row in batch]
                lengths = [len(row) for row in rows]
                width = max(lengths)
                padded = min(lengths) != width
                if padded:
                    pad = getattr(self.tokenizer.raw, "pad_token_id", None)
                    pad = 0 if pad is None else pad
                    rows = [row + [pad] * (width - len(row)) for row in rows]
                scores = self._forward(
                    rows, snapshot, want_scores=True, lengths=lengths if padded else None
                )
                self.stats.sizes.append(len(batch))
                self.stats.padding_tokens += len(batch) * width - sum(lengths)
                for (index, _), score in zip(batch, scores, strict=True):
                    result[index] = score
        return result

    def close(self):
        self.finish_request()
        self.head = self.decoder = self.language_model = self.model = self.processor = None
        self.mx.clear_cache()
