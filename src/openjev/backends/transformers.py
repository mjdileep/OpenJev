from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any

from ..types import TOKENIZER_MODEL, ModelConfig, TokenScore
from .base import HFTokenizer, verdict_ids
from .vision import QwenVisionTokenizer, open_images


@dataclass
class TorchSnapshot:
    cache: Any
    length: int
    rope_delta: Any = None


class TransformersBackend:
    """CUDA safetensors backend, including Qwen images and optional bitsandbytes 4-bit.

    Candidate branches are evaluated sequentially with full cache snapshots. This
    conservative path supports hybrid recurrent caches, unlike generic KV slicing.
    """

    name = "transformers"

    def __init__(self, config: ModelConfig):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError("Install this backend with pip install 'openjev[cuda]'") from exc
        self.torch, self.config = torch, config
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("Use MLX for Metal; the Transformers backend accepts cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in this PyTorch installation")
        self.device = torch.device(device)
        self.model_id = config.model or TOKENIZER_MODEL
        dtype = (
            torch.float32
            if device == "cpu"
            else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        )
        options = dict(revision=config.revision, trust_remote_code=False, dtype=dtype)
        if config.load_in_4bit:
            if device != "cuda":
                raise ValueError("load_in_4bit requires CUDA for this backend")
            from transformers import BitsAndBytesConfig

            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            options["device_map"] = {"": "cuda:0"}
        loader = AutoModelForImageTextToText if config.vision else AutoModelForCausalLM
        self.model = loader.from_pretrained(self.model_id, **options).eval()
        if not config.load_in_4bit:
            self.model.to(self.device)
        tokenizer_id = config.tokenizer or self.model_id
        self.processor = None
        if config.vision:
            if self.model.config.model_type not in {"qwen3_5", "qwen3_5_moe"}:
                raise ValueError("Image scoring currently supports Qwen3.5 models")
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                tokenizer_id,
                revision=config.tokenizer_revision or config.revision,
                trust_remote_code=False,
            )
            self.tokenizer = QwenVisionTokenizer(self.processor.tokenizer)
        else:
            self.tokenizer = HFTokenizer(
                AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    revision=config.tokenizer_revision or config.revision,
                    trust_remote_code=False,
                )
            )
        self.positive_id, self.negative_id = verdict_ids(self.tokenizer, config)
        self.keep_logits = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        self.image_data = None

    def prepare_request(self, images):
        self.image_data = None
        if self.config.vision:
            self.tokenizer.image_counts = []
        if not images:
            return
        prompt = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        prepared = self.processor(
            text=[prompt],
            images=open_images(images),
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.image_data = {}
        for name in ("pixel_values", "image_grid_thw"):
            value = prepared[name].to(self.device)
            if value.is_floating_point():
                value = value.to(self.model.dtype)
            self.image_data[name] = value
        merge = self.model.config.vision_config.spatial_merge_size
        self.tokenizer.image_counts = [
            int(t * h * w // (merge * merge)) for t, h, w in prepared["image_grid_thw"].tolist()
        ]
        if self.tokenizer.encode(prompt) != prepared["input_ids"][0].tolist():
            raise ValueError("Image token expansion differs from the model processor")

    def finish_request(self):
        self.image_data = None
        if self.config.vision:
            self.tokenizer.image_counts = []
            self.model.model.rope_deltas = None

    def _forward(self, tokens, snapshot):
        torch = self.torch
        if not tokens:
            raise ValueError("Forward evaluation needs at least one new token")
        x = torch.tensor([tokens], device=self.device)
        options = dict(
            input_ids=x, past_key_values=snapshot.cache, use_cache=True, return_dict=True
        )
        if self.keep_logits:
            options["logits_to_keep"] = 1
        if self.config.vision:
            # RoPE offsets live outside past_key_values in Qwen. Restore them
            # together with the recurrent/KV state for every branch.
            self.model.model.rope_deltas = snapshot.rope_delta
            if snapshot.length == 0 and self.image_data:
                options.update(self.image_data)
                if self.model.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
                    options["mm_token_type_ids"] = (x == self.tokenizer.image_id).long()
        with torch.inference_mode():
            output = self.model(**options)
        snapshot.cache = output.past_key_values
        snapshot.length += len(tokens)
        if self.config.vision:
            delta = getattr(self.model.model, "rope_deltas", None)
            snapshot.rope_delta = None if delta is None else delta.clone()
        return output.logits[0, -1].float()

    def prefill(self, tokens, parent=None):
        snapshot = TorchSnapshot(None, 0) if parent is None else copy.deepcopy(parent)
        if tokens:
            # Vision prefixes are evaluated intact so image groups and their
            # multidimensional position IDs cannot be split across chunks.
            if self.config.vision:
                self._forward(tokens, snapshot)
            else:
                n = self.config.prefill_chunk_size
                for offset in range(0, len(tokens), n):
                    self._forward(tokens[offset : offset + n], snapshot)
        return snapshot

    def score(self, parent, suffixes):
        result = []
        torch = self.torch
        for suffix in suffixes:
            snapshot = TorchSnapshot(None, 0) if parent is None else copy.deepcopy(parent)
            logits = self._forward(suffix, snapshot)
            pair = logits[[self.positive_id, self.negative_id]]
            z = torch.logsumexp(logits, dim=-1)
            values = torch.stack(
                [pair[0] - z, pair[1] - z, pair[0] - torch.logsumexp(pair, dim=-1)]
            )
            result.append(TokenScore(*values.tolist()))
        return result

    def close(self):
        self.finish_request()
        self.model = self.processor = None
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
