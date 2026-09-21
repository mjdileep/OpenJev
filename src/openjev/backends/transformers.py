from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any

from ..types import TOKENIZER_MODEL, ModelConfig, TokenScore
from .base import BatchStats, HFTokenizer, verdict_ids
from .vision import QwenVisionTokenizer, open_images


@dataclass
class TorchSnapshot:
    cache: Any
    length: int
    rope_delta: Any = None


class TransformersBackend:
    """CUDA safetensors backend, including Qwen images and optional bitsandbytes 4-bit.

    Candidate batches branch the complete hybrid cache. Padding is allowed only
    on terminal scoring branches, whose mutated caches are never reused.
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
        self.decoder = (
            self.model.model
            if config.optimize_head
            and type(self.model).__name__
            in {"Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"}
            else None
        )
        self.image_data = None

    def prepare_request(self, images):
        self.stats = BatchStats()
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

    def _forward(self, rows, snapshot, *, want_scores=True):
        torch = self.torch
        if not rows or any(not row for row in rows):
            raise ValueError("Forward evaluation needs at least one new token")
        lengths = [len(row) for row in rows]
        width = max(lengths)
        pad = getattr(self.tokenizer.raw, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer.raw, "eos_token_id", None)
        pad = 0 if pad is None else pad
        x = torch.tensor([row + [pad] * (width - len(row)) for row in rows], device=self.device)
        options = dict(
            input_ids=x, past_key_values=snapshot.cache, use_cache=True, return_dict=True
        )
        if min(lengths) != width:
            options["attention_mask"] = torch.tensor(
                [[1] * (snapshot.length + n) + [0] * (width - n) for n in lengths],
                device=self.device,
            )
        if self.config.vision:
            # RoPE offsets live outside past_key_values in Qwen. Restore them
            # together with the recurrent/KV state for every branch.
            self.model.model.rope_deltas = snapshot.rope_delta
            if snapshot.length and snapshot.rope_delta is not None:
                # Explicit suffix positions avoid reconstructing positions for
                # the entire prefix from a padded attention mask.
                positions = torch.arange(
                    snapshot.length, snapshot.length + width, device=self.device
                )[None, :] + snapshot.rope_delta.reshape(-1, 1)
                options["position_ids"] = positions[None, ...].expand(3, len(rows), width)
            if snapshot.length == 0 and self.image_data:
                # Full-prompt candidate batches repeat the same image groups
                # for each row. Shared-prefix scoring runs the vision tower once.
                options.update(
                    {
                        name: value.repeat((len(rows),) + (1,) * (value.ndim - 1))
                        if len(rows) > 1
                        else value
                        for name, value in self.image_data.items()
                    }
                )
                if self.model.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
                    options["mm_token_type_ids"] = (x == self.tokenizer.image_id).long()
        with torch.inference_mode():
            if self.decoder is not None:
                output = self.decoder(**options)
                logits = None
                if want_scores:
                    hidden = output.last_hidden_state
                    last = torch.tensor(lengths, device=self.device) - 1
                    logits = self.model.lm_head(
                        hidden[torch.arange(len(rows), device=self.device), last]
                    )
            else:
                # Generic models keep the smallest tail covering all verdict
                # positions. Qwen's direct decoder above projects exactly B rows.
                keep = width - min(lengths) + 1
                if self.keep_logits:
                    options["logits_to_keep"] = keep
                output = self.model(**options)
                last = torch.tensor(lengths, device=self.device) - 1
                if self.keep_logits:
                    last -= width - keep
                logits = (
                    output.logits[torch.arange(len(rows), device=self.device), last]
                    if want_scores
                    else None
                )
        snapshot.cache = output.past_key_values
        snapshot.length += width
        if self.config.vision:
            delta = getattr(self.model.model, "rope_deltas", None)
            snapshot.rope_delta = None if delta is None else delta.clone()
        return None if logits is None else logits.float()

    def prefill(self, tokens, parent=None):
        snapshot = TorchSnapshot(None, 0) if parent is None else copy.deepcopy(parent)
        if tokens:
            # Vision prefixes are evaluated intact so image groups and their
            # multidimensional position IDs cannot be split across chunks.
            if self.config.vision:
                self._forward([tokens], snapshot, want_scores=False)
            else:
                n = self.config.prefill_chunk_size
                for offset in range(0, len(tokens), n):
                    self._forward([tokens[offset : offset + n]], snapshot, want_scores=False)
        return snapshot

    def _fork(self, parent, batch_size):
        snapshot = TorchSnapshot(None, 0) if parent is None else copy.deepcopy(parent)
        if batch_size > 1 and snapshot.cache is not None:
            # Reorder supports duplicated indices and includes conv/recurrent
            # states. batch_repeat_interleave is not implemented by every hybrid
            # layer, so it cannot be used as a generic KV-only shortcut.
            indices = self.torch.zeros(batch_size, dtype=self.torch.long, device=self.device)
            snapshot.cache.reorder_cache(indices)
        if batch_size > 1 and snapshot.rope_delta is not None:
            snapshot.rope_delta = snapshot.rope_delta.repeat_interleave(batch_size, dim=0)
        return snapshot

    def score(self, parent, suffixes):
        result = [None] * len(suffixes)
        torch = self.torch
        size = self.config.batch_size
        # Unknown cache implementations stay serial. With no parent, candidates
        # can be batched directly for the single-question path.
        if parent is not None and (
            parent.cache is not None and not callable(getattr(parent.cache, "reorder_cache", None))
        ):
            size = 1
        ordered = sorted(enumerate(suffixes), key=lambda item: len(item[1]))
        pending = []
        for start in range(0, len(ordered), size):
            batch = ordered[start : start + size]
            rows = [row for _, row in batch]
            snapshot = self._fork(parent, len(batch))
            logits = self._forward(rows, snapshot)
            self.stats.sizes.append(len(batch))
            self.stats.padding_tokens += len(batch) * max(map(len, rows)) - sum(map(len, rows))
            pair = logits[:, [self.positive_id, self.negative_id]]
            z = torch.logsumexp(logits, dim=-1)
            values = torch.stack(
                [pair[:, 0] - z, pair[:, 1] - z, pair[:, 0] - torch.logsumexp(pair, dim=-1)],
                dim=-1,
            )
            pending.append(([index for index, _ in batch], values))
        # One device-to-host transfer after all batches, rather than one sync
        # per candidate. Preserve caller order after sorting by suffix length.
        if pending:
            values = torch.cat([value for _, value in pending]).tolist()
            indices = [index for batch, _ in pending for index in batch]
            for index, value in zip(indices, values, strict=True):
                result[index] = TokenScore(*value)
        return result

    def close(self):
        self.finish_request()
        self.model = self.decoder = self.processor = None
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
