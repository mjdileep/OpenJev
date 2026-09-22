from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..prompt import Tokenizer
from ..types import ModelConfig, TokenScore


@dataclass
class BatchStats:
    sizes: list[int] = field(default_factory=list)
    padding_tokens: int = 0


class Backend(Protocol):
    name: str
    model_id: str
    tokenizer: Tokenizer
    config: ModelConfig
    stats: BatchStats

    def prefill(self, tokens: list[int], parent: Any = None) -> Any: ...
    def score(self, parent: Any, suffixes: list[list[int]]) -> list[TokenScore]: ...
    def prepare_request(self, images: list[str]) -> None: ...
    def finish_request(self) -> None: ...
    def close(self) -> None: ...


class HFTokenizer:
    def __init__(self, tokenizer):
        self.raw = tokenizer
        if not tokenizer.chat_template:
            raise ValueError("Choose an instruction model with a Hugging Face chat template")

    def render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        return self.raw.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def encode(self, text: str) -> list[int]:
        return list(self.raw.encode(text, add_special_tokens=False))

    def encode_many(self, texts: list[str]) -> list[list[int]]:
        # Subclasses may expand image placeholders or use a different tokenizer
        # (GGUF). Preserve their encode semantics instead of bypassing them.
        # MLX forwards encode but blocks special-method lookup. Its bound method
        # identifies the underlying HF tokenizer without replacing chat rendering.
        owner = getattr(self.raw.encode, "__self__", self.raw)
        batch = owner if callable(owner) else getattr(owner, "batch_encode_plus", None)
        if type(self).encode is not HFTokenizer.encode or batch is None:
            return [self.encode(text) for text in texts]
        result = []
        for start in range(0, len(texts), 128):
            result.extend(
                batch(
                    texts[start : start + 128],
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
            )
        return result


def verdict_ids(tokenizer: Tokenizer, config: ModelConfig) -> tuple[int, int]:
    ids = []
    for marker in (config.positive_token, config.negative_token):
        encoded = tokenizer.encode(marker)
        if len(encoded) != 1:
            raise ValueError(f"Verdict {marker!r} must encode as exactly one token: {encoded}")
        ids.append(encoded[0])
    if ids[0] == ids[1]:
        raise ValueError("Verdict markers must map to different token IDs")
    return ids[0], ids[1]
