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
        return self.raw.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def encode(self, text: str) -> list[int]:
        return list(self.raw.encode(text, add_special_tokens=False))


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
