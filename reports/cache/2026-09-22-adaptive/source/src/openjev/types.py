from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ScoreMode = Literal["full", "binary"]
MLX_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
GGUF_MODEL = "unsloth/Qwen3.5-0.8B-GGUF"
GGUF_FILE = "Qwen3.5-0.8B-Q4_K_M.gguf"
TOKENIZER_MODEL = "Qwen/Qwen3.5-0.8B"


@dataclass(frozen=True)
class ModelConfig:
    backend: Literal["auto", "mlx", "gguf", "transformers"] = "auto"
    model: str | None = None
    revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    filename: str | None = None
    device: Literal["auto", "cpu", "cuda", "metal"] = "auto"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    prefill_chunk_size: int = 128
    batch_size: int = 8
    cache_strategy: Literal["shared", "tree", "adaptive"] = "shared"
    positive_token: str = "yes"
    negative_token: str = "no"
    score_mode: ScoreMode = "full"
    optimize_head: bool = True
    vision: bool = False
    load_in_4bit: bool = False
    prompt_style: Literal["full", "short"] = "full"

    def __post_init__(self):
        if self.backend not in {"auto", "mlx", "gguf", "transformers"}:
            raise ValueError("backend must be auto, mlx, gguf, or transformers")
        if self.device not in {"auto", "cpu", "cuda", "metal"}:
            raise ValueError("device must be auto, cpu, cuda, or metal")
        if self.score_mode not in {"full", "binary"}:
            raise ValueError("score_mode must be full or binary")
        if self.cache_strategy not in {"shared", "tree", "adaptive"}:
            raise ValueError("cache_strategy must be shared, tree, or adaptive")
        if self.prompt_style not in {"full", "short"}:
            raise ValueError("prompt_style must be full or short")
        for name in ("n_ctx", "prefill_chunk_size", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.positive_token or not self.negative_token:
            raise ValueError("Verdict tokens must not be empty")
        if self.positive_token == self.negative_token:
            raise ValueError("Positive and negative verdict tokens must differ")


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: dict[str, str]
    type: Literal["choice"] = field(default="choice", init=False)


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: list[str]
    type: Literal["score"] = field(default="score", init=False)


@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: dict[str, str] = field(default_factory=dict)
    type: Literal["noul"] = field(default="noul", init=False)


Question = Choice | Score | Noul


def parse_questions(questions: dict[str, Any]) -> dict[str, Question]:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty mapping")
    result = {}
    for key, q in questions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Question IDs must be nonempty strings")
        if isinstance(q, dict):
            data = dict(q)
            kind = data.pop("type", None)
            cls = {"choice": Choice, "score": Score, "noul": Noul}.get(kind)
            if cls is None:
                raise ValueError(f"{key}: unsupported question type {kind!r}")
            try:
                q = cls(**data)
            except TypeError as exc:
                raise ValueError(f"{key}: invalid question: {exc}") from exc
        if not isinstance(q, (Choice, Score, Noul)):
            raise ValueError(f"{key}: expected Choice, Score, Noul, or question dictionary")
        if not isinstance(q.instructions, str) or not q.instructions.strip():
            raise ValueError(f"{key}: instructions must be a nonempty string")
        if isinstance(q, Score):
            if not isinstance(q.criteria, list) or len(q.criteria) < 2:
                raise ValueError(f"{key}: score requires at least two ordered descriptions")
            values = q.criteria
        else:
            if not isinstance(q.criteria, dict):
                raise ValueError(f"{key}: criteria must be a mapping")
            if isinstance(q, Choice) and len(q.criteria) < 2:
                raise ValueError(f"{key}: choice requires at least two options")
            if any(not isinstance(k, str) or not k for k in q.criteria):
                raise ValueError(f"{key}: criteria keys must be nonempty strings")
            if isinstance(q, Noul) and set(q.criteria) - {"true", "false"}:
                raise ValueError(f"{key}: noul criteria keys must be 'true' or 'false'")
            values = list(q.criteria.values())
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{key}: criteria descriptions must be nonempty strings")
        result[key] = q
    return result


@dataclass(frozen=True)
class TokenScore:
    """Full vocabulary probabilities may be absent with a pruned binary projection."""

    log_p_positive: float | None
    log_p_negative: float | None
    log_p_binary: float

    def log_support(self, mode: ScoreMode) -> float:
        value = self.log_p_positive if mode == "full" else self.log_p_binary
        if value is None or not math.isfinite(value):
            raise ValueError("Backend returned a missing or nonfinite score")
        return value

    def to_dict(self, mode: ScoreMode) -> dict[str, Any]:
        def probability(value):
            return None if value is None else math.exp(value)

        return {
            "support": math.exp(self.log_support(mode)),
            "log_support": self.log_support(mode),
            "p_positive": probability(self.log_p_positive),
            "p_negative": probability(self.log_p_negative),
            "p_positive_given_binary": math.exp(self.log_p_binary),
            "binary_token_mass": (
                None
                if self.log_p_positive is None or self.log_p_negative is None
                else math.exp(self.log_p_positive) + math.exp(self.log_p_negative)
            ),
        }


@dataclass
class Usage:
    uncached_input_tokens: int = 0
    evaluated_input_tokens: int = 0
    reused_input_tokens: int = 0
    content_prefix_tokens: int = 0
    question_prefix_tokens: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
    candidate_batches: list[int] = field(default_factory=list)
    padding_tokens: int = 0
    generated_tokens: int = 0
    elapsed_seconds: float = 0.0
    instruction_prefix_tokens: int = 0
    cache_prefill_tokens: list[int] = field(default_factory=list)
    scoring_prefix_tokens: list[int] = field(default_factory=list)
    planning_seconds: float = 0.0


@dataclass
class DecisionResult:
    model: str
    backend: str
    score_mode: ScoreMode
    answers: dict[str, dict[str, Any]]
    usage: Usage
    cache_strategy: str = "shared"
    calibrated: bool = False
    probability_semantics: str = "normalized_independent_candidate_support"
    prompt_style: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
