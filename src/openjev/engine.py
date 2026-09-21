from __future__ import annotations

import math
import platform
import threading
import time
from typing import Any

from .backends.base import Backend
from .prompt import compile_plan
from .types import DecisionResult, ModelConfig, Noul, Score, Usage, parse_questions


class DecisionEngine:
    """One loaded model. Calls are serialized because backend inference state is mutable."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.config = backend.config
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def from_pretrained(cls, model: str | None = None, **kwargs) -> DecisionEngine:
        return cls.from_config(ModelConfig(model=model, **kwargs))

    @classmethod
    def from_config(cls, config: ModelConfig) -> DecisionEngine:
        backend = config.backend
        if backend == "auto":
            backend = (
                "mlx"
                if platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and config.device in {"auto", "metal"}
                else "transformers"
            )
        if backend == "mlx":
            from .backends.mlx import MLXBackend

            return cls(MLXBackend(config))
        if backend == "transformers":
            from .backends.transformers import TransformersBackend

            return cls(TransformersBackend(config))
        from .backends.gguf import GGUFBackend

        return cls(GGUFBackend(config))

    def decide(
        self,
        state: Any,
        questions: dict,
        *,
        images: list[str] | None = None,
        use_cache: bool = True,
    ) -> DecisionResult:
        with self._lock:
            if self._closed:
                raise RuntimeError("This engine has been closed")
            parsed = parse_questions(questions)
            if images is not None and not isinstance(images, (list, tuple)):
                raise ValueError("images must be a list of local file paths")
            if images and not self.config.vision:
                raise ValueError("Image input requires an engine loaded with vision=True")
            try:
                self.backend.prepare_request(images or [])
                return self._decide(state, parsed, use_cache)
            finally:
                self.backend.finish_request()

    def _decide(self, state, questions, use_cache):
        started = time.perf_counter()
        plan = compile_plan(state, questions, self.backend.tokenizer, self.config)
        usage = Usage(content_prefix_tokens=len(plan.prefix) if use_cache else 0)
        root = None
        if use_cache:
            root = self.backend.prefill(plan.prefix)
            usage.evaluated_input_tokens += len(plan.prefix)
        answers = {}
        for q in plan.questions:
            usage.candidates += len(q.prompts)
            usage.uncached_input_tokens += sum(map(len, q.prompts))
            if use_cache:
                question_suffix = q.prefix[len(plan.prefix) :]
                parent = self.backend.prefill(question_suffix, root)
                usage.evaluated_input_tokens += len(question_suffix)
                usage.question_prefix_tokens[q.key] = len(question_suffix)
                suffixes = [p[len(q.prefix) :] for p in q.prompts]
            else:
                parent = None
                suffixes = q.prompts
                usage.question_prefix_tokens[q.key] = 0
            usage.evaluated_input_tokens += sum(map(len, suffixes))
            scores = self.backend.score(parent, suffixes)
            if len(scores) != len(q.labels):
                raise RuntimeError("Backend returned the wrong number of candidate scores")
            log_support = [score.log_support(self.config.score_mode) for score in scores]
            detail = dict(
                zip(q.labels, [s.to_dict(self.config.score_mode) for s in scores], strict=True)
            )
            if isinstance(q.question, Noul):
                answers[q.key] = {
                    "type": "noul",
                    "noul": math.exp(log_support[0]),
                    "candidates": detail,
                }
                continue
            # Stable normalization of independent supports, NOT a calibrated
            # mutually exclusive posterior. Preserve raw evidence alongside it.
            maximum = max(log_support)
            weights = [math.exp(value - maximum) for value in log_support]
            total = sum(weights)
            probabilities = dict(zip(q.labels, [x / total for x in weights], strict=True))
            answer = {"type": q.question.type, "probabilities": probabilities, "candidates": detail}
            if isinstance(q.question, Score):
                answer["score"] = sum(int(k) * p for k, p in probabilities.items())
                answer["legend"] = dict(enumerate(q.question.criteria))
            else:
                answer["choice"] = max(probabilities, key=probabilities.get)
            answers[q.key] = answer
        usage.reused_input_tokens = usage.uncached_input_tokens - usage.evaluated_input_tokens
        usage.elapsed_seconds = time.perf_counter() - started
        return DecisionResult(
            self.backend.model_id,
            self.backend.name,
            self.config.score_mode,
            answers,
            usage,
        )

    def close(self):
        with self._lock:
            if not self._closed:
                self.backend.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
