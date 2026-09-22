from __future__ import annotations

import math
import platform
import threading
import time
from typing import Any

from .backends.base import Backend
from .cache_plan import CostModel, plan_batches
from .prompt import compile_plan, instruction_prefix
from .types import DecisionResult, ModelConfig, Noul, Score, Usage, parse_questions


class DecisionEngine:
    """One loaded model. Calls are serialized because backend inference state is mutable."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.config = backend.config
        self._lock = threading.Lock()
        self._closed = False
        self._instruction_tokens = []
        self._instruction_cache = None
        self.instruction_cache_seconds = 0.0
        if self.config.cache_strategy == "adaptive" and not self.config.vision:
            started = time.perf_counter()
            tokens = instruction_prefix(backend.tokenizer, self.config)
            if tokens and len(tokens) < self.config.n_ctx:
                try:
                    backend.prepare_request([])
                    self._instruction_cache = backend.prefill(tokens)
                    self._instruction_tokens = tokens
                finally:
                    backend.finish_request()
            self.instruction_cache_seconds = time.perf_counter() - started

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
        adaptive = use_cache and self.config.cache_strategy == "adaptive" and not self.config.vision
        prefix_length = len(plan.prefix) if use_cache and len(plan.questions) > 1 else 0
        usage = Usage(content_prefix_tokens=prefix_length)
        root = None
        if prefix_length and not adaptive:
            root = self.backend.prefill(plan.prefix)
            usage.evaluated_input_tokens += len(plan.prefix)
        # Flatten every question's candidates into one scoring call. The backend
        # can batch across question boundaries while preserving the original order.
        shared_scores = None
        if adaptive:
            shared_scores, usage = self._score_adaptive(plan)
        elif use_cache and (
            self.config.cache_strategy in {"shared", "adaptive"} or len(plan.questions) == 1
        ):
            suffixes = [p[prefix_length:] for q in plan.questions for p in q.prompts]
            shared_scores = self.backend.score(root, suffixes)
            if len(shared_scores) != len(suffixes):
                raise RuntimeError("Backend returned the wrong number of candidate scores")
            usage.evaluated_input_tokens += sum(map(len, suffixes))
        offset = 0
        answers = {}
        for q in plan.questions:
            usage.candidates += len(q.prompts)
            usage.uncached_input_tokens += sum(map(len, q.prompts))
            if shared_scores is not None:
                scores = shared_scores[offset : offset + len(q.prompts)]
                offset += len(q.prompts)
                if not adaptive:
                    usage.question_prefix_tokens[q.key] = 0
            elif use_cache:
                question_suffix = q.prefix[len(plan.prefix) :]
                parent = self.backend.prefill(question_suffix, root)
                usage.evaluated_input_tokens += len(question_suffix)
                usage.question_prefix_tokens[q.key] = len(question_suffix)
                suffixes = [p[len(q.prefix) :] for p in q.prompts]
            else:
                parent = None
                suffixes = q.prompts
                usage.question_prefix_tokens[q.key] = 0
            if shared_scores is None:
                usage.evaluated_input_tokens += sum(map(len, suffixes))
                if use_cache:
                    scores = self.backend.score(parent, suffixes)
                else:
                    # A genuinely independent baseline: full prompts, one row
                    # per model call, with no shared prefix or candidate batch.
                    scores = []
                    for suffix in suffixes:
                        one = self.backend.score(None, [suffix])
                        if len(one) != 1:
                            raise RuntimeError(
                                "Backend returned the wrong number of candidate scores"
                            )
                        scores.extend(one)
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
        stats = getattr(self.backend, "stats", None)
        if stats is not None:
            usage.candidate_batches = list(stats.sizes)
            usage.padding_tokens = stats.padding_tokens
        usage.elapsed_seconds = time.perf_counter() - started
        return DecisionResult(
            self.backend.model_id,
            self.backend.name,
            self.config.score_mode,
            answers,
            usage,
            cache_strategy=(
                (
                    "adaptive"
                    if adaptive
                    else (
                        "single"
                        if len(plan.questions) == 1
                        else (
                            "shared"
                            if self.config.cache_strategy == "adaptive"
                            else self.config.cache_strategy
                        )
                    )
                )
                if use_cache
                else "none"
            ),
            prompt_style=self.config.prompt_style,
        )

    def _score_adaptive(self, plan):
        prompts = [p for q in plan.questions for p in q.prompts]
        n = len(self._instruction_tokens)
        # Never truncate a recurrent cache to repair a tokenizer boundary mismatch.
        matches = n and all(p[:n] == self._instruction_tokens for p in prompts)
        start, root = (n, self._instruction_cache) if matches else (0, None)
        usage = Usage(instruction_prefix_tokens=start)
        started = time.perf_counter()
        tree = plan_batches(
            prompts,
            start,
            CostModel(batch_size=1 if self.backend.name == "gguf" else self.config.batch_size),
        )
        usage.planning_seconds = time.perf_counter() - started
        scores = [None] * len(prompts)
        depths = [0] * len(prompts)

        def execute(node, parent, depth):
            if node.depth > depth:
                tokens = prompts[node.rows[0]][depth : node.depth]
                parent = self.backend.prefill(tokens, parent)
                usage.evaluated_input_tokens += len(tokens)
                usage.cache_prefill_tokens.append(len(tokens))
            if node.pending:
                suffixes = [prompts[i][node.depth :] for i in node.pending]
                values = self.backend.score(parent, suffixes)
                if len(values) != len(suffixes):
                    raise RuntimeError("Backend returned the wrong number of candidate scores")
                usage.evaluated_input_tokens += sum(map(len, suffixes))
                for index, value in zip(node.pending, values, strict=True):
                    scores[index], depths[index] = value, node.depth
            for child in node.children:
                execute(child, parent, node.depth)

        execute(tree, root, start)
        usage.scoring_prefix_tokens = depths
        usage.content_prefix_tokens = min(len(plan.prefix), min(depths))
        offset = 0
        for q in plan.questions:
            depth = min(depths[offset : offset + len(q.prompts)])
            usage.question_prefix_tokens[q.key] = max(
                0, min(depth, len(q.prefix)) - len(plan.prefix)
            )
            offset += len(q.prompts)
        return scores, usage

    def close(self):
        with self._lock:
            if not self._closed:
                self._instruction_cache = None
                self._instruction_tokens = []
                self.backend.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
