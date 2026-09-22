"""Experimental single-question scoring with one prefix and a batch of cache copies."""

from __future__ import annotations

import hashlib
import math
import time

from openjev import Choice, DecisionEngine
from openjev.prompt import compile_plan
from openjev.types import DecisionResult, Usage


def fingerprint(snapshot):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    states = [entry.state for entry in snapshot.cache]
    mx.eval(states)
    checksum = hashlib.sha256()
    for name, array in tree_flatten(states):
        if isinstance(array, mx.array):
            checksum.update(str((name, array.shape, array.dtype)).encode())
            checksum.update(np.array(array.astype(mx.float32)).tobytes())
    checksum.update(
        str((snapshot.length, [getattr(c, "offset", None) for c in snapshot.cache])).encode()
    )
    return checksum.hexdigest()


class SharedPrefixBatchEngine(DecisionEngine):
    """Experiment-only Choice path; production OpenJev is left unchanged."""

    verify_parent = False

    def _decide(self, state, questions, use_cache):
        assert use_cache
        started = time.perf_counter()
        plan = compile_plan(state, questions, self.backend.tokenizer, self.config)
        assert len(plan.questions) == 1
        q = plan.questions[0]
        assert isinstance(q.question, Choice)
        n = len(plan.prefix)
        assert n > 0 and all(p[:n] == plan.prefix for p in q.prompts)
        # Exactly the requested shape: instructions + user context once, then
        # independent batched candidate branches. Question text remains in suffixes.
        parent = self.backend.prefill(plan.prefix)
        before = fingerprint(parent) if self.verify_parent else None
        scores = self.backend.score(parent, [p[n:] for p in q.prompts])
        if self.verify_parent:
            assert fingerprint(parent) == before, "A candidate mutated the retained prefix"
        assert self.backend.stats.sizes == [len(q.prompts)]
        logs = [s.log_support(self.config.score_mode) for s in scores]
        weights = [math.exp(value - max(logs)) for value in logs]
        probabilities = dict(zip(q.labels, [w / sum(weights) for w in weights], strict=True))
        answer = {
            "type": "choice",
            "probabilities": probabilities,
            "choice": max(probabilities, key=probabilities.get),
            "candidates": dict(
                zip(q.labels, [s.to_dict(self.config.score_mode) for s in scores], strict=True)
            ),
        }
        uncached = sum(map(len, q.prompts))
        evaluated = n + sum(len(p) - n for p in q.prompts)
        usage = Usage(
            uncached_input_tokens=uncached,
            evaluated_input_tokens=evaluated,
            reused_input_tokens=uncached - evaluated,
            content_prefix_tokens=n,
            question_prefix_tokens={q.key: 0},
            candidates=len(q.prompts),
            candidate_batches=list(self.backend.stats.sizes),
            padding_tokens=self.backend.stats.padding_tokens,
            elapsed_seconds=time.perf_counter() - started,
        )
        return DecisionResult(
            self.backend.model_id,
            self.backend.name,
            self.config.score_mode,
            {q.key: answer},
            usage,
            cache_strategy="shared_prefix_batch",
            prompt_style=self.config.prompt_style,
        )
