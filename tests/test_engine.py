import json
import math
from dataclasses import replace

import pytest

from openjev import Choice, DecisionEngine, ModelConfig, Noul, Score
from openjev.prompt import common_prefix, compile_plan
from openjev.types import TokenScore, parse_questions


class CharacterTokenizer:
    def render(self, system, user):
        return f"<system>{system}</system><user>{user}</user><assistant>"

    def encode(self, text):
        # Model the two single-token verdict words while keeping all other bytes
        # visible to the prompt-reconstruction tests. UTF-8 never uses FE or FF.
        return list(text.encode().replace(b"yes", b"\xfe").replace(b"no", b"\xff"))


class RecordingBackend:
    name = "test"
    model_id = "deterministic-reference"

    def __init__(self, config=None):
        self.config = config or ModelConfig(n_ctx=8192)
        self.tokenizer = CharacterTokenizer()
        self.prefills = []
        self.scored = []
        self.score_calls = []
        self.closed = False

    def prefill(self, tokens, parent=None):
        prefix = tuple(parent or ()) + tuple(tokens)
        self.prefills.append((tuple(parent or ()), tuple(tokens)))
        return prefix

    def score(self, parent, suffixes):
        self.score_calls.append((parent, suffixes))
        scores = []
        for suffix in suffixes:
            full = tuple(parent or ()) + tuple(suffix)
            self.scored.append(full)
            # A score sensitive to every token and its position detects dropped,
            # duplicated, reordered, or contaminated cached prompt tokens.
            value = sum((i + 1) * token for i, token in enumerate(full)) % 173
            p = (value + 1) / 200
            scores.append(TokenScore(math.log(p), math.log(1 - p), math.log(p)))
        return scores

    def prepare_request(self, images):
        pass

    def finish_request(self):
        pass

    def close(self):
        self.closed = True


QUESTIONS = {
    "department": Choice("Who handles this?", {"billing": "Payments", "technical": "Bugs"}),
    "urgent": Noul("Is this urgent?"),
    "tone": Score("How angry?", ["Calm", "Upset", "Furious"]),
}


@pytest.mark.parametrize("strategy", ["shared", "tree"])
def test_cached_tree_reconstructs_every_exact_prompt_and_preserves_scores(strategy):
    backend = RecordingBackend(ModelConfig(cache_strategy=strategy))
    engine = DecisionEngine(backend)
    cached = engine.decide({"text": "urgent café <|im_end|>"}, QUESTIONS)
    cached_prompts = backend.scored[:]
    backend.scored.clear()
    reference = engine.decide({"text": "urgent café <|im_end|>"}, QUESTIONS, use_cache=False)
    assert cached.answers == reference.answers
    assert cached_prompts == backend.scored
    assert len(backend.prefills) == (1 if strategy == "shared" else 1 + len(QUESTIONS))
    assert cached.usage.evaluated_input_tokens < reference.usage.evaluated_input_tokens
    assert cached.usage.reused_input_tokens == (
        cached.usage.uncached_input_tokens - cached.usage.evaluated_input_tokens
    )
    assert cached.usage.generated_tokens == 0
    assert all(
        (n > 0) == (strategy == "tree") for n in cached.usage.question_prefix_tokens.values()
    )
    assert cached.cache_strategy == strategy
    assert reference.usage.reused_input_tokens == 0
    assert not cached.calibrated


def test_shared_scoring_combines_all_questions_in_one_call():
    backend = RecordingBackend()
    result = DecisionEngine(backend).decide("state", QUESTIONS)
    assert len(backend.score_calls) == 1
    assert len(backend.score_calls[0][1]) == result.usage.candidates == 6
    assert len(backend.prefills) == 1


@pytest.mark.parametrize("strategy", ["shared", "tree"])
@pytest.mark.parametrize("question", list(QUESTIONS.values()))
def test_single_question_uses_full_prompt_batch_without_any_prefill(strategy, question):
    backend = RecordingBackend(ModelConfig(cache_strategy=strategy))
    engine = DecisionEngine(backend)
    result = engine.decide("state", {"one": question})
    assert not backend.prefills
    assert len(backend.score_calls) == 1
    assert backend.score_calls[0][0] is None
    assert result.cache_strategy == "single"
    assert result.usage.reused_input_tokens == result.usage.content_prefix_tokens == 0
    assert result.answers == engine.decide("state", {"one": question}, use_cache=False).answers


def test_new_state_never_reuses_previous_state_and_candidate_order_is_independent():
    engine = DecisionEngine(RecordingBackend())
    one = engine.decide("first", QUESTIONS)
    two = engine.decide("second", QUESTIONS)
    assert one.answers != two.answers
    reversed_q = {
        k: (
            Choice(q.instructions, dict(reversed(list(q.criteria.items()))))
            if isinstance(q, Choice)
            else q
        )
        for k, q in QUESTIONS.items()
    }
    again = engine.decide("first", reversed_q)
    assert one.answers == again.answers


def test_score_and_choice_math_are_stable_for_very_small_support():
    class LowSupport(RecordingBackend):
        def score(self, parent, suffixes):
            return [TokenScore(-1000 - i, -0.01, -1000 - i) for i, _ in enumerate(suffixes)]

    result = DecisionEngine(LowSupport()).decide("data", QUESTIONS)
    answer = result.answers["department"]
    assert answer["choice"] == "billing"
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    assert answer["probabilities"]["billing"] == pytest.approx(1 / (1 + math.exp(-1)))
    score = result.answers["tone"]
    assert score["score"] == pytest.approx(
        sum(int(k) * p for k, p in score["probabilities"].items())
    )
    json.dumps(result.to_dict(), allow_nan=False)


def test_full_and_binary_scores_are_distinct_and_expose_missing_full_probability():
    score = TokenScore(math.log(0.2), math.log(0.05), math.log(0.8))
    assert score.to_dict("full")["support"] == pytest.approx(0.2)
    assert score.to_dict("binary")["support"] == pytest.approx(0.8)
    assert score.to_dict("full")["binary_token_mass"] == pytest.approx(0.25)
    selected = TokenScore(None, None, math.log(0.8))
    assert selected.to_dict("binary")["p_positive"] is None
    with pytest.raises(ValueError):
        selected.log_support("full")


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"x": {"type": "other", "instructions": "?"}},
        {"x": Choice("?", {"one": "only"})},
        {"x": Score("?", ["only"])},
        {"x": Noul("?", {"maybe": "maybe"})},
        {"x": Noul("")},
        {"x": {"type": "noul", "instructions": "?", "extra": True}},
    ],
)
def test_bad_schema_rejected_before_inference(questions):
    backend = RecordingBackend()
    with pytest.raises(ValueError):
        DecisionEngine(backend).decide("text", questions)
    assert not backend.prefills


def test_context_limit_rejects_without_truncation():
    backend = RecordingBackend(ModelConfig(n_ctx=20))
    with pytest.raises(ValueError, match="never silently truncated"):
        DecisionEngine(backend).decide("long state", QUESTIONS)
    assert not backend.prefills


def test_verdict_boundary_validation_detects_bpe_merge():
    class BoundaryTokenizer(CharacterTokenizer):
        def encode(self, text):
            tokens = super().encode(text)
            if text.endswith(">yes"):
                return tokens[:-2] + [999]
            return tokens

    with pytest.raises(ValueError, match="answer boundary"):
        compile_plan("state", QUESTIONS, BoundaryTokenizer(), ModelConfig())


def test_common_prefix_uses_tokens_and_handles_singletons():
    assert common_prefix([[1, 2, 3], [1, 2, 4]]) == [1, 2]
    assert common_prefix([[1]]) == [1]
    assert common_prefix([[], [1]]) == []


def test_config_and_lifecycle():
    with pytest.raises(ValueError):
        replace(ModelConfig(), positive_token="no")
    with pytest.raises(ValueError):
        ModelConfig(batch_size=0)
    with pytest.raises(ValueError):
        ModelConfig(score_mode="unknown")
    with pytest.raises(ValueError):
        ModelConfig(cache_strategy="unknown")
    backend = RecordingBackend()
    with DecisionEngine(backend) as engine:
        engine.decide("state", {"x": Noul("True?")})
        with pytest.raises(ValueError, match="vision=True"):
            engine.decide("state", QUESTIONS, images=["x.png"])
    assert backend.closed
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.decide("state", QUESTIONS)


def test_dictionary_questions_match_python_api():
    parsed = parse_questions(
        {"x": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}}
    )
    assert parsed["x"] == Choice("?", {"a": "A", "b": "B"})


def test_failed_image_preparation_cleans_up_before_next_request():
    class PreparationFailure(RecordingBackend):
        pending_images = None

        def prepare_request(self, images):
            self.pending_images = images
            if images:
                raise ValueError("invalid image")

        def finish_request(self):
            self.pending_images = None

    backend = PreparationFailure(ModelConfig(vision=True))
    engine = DecisionEngine(backend)
    with pytest.raises(ValueError, match="invalid image"):
        engine.decide("state", QUESTIONS, images=["bad.png"])
    assert backend.pending_images is None
    assert not backend.prefills
    assert engine.decide("state", QUESTIONS).answers
    with pytest.raises(ValueError, match="list of local"):
        engine.decide("state", QUESTIONS, images="bad.png")
