from dataclasses import replace

import pytest
from test_engine import QUESTIONS, CharacterTokenizer, RecordingBackend

from openjev import Choice, DecisionEngine, ModelConfig, Noul
from openjev.cache_plan import CostModel, plan_batches
from openjev.prompt import compile_plan


@pytest.mark.parametrize("style", ["full", "short"])
def test_permanent_cache_reconstructs_prompts_across_requests_and_question_order(style):
    backend = RecordingBackend(ModelConfig(cache_strategy="adaptive", prompt_style=style))
    engine = DecisionEngine(backend)
    assert len(backend.prefills) == 1
    retained = engine._instruction_cache
    for state in ("First café <|im_end|>", {"context": "Entirely different"}, "First again"):
        backend.scored.clear()
        cached = engine.decide(state, QUESTIONS)
        reconstructed = sorted(backend.scored)
        backend.scored.clear()
        reference = engine.decide(state, QUESTIONS, use_cache=False)
        assert cached.answers == reference.answers
        assert reconstructed == sorted(backend.scored)
        assert cached.usage.instruction_prefix_tokens == len(engine._instruction_tokens)
        assert cached.usage.reused_input_tokens > 0
        assert engine._instruction_cache == retained
        swapped = dict(reversed(list(QUESTIONS.items())))
        swapped["department"] = replace(
            QUESTIONS["department"],
            criteria=dict(reversed(list(QUESTIONS["department"].criteria.items()))),
        )
        assert engine.decide(state, swapped).answers == cached.answers
    engine.close()
    assert engine._instruction_cache is None


def test_single_candidate_uses_one_pass_from_permanent_cache():
    backend = RecordingBackend(ModelConfig(cache_strategy="adaptive"))
    engine = DecisionEngine(backend)
    result = engine.decide("A red apple", {"fruit": Noul("Is it fruit?")})
    assert len(backend.prefills) == 1  # Only initialization, never per-request prefill.
    assert len(backend.score_calls) == 1
    assert backend.score_calls[0][0] == engine._instruction_cache
    assert not result.usage.cache_prefill_tokens
    assert result.usage.reused_input_tokens == len(engine._instruction_tokens)


def test_one_question_caches_question_and_keeps_candidates_in_one_batch():
    backend = RecordingBackend(ModelConfig(cache_strategy="adaptive"))
    engine = DecisionEngine(backend)
    questions = {"q": Choice("Shared rules. " * 50, {"a": "Alpha", "b": "Beta", "c": "Gamma"})}
    result = engine.decide("state", questions)
    plan = compile_plan("state", questions, backend.tokenizer, backend.config)
    assert len(backend.score_calls) == 1
    assert len(backend.score_calls[0][1]) == 3
    assert min(result.usage.scoring_prefix_tokens) >= len(plan.questions[0].prefix)
    assert result.usage.question_prefix_tokens["q"] > 0
    assert result.answers == engine.decide("state", questions, use_cache=False).answers


def test_short_unrelated_questions_share_batch_without_cross_question_normalization():
    backend = RecordingBackend(ModelConfig(cache_strategy="adaptive"))
    engine = DecisionEngine(backend)
    questions = {"a": Noul("A?"), "z": Noul("Z?")}
    result = engine.decide("state", questions)
    assert len(backend.score_calls) == 1
    assert len(backend.score_calls[0][1]) == 2
    assert result.answers == engine.decide("state", questions, use_cache=False).answers


def test_planner_selects_long_branches_but_pools_short_branches():
    prompts = [
        [1] * 100 + [2] * 800 + [10],
        [1] * 100 + [2] * 800 + [11],
        [1] * 100 + [3] * 800 + [12],
        [1] * 100 + [3] * 800 + [13],
        [1] * 100 + [4, 14],
        [1] * 100 + [5, 15],
    ]
    plan = plan_batches(prompts, 5, CostModel())
    assert plan.depth == 100
    assert len(plan.children) == 2
    assert set(plan.pending) == {4, 5}
    assert all(child.depth == 900 for child in plan.children)


def test_duplicate_prompts_keep_a_final_token_and_all_rows():
    prompts = [[1] * 200 + [2]] * 3
    plan = plan_batches(prompts, 0, CostModel())
    assert plan.depth == 200
    assert plan.pending == (0, 1, 2)
    assert plan_batches([[1, 2, 3]], 1, CostModel()).depth == 1
    with pytest.raises(ValueError):
        plan_batches([[1, 2], [9, 2]], 1, CostModel())


def test_tiny_intermediate_edge_is_not_materialized_as_an_extra_cache():
    prompts = [
        [1] * 100 + [2] * 3 + [3] * 800 + [10],
        [1] * 100 + [2] * 3 + [3] * 800 + [11],
        [1] * 100 + [2] * 3 + [4] * 800 + [12],
        [1] * 100 + [2] * 3 + [4] * 800 + [13],
        [1] * 100 + [5, 14],
    ]
    plan = plan_batches(prompts, 5, CostModel())

    def check(node, parent):
        assert node.depth - parent != 3
        for child in node.children:
            check(child, node.depth)

    check(plan, 5)


def test_boundary_mismatch_discards_static_cache_without_truncating_it():
    class UnusualTemplate(CharacterTokenizer):
        def encode(self, text):
            tokens = super().encode(text)
            return ([999] + tokens[1:]) if '"special"' in text else tokens

    backend = RecordingBackend(ModelConfig(cache_strategy="adaptive"))
    backend.tokenizer = UnusualTemplate()
    engine = DecisionEngine(backend)
    cached = engine.decide("special", QUESTIONS)
    assert cached.usage.instruction_prefix_tokens == 0
    assert cached.answers == engine.decide("special", QUESTIONS, use_cache=False).answers
    assert engine.decide("normal", QUESTIONS).usage.instruction_prefix_tokens > 0


def test_vision_uses_existing_image_aware_path():
    backend = RecordingBackend(ModelConfig(cache_strategy="adaptive", vision=True))
    engine = DecisionEngine(backend)
    assert not backend.prefills
    result = engine.decide("state", QUESTIONS)
    assert result.cache_strategy == "shared"
    assert result.usage.instruction_prefix_tokens == 0
    assert result.answers == engine.decide("state", QUESTIONS, use_cache=False).answers
