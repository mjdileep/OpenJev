"""Real Liquid MLX checks: OPENJEV_TEST_LIQUID=1 pytest tests/test_liquid_mlx.py."""

import os

import pytest

from openjev import Choice, DecisionEngine, Noul
from openjev.cli import compare_results
from openjev.prompt import compile_plan

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("OPENJEV_TEST_LIQUID"), reason="Opt-in Liquid model test"),
]


@pytest.fixture(scope="module")
def liquid():
    with DecisionEngine.from_pretrained(
        "LiquidAI/LFM2.5-350M-MLX-4bit",
        revision="f6cb4e006bb7a2d8a6afa14ec0a53e0586f65a5b",
        backend="mlx",
        prompt_style="full",
        score_mode="full",
        batch_size=4,
    ) as engine:
        yield engine


def test_liquid_last_position_projection_matches_native(liquid):
    backend = liquid.backend
    mx = backend.mx
    questions = {"fruit": Noul("Is the object a fruit?")}
    plan = compile_plan("The apple is red.", questions, backend.tokenizer, liquid.config)
    ids = plan.questions[0].prompts[0]
    tokens = mx.array([ids])
    expected = backend.model(tokens)[:, -1, :].astype(mx.float32)
    actual, selected = backend._head(backend.decoder(tokens)[:, -1, :])
    assert not selected  # The original full-vocabulary probability is retained.
    actual = actual.astype(mx.float32)
    for token in (backend.positive_id, backend.negative_id):
        a = actual[:, token] - mx.logsumexp(actual, axis=-1)
        b = expected[:, token] - mx.logsumexp(expected, axis=-1)
        assert float(mx.max(mx.abs(a - b))) < 0.05


def test_liquid_mixed_candidates_and_shared_cache(liquid):
    questions = {
        "color": Choice(
            "What color is the apple?",
            {"red": "Red", "blue": "The apple has a blue skin.", "green": "Green."},
        ),
        "fruit": Noul("Is the object a fruit?"),
    }
    state = "The apple is red. It is a fruit."
    reference = liquid.decide(state, questions, use_cache=False)
    cached = liquid.decide(state, questions)
    batched = liquid.decide(state, {"color": questions["color"]})
    for result in (cached, batched):
        # Quantized batching can choose different reduction kernels.
        assert compare_results(result, reference)["max_candidate_support_difference"] < 0.02
        assert result.answers["color"]["choice"] == reference.answers["color"]["choice"]
        assert result.usage.generated_tokens == 0
    assert cached.usage.reused_input_tokens > 0
    assert batched.usage.candidate_batches == [3]
