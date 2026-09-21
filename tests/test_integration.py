"""Opt-in real model checks: OPENJEV_TEST_BACKEND=mlx pytest -m integration."""

import os

import pytest

from openjev import Choice, DecisionEngine, Noul
from openjev.cli import compare_results

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("OPENJEV_TEST_BACKEND"), reason="Opt-in model test"),
]


@pytest.fixture(scope="module")
def engine():
    backend = os.environ["OPENJEV_TEST_BACKEND"]
    device = os.getenv("OPENJEV_TEST_DEVICE", "auto")
    # Identical token-by-token arithmetic isolates cache correctness from the
    # different reduction kernels used by quantized prefill and batched decode.
    with DecisionEngine.from_pretrained(
        backend=backend,
        device=device,
        batch_size=1,
        prefill_chunk_size=1,
    ) as instance:
        yield instance


def test_real_cached_scores_match_independent_prompts(engine):
    questions = {
        "color": Choice(
            "What color is the apple?", {"red": "Red", "blue": "Blue", "green": "Green"}
        ),
        "fruit": Noul("Is the object a fruit?"),
    }
    cached = engine.decide("The apple is red.", questions)
    reference = engine.decide("The apple is red.", questions, use_cache=False)
    check = compare_results(cached, reference)
    assert check["max_candidate_support_difference"] < 1e-5
    assert check["same_choices"]
    assert cached.answers["color"]["choice"] == "red"
    assert cached.usage.reused_input_tokens > 0
    assert cached.usage.generated_tokens == 0
    # Reverse candidate and question order to catch mutable branch contamination.
    swapped = dict(reversed(list(questions.items())))
    swapped["color"] = Choice(
        questions["color"].instructions, {"green": "Green", "blue": "Blue", "red": "Red"}
    )
    changed = engine.decide("The apple is red.", swapped)
    assert compare_results(cached, changed)["max_candidate_support_difference"] < 1e-5


def test_mlx_binary_projection_matches_full_projection():
    if os.environ["OPENJEV_TEST_BACKEND"] != "mlx":
        pytest.skip("Selected-row projection is MLX-specific")
    questions = {"color": Choice("What color?", {"red": "Red", "blue": "Blue"})}
    with DecisionEngine.from_pretrained(backend="mlx", score_mode="binary") as selected:
        a = selected.decide("The apple is red.", questions)
    with DecisionEngine.from_pretrained(
        backend="mlx", score_mode="binary", optimize_head=False
    ) as full:
        b = full.decide("The apple is red.", questions)
    assert compare_results(a, b)["max_candidate_support_difference"] < 0.02
    assert a.answers["color"]["candidates"]["red"]["p_positive"] is None


def test_image_cache_and_image_changes(tmp_path):
    backend = os.environ["OPENJEV_TEST_BACKEND"]
    if backend == "gguf":
        pytest.skip("Use MLX or Transformers for images")
    from PIL import Image

    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", (112, 112), (255, 0, 0)).save(red)
    Image.new("RGB", (112, 112), (0, 0, 255)).save(blue)
    questions = {"color": Choice("What is the main color?", {"red": "Red", "blue": "Blue"})}
    with DecisionEngine.from_pretrained(
        backend=backend,
        vision=True,
        device=os.getenv("OPENJEV_TEST_DEVICE", "auto"),
        batch_size=1,
        prefill_chunk_size=1,
    ) as vision:
        a = vision.decide("Inspect the image.", questions, images=[str(red)])
        b = vision.decide("Inspect the image.", questions, images=[str(red)], use_cache=False)
        tolerance = 1e-5 if backend == "mlx" else 0.002
        assert compare_results(a, b)["max_candidate_support_difference"] < tolerance
        c = vision.decide("Inspect the image.", questions, images=[str(blue)])
        assert a.answers["color"]["choice"] == "red"
        assert c.answers["color"]["choice"] == "blue"
        assert a.usage.generated_tokens == 0
