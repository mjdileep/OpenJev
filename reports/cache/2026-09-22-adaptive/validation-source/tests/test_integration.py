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


def test_real_adaptive_cache_matches_independent_tokenwise_inference():
    if os.environ["OPENJEV_TEST_BACKEND"] != "mlx":
        pytest.skip("This strict tokenwise diagnostic uses MLX")
    questions = {
        "color": Choice("What color?", {"red": "Red", "blue": "Blue"}),
        "fruit": Noul("Is it fruit?"),
    }
    with DecisionEngine.from_pretrained(
        backend="mlx",
        cache_strategy="adaptive",
        batch_size=1,
        prefill_chunk_size=1,
    ) as adaptive:
        for state in ("A red apple.", "A blue car."):
            cached = adaptive.decide(state, questions)
            reference = adaptive.decide(state, questions, use_cache=False)
            assert compare_results(cached, reference)["max_candidate_support_difference"] < 1e-5
            assert compare_results(cached, reference)["same_choices"]
            assert cached.usage.instruction_prefix_tokens > 0


@pytest.mark.parametrize("style", ["full", "short"])
def test_mlx_binary_projection_matches_full_projection(style):
    if os.environ["OPENJEV_TEST_BACKEND"] != "mlx":
        pytest.skip("Selected-row projection is MLX-specific")
    questions = {"color": Choice("What color?", {"red": "Red", "blue": "Blue"})}
    with DecisionEngine.from_pretrained(
        backend="mlx", score_mode="binary", prompt_style=style
    ) as selected:
        a = selected.decide("The apple is red.", questions)
    with DecisionEngine.from_pretrained(
        backend="mlx", score_mode="binary", optimize_head=False, prompt_style=style
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
    questions = {
        "color": Choice("What is the main color?", {"red": "Red", "blue": "Blue"}),
        "red": Noul("Is the image primarily red?"),
    }
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
        assert a.usage.reused_input_tokens > 0
        assert a.usage.generated_tokens == 0


@pytest.mark.parametrize("vision", [False, True], ids=["text", "image"])
@pytest.mark.parametrize("style", ["full", "short"])
def test_real_mixed_length_batches_and_single_question(vision, style):
    backend = os.environ["OPENJEV_TEST_BACKEND"]
    if backend == "gguf":
        pytest.skip("GGUF scores branches serially")
    questions = {
        "color": Choice(
            "What is the main color?", {"red": "Red", "blue": "Blue", "green": "Mostly green"}
        ),
        "red": Noul("Is it primarily red?"),
    }
    images = ["examples/images/red-square.png"] if vision else []
    state = "Inspect the image." if vision else "The square is red."
    # This checks the normal batched execution path. Quantized MLX kernels
    # differ by shape; the strict singleton cache test above isolates that from
    # correctness, while tiny FP32 hybrid-model tests check padding to 1e-5.
    tolerance = 0.08 if backend == "mlx" else 0.002
    with DecisionEngine.from_pretrained(
        backend=backend,
        vision=vision,
        prompt_style=style,
        score_mode="binary" if style == "short" else "full",
        device=os.getenv("OPENJEV_TEST_DEVICE", "auto"),
    ) as engine:
        batched = engine.decide(state, questions, images=images)
        independent = engine.decide(state, questions, images=images, use_cache=False)
        assert batched.usage.candidate_batches == [4]
        assert batched.usage.padding_tokens > 0
        assert batched.usage.content_prefix_tokens > 0
        assert batched.answers["color"]["choice"] == "red"
        assert compare_results(batched, independent)["max_candidate_support_difference"] < tolerance
        swapped = {
            "red": questions["red"],
            "color": Choice(
                questions["color"].instructions,
                dict(reversed(list(questions["color"].criteria.items()))),
            ),
        }
        reordered = engine.decide(state, swapped, images=images)
        assert compare_results(batched, reordered)["max_candidate_support_difference"] < 1e-5
        single = engine.decide(state, {"color": questions["color"]}, images=images)
        reference = engine.decide(
            state, {"color": questions["color"]}, images=images, use_cache=False
        )
        assert single.cache_strategy == "single"
        assert single.usage.content_prefix_tokens == single.usage.reused_input_tokens == 0
        assert single.usage.candidate_batches == [3]
        assert single.answers["color"]["choice"] == "red"
        assert compare_results(single, reference)["max_candidate_support_difference"] < tolerance
