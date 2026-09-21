"""MLX checks: OPENJEV_TEST_BONSAI=1 pytest tests/test_bonsai.py.

Add OPENJEV_TEST_BONSAI_MODEL=1 for the cached 27B integration check.
"""

import json
import os
from dataclasses import replace

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENJEV_TEST_BONSAI"), reason="Opt-in Apple Silicon test"
)


def test_packed_transforms_and_selected_head():
    import mlx.core as mx
    import numpy as np

    from openjev.backends.bonsai import Packed, transform

    rng = np.random.default_rng(42)
    values = rng.normal(size=(8, 1024)).astype(np.float16)
    arrays = mx.quantize(mx.array(values), group_size=128, bits=2)
    signs = mx.array(rng.choice([-1, 1], 1024).astype(np.float32))
    x = mx.array(rng.normal(size=(3, 1024)).astype(np.float16))
    # Independent Sylvester matrix checks transform direction, sign placement, and scale.
    h = np.ones((1, 1), dtype=np.float32)
    while len(h) < 1024:
        h = np.block([[h, h], [h, -h]])
    expected = (np.asarray(x).astype(np.float32) * np.asarray(signs)) @ h / 32
    np.testing.assert_allclose(np.asarray(transform(x, 1024, signs)), expected, atol=0.002)
    embedding = Packed(arrays, 1024, signs, True)
    linear = Packed(arrays, 1024, signs, False)
    dense = np.asarray(mx.dequantize(*arrays, group_size=128, bits=2)).astype(np.float32)
    ids = mx.array([2, 5])
    expected_embedding = (dense[[2, 5]] @ h / 32) * np.asarray(signs)
    np.testing.assert_allclose(np.asarray(embedding(ids)), expected_embedding, atol=0.003)
    full = linear(x)
    selected = linear.project_selected(x, ids)
    np.testing.assert_allclose(np.asarray(selected), np.asarray(full[:, [2, 5]]), atol=0.05)


def test_loader_rejects_unknown_contract(tmp_path):
    from openjev.backends.bonsai import load_bonsai

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "prism_hadamard_qwen35"}))
    with pytest.raises(ValueError, match="Unsupported Bonsai pack"):
        load_bonsai(tmp_path)


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENJEV_TEST_BONSAI_MODEL"), reason="Requires Bonsai weights")
def test_real_bonsai_caching_binary_and_tokenizer():
    import mlx.core as mx
    from tokenizers import Tokenizer

    from openjev import Choice, DecisionEngine, Noul
    from openjev.cli import compare_results

    mx.set_cache_limit(256 * 1024**2)
    with DecisionEngine.from_pretrained(
        "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        revision="3f926b415992eaa2ae9dd7b573706494d6bbf787",
        backend="mlx",
        batch_size=3,
    ) as engine:
        tokenizer = engine.backend.tokenizer.raw
        raw = Tokenizer.from_file(str(tokenizer.name_or_path) + "/tokenizer.json")
        for text in ["yes", "no", "2048 128 12.34\n", "こんにちは € é 中"]:
            assert (
                tokenizer.encode(text, add_special_tokens=False)
                == raw.encode(text, add_special_tokens=False).ids
            )
        rendered = engine.backend.tokenizer.render("Output yes or no.", "Is 2+2=4?")
        assert rendered.endswith("<think>\n\n</think>\n\n")
        state = "The apple is red."
        questions = {
            "color": Choice("What color is the apple?", {"red": "Red", "blue": "Blue"}),
            "fruit": Noul("Is the object a fruit?"),
        }
        full = engine.decide(state, questions)
        reference = engine.decide(state, questions, use_cache=False)
        assert full.answers["color"]["choice"] == "red"
        assert full.answers["fruit"]["noul"] > 0.5
        assert full.usage.reused_input_tokens > 0
        assert full.usage.generated_tokens == 0
        assert compare_results(full, reference)["max_candidate_support_difference"] < 0.03
        binary_config = replace(engine.config, score_mode="binary")
        engine.config = engine.backend.config = binary_config
        binary = engine.decide(state, questions)
        for key in questions:
            for label, candidate in binary.answers[key]["candidates"].items():
                expected = full.answers[key]["candidates"][label]["p_positive_given_binary"]
                assert candidate["support"] == pytest.approx(expected, abs=0.01)
                assert candidate["p_positive"] is None
