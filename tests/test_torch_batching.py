"""Small real hybrid models: no weight downloads or GPU required."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import (  # noqa: E402
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from openjev.backends.transformers import TransformersBackend  # noqa: E402
from openjev.types import ModelConfig  # noqa: E402


@pytest.fixture(params=[False, True], ids=["text", "vision"])
def backend(request):
    torch.manual_seed(42)
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
    )
    b = TransformersBackend.__new__(TransformersBackend)
    b.torch, b.device = torch, torch.device("cpu")
    b.config = ModelConfig(backend="transformers", vision=request.param, batch_size=3)
    if request.param:
        vision = Qwen3_5VisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            out_hidden_size=32,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=1,
            num_position_embeddings=16,
        )
        b.model = Qwen3_5ForConditionalGeneration(
            Qwen3_5Config(
                text_config=text,
                vision_config=vision,
                image_token_id=60,
                video_token_id=63,
                vision_start_token_id=61,
                vision_end_token_id=62,
            )
        ).eval()
    else:
        b.model = Qwen3_5ForCausalLM(text).eval()
    b.decoder = b.model.model
    b.keep_logits = True
    b.tokenizer = SimpleNamespace(raw=SimpleNamespace(pad_token_id=0), image_id=60, image_counts=[])
    b.positive_id, b.negative_id = 3, 4
    b.processor = None
    b.prepare_request([])
    yield b
    b.close()


def as_tensor(scores):
    return torch.tensor([[s.log_p_positive, s.log_p_negative, s.log_p_binary] for s in scores])


def cache_tensors(cache):
    result = []
    for layer in cache.layers:
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            values = value.values() if isinstance(value, dict) else [value]
            result.extend(v for v in values if isinstance(v, torch.Tensor))
    return result


def test_padded_batches_match_independent_prompts_and_do_not_mutate_parent(backend):
    prefix = [5, 8, 13, 9, 6]
    # Includes one-token suffix, padding, length sorting, and a final partial batch.
    suffixes = [[15, 16, 17, 18, 19], [20], [23, 22, 21], [14, 9]]
    parent = backend.prefill(prefix)
    before = [v.clone() for v in cache_tensors(parent.cache)]
    actual = backend.score(parent, suffixes)
    assert backend.stats.sizes == [3, 1]
    assert backend.stats.padding_tokens == 3
    for a, b in zip(before, cache_tensors(parent.cache), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    expected = [backend.score(None, [prefix + suffix])[0] for suffix in suffixes]
    torch.testing.assert_close(as_tensor(actual), as_tensor(expected), rtol=1e-5, atol=1e-5)
    reversed_scores = backend.score(parent, list(reversed(suffixes)))
    torch.testing.assert_close(
        as_tensor(actual), as_tensor(reversed_scores[::-1]), rtol=1e-5, atol=1e-5
    )


def test_single_question_batch_is_one_forward_and_prefill_skips_head(backend):
    forwards, projections = [], []
    decoder_hook = backend.model.model.register_forward_hook(lambda *args: forwards.append(1))
    head_hook = backend.model.lm_head.register_forward_hook(lambda *args: projections.append(1))
    try:
        backend.prefill([5, 8, 13])
        assert len(forwards) == 1 and not projections
        forwards.clear()
        rows = [[8, 9, 10], [9, 8], [7, 8, 9, 10]]
        result = backend.score(None, rows)
        assert len(forwards) == len(projections) == 1
        assert len(result) == 3
    finally:
        decoder_hook.remove()
        head_hook.remove()


def test_generic_head_path_and_batch_size_one_match_optimized_path(backend):
    parent = backend.prefill([7, 9, 8])
    rows = [[5, 6], [7], [8, 9, 10]]
    expected = backend.score(parent, rows)
    backend.decoder = None
    actual = backend.score(parent, rows)
    torch.testing.assert_close(as_tensor(actual), as_tensor(expected), rtol=1e-5, atol=1e-5)
    backend.config = replace(backend.config, batch_size=1)
    singleton = backend.score(parent, rows)
    torch.testing.assert_close(as_tensor(singleton), as_tensor(expected), rtol=1e-5, atol=1e-5)


def test_image_prefix_and_full_prompt_batches_preserve_rope_offsets(backend):
    if not backend.config.vision:
        pytest.skip("Vision model only")
    backend.image_data = {
        "pixel_values": torch.randn(4, 12),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    prefix = [1, 61, 60, 60, 60, 60, 62, 2]
    suffixes = [[5, 6, 7], [8], [10, 11]]
    parent = backend.prefill(prefix)
    delta = copy.deepcopy(parent.rope_delta)
    shared = backend.score(parent, suffixes)
    full_batch = backend.score(None, [prefix + row for row in suffixes])
    independent = [backend.score(None, [prefix + row])[0] for row in suffixes]
    torch.testing.assert_close(parent.rope_delta, delta, rtol=0, atol=0)
    torch.testing.assert_close(as_tensor(shared), as_tensor(independent), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(as_tensor(full_batch), as_tensor(independent), rtol=1e-5, atol=1e-5)
