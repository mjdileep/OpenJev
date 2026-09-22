"""Actual MLX cache isolation: OPENJEV_TEST_BACKEND=mlx pytest tests/test_mlx_cache.py."""

import copy
import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("OPENJEV_TEST_BACKEND") != "mlx", reason="Opt-in MLX cache test"),
]


@pytest.mark.parametrize("filled", [False, True])
def test_forked_kv_matches_merge_and_keeps_parent_and_sibling_independent(filled):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from openjev.backends.mlx import fork_cache

    parent = KVCache()
    if filled:
        x = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float16)
        parent.update_and_fetch(x, x + 1)
    child = fork_cache(parent, 3)
    expected = type(parent).merge([copy.deepcopy(parent) for _ in range(3)])
    sibling = fork_cache(parent, 3)
    for cache in (child, expected):
        x = mx.arange(48).reshape(3, 2, 2, 4).astype(mx.float16)
        cache.update_and_fetch(x, x + 5)
    assert mx.array_equal(child.keys, expected.keys).item()
    assert mx.array_equal(child.values, expected.values).item()
    assert mx.array_equal(child.offset, expected.offset).item()
    child.keys[0] = -1
    mx.eval(child.keys)
    assert not mx.all(child.keys[1] == -1).item()
    if filled:
        assert mx.array_equal(parent.keys[..., :3, :], sibling.keys[:1]).item()
        assert parent.offset == sibling._idx == 3
    else:
        assert parent.keys is None and sibling.keys is None


@pytest.mark.parametrize("filled", [False, True])
def test_forked_recurrent_and_convolution_states_are_independent(filled):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    from openjev.backends.mlx import fork_cache

    parent = ArraysCache(2)
    if filled:
        parent[0] = mx.arange(12).reshape(1, 3, 4).astype(mx.float16)
        parent[1] = mx.arange(24).reshape(1, 2, 3, 4).astype(mx.float32)
    child = fork_cache(parent, 4)
    expected = type(parent).merge([copy.deepcopy(parent) for _ in range(4)])
    sibling = fork_cache(parent, 4)
    if filled:
        for i in range(2):
            assert mx.array_equal(child[i], expected[i]).item()
            child[i][0] = -1
            mx.eval(child[i])
            assert mx.array_equal(child[i][1:2], parent[i]).item()
            assert mx.array_equal(sibling[i][:1], parent[i]).item()
        assert parent.left_padding is None and parent.lengths is None
    else:
        assert child.cache == [None, None]
        assert mx.array_equal(child.left_padding, expected.left_padding).item()
        child[0] = mx.ones((4, 3, 4))
        assert parent[0] is None and sibling[0] is None


def test_unknown_cache_uses_defensive_merge():
    from openjev.backends.mlx import fork_cache

    class CustomCache:
        def __init__(self):
            self.data = [1]

        @classmethod
        def merge(cls, caches):
            caches[0].data.append(2)
            return caches

    parent = CustomCache()
    branches = fork_cache(parent, 3)
    assert parent.data == branches[1].data == branches[2].data == [1]
    assert branches[0].data == [1, 2]
