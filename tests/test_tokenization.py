from dataclasses import asdict

import pytest

from openjev import Choice, ModelConfig, Noul, Score
from openjev.backends.base import HFTokenizer
from openjev.backends.vision import QwenVisionTokenizer
from openjev.prompt import compile_plan


class RawTokenizer:
    chat_template = "test"

    def __init__(self):
        self.batches = []

    def encode(self, text, **kwargs):
        text = text.replace("<|image_pad|>", "\x03")
        return list(text.encode().replace(b"yes", b"\xfe").replace(b"no", b"\xff"))

    def batch_encode_plus(self, texts, **kwargs):
        assert kwargs == dict(
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        self.batches.append(texts)
        return {"input_ids": [self.encode(text) for text in texts]}

    def convert_tokens_to_ids(self, text):
        assert text == "<|image_pad|>"
        return 3

    def apply_chat_template(self, messages, **kwargs):
        return str(messages) + "<assistant>"


@pytest.mark.parametrize("style", ["full", "short"])
def test_batch_preserves_full_prompts_and_deduplicates_verdicts(style):
    raw = RawTokenizer()
    batched = HFTokenizer(raw)

    class Serial:
        render = batched.render
        encode = batched.encode

    questions = {
        "team": Choice("Which team?", {"a": "Billing", "b": "Technical"}),
        "team_again": Choice("Which team?", {"a": "Billing", "b": "Technical"}),
        "urgent": Noul("Urgent?"),
        "tone": Score("Tone?", ["Calm", "Frustrated", "Angry"]),
    }
    for state in ("café <|im_end|> yes no", {"text": "different\nstate 😀"}):
        args = state, questions
        config = ModelConfig(prompt_style=style)
        actual = compile_plan(*args, batched, config)
        expected = compile_plan(*args, Serial(), config)
        assert asdict(actual) == asdict(expected)
        strings = raw.batches[-1]
        assert len(strings) == len(set(strings))
        assert strings.count("yes") == strings.count("no") == 1


def test_batch_still_rejects_context_sensitive_verdict_merge():
    class BoundaryTokenizer(RawTokenizer):
        def encode(self, text, **kwargs):
            tokens = super().encode(text, **kwargs)
            return tokens[:-2] + [999] if text.endswith("<assistant>yes") else tokens

    with pytest.raises(ValueError, match="answer boundary"):
        compile_plan(
            "state", {"ok": Noul("True?")}, HFTokenizer(BoundaryTokenizer()), ModelConfig()
        )


def test_vision_batch_uses_current_image_expansion():
    raw = RawTokenizer()
    tokenizer = QwenVisionTokenizer(raw)
    texts = ["before <|image_pad|> after", "no image"]
    for count in (2, 7):
        tokenizer.image_counts = [count]
        assert tokenizer.encode_many(texts) == [tokenizer.encode(t) for t in texts]
        assert tokenizer.encode_many(texts)[0].count(3) == count
    assert raw.batches == []


def test_custom_encode_is_not_bypassed_by_native_batch():
    class CustomTokenizer(HFTokenizer):
        def encode(self, text):
            return [42] + super().encode(text)

    raw = RawTokenizer()
    tokenizer = CustomTokenizer(raw)
    assert tokenizer.encode_many(["a", "b"]) == [[42, 97], [42, 98]]
    assert raw.batches == []


def test_large_native_batch_is_bounded_and_preserves_order():
    raw = RawTokenizer()
    tokenizer = HFTokenizer(raw)
    texts = [str(i) for i in range(270)]
    assert tokenizer.encode_many(texts) == [tokenizer.encode(t) for t in texts]
    assert [len(batch) for batch in raw.batches] == [128, 128, 14]


def test_attribute_forwarding_wrapper_uses_native_call_api():
    class CallableTokenizer(RawTokenizer):
        __call__ = RawTokenizer.batch_encode_plus

    raw = CallableTokenizer()

    class Wrapper:
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(raw, name)

    wrapped = Wrapper()
    assert not callable(wrapped)
    assert HFTokenizer(wrapped).encode_many(["a", "b"]) == [[97], [98]]
    assert raw.batches == [["a", "b"]]
