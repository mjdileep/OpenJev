from __future__ import annotations

from pathlib import Path

from .base import HFTokenizer


class QwenVisionTokenizer(HFTokenizer):
    """Qwen's image placeholders, expanded once image dimensions are known."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.image_counts: list[int] = []
        self.image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def render(self, system, user):
        if self.image_counts:
            user = (
                "<|vision_start|><|image_pad|><|vision_end|>" * len(self.image_counts) + "\n" + user
            )
        return super().render(system, user)

    def encode(self, text):
        tokens = super().encode(text)
        if not self.image_counts or self.image_id not in tokens:
            return tokens
        result = []
        image_index = 0
        for token in tokens:
            if token == self.image_id:
                if image_index >= len(self.image_counts):
                    raise ValueError("Unexpected image placeholder in prompt")
                result.extend([token] * self.image_counts[image_index])
                image_index += 1
            else:
                result.append(token)
        if image_index != len(self.image_counts):
            raise ValueError("Image placeholders do not match supplied images")
        return result


def open_images(paths: list[str]):
    from PIL import Image, ImageOps

    images = []
    for name in paths:
        path = Path(name).expanduser()
        if not path.is_file():
            raise ValueError(f"Image must be an existing local file: {name}")
        with Image.open(path) as source:
            images.append(ImageOps.exif_transpose(source).convert("RGB"))
    return images
