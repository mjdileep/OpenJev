from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from ..types import GGUF_FILE, GGUF_MODEL, TOKENIZER_MODEL, ModelConfig, TokenScore
from .base import HFTokenizer, verdict_ids


@dataclass(frozen=True)
class GGUFState:
    """Native state includes recurrent memory; truncating KV alone is insufficient."""

    data: bytes
    tokens: tuple[int, ...]


class GGUFTokenizer(HFTokenizer):
    def __init__(self, hf_tokenizer, llama):
        super().__init__(hf_tokenizer)
        self.llama = llama

    def encode(self, text):
        return self.llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)


class GGUFBackend:
    name = "gguf"

    def __init__(self, config: ModelConfig):
        if config.vision:
            raise ValueError(
                "GGUF currently scores text only. Use backend='mlx' or 'transformers' "
                "with vision=True for images."
            )
        if config.load_in_4bit:
            raise ValueError("GGUF quantization is selected by filename, such as Q4_K_M.gguf")
        try:
            import llama_cpp
            import numpy as np
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install GGUF support with pip install 'openjev[gguf]'") from exc
        self.config, self.native, self.np = config, llama_cpp, np
        self.model_id = config.model or GGUF_MODEL
        tokenizer_id = config.tokenizer
        if tokenizer_id is None:
            if self.model_id != GGUF_MODEL:
                raise ValueError("A custom GGUF model requires tokenizer='matching/HF-tokenizer'")
            tokenizer_id = TOKENIZER_MODEL
        model_path = Path(self.model_id).expanduser()
        if model_path.is_file():
            path = str(model_path)
        else:
            filename = config.filename or (GGUF_FILE if self.model_id == GGUF_MODEL else None)
            if filename is None:
                raise ValueError("A GGUF Hugging Face repository requires an exact filename")
            path = hf_hub_download(self.model_id, filename, revision=config.revision)
        gpu_layers = 0 if config.device == "cpu" else config.n_gpu_layers
        if config.device in {"cuda", "metal"}:
            info = llama_cpp.llama_print_system_info().decode().lower()
            if config.device not in info or not llama_cpp.llama_supports_gpu_offload():
                raise RuntimeError(
                    f"llama-cpp-python was not built with {config.device} support. "
                    "Rebuild with GGML_CUDA=ON or GGML_METAL=ON; see README."
                )
            if gpu_layers == 0:
                raise ValueError("A requested GPU device requires n_gpu_layers != 0")
        self.llama = llama_cpp.Llama(
            model_path=path,
            n_ctx=config.n_ctx,
            n_batch=min(config.prefill_chunk_size, config.n_ctx),
            n_gpu_layers=gpu_layers,
            logits_all=False,
            verbose=False,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                revision=config.tokenizer_revision,
                trust_remote_code=False,
            )
            self.tokenizer = GGUFTokenizer(tokenizer, self.llama)
            self.positive_id, self.negative_id = verdict_ids(self.tokenizer, config)
        except Exception:
            self.llama.close()
            raise

    def prepare_request(self, images):
        if images:
            raise ValueError("GGUF image input is not supported; select an image-capable backend")

    def finish_request(self):
        self.llama.reset()

    def _restore(self, state):
        if state is None:
            self.llama.reset()
            return
        buffer = (ctypes.c_uint8 * len(state.data)).from_buffer_copy(state.data)
        restored = self.native.llama_state_set_data(self.llama._ctx.ctx, buffer, len(state.data))
        if restored != len(state.data):
            raise RuntimeError("Could not restore the complete llama.cpp model state")
        self.llama.n_tokens = len(state.tokens)
        self.llama.input_ids[: len(state.tokens)] = state.tokens
        self.llama._requires_eval = True

    def _snapshot(self):
        n = self.native.llama_state_get_size(self.llama._ctx.ctx)
        buffer = (ctypes.c_uint8 * n)()
        written = self.native.llama_state_get_data(self.llama._ctx.ctx, buffer, n)
        if written <= 0 or written > n:
            raise RuntimeError("Could not snapshot the complete llama.cpp model state")
        # Avoid Llama.save_state(): it also copies unused Python logits history.
        return GGUFState(
            bytes(buffer[:written]),
            tuple(self.llama.input_ids[: self.llama.n_tokens].tolist()),
        )

    def prefill(self, tokens, parent=None):
        self._restore(parent)
        if tokens:
            self.llama.eval(tokens)
        return self._snapshot()

    def score(self, parent, suffixes):
        results = []
        np = self.np
        for suffix in suffixes:
            self._restore(parent)
            if not suffix:
                raise ValueError("Candidate suffix must not be empty")
            self.llama.eval(suffix)
            # Read the live final-position logits before ANY sampling, grammar,
            # logit bias, temperature, top-k, or vocabulary restriction.
            ptr = self.native.llama_get_logits_ith(self.llama._ctx.ctx, -1)
            if not ptr:
                raise RuntimeError("llama.cpp returned no logits at the verdict position")
            logits = np.ctypeslib.as_array(ptr, shape=(self.llama.n_vocab(),)).astype(np.float64)
            positive, negative = logits[self.positive_id], logits[self.negative_id]
            maximum = logits.max()
            log_z = maximum + np.log(np.exp(logits - maximum).sum())
            binary = positive - np.logaddexp(positive, negative)
            results.append(
                TokenScore(float(positive - log_z), float(negative - log_z), float(binary))
            )
        return results

    def close(self):
        self.llama.close()
