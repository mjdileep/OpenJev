"""Laya's native choice head applied to legal 2048 moves."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from pathlib import Path

# The sibling example supplies the same tested game rules for both demos.
GAME = Path(__file__).resolve().parents[1] / "game_2048"
sys.path.insert(0, str(GAME))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from game import ACTIONS, preview, validate_board  # noqa: E402

MODEL = "convaiinnovations/laya"
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


def game_question(board):
    """Keep full board previews in state, outside Laya's short option budget."""
    validate_board(board)

    def grid(value):
        return " / ".join(" ".join(map(str, row)) for row in value)

    state = ["Current board: " + grid(board)]
    options = {}
    for action in ACTIONS:
        after, points = preview(board, action)
        if after == board:
            continue
        empty = sum(value == 0 for row in after for value in row)
        state.append(f"After {action}: {grid(after)}")
        options[action] = f"Swipe {action}: merge {points} points; {empty} empty cells."
    if not options:
        raise ValueError("Game over: there are no legal moves")
    question = {
        "type": "choice",
        "instructions": (
            "Choose the best legal swipe to reach 2048 and stay alive. "
            "Merge equal tiles, preserve empty cells, keep large tiles together at an edge. "
            "Compare the resulting boards. Rows run top to bottom, 0 is empty. "
            "After each move a random empty cell gets 2 (90%) or 4 (10%)."
        ),
        "criteria": options,
    }
    return "\n".join(state), {"move": question}


def audit_input(agent, state, questions):
    """Reject silent SDK truncation of instructions, options, or board state."""
    from laya.common import build_sequence, render_options

    q = agent._to_internal(questions["move"])
    tok = agent.tok
    options = [
        [tok.mask_token_id] + tok(" " + option, add_special_tokens=False)["input_ids"]
        for option in render_options(q)
    ]
    head = tok(f"{q['t']} question: {q['ins']}", add_special_tokens=False)["input_ids"]
    state_ids = tok(state, add_special_tokens=False)["input_ids"]
    if any(len(option) > 49 for option in options):
        raise ValueError("Option exceeds the Laya SDK token limit")
    if len(head) + sum(map(len, options)) > agent.cfg["head_max_len"]:
        raise ValueError("Question exceeds the Laya head token budget")
    ids, markers = build_sequence(tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
    expected = [tok.cls_token_id] + head + [tok.sep_token_id]
    for option in options:
        expected.extend(option)
    expected += [tok.sep_token_id] + state_ids + [tok.sep_token_id]
    if ids != expected or len(markers) != len(options):
        raise ValueError("Board exceeds the model token budget; refusing to truncate")
    return {"input_tokens": len(ids), "option_count": len(options), "truncated": False}


class Player:
    def __init__(self, device="auto"):
        if device not in {"auto", "mps", "cuda", "cpu"}:
            raise ValueError("device must be auto, mps, cuda, or cpu")
        try:
            import laya
            import torch
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("Install this example with pip install -e '.[laya]'") from exc

        self.torch = torch
        # Pin the root checkpoint and avoid downloading the sibling models.
        path = snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=[
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/*",
                "encoder/*",
            ],
        )
        self.agent = laya.load(path, device=None if device == "auto" else device)
        self.metadata = {
            "model": MODEL,
            "revision": REVISION,
            "sdk_version": laya.__version__,
            "torch_version": torch.__version__,
            "backend": f"pytorch {self.agent.device}",
            "hardware": f"{platform.system()} · {platform.machine()}",
            "dtype": str(next(self.agent.model.parameters()).dtype),
            "scoring": "Laya native choice head",
            "thinking": False,
            "calibrated_for_2048": False,
            "max_tokens": self.agent.cfg["max_len"],
            "head_max_tokens": self.agent.cfg["head_max_len"],
        }

    def decide(self, board):
        started = time.perf_counter()
        state, questions = game_question(board)
        audit = audit_input(self.agent, state, questions)
        native = self.agent.predict(state, questions)
        answer = native["answers"]["move"]
        action = answer["choice"]
        probabilities = answer["probabilities"]
        legal = questions["move"]["criteria"]
        if action not in legal or set(probabilities) != set(legal):
            raise RuntimeError("Model returned an invalid action set")
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
            raise RuntimeError("Model returned invalid scores")
        if abs(sum(probabilities.values()) - 1) > 0.001:
            raise RuntimeError("Model scores do not sum to one")
        if native["usage"]["input_tokens"] != audit["input_tokens"]:
            raise RuntimeError("Laya token usage differs from the audited input")
        if native["usage"]["output_tokens"] != 0:
            raise RuntimeError("Expected a decision without generated tokens")
        response = {
            "action": action,
            "forced": len(legal) == 1,
            "probabilities": probabilities,
            "usage": {
                "elapsed_seconds": time.perf_counter() - started,
                "generated_tokens": 0,
                "input_tokens": audit["input_tokens"],
                "candidate_batches": [len(legal)],
                "forward_passes": 1,
                "question_batch_size": 1,
                "reused_input_tokens": 0,
            },
            "result": native,
            "board": board,
            "timestamp": time.time(),
            "input": {"state": state, "questions": questions, "audit": audit},
            "metadata": {**self.metadata, "backend": f"pytorch {self.agent.device}"},
        }
        if self.agent.device.type == "mps":
            response["mps_allocated_gb"] = self.torch.mps.current_allocated_memory() / 1e9
        return response
