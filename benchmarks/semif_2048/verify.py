"""Replay and verify a completed comparison: python verify.py RUN_DIRECTORY."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from run import (
    METHODS,
    digest,
    initial_board,
    legal_moves,
    preview,
    question_for,
    spawn,
    statistics_ms,
    summarize,
)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    artifacts = json.loads((root / "model-artifacts.json").read_text())
    if protocol.get("comparison_type") == "different_models":
        assert artifacts["comparison_type"] == "different_models"
        for method in METHODS:
            assert artifacts[method]["source"] == protocol[method]["model"]
            assert artifacts[method]["revision"] == protocol[method]["revision"]
        assert protocol["openjev"]["model"] != protocol["semif"]["model"]
    else:
        assert artifacts["identical_loaded_parameters"]
    if (root / "source").is_dir():
        for relative, expected in protocol["source_sha256"].items():
            assert hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() == expected
    corpus = json.loads((root / "fixed-boards.json").read_text())
    assert digest(corpus) == protocol["fixed_board_sha256"]
    if protocol["openjev"].get("cache_strategy") == "shared_prefix_batch":
        cache_check = json.loads((root / "cache-validation.json").read_text())
        assert cache_check["status"] == "verified"
        assert cache_check["boards"] == len(corpus)
        assert not cache_check["included_in_timing"]
    fixed = records(root / "fixed-board-decisions.jsonl")
    assert len(fixed) == len(corpus) * protocol["fixed_board_repeats"] * 2
    expected_games = {(seed, method) for seed in protocol["seed_list"] for method in METHODS}
    assert len(summary["games"]) == len(expected_games)
    game_summaries = {(g["seed"], g["method"]): g for g in summary["games"]}
    assert set(game_summaries) == expected_games
    keys = set()
    for row in fixed:
        assert row["method"] in METHODS
        assert 0 <= row["board_index"] < len(corpus)
        assert 0 <= row["repeat"] < protocol["fixed_board_repeats"]
        key = row["method"], row["board_index"], row["repeat"]
        assert key not in keys
        keys.add(key)
        verify_decision(row, corpus[row["board_index"]], protocol)
    checked = 0
    for seed in protocol["seed_list"]:
        for method in METHODS:
            board, rng = initial_board(seed)
            score = forced = 0
            times = []
            rows = records(root / f"game-{seed}-{method}.jsonl")
            for move, row in enumerate(rows, start=1):
                assert (row["method"], row["seed"], row["move"]) == (method, seed, move)
                assert board == row["board"]
                verify_decision(row, board, protocol)
                after, points = preview(board, row["action"])
                assert after != board
                assert spawn(after, rng) == row["spawn"]
                assert after == row["after"]
                score += points
                assert score == row["score"]
                board = after
                forced += row["forced"]
                if not row["forced"]:
                    times.append(row["elapsed_seconds"])
            recorded = json.loads((root / f"game-{seed}-{method}-summary.json").read_text())
            assert recorded == game_summaries[seed, method]
            assert recorded["initial_board"] == initial_board(seed)[0]
            assert recorded["score"] == score
            assert recorded["moves"] == len(rows)
            assert recorded["forced_moves"] == forced
            assert recorded["max_tile"] == max(map(max, board))
            assert recorded["final_board"] == board
            assert recorded["game_over"] == (not legal_moves(board))
            assert recorded["won"] == (max(map(max, board)) >= 2048)
            assert len(rows) <= protocol["max_moves"]
            assert recorded["capped"] == (
                len(rows) == protocol["max_moves"]
                and not recorded["won"]
                and not recorded["game_over"]
            )
            assert recorded["game_over"] or recorded["won"] or recorded["capped"]
            assert math.isclose(recorded["decision_seconds"], sum(times))
            assert recorded["latency"] == statistics_ms(times)
            checked += len(rows)
    assert summarize(summary["games"], fixed) == summary
    result = {
        "status": "verified",
        "game_transitions": checked,
        "fixed_board_calls": len(fixed),
        "games": len(summary["games"]),
    }
    print(json.dumps(result))
    return result


def verify_decision(row, board, protocol):
    legal = legal_moves(board)
    assert row["action"] in legal
    assert row["forced"] == (len(legal) == 1)
    assert set(row["probabilities"]) == set(legal)
    assert row["generated_tokens"] == 0
    assert row["elapsed_seconds"] >= 0
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in row["probabilities"].values())
    assert math.isclose(sum(row["probabilities"].values()), 1, abs_tol=1e-6)
    if row["forced"]:
        assert row["elapsed_seconds"] == 0
        return
    assert row["action"] == max(row["probabilities"], key=row["probabilities"].get)
    state, questions, _ = question_for(board)
    q = questions["move"]
    common = {
        "state": state,
        "question": q.instructions,
        "options": [{"id": k, "description": v} for k, v in q.criteria.items()],
    }
    assert digest(common) == row["common_input_sha256"]
    if row["method"] == "openjev":
        assert row["native"]["score_mode"] == protocol["openjev"]["score_mode"]
        assert row["native"].get("prompt_style", "full") == protocol["openjev"].get(
            "prompt_style", "full"
        )
        assert row["native"]["usage"]["generated_tokens"] == 0
        strategy = protocol["openjev"].get("cache_strategy", "single")
        assert row["native"]["cache_strategy"] == strategy
        if strategy == "shared_prefix_batch":
            usage = row["native"]["usage"]
            assert usage["content_prefix_tokens"] > 0
            assert usage["candidate_batches"] == [len(legal)]
            assert usage["candidates"] == len(legal)
            assert usage["question_prefix_tokens"] == {"move": 0}
            assert usage["reused_input_tokens"] == (
                (len(legal) - 1) * usage["content_prefix_tokens"]
            )
            assert (
                usage["evaluated_input_tokens"] + usage["reused_input_tokens"]
                == (usage["uncached_input_tokens"])
            )
        assert row["native"]["answers"]["move"]["choice"] == row["action"]
        assert row["native"]["answers"]["move"]["probabilities"] == row["probabilities"]
        assert row["native"]["usage"]["evaluated_input_tokens"] == row["input_tokens"]
    else:
        assert row["native"]["prompt_version"] == "direct-options-v1"
        assert (
            dict(zip(row["native"]["option_ids"], row["native"]["probabilities"], strict=True))
            == row["probabilities"]
        )
        assert row["native"]["input_tokens"] == row["input_tokens"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    verify(args.run_directory)
