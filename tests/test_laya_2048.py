import importlib.util
import json
import math
import os
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "laya_player", ROOT / "examples/laya_2048/laya_player.py"
)
laya_game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(laya_game)


def test_recorded_laya_game_and_reported_metrics():
    """The published sample must replay under the same rules and support its claims."""
    run = json.loads((ROOT / "docs/laya-2048-run.json").read_text())
    history, summary = run["history"], run["summary"]
    score = 0
    for i, row in enumerate(history):
        _, questions = laya_game.game_question(row["board"])
        legal = questions["move"]["criteria"]
        assert row["action"] in legal
        assert set(row["probabilities"]) == set(legal)
        assert sum(row["probabilities"].values()) == pytest.approx(1, abs=0.001)
        assert row["forced"] == (len(legal) == 1)
        moved, points = laya_game.preview(row["board"], row["action"])
        assert moved != row["board"]
        after = row["after"]
        changes = [
            (before, new)
            for a, b in zip(moved, after, strict=True)
            for before, new in zip(a, b, strict=True)
            if before != new
        ]
        assert len(changes) == 1 and changes[0][0] == 0 and changes[0][1] in (2, 4)
        if i + 1 < len(history):
            assert after == history[i + 1]["board"]
        score += points
        assert row["score"] == score
        assert row["usage"]["generated_tokens"] == 0
        assert row["usage"]["forward_passes"] == 1
    assert score == summary["score"]
    assert len(history) == summary["moves"]
    final = history[-1]["after"]
    assert max(map(max, final)) == summary["best_tile"]
    assert all(laya_game.preview(final, a)[0] == final for a in laya_game.ACTIONS)
    assert summary["game_over"] and not summary["won"]
    ms = [row["usage"]["elapsed_seconds"] * 1000 for row in history]
    assert statistics.mean(ms) == pytest.approx(summary["mean_ms"])
    assert statistics.median(ms) == pytest.approx(summary["median_ms"])
    assert sorted(ms)[math.ceil(len(ms) * 0.95) - 1] == pytest.approx(summary["p95_ms"])
    assert summary["revision"] == laya_game.REVISION


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENJEV_TEST_LAYA"), reason="Opt-in Laya model test")
def test_real_laya_choices_forced_move_and_truncation():
    player = laya_game.Player(device=os.getenv("OPENJEV_TEST_DEVICE", "auto"))
    board = [[0, 0, 0, 0], [0, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    result = player.decide(board)
    assert not result["forced"]
    assert set(result["probabilities"]) == set(laya_game.ACTIONS)
    assert result["usage"]["generated_tokens"] == 0
    assert result["input"]["audit"]["truncated"] is False
    forced = player.decide([[0] * 4, [2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4]])
    assert forced["forced"] and forced["action"] == "up"
    assert forced["probabilities"] == {"up": 1.0}
    state, questions = laya_game.game_question(board)
    with pytest.raises(ValueError, match="refusing to truncate"):
        laya_game.audit_input(player.agent, state * 20, questions)
