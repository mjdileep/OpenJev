import importlib.util
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("game_2048", ROOT / "examples/game_2048/game.py")
game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(game)


def test_previews_match_original_game():
    if not shutil.which("node"):
        pytest.skip("Node.js required to compare with original 2048 engine")
    rng = random.Random(2026)
    boards = [
        [[rng.choice([0, 0, 2, 4, 8, 16, 32]) for _ in range(4)] for _ in range(4)]
        for _ in range(1000)
    ]
    reference = json.loads(
        subprocess.check_output(
            ["node", str(ROOT / "tests/game_2048_reference.cjs")],
            input=json.dumps(boards).encode(),
            timeout=30,
        )
    )
    for board, moves in zip(boards, reference, strict=True):
        for action, expected in zip(game.ACTIONS, moves, strict=True):
            assert list(game.preview(board, action)) == expected


def test_dead_board_and_illegal_moves():
    dead = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    with pytest.raises(ValueError, match="Game over"):
        game.question_for(dead)
    board = [[2, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4]
    state, questions, forced = game.question_for(board)
    assert state["legal_moves"] == ["right", "down"]
    assert list(questions["move"].criteria) == ["right", "down"]
    assert forced is None


@pytest.mark.parametrize("tile", [True, -2, 3, 2.5, "2", 2**21])
def test_reject_invalid_tiles(tile):
    with pytest.raises(ValueError):
        game.question_for([[tile, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4])


def test_forced_move():
    board = [[0] * 4, [2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4]]
    state, questions, forced = game.question_for(board)
    assert forced == "up"
    assert state["legal_moves"] == ["up"]
    assert questions["move"].type == "noul"
