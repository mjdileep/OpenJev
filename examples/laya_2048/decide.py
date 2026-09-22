"""Choose one move: python examples/laya_2048/decide.py."""

from laya_player import Player

player = Player()  # Automatically selects CUDA, Apple Silicon MPS, or CPU.
board = [
    [2, 2, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
result = player.decide(board)
print("Move:", result["action"])
print("Choice scores:", result["probabilities"])
print("Seconds:", result["usage"]["elapsed_seconds"])
