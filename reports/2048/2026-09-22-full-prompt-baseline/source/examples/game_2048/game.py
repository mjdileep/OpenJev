"""Public state/action adapter. Previews apply rules only; there is no search policy."""

from openjev import Choice, Noul

ACTIONS = ("up", "right", "down", "left")


def validate_board(board):
    if not isinstance(board, list) or len(board) != 4:
        raise ValueError("Expected a 4 by 4 board")
    for row in board:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("Expected a 4 by 4 board")
        for value in row:
            if type(value) is not int or value < 0 or value > 2**20:
                raise ValueError("Invalid tile")
            if value and (value < 2 or value & (value - 1)):
                raise ValueError("Tiles must be zero or powers of two")


def preview(board, action):
    """Slide/merge without spawning a random tile. Rows run top to bottom."""
    validate_board(board)
    if action not in ACTIONS:
        raise ValueError("Unknown action")
    result = [[0] * 4 for _ in range(4)]
    points = 0
    for line in range(4):
        positions = (
            [(line, i) for i in range(4)]
            if action in ("left", "right")
            else [(i, line) for i in range(4)]
        )
        if action in ("right", "down"):
            positions.reverse()
        values = [board[y][x] for y, x in positions if board[y][x]]
        merged = []
        i = 0
        while i < len(values):
            value = values[i]
            if i + 1 < len(values) and value == values[i + 1]:
                value *= 2
                points += value
                i += 1
            merged.append(value)
            i += 1
        for (y, x), value in zip(positions, merged, strict=False):
            result[y][x] = value
    return result, points


def question_for(board):
    validate_board(board)
    options = {}
    for action in ACTIONS:
        after, points = preview(board, action)
        if after != board:
            empty = sum(value == 0 for row in after for value in row)
            grid = "/".join(",".join(map(str, row)) for row in after)
            options[action] = (
                f"Swipe {action}. Result before random spawn: {grid}; "
                f"merge points={points}; empty cells={empty}."
            )
    if not options:
        raise ValueError("Game over: there are no legal moves")
    instructions = (
        "Which legal swipe is best for reaching 2048 and keeping the game alive? "
        "Combine equal tiles, preserve empty cells, and keep large tiles together at an edge. "
        "Compare the possible moves; avoid scattering large tiles. "
        "Rows are top to bottom, columns left to right; 0 is empty. "
        "After each swipe a random empty cell receives a 2 (90%) or 4 (10%)."
    )
    state = {"board": board, "legal_moves": list(options)}
    if len(options) == 1:
        action = next(iter(options))
        return state, {"move": Noul("Is this the only legal move? " + options[action])}, action
    return state, {"move": Choice(instructions, options)}, None
