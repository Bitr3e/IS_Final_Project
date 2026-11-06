"""
Tic-Tac-Toe rules and a perfect (minimax) player.

A board is a tuple of 9 numbers: 1 = X, -1 = O, 0 = empty.
Squares are numbered 0-8 in code and shown to the player as 1-9.
X always moves first.

Shared by all three paradigms: the supervised models learn from this perfect
player, the unsupervised part clusters these boards, and the Q-learning bot
plays this game against itself.
"""

from functools import lru_cache

EMPTY = (0,) * 9
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
CORNERS, EDGES = [0, 2, 6, 8], [1, 3, 5, 7]
PRIORITY = [4, 0, 2, 6, 8, 1, 3, 5, 7]       # tie-break: centre, corners, edges
SYMBOL = {1: "X", -1: "O", 0: "."}
# Demo position used in the supervised and reinforcement parts: O to move,
# X has squares 1 and 2 and threatens to complete the top row on square 3.
EXAMPLE = (1, 1, 0, 0, -1, 0, 0, 0, 0)


def winner(b):
    for a, c, d in LINES:
        if b[a] != 0 and b[a] == b[c] == b[d]:
            return b[a]
    return 0


def to_move(b):
    return 1 if b.count(1) == b.count(-1) else -1


def legal(b):
    return [i for i in range(9) if b[i] == 0]


def play(b, i, p):
    l = list(b)
    l[i] = p
    return tuple(l)


@lru_cache(maxsize=None)
def minimax_value(b):
    """Value for the player to move: +1 forced win, 0 draw, -1 forced loss."""
    if winner(b) != 0:
        return -1                      # the previous player just won
    moves = legal(b)
    if not moves:
        return 0
    p = to_move(b)
    return max(-minimax_value(play(b, m, p)) for m in moves)


def optimal_moves(b):
    p = to_move(b)
    vals = {m: -minimax_value(play(b, m, p)) for m in legal(b)}
    best = max(vals.values())
    return [m for m, v in vals.items() if v == best]


def all_positions():
    """Every board reachable in a legal game (including finished ones)."""
    seen, stack = {EMPTY}, [EMPTY]
    while stack:
        b = stack.pop()
        if winner(b) or not legal(b):
            continue
        p = to_move(b)
        for m in legal(b):
            nb = play(b, m, p)
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return sorted(seen)


def text_board(b):
    rows = []
    for r in range(3):
        cells = [SYMBOL[b[3 * r + c]] if b[3 * r + c] else str(3 * r + c + 1)
                 for c in range(3)]
        rows.append(" " + " | ".join(cells))
    return "\n---+---+---\n".join(rows)
