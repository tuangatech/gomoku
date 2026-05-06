"""
test_game.py — Unit tests for game.py.
Run with: python test_game.py
"""

import numpy as np
import sys

from config import Config
from game import GomokuGame


def test(name: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if not condition:
        sys.exit(1)


def run_tests():
    cfg = Config.for_validation()  # 6x6, win=4
    print(f"Testing GomokuGame ({cfg.BOARD_SIZE}x{cfg.BOARD_SIZE}, win={cfg.WIN_LENGTH})")

    # ── Basic setup ───────────────────────────────────────────────────────────
    g = GomokuGame(cfg)
    test("Starts empty", g.board.sum() == 0)
    test("100 legal moves on 6x6", len(g.get_legal_moves()) == 36)
    test("Current player is Black (1)", g.current_player == 1)
    test("Not terminal at start", not g.is_terminal()[0])

    # ── Move mechanics ────────────────────────────────────────────────────────
    g.make_move(0)   # Black at (0,0)
    test("Legal moves decremented", len(g.get_legal_moves()) == 35)
    test("Player switched to White", g.current_player == -1)
    test("Stone placed correctly", g.board[0, 0] == 1)

    # ── Horizontal win ────────────────────────────────────────────────────────
    g = GomokuGame(cfg)
    # Black plays row 0, cols 0-3 (4 in a row)
    # White plays row 1 (filler)
    moves = [0, 6, 1, 7, 2, 8, 3]   # Black: 0,1,2,3 | White: 6,7,8
    for m in moves:
        g.make_move(m)
    test("Horizontal win detected", g._done)
    test("Black wins horizontally", g.winner == 1)

    # ── Vertical win ──────────────────────────────────────────────────────────
    g = GomokuGame(cfg)
    # Black plays col 0, rows 0-3
    # White plays col 1
    moves = [0, 1, 6, 7, 12, 13, 18]   # Black: 0,6,12,18 | White: 1,7,13
    for m in moves:
        g.make_move(m)
    test("Vertical win detected", g._done)
    test("Black wins vertically", g.winner == 1)

    # ── Diagonal win (top-left to bottom-right) ───────────────────────────────
    g = GomokuGame(cfg)
    # Black: (0,0)=0, (1,1)=7, (2,2)=14, (3,3)=21
    # White: (0,1)=1, (1,2)=8, (2,3)=15
    moves = [0, 1, 7, 8, 14, 15, 21]
    for m in moves:
        g.make_move(m)
    test("Diagonal win (\\) detected", g._done)
    test("Black wins diagonally", g.winner == 1)

    # ── Diagonal win (top-right to bottom-left) ───────────────────────────────
    g = GomokuGame(cfg)
    # Black: (0,3)=3, (1,2)=8, (2,1)=13, (3,0)=18
    # White: (0,0)=0, (1,0)=6, (2,0)=12
    moves = [3, 0, 8, 6, 13, 12, 18]
    for m in moves:
        g.make_move(m)
    test("Diagonal win (/) detected", g._done)
    test("Black wins anti-diagonally", g.winner == 1)

    # ── No premature win ──────────────────────────────────────────────────────
    g = GomokuGame(cfg)
    g.make_move(0)   # Black
    g.make_move(6)   # White
    g.make_move(1)   # Black (2 in a row)
    test("No win after 2 in a row", not g._done)

    # ── State tensor shape ─────────────────────────────────────────────────────
    g = GomokuGame(cfg)
    g.make_move(0)
    state = g.get_state_tensor()
    test("State tensor shape [3, S, S]", state.shape == (3, cfg.BOARD_SIZE, cfg.BOARD_SIZE))
    test("Channel 0 has opponent's stone (after move, player switched)", state[1, 0, 0] == 1.0)
    test("Channel 1 empty (no white stones yet)", state[0].sum() == 0.0)

    # ── Clone independence ────────────────────────────────────────────────────
    g = GomokuGame(cfg)
    g.make_move(0)
    g2 = g.clone()
    g2.make_move(1)
    test("Clone is independent", g.board[0, 1] == 0 and g2.board[0, 1] != 0)

    # ── Winning cells ─────────────────────────────────────────────────────────
    g = GomokuGame(cfg)
    for m in [0, 6, 1, 7, 2, 8, 3]:
        g.make_move(m)
    cells = g.get_winning_cells()
    test("Winning cells returned", len(cells) >= cfg.WIN_LENGTH)

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    run_tests()
