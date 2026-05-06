"""
game.py — Gomoku game logic.

Board-size agnostic: driven entirely by Config.BOARD_SIZE and Config.WIN_LENGTH.
Uses NumPy for fast array operations throughout.

Board encoding:
    0  = empty
    1  = Black (first player)
   -1  = White (second player)

State tensor (for neural network input):
    Channel 0: current player's stones  (binary)
    Channel 1: opponent's stones        (binary)
    Channel 2: turn indicator           (all 1s = Black's turn, all 0s = White's turn)
"""

import numpy as np
from typing import List, Optional, Tuple
from copy import deepcopy

from config import Config


class GomokuGame:
    """
    Immutable-friendly Gomoku game state.
    Call .clone() before make_move() to preserve the original state (used heavily in MCTS).
    """

    # Directions to check for winning lines: (row_delta, col_delta)
    _DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def __init__(self, config: Config):
        self.cfg = config
        self.size = config.BOARD_SIZE
        self.win_len = config.WIN_LENGTH

        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player: int = 1          # 1 = Black, -1 = White
        self.move_history: List[int] = []     # flat action indices
        self.winner: Optional[int] = None     # 1, -1, or 0 (draw)
        self._done: bool = False

    # ── Core interface ────────────────────────────────────────────────────────

    def get_legal_moves(self) -> List[int]:
        """Return flat indices of all empty cells."""
        if self._done:
            return []
        return [int(i) for i in np.where(self.board.flatten() == 0)[0]]

    def make_move(self, action: int) -> None:
        """
        Place current player's stone at flat index `action`.
        Mutates in place. Use .clone() first if you need to preserve state.
        """
        assert not self._done, "Game is already over"
        row, col = divmod(action, self.size)
        assert self.board[row, col] == 0, f"Cell ({row},{col}) is already occupied"

        self.board[row, col] = self.current_player
        self.move_history.append(action)

        if self._check_win(row, col):
            self.winner = self.current_player
            self._done = True
        elif len(self.move_history) == self.size * self.size:
            self.winner = 0   # draw
            self._done = True
        else:
            self.current_player = -self.current_player

    def is_terminal(self) -> Tuple[bool, float]:
        """
        Returns (done, outcome_for_current_player).
        outcome: +1 = current player wins, -1 = current player loses, 0 = draw.

        NOTE: Call BEFORE switching player after make_move, or interpret carefully.
        We return the outcome from the perspective of the player who just moved
        when the game ends — handled correctly in self_play.py.
        """
        if not self._done:
            return False, 0.0
        if self.winner == 0:
            return True, 0.0
        # winner is the player who just moved (current_player was flipped only if game continues)
        # After make_move, if game ended, current_player was NOT flipped
        return True, 1.0 if self.winner == self.current_player else -1.0

    def get_state_tensor(self) -> np.ndarray:
        """
        Returns [3, BOARD_SIZE, BOARD_SIZE] float32 tensor for network input.
        Always from the perspective of the CURRENT player.
        """
        cp = self.current_player
        ch0 = (self.board == cp).astype(np.float32)       # current player's stones
        ch1 = (self.board == -cp).astype(np.float32)      # opponent's stones
        ch2 = np.full((self.size, self.size),
                      1.0 if cp == 1 else 0.0, dtype=np.float32)  # turn indicator
        return np.stack([ch0, ch1, ch2], axis=0)           # [3, S, S]

    def clone(self) -> "GomokuGame":
        """Fast deep copy for MCTS tree exploration."""
        g = GomokuGame.__new__(GomokuGame)
        g.cfg = self.cfg
        g.size = self.size
        g.win_len = self.win_len
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.move_history = self.move_history.copy()
        g.winner = self.winner
        g._done = self._done
        return g

    # ── Win detection ─────────────────────────────────────────────────────────

    def _check_win(self, row: int, col: int) -> bool:
        """
        Check if the last move at (row, col) created a winning line.
        Only scans through the last-placed stone — O(win_len * 4) not O(board²).
        """
        player = self.board[row, col]
        for dr, dc in self._DIRECTIONS:
            count = 1
            # scan in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r += dr; c += dc
            # scan in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r -= dr; c -= dc
            if count >= self.win_len:
                return True
        return False

    def get_winning_cells(self) -> List[Tuple[int, int]]:
        """
        Return (row, col) pairs of the winning five-in-a-row, for UI highlighting.
        Returns [] if no winner yet.
        """
        if self.winner is None or self.winner == 0:
            return []
        if not self.move_history:
            return []

        last_action = self.move_history[-1]
        row, col = divmod(last_action, self.size)
        player = self.winner

        for dr, dc in self._DIRECTIONS:
            line = [(row, col)]
            for sign in [1, -1]:
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                    line.append((r, c))
                    r += sign * dr; c += sign * dc
            if len(line) >= self.win_len:
                return line
        return []

    # ── Utility ───────────────────────────────────────────────────────────────

    def render(self) -> str:
        """ASCII board for debugging."""
        symbols = {0: "·", 1: "●", -1: "○"}
        header = "  " + " ".join(f"{c:2}" for c in range(self.size))
        rows = [header]
        for r in range(self.size):
            row_str = f"{r:2} " + "  ".join(symbols[self.board[r, c]] for c in range(self.size))
            rows.append(row_str)
        turn = "Black" if self.current_player == 1 else "White"
        rows.append(f"Turn: {turn}  Moves: {len(self.move_history)}")
        if self._done:
            if self.winner == 0:
                rows.append("Result: Draw")
            else:
                w = "Black" if self.winner == 1 else "White"
                rows.append(f"Result: {w} wins!")
        return "\n".join(rows)

    def __repr__(self) -> str:
        return self.render()
