"""
play_gui.py — Pygame desktop UI: Human vs AlphaZero bot.

Usage:
    python play_gui.py --mode quick       # play vs 5x5 champion
    python play_gui.py --mode validate    # play vs 6x6 champion
    python play_gui.py --mode full        # play vs 10x10 champion
    python play_gui.py --sims 400         # adjust bot thinking depth

Controls:
    Click        → place your stone
    R            → new game
    U            → undo last move pair (your move + bot move)
    H            → toggle policy heatmap (see what bot is considering)
    Q / Escape   → quit
"""

import argparse
import sys
import threading
import numpy as np
import torch

try:
    import pygame
except ImportError:
    print("ERROR: pygame not installed. Run: pip install pygame")
    sys.exit(1)

from config import Config
from game import GomokuGame
from mcts import MCTS
from network import PolicyValueNet
from utils import load_champion


# ── Visual constants ──────────────────────────────────────────────────────────

CELL      = 60      # pixels per cell
MARGIN    = 40      # board margin
PANEL_H   = 100     # bottom panel height

# Colors
BG_COLOR       = (240, 217, 181)   # classic board wood color
GRID_COLOR     = (0, 0, 0)
BLACK_STONE    = (30, 30, 30)
WHITE_STONE    = (240, 240, 240)
WIN_HIGHLIGHT  = (255, 80, 0)
HEAT_COLOR     = (255, 0, 0)       # heatmap overlay color
TEXT_COLOR     = (20, 20, 20)
PANEL_BG       = (50, 50, 50)
PANEL_TEXT     = (220, 220, 220)
THINKING_COLOR = (100, 180, 255)


def board_to_pixel(row: int, col: int) -> tuple:
    return (MARGIN + col * CELL, MARGIN + row * CELL)

def pixel_to_cell(x: int, y: int, size: int) -> tuple:
    col = round((x - MARGIN) / CELL)
    row = round((y - MARGIN) / CELL)
    if 0 <= row < size and 0 <= col < size:
        return row, col
    return None, None


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, config: Config):
        self.cfg = config
        self.size = config.BOARD_SIZE
        self.width  = MARGIN * 2 + CELL * (self.size - 1)
        self.height = MARGIN * 2 + CELL * (self.size - 1) + PANEL_H

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Gomoku — AlphaZero Bot")
        self.font_sm = pygame.font.SysFont("Arial", 16)
        self.font_md = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_lg = pygame.font.SysFont("Arial", 28, bold=True)
        self.clock = pygame.time.Clock()

    def draw(
        self,
        game: GomokuGame,
        thinking: bool = False,
        show_heatmap: bool = False,
        heatmap: np.ndarray = None,
        status: str = "",
    ):
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        if show_heatmap and heatmap is not None:
            self._draw_heatmap(heatmap, game)
        self._draw_stones(game)
        self._draw_last_move(game)
        if game._done:
            self._draw_win_highlight(game)
        self._draw_panel(game, thinking, show_heatmap, status)
        pygame.display.flip()

    def _draw_grid(self):
        s = self.size
        for i in range(s):
            # Horizontal lines
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (MARGIN, MARGIN + i * CELL),
                (MARGIN + (s-1) * CELL, MARGIN + i * CELL), 1
            )
            # Vertical lines
            pygame.draw.line(
                self.screen, GRID_COLOR,
                (MARGIN + i * CELL, MARGIN),
                (MARGIN + i * CELL, MARGIN + (s-1) * CELL), 1
            )

    def _draw_stones(self, game: GomokuGame):
        r = int(CELL * 0.42)
        for row in range(self.size):
            for col in range(self.size):
                v = game.board[row, col]
                if v == 0:
                    continue
                px, py = board_to_pixel(row, col)
                color = BLACK_STONE if v == 1 else WHITE_STONE
                pygame.draw.circle(self.screen, color, (px, py), r)
                outline = (200, 200, 200) if v == 1 else (100, 100, 100)
                pygame.draw.circle(self.screen, outline, (px, py), r, 1)

    def _draw_last_move(self, game: GomokuGame):
        if not game.move_history:
            return
        action = game.move_history[-1]
        row, col = divmod(action, self.size)
        px, py = board_to_pixel(row, col)
        v = game.board[row, col]
        dot_color = (200, 200, 200) if v == 1 else (80, 80, 80)
        pygame.draw.circle(self.screen, dot_color, (px, py), 5)

    def _draw_win_highlight(self, game: GomokuGame):
        cells = game.get_winning_cells()
        r = int(CELL * 0.42)
        for row, col in cells:
            px, py = board_to_pixel(row, col)
            pygame.draw.circle(self.screen, WIN_HIGHLIGHT, (px, py), r, 4)

    def _draw_heatmap(self, heatmap: np.ndarray, game: GomokuGame):
        """Draw semi-transparent red overlay proportional to policy probs."""
        max_prob = heatmap.max()
        if max_prob <= 0:
            return
        r_base = int(CELL * 0.4)
        for action in range(self.cfg.ACTION_SIZE):
            row, col = divmod(action, self.size)
            if game.board[row, col] != 0:
                continue
            prob = heatmap[action] / max_prob
            if prob < 0.01:
                continue
            alpha = int(prob * 180)
            px, py = board_to_pixel(row, col)
            surf = pygame.Surface((r_base*2, r_base*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*HEAT_COLOR, alpha), (r_base, r_base), r_base)
            self.screen.blit(surf, (px - r_base, py - r_base))

    def _draw_panel(self, game: GomokuGame, thinking: bool, show_heatmap: bool, status: str):
        panel_y = self.height - PANEL_H
        pygame.draw.rect(self.screen, PANEL_BG, (0, panel_y, self.width, PANEL_H))

        # Status line
        if game._done:
            if game.winner == 0:
                msg = "Draw!"
            else:
                w = "Black" if game.winner == 1 else "White"
                msg = f"{w} wins! {'You win! 🎉' if game.winner == 1 else 'Bot wins!'}"
            txt = self.font_lg.render(msg, True, WIN_HIGHLIGHT)
        elif thinking:
            txt = self.font_lg.render("Bot is thinking...", True, THINKING_COLOR)
        else:
            cp = "● Black (You)" if game.current_player == 1 else "○ White (Bot)"
            txt = self.font_md.render(f"Your turn  |  {cp}  |  Move {len(game.move_history)+1}", True, PANEL_TEXT)
        self.screen.blit(txt, (10, panel_y + 10))

        # Controls hint
        hint = "[R] New Game   [U] Undo   [H] Heatmap" + ("  [ON]" if show_heatmap else "")
        if status:
            hint += f"   {status}"
        htxt = self.font_sm.render(hint, True, (160, 160, 160))
        self.screen.blit(htxt, (10, panel_y + 50))

        # Move counter
        moves_txt = self.font_sm.render(f"Moves: {len(game.move_history)}", True, (160, 160, 160))
        self.screen.blit(moves_txt, (self.width - 100, panel_y + 10))

    def tick(self):
        self.clock.tick(30)


# ── Main game loop ────────────────────────────────────────────────────────────

def run(cfg: Config, num_simulations: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load champion
    net = load_champion(cfg, device)
    if net is None:
        print(f"No champion.pth found in {cfg.CHECKPOINT_DIR}/")
        print("Train first: python train.py --mode quick")
        sys.exit(1)

    net.eval()
    mcts = MCTS(net, cfg, device)

    renderer = Renderer(cfg)
    game = GomokuGame(cfg)
    game_history_stack = []   # for undo: list of (game_clone, mcts_state)

    thinking = False
    show_heatmap = False
    last_heatmap = None
    bot_thread = None
    pending_bot_action = [None]

    def bot_move():
        """Run in background thread — doesn't block UI."""
        policy = mcts.get_action_probs(
            game,
            num_simulations=num_simulations,
            temperature=0.0,
            add_noise=False,
        )
        pending_bot_action[0] = (int(np.argmax(policy)), policy.copy())

    human_color = 1    # Human plays Black (first)
    bot_color   = -1

    print(f"Gomoku {cfg.BOARD_SIZE}×{cfg.BOARD_SIZE} — You play Black (●), Bot plays White (○)")
    print(f"Bot MCTS simulations: {num_simulations}")

    running = True
    while running:
        # ── Handle events ─────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_r:
                    # New game
                    game = GomokuGame(cfg)
                    mcts.reset()
                    game_history_stack.clear()
                    pending_bot_action[0] = None
                    thinking = False
                    last_heatmap = None

                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap

                elif event.key == pygame.K_u:
                    # Undo last two moves (human + bot)
                    if len(game_history_stack) >= 1 and not thinking:
                        game, saved_mcts_root = game_history_stack.pop()
                        mcts._root = saved_mcts_root
                        last_heatmap = None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if (not game._done
                        and game.current_player == human_color
                        and not thinking
                        and bot_thread is None):
                    x, y = event.pos
                    row, col = pixel_to_cell(x, y, cfg.BOARD_SIZE)
                    if row is not None:
                        action = row * cfg.BOARD_SIZE + col
                        if action in game.get_legal_moves():
                            # Save state for undo (before human move)
                            game_history_stack.append((game.clone(), mcts._root))
                            mcts.update_with_move(action)
                            game.make_move(action)

        # ── Bot turn ──────────────────────────────────────────────────────────
        if (not game._done
                and game.current_player == bot_color
                and not thinking
                and bot_thread is None):
            thinking = True
            pending_bot_action[0] = None
            bot_thread = threading.Thread(target=bot_move, daemon=True)
            bot_thread.start()

        # ── Collect bot result ────────────────────────────────────────────────
        if bot_thread is not None and not bot_thread.is_alive():
            bot_thread = None
            thinking = False
            if pending_bot_action[0] is not None:
                action, policy = pending_bot_action[0]
                last_heatmap = policy
                mcts.update_with_move(action)
                game.make_move(action)

        # ── Draw ──────────────────────────────────────────────────────────────
        renderer.draw(
            game,
            thinking=thinking,
            show_heatmap=show_heatmap,
            heatmap=last_heatmap,
            status=f"Sims={num_simulations}",
        )
        renderer.tick()

    pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Gomoku vs AlphaZero bot")
    parser.add_argument("--mode", choices=["quick", "validate", "full"], default="quick",
                        help="'quick' = 5x5, 'validate' = 6x6, 'full' = 10x10")
    parser.add_argument("--sims", type=int, default=800,
                        help="MCTS simulations per bot move (more = stronger & slower)")
    args = parser.parse_args()

    mode_map = {"quick": Config.for_quick_test, "validate": Config.for_validation, "full": Config.for_full}
    cfg = mode_map[args.mode]()
    run(cfg, args.sims)
