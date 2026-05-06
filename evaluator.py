"""
evaluator.py — Evaluation: champion vs challenger, and vs heuristic bot.

Two evaluation modes:
  1. Champion vs Challenger: decides if new model gets promoted.
  2. vs Heuristic Bot: stable external benchmark to track progress over time.

Colors alternate across games to neutralize first-mover advantage.
"""

import numpy as np
import torch
from typing import Tuple

from game import GomokuGame
from mcts import MCTS
from network import PolicyValueNet
from config import Config


# ── Heuristic bot ─────────────────────────────────────────────────────────────

class HeuristicBot:
    """
    Simple rule-based opponent. Useful as a stable external benchmark.

    Priority order:
      1. Win immediately if possible
      2. Block opponent's immediate win
      3. Play closest to board center
    """

    def __init__(self, config: Config):
        self.cfg = config

    def pick_move(self, game: GomokuGame) -> int:
        legal = game.get_legal_moves()
        size = game.size

        # Priority 1: Can we win right now?
        for action in legal:
            g = game.clone()
            g.make_move(action)
            done, outcome = g.is_terminal()
            if done and outcome > 0:   # we won (outcome is from g's current_player perspective, which flipped)
                # Actually: after make_move, if done, winner is game.current_player (before flip)
                if g.winner == game.current_player:
                    return action

        # Priority 2: Block opponent's immediate win
        opponent = -game.current_player
        for action in legal:
            g = game.clone()
            # Temporarily place opponent's stone
            g.board[action // size, action % size] = opponent
            g.move_history.append(action)
            if g._check_win(action // size, action % size):
                return action   # block here

        # Priority 3: Closest to center (with small random tiebreak)
        center = (size - 1) / 2.0
        def dist_to_center(a):
            r, c = divmod(a, size)
            return (r - center) ** 2 + (c - center) ** 2 + np.random.uniform(0, 0.01)

        return min(legal, key=dist_to_center)


# ── One game between two agents ───────────────────────────────────────────────

def play_evaluation_game(
    black_mcts: MCTS,
    white_mcts: MCTS,
    config: Config,
    num_simulations: int,
) -> int:
    """
    Play one game between two MCTS agents.
    Returns the winner: 1 (Black), -1 (White), 0 (Draw).
    """
    game = GomokuGame(config)
    mcts_map = {1: black_mcts, -1: white_mcts}

    for m in mcts_map.values():
        m.reset()

    while True:
        done, _ = game.is_terminal()
        if done:
            break

        current_mcts = mcts_map[game.current_player]
        policy = current_mcts.get_action_probs(
            game,
            num_simulations=num_simulations,
            temperature=0.0,   # greedy during evaluation
            add_noise=False,
        )
        action = int(np.argmax(policy))

        for m in mcts_map.values():
            m.update_with_move(action)

        game.make_move(action)

    return game.winner or 0


def play_vs_heuristic_game(
    net_mcts: MCTS,
    heuristic: HeuristicBot,
    net_plays_black: bool,
    config: Config,
    num_simulations: int,
) -> int:
    """
    Play one game between neural net (MCTS) and heuristic bot.
    Returns winner: 1 (Black), -1 (White), 0 (Draw).
    """
    game = GomokuGame(config)
    net_mcts.reset()
    net_color = 1 if net_plays_black else -1

    while True:
        done, _ = game.is_terminal()
        if done:
            break

        if game.current_player == net_color:
            policy = net_mcts.get_action_probs(
                game,
                num_simulations=num_simulations,
                temperature=0.0,
                add_noise=False,
            )
            action = int(np.argmax(policy))
        else:
            action = heuristic.pick_move(game)

        net_mcts.update_with_move(action)
        game.make_move(action)

    return game.winner or 0


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_models(
    challenger: PolicyValueNet,
    champion: PolicyValueNet,
    config: Config,
    device: torch.device,
    num_games: int = None,
    num_simulations: int = None,
) -> Tuple[float, int, int, int]:
    """
    Run champion vs challenger evaluation.
    Colors alternate: challenger plays Black in even games, White in odd games.

    Returns:
        (challenger_win_rate, wins, losses, draws)
    """
    num_games = num_games or config.EVAL_GAMES
    num_simulations = num_simulations or config.MCTS_SIMULATIONS_EVAL

    challenger_mcts = MCTS(challenger, config, device)
    champion_mcts = MCTS(champion, config, device)

    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        challenger_is_black = (i % 2 == 0)
        if challenger_is_black:
            winner = play_evaluation_game(challenger_mcts, champion_mcts, config, num_simulations)
        else:
            winner = play_evaluation_game(champion_mcts, challenger_mcts, config, num_simulations)

        challenger_color = 1 if challenger_is_black else -1
        if winner == challenger_color:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0
    return win_rate, wins, losses, draws


def evaluate_vs_snapshot(
    challenger: PolicyValueNet,
    snapshot: PolicyValueNet,
    config: Config,
    device: torch.device,
    num_games: int = 20,
    num_simulations: int = None,
) -> Tuple[float, int, int, int]:
    """
    Evaluate current champion against a frozen earlier snapshot.
    Stronger benchmark once heuristic is saturated.
    Returns (win_rate, wins, losses, draws) for the challenger.
    """
    return evaluate_models(
        challenger, snapshot, config, device,
        num_games=num_games,
        num_simulations=num_simulations,
    )


def evaluate_vs_heuristic(
    network: PolicyValueNet,
    config: Config,
    device: torch.device,
    num_games: int = 20,
    num_simulations: int = None,
) -> Tuple[float, int, int, int]:
    """
    Evaluate network against the heuristic bot.
    Returns (win_rate, wins, losses, draws) for the network.
    """
    num_simulations = num_simulations or config.MCTS_SIMULATIONS_EVAL
    net_mcts = MCTS(network, config, device)
    heuristic = HeuristicBot(config)

    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        net_plays_black = (i % 2 == 0)
        winner = play_vs_heuristic_game(net_mcts, heuristic, net_plays_black, config, num_simulations)
        net_color = 1 if net_plays_black else -1
        if winner == net_color:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0
    return win_rate, wins, losses, draws
