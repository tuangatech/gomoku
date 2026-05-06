"""
self_play.py — Self-play data generation.

Each game produces a list of training examples:
    (state_tensor [3,S,S], mcts_policy [A], outcome float)

outcome is always from the perspective of the player who was to move
at that board state: +1 if they won, -1 if they lost, 0 for draw.

Parallelization: run_self_play_batch() spawns worker processes that each
play complete games on CPU, sending results back via a Queue.
The GPU is used only for network inference — workers share weights via
a state_dict passed through the queue.
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Tuple
import time

from game import GomokuGame
from mcts import MCTS
from network import PolicyValueNet
from config import Config


Example = Tuple[np.ndarray, np.ndarray, float]   # (state, policy, outcome)


# ── Single game ───────────────────────────────────────────────────────────────

def play_one_game(
    network: PolicyValueNet,
    config: Config,
    device: torch.device,
    num_simulations: int,
    verbose: bool = False,
) -> List[Example]:
    """
    Play one complete self-play game.
    Returns list of (state, mcts_policy, outcome) training examples.
    """
    game = GomokuGame(config)
    mcts = MCTS(network, config, device)
    history: List[Tuple[np.ndarray, np.ndarray, int]] = []
    # history stores (state, policy, player_who_moved)

    move_num = 0
    while True:
        done, _ = game.is_terminal()
        if done:
            break

        # Temperature schedule: exploratory early, greedy late
        temperature = 1.0 if move_num < config.TEMPERATURE_THRESHOLD else 0.05

        state = game.get_state_tensor()
        policy = mcts.get_action_probs(
            game,
            num_simulations=num_simulations,
            temperature=temperature,
            add_noise=True,   # always True during self-play
        )

        # Sample action from policy distribution
        action = int(np.random.choice(config.ACTION_SIZE, p=policy))

        history.append((state.copy(), policy.copy(), game.current_player))

        mcts.update_with_move(action)
        game.make_move(action)
        move_num += 1

        if verbose and move_num % 5 == 0:
            print(game.render())

    # Assign outcomes retroactively
    winner = game.winner   # 1, -1, or 0
    examples: List[Example] = []
    for (state, policy, player) in history:
        if winner == 0:
            outcome = 0.0
        elif winner == player:
            outcome = 1.0
        else:
            outcome = -1.0
        examples.append((state, policy, outcome))

    if verbose:
        w = {1: "Black", -1: "White", 0: "Draw"}[winner]
        print(f"Game over: {w}  ({len(examples)} positions)")

    return examples


# ── Worker process (CPU) ──────────────────────────────────────────────────────

def _worker_fn(
    worker_id: int,
    state_dict: dict,
    config: Config,
    num_games: int,
    num_simulations: int,
    result_queue: mp.Queue,
) -> None:
    """
    Worker process: load network weights on CPU, play `num_games` games,
    put results in queue.
    """
    device = torch.device("cpu")
    net = PolicyValueNet(config)
    net.load_state_dict(state_dict)
    net.eval()

    all_examples: List[Example] = []
    for i in range(num_games):
        examples = play_one_game(net, config, device, num_simulations)
        all_examples.extend(examples)

    result_queue.put((worker_id, all_examples))


# ── Parallel batch ────────────────────────────────────────────────────────────

def run_self_play_batch(
    network: PolicyValueNet,
    config: Config,
    num_games: int,
    num_simulations: int,
) -> List[Example]:
    """
    Play `num_games` self-play games using parallel CPU workers.
    Returns all training examples from all games.

    The network is on GPU for training; workers receive a CPU copy of weights.
    """
    num_workers = min(config.NUM_SELF_PLAY_WORKERS, num_games)
    games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    # Snapshot weights to CPU (safe to share across processes)
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}

    ctx = mp.get_context("spawn")   # Windows-safe (fork not available on Windows)
    result_queue = ctx.Queue()
    processes = []

    t0 = time.time()
    for wid in range(num_workers):
        n = games_per_worker + (1 if wid < remainder else 0)
        p = ctx.Process(
            target=_worker_fn,
            args=(wid, state_dict, config, n, num_simulations, result_queue),
            daemon=True,
        )
        p.start()
        processes.append(p)

    all_examples: List[Example] = []
    for _ in range(num_workers):
        wid, examples = result_queue.get()
        all_examples.extend(examples)

    for p in processes:
        p.join()

    elapsed = time.time() - t0
    print(f"  Self-play: {num_games} games, {len(all_examples)} positions in {elapsed:.1f}s")
    return all_examples


# ── Fallback: single-process (for debugging / small boards) ──────────────────

def run_self_play_single(
    network: PolicyValueNet,
    config: Config,
    device: torch.device,
    num_games: int,
    num_simulations: int,
) -> List[Example]:
    """
    Sequential self-play (no multiprocessing).
    Use this for debugging or when NUM_SELF_PLAY_WORKERS=1.
    """
    all_examples: List[Example] = []
    t0 = time.time()
    for i in range(num_games):
        examples = play_one_game(network, config, device, num_simulations)
        all_examples.extend(examples)
        if (i + 1) % 10 == 0:
            print(f"  Self-play game {i+1}/{num_games} ({len(all_examples)} positions so far)")
    elapsed = time.time() - t0
    print(f"  Self-play done: {len(all_examples)} positions in {elapsed:.1f}s")
    return all_examples
