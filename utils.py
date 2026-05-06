"""
utils.py — Logging, Elo rating, and checkpoint management.
"""

import csv
import copy
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from network import PolicyValueNet
from config import Config


# ── Elo ───────────────────────────────────────────────────────────────────────

class EloTracker:
    """
    Maintains rolling Elo ratings for champion model across iterations.
    Useful for detecting convergence (Elo gain < threshold for N iterations).
    """

    K = 32        # Elo K-factor
    BASE = 1000   # Starting Elo

    def __init__(self):
        self.elo = self.BASE
        self.history: list[float] = [self.BASE]

    def update(self, win_rate_against_prev: float) -> float:
        """
        Update Elo after playing against previous champion (assumed Elo = current - some gap).
        Approximation: treat previous champion as having the same Elo, use win rate directly.
        """
        # Expected score at equal Elo = 0.5
        expected = 0.5
        self.elo += self.K * (win_rate_against_prev - expected)
        self.history.append(self.elo)
        return self.elo

    def recent_gain(self, n: int = 10) -> float:
        """Average Elo gain per iteration over last n iterations."""
        if len(self.history) < 2:
            return float("inf")
        recent = self.history[-min(n, len(self.history)):]
        return (recent[-1] - recent[0]) / max(len(recent) - 1, 1)

    def has_plateaued(self, threshold: float = 5.0, window: int = 10) -> bool:
        """True if average Elo gain < threshold for the last `window` iterations."""
        return len(self.history) >= window and abs(self.recent_gain(window)) < threshold


# ── Policy Loss Monitor ──────────────────────────────────────────────────────

class PolicyLossMonitor:
    """
    Tracks policy loss trend and warns when it rises persistently
    without corresponding Elo gains — a sign the model is forgetting
    how to play while the value head improves.
    """

    def __init__(self, window: int = 10):
        self.window = window
        self.policy_losses: list[float] = []
        self.elo_at_record: list[float] = []

    def record(self, policy_loss: float, current_elo: float) -> Optional[str]:
        """
        Record a policy loss observation. Returns a warning string
        if policy loss has risen for `window` consecutive iterations
        while Elo has stalled (< 5 gain over the same window).
        """
        self.policy_losses.append(policy_loss)
        self.elo_at_record.append(current_elo)

        if len(self.policy_losses) < self.window + 1:
            return None

        recent = self.policy_losses[-self.window:]
        rising = all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))
        elo_gain = self.elo_at_record[-1] - self.elo_at_record[-self.window]
        elo_stalled = abs(elo_gain) < 5.0

        if rising and elo_stalled:
            return (
                f"Policy loss has risen for {self.window} consecutive iterations "
                f"({recent[0]:.4f} → {recent[-1]:.4f}) while Elo is stalled "
                f"(gain={elo_gain:+.1f}). The model may be forgetting how to play. "
                f"Consider lowering LR or checking training data diversity."
            )
        return None


# ── CSV Logger ────────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    Append-only CSV log. One row per training iteration.
    Columns: iteration, elo, vs_heuristic_winrate, value_loss, policy_loss,
             total_loss, buffer_size, lr, elapsed_seconds
    """

    COLUMNS = [
        "iteration", "elo", "vs_heuristic_wr", "vs_snapshot_wr",
        "value_loss", "policy_loss", "total_loss",
        "buffer_size", "lr", "elapsed_s",
    ]

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._start_time = time.time()

        if not log_path.exists():
            with open(log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(self, **kwargs) -> None:
        kwargs.setdefault("elapsed_s", round(time.time() - self._start_time, 1))
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore")
            writer.writerow(kwargs)

    def elapsed_str(self) -> str:
        """Human-readable elapsed time since training started."""
        s = int(time.time() - self._start_time)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def print_row(self, iteration: int, **kwargs) -> None:
        elo = kwargs.get("elo", "?")
        vl  = kwargs.get("value_loss", 0)
        pl  = kwargs.get("policy_loss", 0)
        hwr = kwargs.get("vs_heuristic_wr", 0)
        buf = kwargs.get("buffer_size", 0)
        print(
            f"[Iter {iteration:3d}] "
            f"Elo={elo:6.1f}  "
            f"v_loss={vl:.4f}  p_loss={pl:.4f}  "
            f"vs_heuristic={hwr:.1%}  "
            f"buf={buf:,}  "
            f"t={self.elapsed_str()}"
        )


# ── TensorBoard Logger ───────────────────────────────────────────────────────

class TBLogger:
    """Thin wrapper around TensorBoard SummaryWriter."""

    def __init__(self, log_dir: Path, cfg: Config):
        run_name = f"{cfg.BOARD_SIZE}x{cfg.BOARD_SIZE}_win{cfg.WIN_LENGTH}"
        self.writer = SummaryWriter(log_dir / run_name)
        self.writer.add_text(
            "config",
            f"board={cfg.BOARD_SIZE}x{cfg.BOARD_SIZE}  win={cfg.WIN_LENGTH}  "
            f"blocks={cfg.NUM_RES_BLOCKS}  filters={cfg.NUM_FILTERS}  "
            f"lr={cfg.LEARNING_RATE}  batch={cfg.BATCH_SIZE}  "
            f"mcts_train={cfg.MCTS_SIMULATIONS_TRAIN}  "
            f"buffer={cfg.REPLAY_BUFFER_SIZE}",
        )

    def log(self, iteration: int, **kwargs) -> None:
        for key, value in kwargs.items():
            self.writer.add_scalar(key, value, iteration)

    def log_network(self, network: PolicyValueNet, iteration: int) -> None:
        for name, param in network.named_parameters():
            self.writer.add_histogram(f"params/{name}", param, iteration)
            if param.grad is not None:
                self.writer.add_histogram(f"grads/{name}", param.grad, iteration)

    def close(self) -> None:
        self.writer.close()


# ── Checkpoint management ─────────────────────────────────────────────────────

def save_checkpoint(
    network: PolicyValueNet,
    iteration: int,
    elo: float,
    config: Config,
    filename: Optional[str] = None,
) -> Path:
    """Save network weights and metadata."""
    if filename is None:
        filename = f"checkpoint_{iteration:04d}.pth"
    path = config.CHECKPOINT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": iteration,
        "elo": elo,
        "board_size": config.BOARD_SIZE,
        "win_length": config.WIN_LENGTH,
        "state_dict": network.state_dict(),
    }, path)
    return path


def save_champion(network: PolicyValueNet, config: Config) -> Path:
    """Overwrite the champion checkpoint (always named champion.pth)."""
    path = config.CHECKPOINT_DIR / "champion.pth"
    torch.save({
        "board_size": config.BOARD_SIZE,
        "win_length": config.WIN_LENGTH,
        "state_dict": network.state_dict(),
    }, path)
    return path


def load_champion(config: Config, device: torch.device) -> Optional[PolicyValueNet]:
    """Load champion network if checkpoint exists, else return None."""
    path = config.CHECKPOINT_DIR / "champion.pth"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    net = PolicyValueNet(config)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()
    print(f"Loaded champion from {path}")
    return net


def save_snapshot(network: PolicyValueNet, iteration: int, config: Config) -> Path:
    """Save a frozen snapshot for use as a stronger eval benchmark."""
    path = config.CHECKPOINT_DIR / "snapshot_benchmark.pth"
    torch.save({
        "iteration": iteration,
        "board_size": config.BOARD_SIZE,
        "win_length": config.WIN_LENGTH,
        "state_dict": network.state_dict(),
    }, path)
    return path


def load_snapshot(config: Config, device: torch.device) -> Optional[PolicyValueNet]:
    """Load the frozen snapshot benchmark if it exists."""
    path = config.CHECKPOINT_DIR / "snapshot_benchmark.pth"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    net = PolicyValueNet(config)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    net.eval()
    return net


def clone_network(network: PolicyValueNet) -> PolicyValueNet:
    """Deep copy a network (for champion/challenger split)."""
    clone = copy.deepcopy(network)
    return clone

