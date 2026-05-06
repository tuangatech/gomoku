"""
config.py — Single source of truth for all hyperparameters.

To switch between validation mode (6x6) and full training (10x10),
change BOARD_SIZE and WIN_LENGTH here. Nothing else needs to change.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Board ────────────────────────────────────────────────────────────────
    BOARD_SIZE: int = 6          # 6 for validation, 10 for full training
    WIN_LENGTH: int = 4          # 4 for 6x6 validation, 5 for 10x10

    # ── Network ──────────────────────────────────────────────────────────────
    NUM_RES_BLOCKS: int = 4      # 4 for 6x6 (fast), 6-8 for 10x10
    NUM_FILTERS: int = 64        # 64 for 6x6, 128 for 10x10

    # ── MCTS ─────────────────────────────────────────────────────────────────
    MCTS_SIMULATIONS_TRAIN: int = 200   # per move during self-play (speed vs quality)
    MCTS_SIMULATIONS_EVAL: int = 400    # per move during evaluation
    MCTS_SIMULATIONS_PLAY: int = 800    # per move during human vs bot
    C_PUCT: float = 2.5                 # exploration constant
    DIRICHLET_ALPHA: float = 0.6        # higher alpha = more uniform noise (use 0.3 for 10x10)
    DIRICHLET_EPSILON: float = 0.25     # weight of noise vs network prior at root
    TEMPERATURE_THRESHOLD: int = 6      # moves before switching to greedy (10-15 for 10x10)

    # ── Self-Play ────────────────────────────────────────────────────────────
    SELF_PLAY_GAMES_PER_ITER: int = 50  # 50 for 6x6, 100-150 for 10x10
    NUM_SELF_PLAY_WORKERS: int = 4      # parallel CPU workers for self-play

    # ── Training ─────────────────────────────────────────────────────────────
    REPLAY_BUFFER_SIZE: int = 20_000    # 20k for 6x6, 50k for 10x10
    BATCH_SIZE: int = 256               # 256 for 6x6, 512 for 10x10
    TRAIN_STEPS_PER_ITER: int = 200     # gradient steps after each self-play batch
    LEARNING_RATE: float = 1e-3
    LR_MILESTONES: list = field(default_factory=lambda: [100, 200])  # iteration-based fallback
    LR_GAMMA: float = 0.1               # LR multiplier at each milestone
    LR_BUFFER_FULL_DROP: float = 3e-4   # LR to use once buffer first fills
    LR_BUFFER_FULL_DECAY: float = 1e-4  # LR to decay to after sustained buffer-full training
    L2_WEIGHT_DECAY: float = 1e-4

    # ── Evaluation ───────────────────────────────────────────────────────────
    EVAL_GAMES: int = 20                # 20 for 6x6 speed, 40 for 10x10
    PROMOTION_THRESHOLD: float = 0.55   # challenger must win >55% to become champion
    SNAPSHOT_EVAL_GAMES: int = 20       # games per snapshot evaluation
    SNAPSHOT_HEURISTIC_WR_TRIGGER: float = 0.80  # save snapshot when heuristic WR first hits this

    # ── Training Loop ────────────────────────────────────────────────────────
    NUM_ITERATIONS: int = 100           # 100 for 6x6 validation, 300+ for 10x10
    CHECKPOINT_INTERVAL: int = 5        # save champion every N iterations
    EVAL_INTERVAL: int = 5             # run evaluator every N iterations

    # ── Paths ─────────────────────────────────────────────────────────────────
    CHECKPOINT_DIR: Path = Path("checkpoints")
    LOG_FILE: Path = Path("training_log.csv")
    TENSORBOARD_DIR: Path = Path("runs")

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE: str = "cuda"               # "cuda" or "cpu"

    def __post_init__(self):
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.ACTION_SIZE = self.BOARD_SIZE * self.BOARD_SIZE

    @classmethod
    def for_quick_test(cls) -> "Config":
        """5x5 / 3-in-a-row smoke test. Should converge in ~30 minutes."""
        return cls(
            BOARD_SIZE=5, WIN_LENGTH=3,
            NUM_RES_BLOCKS=4, NUM_FILTERS=64,
            MCTS_SIMULATIONS_TRAIN=200, MCTS_SIMULATIONS_EVAL=400,
            MCTS_SIMULATIONS_PLAY=400,
            SELF_PLAY_GAMES_PER_ITER=50,
            REPLAY_BUFFER_SIZE=10_000, BATCH_SIZE=128,
            TRAIN_STEPS_PER_ITER=200,
            NUM_ITERATIONS=50,
            EVAL_GAMES=20,
            EVAL_INTERVAL=3,
            CHECKPOINT_INTERVAL=5,
            DIRICHLET_ALPHA=1.0,
            TEMPERATURE_THRESHOLD=4,
        )

    @classmethod
    def for_validation(cls) -> "Config":
        """6x6 / 4-in-a-row validation config (~2-4 hours)."""
        return cls(
            BOARD_SIZE=6, WIN_LENGTH=4,
            NUM_RES_BLOCKS=6, NUM_FILTERS=128,
            MCTS_SIMULATIONS_TRAIN=400, MCTS_SIMULATIONS_EVAL=800,
            SELF_PLAY_GAMES_PER_ITER=100,
            REPLAY_BUFFER_SIZE=30_000, BATCH_SIZE=256,
            TRAIN_STEPS_PER_ITER=400,
            NUM_ITERATIONS=120,
            DIRICHLET_ALPHA=0.6,
            TEMPERATURE_THRESHOLD=8,
        )

    @classmethod
    def for_full(cls) -> "Config":
        """10x10 / 5-in-a-row full training config (~3-5 days)."""
        return cls(
            BOARD_SIZE=10, WIN_LENGTH=5,
            NUM_RES_BLOCKS=8, NUM_FILTERS=128,
            MCTS_SIMULATIONS_TRAIN=600, MCTS_SIMULATIONS_EVAL=1200,
            SELF_PLAY_GAMES_PER_ITER=150,
            REPLAY_BUFFER_SIZE=75_000, BATCH_SIZE=512,
            TRAIN_STEPS_PER_ITER=600,
            NUM_ITERATIONS=300,
            DIRICHLET_ALPHA=0.3,
            TEMPERATURE_THRESHOLD=15,
        )
