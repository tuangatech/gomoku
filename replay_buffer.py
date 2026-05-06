"""
replay_buffer.py — Experience replay buffer.

Stores (state, policy, outcome) examples from self-play games.
Uses a deque with maxlen so old experiences are automatically evicted.
Sampling is uniform random across all stored positions.
"""

import numpy as np
from collections import deque
from typing import List, Tuple
import torch

from config import Config

Example = Tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    """
    Rolling window replay buffer.

    Capacity = config.REPLAY_BUFFER_SIZE positions.
    Once full, oldest positions are silently dropped.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._buffer: deque[Example] = deque(maxlen=config.REPLAY_BUFFER_SIZE)

    def add(self, examples: List[Example]) -> None:
        """Add a batch of (state, policy, outcome) examples."""
        self._buffer.extend(examples)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch.
        Returns GPU tensors ready for training.

        Returns:
            states:   [batch, 3, S, S]  float32
            policies: [batch, A]        float32
            outcomes: [batch]           float32
        """
        assert len(self._buffer) >= batch_size, (
            f"Buffer has {len(self._buffer)} examples, need {batch_size}"
        )
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        states   = np.stack([b[0] for b in batch], axis=0)   # [B, 3, S, S]
        policies = np.stack([b[1] for b in batch], axis=0)   # [B, A]
        outcomes = np.array([b[2] for b in batch], dtype=np.float32)  # [B]

        return (
            torch.tensor(states,   dtype=torch.float32).to(device),
            torch.tensor(policies, dtype=torch.float32).to(device),
            torch.tensor(outcomes, dtype=torch.float32).to(device),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_size: int = None) -> bool:
        """True if buffer has enough examples to start training."""
        min_size = min_size or self.cfg.BATCH_SIZE
        return len(self._buffer) >= min_size
