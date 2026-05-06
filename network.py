"""
network.py — Dual-headed Residual CNN (AlphaZero architecture).

Input:  [batch, 3, BOARD_SIZE, BOARD_SIZE]  (board state tensor)
Output: policy logits [batch, ACTION_SIZE]   (one per cell)
        value scalar  [batch, 1]             (tanh, range -1 to +1)

Board-size agnostic: all shapes derived from Config at init time.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import Config


# ── Building blocks ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Standard pre-activation residual block used in AlphaZero.
    Conv → BN → ReLU → Conv → BN  +  skip connection → ReLU
    """
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# ── Main network ─────────────────────────────────────────────────────────────

class PolicyValueNet(nn.Module):
    """
    AlphaZero-style dual-headed network.

    Policy head: outputs a probability distribution over all board cells.
    Value head:  outputs a scalar estimate of the current position's value.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        F = config.NUM_FILTERS
        S = config.BOARD_SIZE
        A = config.ACTION_SIZE   # = S * S

        # ── Shared body ──────────────────────────────────────────────────────
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, F, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(F) for _ in range(config.NUM_RES_BLOCKS)]
        )

        # ── Policy head ──────────────────────────────────────────────────────
        # 1×1 conv to compress channels → flatten → FC to action space
        self.policy_head = nn.Sequential(
            nn.Conv2d(F, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * S * S, A),
            # No softmax here — use log_softmax in loss, softmax in inference
        )

        # ── Value head ───────────────────────────────────────────────────────
        # 1×1 conv → flatten → FC → ReLU → FC → tanh
        self.value_head = nn.Sequential(
            nn.Conv2d(F, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(S * S, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),   # output in [-1, +1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, 3, S, S]
        returns: (policy_logits [batch, A], value [batch, 1])
        """
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)   # [batch, A] — raw logits
        value  = self.value_head(x)    # [batch, 1] — tanh output
        return policy, value

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        state: np.ndarray,
        legal_moves: list,
        device: torch.device,
    ) -> Tuple[np.ndarray, float]:
        """
        Single-position inference used by MCTS.

        Args:
            state:       [3, S, S] float32 numpy array
            legal_moves: list of legal flat action indices
            device:      torch device

        Returns:
            policy_probs: np.ndarray of shape [ACTION_SIZE], illegal moves zeroed & renormalized
            value:        float scalar in [-1, +1]
        """
        self.eval()
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 3, S, S]
        policy_logits, value_tensor = self(x)

        # Mask illegal moves before softmax
        policy_logits = policy_logits.squeeze(0).cpu().numpy()   # [A]
        mask = np.full(self.cfg.ACTION_SIZE, -1e9)
        mask[legal_moves] = policy_logits[legal_moves]
        policy_probs = np.exp(mask - mask.max())                 # stable softmax numerator
        policy_probs /= policy_probs.sum()                       # normalize

        value = float(value_tensor.squeeze().cpu().item())
        return policy_probs, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
