"""
trainer.py — Neural network training.

Uses PyTorch mixed precision (AMP) for ~2x speedup on RTX 4070.
Combined loss = MSE(value) + CrossEntropy(policy) + L2 regularization.
L2 is handled via weight_decay in the optimizer, not the loss directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from typing import Tuple

from network import PolicyValueNet
from replay_buffer import ReplayBuffer
from config import Config


class Trainer:
    """Handles all gradient updates for the policy-value network."""

    def __init__(self, network: PolicyValueNet, config: Config, device: torch.device):
        self.net = network
        self.cfg = config
        self.device = device

        self.optimizer = optim.Adam(
            network.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.L2_WEIGHT_DECAY,
        )

        # LR scheduler: step decay at configured iteration milestones
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.LR_MILESTONES,
            gamma=config.LR_GAMMA,
        )

        # Mixed precision scaler (safe no-op on CPU)
        self.scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    def train_epoch(self, buffer: ReplayBuffer) -> Tuple[float, float, float]:
        """
        Run TRAIN_STEPS_PER_ITER gradient steps on mini-batches from buffer.

        Returns:
            (avg_total_loss, avg_value_loss, avg_policy_loss)
        """
        self.net.train()
        total_loss_sum = 0.0
        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        steps = self.cfg.TRAIN_STEPS_PER_ITER

        for _ in range(steps):
            states, policies, outcomes = buffer.sample(self.cfg.BATCH_SIZE, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(self.device.type, enabled=(self.device.type == "cuda")):
                pred_policies, pred_values = self.net(states)

                # Value loss: MSE between predicted value and actual outcome
                value_loss = F.mse_loss(pred_values.squeeze(1), outcomes)

                # Policy loss: cross-entropy between predicted distribution and MCTS distribution
                # policies are soft targets from MCTS visit counts (already normalized)
                log_probs = F.log_softmax(pred_policies, dim=1)
                policy_loss = -torch.mean(torch.sum(policies * log_probs, dim=1))

                loss = value_loss + policy_loss

            self.scaler.scale(loss).backward()
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_sum += loss.item()
            value_loss_sum += value_loss.item()
            policy_loss_sum += policy_loss.item()

        return (
            total_loss_sum / steps,
            value_loss_sum / steps,
            policy_loss_sum / steps,
        )

    def step_scheduler(self) -> None:
        """Call once per training iteration (not per gradient step)."""
        self.scheduler.step()

    def set_lr(self, new_lr: float) -> None:
        """Manually override learning rate (disengages scheduler)."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
