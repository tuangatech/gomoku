# Gomoku AlphaZero — From Scratch

- AlphaZero-style Gomoku bot built from scratch in Python + PyTorch.
- Self-play reinforcement learning, MCTS + ResNet, Pygame UI.
- Trained on an RTX-4070

## Quick Start

```bash
# 1. Install dependencies (PyTorch CUDA: https://pytorch.org/get-started/locally/)
pip install -r requirements.txt

# 2. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. Run unit tests
python test_game.py

# 4. Quick smoke test — 5x5 / 3-in-a-row (~2 hours)
python train.py --mode quick
python play_gui.py --mode quick

# 5. Validation — 6x6 / 4-in-a-row (~15 hours)
python train.py --mode validate
python play_gui.py --mode validate

# 6. Full training — 10x10 / 5-in-a-row (~5 days)
python train.py --mode full
python play_gui.py --mode full

# 7. Resume training (warm-start from saved champion weights)
python train.py --mode full --resume

# 8. Monitor training with TensorBoard
tensorboard --logdir runs
```

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters. Change `BOARD_SIZE` here only. |
| `game.py` | Board logic, win detection, state tensor |
| `network.py` | Dual-headed ResNet (policy + value) |
| `mcts.py` | MCTS with PUCT selection, Dirichlet noise, tree reuse |
| `self_play.py` | Self-play data generation (parallel workers) |
| `replay_buffer.py` | Rolling experience replay buffer |
| `trainer.py` | Training loop with mixed precision |
| `evaluator.py` | Heuristic bot evaluation + regression guard |
| `utils.py` | Elo tracking, CSV logging, TensorBoard logging, checkpoint save/load |
| `train.py` | Main training loop |
| `play_gui.py` | Pygame play UI |
| `test_game.py` | Unit tests for game logic |
| `requirements.txt` | Python dependencies (PyTorch, NumPy, Pygame, TensorBoard) |

## Training Progression

| Mode | Board | Win | Time | Purpose |
|------|-------|-----|------|---------|
| `quick` | 5×5 | 3-in-a-row | ~2 hours | Smoke test — verify the pipeline works |
| `validate` | 6×6 | 4-in-a-row | ~15 hrs | Validate learning — bot should beat heuristic >70% by iter 10 |
| `full` | 10×10 | 5-in-a-row | ~5 days | Full training — bot develops real strategic play |

Start with `quick` to verify everything runs, then `validate`, then `full`.

## Play Against the Bot

You can play against the latest checkpoint at any time — even while training is still running (or after stopping it with Ctrl+C). Just match the `--mode` to whichever mode you trained with:

```bash
# Play the 5x5 bot
python play_gui.py --mode quick

# Play the 6x6 bot
python play_gui.py --mode validate

# Play the 10x10 bot
python play_gui.py --mode full
```

The GUI loads `checkpoints/champion.pth` automatically. No need to specify a checkpoint path.

## Controls (play_gui.py)

| Key | Action |
|-----|--------|
| Click | Place your stone |
| R | New game |
| U | Undo last move pair |
| H | Toggle policy heatmap |
| Q / Escape | Quit |
