"""
train.py — Main training loop.

Usage:
    # Validation run (6x6, fast — recommended first)
    python train.py --mode validate

    # Full training (10x10)
    python train.py --mode full

    # Resume from champion checkpoint
    python train.py --mode full --resume

Run from the gomoku_az/ directory.
"""

import argparse
import shutil
import sys
import torch
import torch.multiprocessing as mp

from config import Config
from game import GomokuGame
from network import PolicyValueNet
from replay_buffer import ReplayBuffer
from self_play import run_self_play_batch, run_self_play_single
from trainer import Trainer
from evaluator import evaluate_vs_heuristic, evaluate_vs_snapshot
from utils import (
    EloTracker, TrainingLogger, TBLogger, PolicyLossMonitor,
    save_checkpoint, save_champion, load_champion, clone_network,
    save_snapshot, load_snapshot,
)


def main(cfg: Config, resume: bool = False, single_process: bool = False):
    # ── Clean previous training data (unless resuming) ─────────────────────────
    if not resume:
        if cfg.CHECKPOINT_DIR.exists():
            shutil.rmtree(cfg.CHECKPOINT_DIR)
            print(f"Cleared old checkpoints: {cfg.CHECKPOINT_DIR}/")
        if cfg.LOG_FILE.exists():
            cfg.LOG_FILE.unlink()
            print(f"Cleared old log: {cfg.LOG_FILE}")
        if cfg.TENSORBOARD_DIR.exists():
            shutil.rmtree(cfg.TENSORBOARD_DIR)
            print(f"Cleared old TensorBoard logs: {cfg.TENSORBOARD_DIR}/")

    # ── Device setup ──────────────────────────────────────────────────────────
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, running on CPU (will be slow)")

    print(f"Board: {cfg.BOARD_SIZE}×{cfg.BOARD_SIZE}, win={cfg.WIN_LENGTH}")
    print(f"Network: {cfg.NUM_RES_BLOCKS} res blocks × {cfg.NUM_FILTERS} filters")

    # ── Network initialization ────────────────────────────────────────────────
    if resume:
        champion = load_champion(cfg, device)
        if champion is None:
            print("No champion checkpoint found — starting fresh.")
            champion = PolicyValueNet(cfg).to(device)
    else:
        champion = PolicyValueNet(cfg).to(device)

    print(f"Parameters: {champion.count_parameters():,}")

    # ── Supporting objects ────────────────────────────────────────────────────
    buffer   = ReplayBuffer(cfg)
    trainer  = Trainer(champion, cfg, device)
    elo      = EloTracker()
    logger   = TrainingLogger(cfg.LOG_FILE)
    tb       = TBLogger(cfg.TENSORBOARD_DIR, cfg)
    plm      = PolicyLossMonitor(window=10)
    best_heuristic_wr = 0.0
    snapshot_saved = False
    snapshot_net = None

    if resume:
        snapshot_net = load_snapshot(cfg, device)
        if snapshot_net is not None:
            snapshot_saved = True
            print("Loaded snapshot benchmark from previous run")

    buffer_was_full = False
    lr_drop_stage = 0  # 0=initial, 1=first drop, 2=second drop

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting training: {cfg.NUM_ITERATIONS} iterations")
    print(f"{'='*60}\n")

    for iteration in range(1, cfg.NUM_ITERATIONS + 1):
        print(f"\n── Iteration {iteration}/{cfg.NUM_ITERATIONS} ──────────────────────────")

        # ── 1. Self-play ──────────────────────────────────────────────────────
        champion.eval()
        print(f"  Generating {cfg.SELF_PLAY_GAMES_PER_ITER} self-play games...")

        if single_process or cfg.NUM_SELF_PLAY_WORKERS <= 1:
            examples = run_self_play_single(
                champion, cfg, device,
                cfg.SELF_PLAY_GAMES_PER_ITER,
                cfg.MCTS_SIMULATIONS_TRAIN,
            )
        else:
            examples = run_self_play_batch(
                champion, cfg,
                cfg.SELF_PLAY_GAMES_PER_ITER,
                cfg.MCTS_SIMULATIONS_TRAIN,
            )

        buffer.add(examples)
        print(f"  Buffer size: {len(buffer):,} / {cfg.REPLAY_BUFFER_SIZE:,}")

        # ── 2. Train ──────────────────────────────────────────────────────────
        if not buffer.is_ready():
            print(f"  Buffer not ready yet ({len(buffer)} < {cfg.BATCH_SIZE}), skipping train")
            continue

        pre_train_state = clone_network(champion).state_dict()

        print(f"  Training {cfg.TRAIN_STEPS_PER_ITER} steps (batch={cfg.BATCH_SIZE})...")
        champion.train()
        total_l, value_l, policy_l = trainer.train_epoch(buffer)
        trainer.step_scheduler()

        # Buffer-triggered LR decay: drop LR when buffer first fills,
        # then drop again after 20 more iterations at capacity
        buffer_full = len(buffer) >= cfg.REPLAY_BUFFER_SIZE
        if buffer_full and not buffer_was_full:
            buffer_was_full = True
            buffer_full_iter = iteration
        if buffer_was_full and lr_drop_stage == 0:
            trainer.set_lr(cfg.LR_BUFFER_FULL_DROP)
            lr_drop_stage = 1
            print(f"  LR → {cfg.LR_BUFFER_FULL_DROP} (buffer full at iter {iteration})")
        if buffer_was_full and lr_drop_stage == 1 and iteration >= buffer_full_iter + 20:
            trainer.set_lr(cfg.LR_BUFFER_FULL_DECAY)
            lr_drop_stage = 2
            print(f"  LR → {cfg.LR_BUFFER_FULL_DECAY} (sustained buffer-full training)")

        print(f"  Loss: total={total_l:.4f}  value={value_l:.4f}  policy={policy_l:.4f}")

        warning = plm.record(policy_l, elo.elo)
        if warning:
            print(f"  ⚠ WARNING: {warning}")

        tb.log(iteration, **{
            "loss/total": total_l,
            "loss/value": value_l,
            "loss/policy": policy_l,
            "training/lr": trainer.current_lr(),
            "training/buffer_size": len(buffer),
        })
        if iteration % cfg.CHECKPOINT_INTERVAL == 0:
            tb.log_network(champion, iteration)

        # ── 3. Evaluate every EVAL_INTERVAL iterations ────────────────────────
        heuristic_wr = 0.0
        snap_wr = 0.0

        if iteration % cfg.EVAL_INTERVAL == 0:
            # vs Heuristic — stable external benchmark for tracking progress
            print(f"  Evaluating vs heuristic ({cfg.EVAL_GAMES} games)...")
            champion.eval()
            heuristic_wr, hw, hl, hd = evaluate_vs_heuristic(
                champion, cfg, device,
                num_games=cfg.EVAL_GAMES,
                num_simulations=cfg.MCTS_SIMULATIONS_EVAL,
            )
            print(f"  vs Heuristic: {heuristic_wr:.1%}  (W{hw} L{hl} D{hd})")

            # Regression guard: if win rate drops significantly, revert weights
            regression_margin = 0.15
            if heuristic_wr < best_heuristic_wr - regression_margin and best_heuristic_wr > 0:
                print(f"  Regression detected ({heuristic_wr:.1%} vs best {best_heuristic_wr:.1%}), "
                      f"reverting to pre-train weights")
                champion.load_state_dict(pre_train_state)
                champion.to(device)
            else:
                best_heuristic_wr = max(best_heuristic_wr, heuristic_wr)

            # Save a snapshot once heuristic WR reaches the trigger threshold
            if not snapshot_saved and heuristic_wr >= cfg.SNAPSHOT_HEURISTIC_WR_TRIGGER:
                snapshot_path = save_snapshot(champion, iteration, cfg)
                snapshot_net = clone_network(champion)
                snapshot_net.eval()
                snapshot_saved = True
                print(f"  Snapshot benchmark saved at iteration {iteration} "
                      f"(heuristic WR={heuristic_wr:.0%}) → {snapshot_path}")

            # Evaluate vs snapshot once it exists
            if snapshot_net is not None:
                print(f"  Evaluating vs snapshot ({cfg.SNAPSHOT_EVAL_GAMES} games)...")
                champion.eval()
                snap_wr, sw, sl, sd = evaluate_vs_snapshot(
                    champion, snapshot_net, cfg, device,
                    num_games=cfg.SNAPSHOT_EVAL_GAMES,
                    num_simulations=cfg.MCTS_SIMULATIONS_EVAL,
                )
                print(f"  vs Snapshot: {snap_wr:.1%}  (W{sw} L{sl} D{sd})")
                tb.log(iteration, **{"eval/vs_snapshot_wr": snap_wr})

        # Update Elo only when evaluation actually ran
        if iteration % cfg.EVAL_INTERVAL == 0:
            current_elo = elo.update(heuristic_wr)
        else:
            current_elo = elo.elo

        # ── 4. Checkpoint ─────────────────────────────────────────────────────
        if iteration % cfg.CHECKPOINT_INTERVAL == 0:
            path = save_checkpoint(champion, iteration, current_elo, cfg)
            save_champion(champion, cfg)
            print(f"  Checkpoint saved: {path}")

        # ── 5. Log ────────────────────────────────────────────────────────────
        logger.log(
            iteration=iteration,
            elo=round(current_elo, 1),
            vs_heuristic_wr=round(heuristic_wr, 4),
            vs_snapshot_wr=round(snap_wr, 4),
            value_loss=round(value_l, 5),
            policy_loss=round(policy_l, 5),
            total_loss=round(total_l, 5),
            buffer_size=len(buffer),
            lr=round(trainer.current_lr(), 6),
        )
        logger.print_row(
            iteration,
            elo=current_elo,
            value_loss=value_l,
            policy_loss=policy_l,
            vs_heuristic_wr=heuristic_wr,
            buffer_size=len(buffer),
        )
        tb.log(iteration, **{
            "eval/elo": current_elo,
            "eval/vs_heuristic_wr": heuristic_wr,
        })

        # ── 6. Convergence check (only after enough evaluations) ─────────────
        num_evals = iteration // cfg.EVAL_INTERVAL
        if num_evals >= 10 and elo.has_plateaued(threshold=5.0, window=10):
            print("\n✓ Elo has plateaued (<5 gain/iter for 10 evaluations). Training converged.")
            break

    # ── Final save ────────────────────────────────────────────────────────────
    tb.close()
    save_champion(champion, cfg)
    print(f"\nTraining complete. Champion saved to {cfg.CHECKPOINT_DIR}/champion.pth")
    print(f"Final Elo: {elo.elo:.1f}")
    print(f"Total time: {logger.elapsed_str()}")
    print(f"Log: {cfg.LOG_FILE}")
    print(f"TensorBoard: tensorboard --logdir {cfg.TENSORBOARD_DIR}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gomoku AlphaZero")
    parser.add_argument(
        "--mode", choices=["quick", "validate", "full"], default="quick",
        help="'quick' = 5x5/3-in-a-row (~15 min), 'validate' = 6x6/4-in-a-row (~2 hrs), 'full' = 10x10/5-in-a-row"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from champion checkpoint")
    parser.add_argument("--single", action="store_true", help="Disable multiprocessing (debug mode)")
    args = parser.parse_args()

    if args.mode == "quick":
        cfg = Config.for_quick_test()
        print("Mode: QUICK TEST (5×5, 3-in-a-row)")
    elif args.mode == "validate":
        cfg = Config.for_validation()
        print("Mode: VALIDATION (6×6, 4-in-a-row)")
    else:
        cfg = Config.for_full()
        print("Mode: FULL TRAINING (10×10, 5-in-a-row)")

    # Windows requires this guard for multiprocessing
    mp.freeze_support()

    main(cfg, resume=args.resume, single_process=args.single)
