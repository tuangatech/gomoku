# Building a Gomoku AI from Scratch with AlphaZero & Reinforcement Learning

> Training a bot that starts knowing nothing and teaches itself to beat me — on a 10×10 board, five-in-a-row, using the same algorithm that mastered Go, chess, and shogi. Result: trained 6x6 board, 4-win took me 15 hours on a RTX-4070 and I stoped there :) 

---

## Table of Contents

1. [The Game: Gomoku 10×10](#1-the-game-gomoku-10x10)
2. [What is Reinforcement Learning?](#2-what-is-reinforcement-learning)
3. [AlphaZero: The Algorithm That Changed Everything](#3-alphazero-the-algorithm-that-changed-everything)
4. [The Proposed Solution & Why](#4-the-proposed-solution--why)
5. [Hardware: RTX 4070 & What It Means](#5-hardware-rtx-4070--what-it-means)
6. [Neural Network Architecture](#6-neural-network-architecture)
7. [Training Hyperparameters](#7-training-hyperparameters)
8. [Training Stopping Criteria](#8-training-stopping-criteria)
9. [Validation Strategy: Start with 6×6](#9-validation-strategy-start-with-6x6)
10. [Implementation Phases](#10-implementation-phases)
11. [The Play UI](#11-the-play-ui)
12. [On Rewards: Why Sparse is Better](#12-on-rewards-why-sparse-is-better)
13. [Common Pitfalls & How to Avoid Them](#13-common-pitfalls--how-to-avoid-them)
14. [Lessons Learned](#14-lessons-learned)
15. [Suggested Additions for Your Medium Post](#15-suggested-additions-for-your-medium-post)
16. [Glossary](#16-glossary)

---

## 1. The Game: Gomoku 10×10

**Gomoku** (五目並べ) is a two-player strategy game with ancient roots in East Asia. The name literally means "five pieces" in Japanese.

### Rules

- The board is **10×10** (100 intersections)
- Players alternate placing a stone of their color (Black, White)
- **Goal:** Be the first to place **exactly five stones in an unbroken row** — horizontally, vertically, or diagonally
- On standard Gomoku (free-style), exactly five wins; six-in-a-row does *not* count (this project uses standard free-style rules)

### Why 10×10 Is Interesting

The standard competitive Gomoku board is 15×15. A 10×10 board is a sweet spot for a personal project:

| Board | Cells | State Space (est.) | Training Time (single GPU) |
|-------|-------|-------------------|---------------------------|
| 5×5   | 25    | ~10^7              | ~2 hours (smoke test)    |
| 6×6   | 36    | ~10^12             | 15 hours                  |
| 10×10 | 100   | ~10^40             | ~5 days                   |
| 15×15 | 225   | ~10^105            | Weeks–months              |

At 10×10, the game is complex enough that the bot must develop real strategic intuition — threats, double-fours, open threes — yet tractable enough to train from scratch in a weekend.

### Strategic Depth on 10×10

A strong Gomoku player needs to recognize and execute:

- **Open-four** (○○○○_): Four in a row with both ends open — unstoppable in one move
- **Broken-four** (○○○_○): Four with a gap — also wins in one move
- **Double-three**: Creating two simultaneous open-threes — opponent can only block one
- **Fork attacks**: Forcing the opponent to defend in two places simultaneously

The bot will discover all of these patterns through self-play, with zero hand-coded knowledge.

---

## 2. What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of machine learning where an agent learns by **interacting with an environment**, receiving feedback, and improving its strategy over time.

### The Core Loop

```
State (s) → Agent picks Action (a) → Environment gives Reward (r) + Next State (s')
                    ↑                                                        |
                    └────────────── Agent updates its Policy ←──────────────┘
```

### Key Concepts

| Term | Meaning |
|------|---------|
| **State** | A snapshot of the board (who has stones where, whose turn) |
| **Action** | Placing a stone on one of the legal empty cells |
| **Policy (π)** | A function that maps a board state to a probability distribution over moves |
| **Value (V)** | The expected outcome (+1/0/−1) from a given board position |
| **Reward** | Feedback signal: +1 for winning, −1 for losing, 0 for draw |
| **Self-play** | The agent plays against itself to generate training data |

### Why RL for Games?

Games have three properties that make RL extremely powerful:
1. **Perfect simulation** — you can play millions of games in software
2. **Clear reward signal** — winning/losing is unambiguous
3. **No labeled training data needed** — the agent generates its own experience

---

## 3. AlphaZero: The Algorithm That Changed Everything

AlphaZero (DeepMind, 2017) started a revolution by mastering Go, chess, and shogi *from scratch* — no human game databases, no hand-crafted evaluation functions. Just the rules, compute, and self-play.

### What Came Before

Before AlphaZero, game AIs used **handcrafted evaluation functions**: thousands of rules like "a knight on e5 is worth +0.3 pawns." These were brittle, required domain experts, and had hard ceilings.

### The AlphaZero Insight

Instead of handcrafted evaluation, use a **neural network that learns what positions are good** — and combine it with **Monte Carlo Tree Search** to look ahead intelligently. Let the network and the tree search bootstrap each other through self-play.

### The Two Predecessors

```
AlphaGo (2016)          AlphaGo Zero (2017)        AlphaZero (2017)
─────────────────────   ───────────────────────    ──────────────────────
Trained on human games  Pure self-play, Go only    Pure self-play, any game
Separate policy + value Single dual-headed net      Same architecture
Go only                 Tabula rasa                 Generalized to chess/shogi
```

### How AlphaZero Works: The 5-Step Dance

```
┌─────────────────────────────────────────────────────────┐
│  ITERATION  (repeat ~200–500 times)                      │
│                                                          │
│  Step 1: SELF-PLAY                                       │
│    Current champion network plays N games vs itself      │
│    Each move uses MCTS (400–800 simulated games)         │
│    Store (state, MCTS_policy, final_outcome) triples     │
│                                                          │
│  Step 2: SAMPLE                                          │
│    Draw random mini-batches from replay buffer           │
│    Buffer holds last ~50,000 positions                   │
│                                                          │
│  Step 3: TRAIN                                           │
│    Update network to minimize combined loss              │
│    → Value head: predict who won from this position      │
│    → Policy head: predict MCTS visit distribution       │
│                                                          │
│  Step 4: EVALUATE                                        │
│    Network plays vs fixed heuristic bot (stable benchmark)│
│    If win rate regresses >15%, revert to previous weights│
│                                                          │
│  Step 5: CHECKPOINT                                      │
│    Save model weights. Go to Step 1.                     │
└─────────────────────────────────────────────────────────┘
```

### Monte Carlo Tree Search (MCTS) — The "Thinking" Part

MCTS is how the bot looks ahead during both self-play and actual play. Each MCTS call runs hundreds of **simulated games** from the current position, guided by the neural network.

```
For each simulation:
  1. SELECT:   Walk down tree using UCB/PUCT score
               (balances exploitation vs exploration)
  2. EXPAND:   At a leaf, add child nodes for legal moves
  3. EVALUATE: Neural network scores the leaf position
  4. BACKUP:   Propagate value back up to root

After 400 simulations:
  → Visit counts become the move probability distribution
  → Pick the move with highest visit count
```

The **PUCT formula** that guides selection:

```
score(s, a) = Q(s,a) + c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))

Q(s,a) = average value from simulations through this action
P(s,a) = neural network's prior probability for this action
N(s)   = total visit count of parent node
N(s,a) = visit count of this action
c_puct = exploration constant (tune: 1.0–5.0)
```

This is the heart of AlphaZero. The neural network doesn't need to be perfect — it just needs to be good enough to guide MCTS toward promising branches.

---

## 4. The Proposed Solution & Why

### Full Stack

```
Language:     Python 3.11+
DL Framework: PyTorch 2.x  (best GPU support, most AlphaZero reference code)
Board Logic:  NumPy         (fast array operations)
Training:     PyTorch + CUDA (RTX 4070)
Self-play:    Python multiprocessing (CPU cores) + GPU for net inference
Play UI:      Pygame         (self-contained desktop app, no server needed)
Monitoring:   TensorBoard    (loss curves, Elo, weight histograms)
```

### Why This Architecture Over Alternatives

| Alternative | Why Rejected |
|-------------|-------------|
| DQN / PPO | These are single-agent algorithms. Board games are two-player zero-sum — AlphaZero's MCTS + self-play handles this natively. DQN would require a separate opponent to train against. |
| Pure MCTS (no net) | Works, but plateaus quickly. The neural network is what allows AlphaZero to prune the search tree intelligently. Pure MCTS on a 10×10 board is too slow to explore deeply. |
| MuZero | More powerful (learns its own world model), but significantly more complex to implement. Overkill for this project. |
| TensorFlow | PyTorch has cleaner imperative debugging, easier custom MCTS integration, and more AlphaZero reference implementations. |
| Web UI | A Pygame desktop app requires zero server setup, works offline, and ships as a single Python script. |

### Why Sparse Rewards (No Shaping)

AlphaZero's MCTS **implicitly discovers** the value of four-in-a-row, blocking, and center control through tree search. Shaped intermediate rewards can introduce bias — the bot may chase sub-goals (making threats) rather than winning. Pure sparse rewards (+1/−1/0) keep the learning signal clean and aligned with the actual objective.

This is one of AlphaZero's core philosophical contributions: **let the algorithm discover strategy, don't tell it what strategy looks like.**

---

## 5. Hardware: RTX 4070 & What It Means

The RTX 4070 is an excellent single-GPU training platform for this scale of project.

| Spec | RTX 4070 | Impact |
|------|----------|--------|
| VRAM | 12 GB | Comfortably fits the network + batch in memory |
| CUDA Cores | 5,888 | Fast parallel convolution |
| Tensor Cores | 184 (4th gen) | Mixed-precision (FP16) training speedup |
| Memory BW | 504 GB/s | Fast data feeding |

### Recommended Optimizations for RTX 4070

```python
# Enable TF32 for extra speed on Ampere/Ada GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Mixed precision training (nearly 2x speedup, negligible quality loss)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# cuDNN benchmark mode (speeds up fixed-size convolutions)
torch.backends.cudnn.benchmark = True
```

### Expected Training Timeline on RTX 4070

| Phase | Wall-clock Time | Bot Strength |
|-------|----------------|-------------|
| Iteration 1–5 | ~4–8 hours | Plays legally, no strategy |
| Iteration 10–20 | ~12–24 hours | Blocks obvious threats |
| Iteration 30–50 | ~1.5–2 days | Creates double threats, hard to beat casually |
| Iteration 80–150 | ~3–5 days | Consistently beats most humans |

You can interrupt and resume training at any checkpoint. Playing against the bot after just 12–24 hours of training already demonstrates the dramatic improvement in early iterations.

---

## 6. Neural Network Architecture

The network takes a board state and outputs two things simultaneously: **where to play** (policy) and **who is winning** (value).

### Input Representation

```
Input tensor: shape [batch, 3, 10, 10]

Channel 0: current player's stones   (1.0 where current player has stone, else 0.0)
Channel 1: opponent's stones          (1.0 where opponent has stone, else 0.0)
Channel 2: turn indicator             (all 1.0 if Black's turn, all 0.0 if White's turn)
```

The turn channel is critical — the same board position looks different to each player.

### Network Body: Residual CNN

```
Input [3, 10, 10]
    ↓
Conv2d(3 → 128, kernel=3, pad=1) + BN + ReLU
    ↓
ResBlock × 6  (each: Conv→BN→ReLU→Conv→BN + skip, 128 filters)
    ↓
    ├──────────────────────────────┐
    ↓                              ↓
POLICY HEAD                   VALUE HEAD
Conv(128→2, 1×1) + BN + ReLU  Conv(128→1, 1×1) + BN + ReLU
Flatten → FC(200→100)          Flatten → FC(100→256) → ReLU
Softmax over 100 cells          FC(256→1) → tanh
    ↓                              ↓
P(s,a): prob distribution      V(s): scalar in [-1, +1]
over 100 legal moves
```

### Why Residual Blocks?

Skip connections allow gradients to flow cleanly through deep networks. Without them, training 6+ convolutional layers on a 10×10 board would suffer from vanishing gradients and slow convergence. This architecture is directly from the original AlphaZero paper.

### Parameter Count (approximate)

| Component | Parameters |
|-----------|-----------|
| Initial conv | ~3,500 |
| 6 Residual blocks | ~4.7M |
| Policy head | ~25,800 |
| Value head | ~32,000 |
| **Total** | **~4.8M** |

4.8M parameters fits comfortably in the RTX 4070's 12GB VRAM, even with a large batch size.

### Combined Training Loss

```
L = MSE(v, z) + CrossEntropy(π, p) + λ||θ||²

v = network value output
z = actual game outcome (+1 win, -1 loss, 0 draw) from current player's perspective
π = network policy output (probability over 100 cells)
p = MCTS visit count distribution (normalized)
λ = L2 regularization coefficient (e.g., 1e-4)
```

The two losses train the network simultaneously to be a good evaluator *and* a good move predictor.

---

## 7. Training Hyperparameters

### Core Hyperparameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| **MCTS simulations / move** | 400 (train), 800 (eval/play) | More = stronger but slower |
| **Temperature τ (early game)** | 1.0 (first 15 moves) | Encourages exploration of diverse positions |
| **Temperature τ (late game)** | → 0 (greedy) | Deterministic after exploration phase |
| **Dirichlet noise α** | 0.3 | Added to root node priors — forces exploration |
| **Dirichlet ε** | 0.25 | Weight of noise vs network prior at root |
| **c_puct** | 2.5 | Exploration constant in PUCT formula |
| **Replay buffer size** | 50,000 positions | Rolling window — older data expires |
| **Batch size** | 512 | Good throughput on RTX 4070 |
| **Learning rate** | 1e-3 → 3e-4 → 1e-4 | Buffer-triggered decay: drop when buffer fills, then again after 20 iterations |
| **L2 regularization** | 1e-4 | Prevents overfitting to recent games |
| **Self-play games / iteration** | 100–150 | Before each training update cycle |
| **Training steps / iteration** | 500–1000 mini-batches | After each self-play batch |
| **Evaluation games** | 40 | Challenger vs champion |
| **Promotion threshold** | 55% win rate | New model becomes champion |
| **Residual blocks** | 6–8 | Start at 6; scale up if training is stable |
| **Filters per layer** | 128 | Standard from paper |

### Dirichlet Noise — Why It Matters

Without noise, the bot would always explore the same moves at the root and never discover unconventional strategies. Adding Dirichlet noise at the root node ensures each game explores slightly different opening lines, which diversifies the training data.

```python
# At root node only — blend network prior with noise
noise = np.random.dirichlet([alpha] * n_legal_moves)
noisy_prior = (1 - epsilon) * network_prior + epsilon * noise
```

### Temperature Schedule

```
Moves 1–15:  τ = 1.0   → sample proportionally to visit counts (diverse play)
Move 16+:    τ → 0     → pick highest visit count (strong play)
```

This gives the replay buffer diverse early positions (critical for generalization) while ensuring the bot plays its best moves late in the game.

### Learning Rate Schedule: Buffer-Triggered Decay

The original plan used iteration-based step decay (drop at iterations 100 and 200). Validation training revealed a better signal: **the replay buffer reaching capacity**.

When the buffer fills, the training distribution shifts — old weak-play examples are being evicted and replaced by current-strength games. The data becomes more homogeneous and the gradients more correlated. A high learning rate at this point causes the policy head to overshoot, producing the rising policy loss observed in validation.

```
Phase 1 (buffer filling):    LR = 1e-3     (aggressive learning from diverse data)
Phase 2 (buffer just full):  LR = 3e-4     (fine-tuning on stabilized distribution)
Phase 3 (+20 iters at cap):  LR = 1e-4     (precision tuning)
```

The iteration-based `MultiStepLR` scheduler still runs as a fallback (for very long training runs where iterations 100/200 matter), but the buffer-triggered drops typically fire first and override it.

---

## 8. Training Stopping Criteria

### Use All Three Signals Together

**Signal 1: Internal Elo Stabilization (Primary)**

Maintain a rolling Elo rating by having each checkpoint play the previous champion. Track Elo gain per iteration. When gain drops below ~5 Elo points for **10+ consecutive iterations**, the bot has plateaued.

```
Iteration 10:  +80 Elo  ← learning fast
Iteration 30:  +40 Elo  ← still improving
Iteration 80:  +15 Elo  ← slowing down
Iteration 120: +4  Elo  ← plateau → stop or lower LR
```

**Signal 2: Training Loss Convergence**

Plot value loss and policy loss per iteration. When both curves flatten over 20+ iterations with no meaningful downward trend, you've extracted most of the learning signal from the current architecture.

**Signal 3: The Human Test (Practical)**

When you cannot win a single game even when you try your hardest — that's your subjective "done." This is the most personally satisfying stopping criterion.

### What "Good Enough" Looks Like

| Milestone | Approximate Iterations | Wall-Clock |
|-----------|------------------------|------------|
| Stops making random moves | 5–10 | ~8 hours |
| Blocks 4-in-a-rows reliably | 15–25 | ~16 hours |
| Creates double-threat attacks | 30–50 | ~2 days |
| Beats casual human players | 50–80 | ~2.5–3 days |
| **Consistently beats you** | 80–150 | **~3–5 days** |

### Training Resume Strategy

Save checkpoints every 10 iterations. On Windows with PyTorch, you can:
- Stop training overnight
- Resume from the latest champion checkpoint next morning with `--resume`

The `--resume` flag warm-starts from the saved champion weights. Optimizer state (Adam momentum), replay buffer, and iteration count are **not** preserved — they rebuild from scratch. This causes a brief wobble in loss (Adam re-estimates momentum within a few hundred steps) and a short warm-up period (the replay buffer refills via self-play at the champion's current strength). Neither affects where training converges — the weights continue improving normally.

Without `--resume`, old checkpoints and logs are automatically cleared at the start of a fresh run.

---

## 9. Validation Strategy: Start Small

Before committing 4 days of GPU time to 10×10 training, validate your entire implementation in stages. The codebase is board-size agnostic by design (`BOARD_SIZE` and `WIN_LENGTH` are single config values).

### Three-stage validation

| Stage | Mode | Board | Time | What it proves |
|-------|------|-------|------|----------------|
| 1 | `--mode quick` | 5×5 / 3-in-a-row | ~15 min | Pipeline runs end-to-end, no crashes or NaNs |
| 2 | `--mode validate` | 6×6 / 4-in-a-row | ~2-4 hrs | Bot learns real tactics, beats heuristic >95%, beats snapshot |
| 3 | `--mode full` | 10×10 / 5-in-a-row | ~3-5 days | Full strategic depth |

### Why start with 5×5?

The 5×5/3-in-a-row game has a tiny state space (~10^7 positions). The bot should converge within ~15 iterations. If it doesn't improve at all, something is broken — and you've spent 15 minutes, not 2 hours, finding out.

### Why 6×6 next?

Training on 6×6 is not a compromise — it is a **debugging tool**. Research confirms this is sufficient: one AlphaZero Gomoku study showed the model learned a winning strategy on a 6×6 board after just a few hours on a budget GPU.

In 2 hours of training on 6×6, you can verify:

| Check | What It Tells You |
|-------|------------------|
| Value loss decreasing | Network is getting gradient signal |
| Policy loss decreasing | MCTS priors are improving |
| vs Heuristic >70% by iter 10 | Self-play and evaluation logic correct |
| Bot blocks your 3-in-a-row | Win detection and MCTS backup correct |
| MCTS visit counts vary | Dirichlet noise working |

### The Heuristic Bot: Your Stable Benchmark

A simple rule-based bot (win if possible → block opponent's win → play toward center) serves as a **fixed external yardstick** throughout training. Unlike the champion model which keeps improving, the heuristic never changes — so you can plot "% wins vs heuristic" across iterations and get a clean convergence curve.

**Critical rule:** Use the heuristic only for evaluation, never for training. Training against a fixed opponent biases the bot toward beating that specific style.

### Validation Milestones (6×6, ~4 hours)

```
Iteration 1–3:   vs Heuristic  20–40%  (random play, learning basic moves)
Iteration 5–10:  vs Heuristic  50–70%  (blocks threats, creates simple attacks)
Iteration 15–25: vs Heuristic  80–95%  (double threats; snapshot benchmark saved at ~80%)
Iteration 25–50: vs Heuristic  95%+    (saturated — watch vs Snapshot instead)
Iteration 50+:   vs Snapshot   60–80%  (clearly stronger than earlier self)
Iteration 80+:   vs Snapshot   80%+    (strong convergence — switch to 10x10)
```

If you reach iteration 15 with vs-heuristic win rate still below 50%, something is broken — check the MCTS value backup (missing negation is the most common bug).

Once the heuristic is saturated at 95%, **stop watching it** and focus on the snapshot win rate — that's your real progress indicator for the rest of training.

---

## 10. Implementation Phases

### Phase 0: Environment Setup (Day 1, ~1 hour)

**Goal:** Python environment, GPU verification, project skeleton

```
gomoku_az/
├── game.py          # Board logic
├── mcts.py          # MCTS algorithm
├── network.py       # ResNet policy-value network
├── self_play.py     # Self-play data generation
├── trainer.py       # Training loop
├── evaluator.py     # Champion vs challenger
├── replay_buffer.py # Experience replay
├── config.py        # All hyperparameters in one place
├── play_gui.py      # Pygame play interface
├── utils.py         # Logging, Elo calculation, helpers
└── checkpoints/     # Saved models
```

**Deliverable:** `python -c "import torch; print(torch.cuda.is_available())"` prints `True`

---

### Phase 1: Game Logic (`game.py`) (Day 1, ~2 hours)

**Goal:** A correct, fast board implementation

```python
class GomokuGame:
    BOARD_SIZE = 10
    WIN_LENGTH = 5

    def __init__(self):
        self.board = np.zeros((10, 10), dtype=np.int8)  # 0=empty, 1=black, -1=white
        self.current_player = 1
        self.move_history = []

    def get_state_tensor(self) -> np.ndarray:
        # Returns [3, 10, 10] input for network

    def get_legal_moves(self) -> List[int]:
        # Returns list of flat indices (0–99) of empty cells

    def make_move(self, action: int) -> None:
        # Places stone, switches player

    def check_winner(self) -> Optional[int]:
        # Returns 1, -1, or None (no winner yet)

    def is_terminal(self) -> Tuple[bool, float]:
        # Returns (done, outcome_for_current_player)

    def clone(self) -> 'GomokuGame':
        # Deep copy for MCTS simulations
```

**Key implementation note:** Win detection runs on every move. Optimize it — check only the 4 directions from the last-placed stone, not the entire board.

**Tests to write:**
- Five horizontal stones → correct winner
- Five vertical stones → correct winner  
- Five diagonal stones (both diagonals) → correct winner
- Full board, no winner → draw
- Legal moves decrease by 1 per move

---

### Phase 2: Neural Network (`network.py`) (Day 1, ~2 hours)

**Goal:** Dual-headed ResNet in PyTorch with correct input/output shapes

```python
class ResidualBlock(nn.Module):
    def __init__(self, filters=128):
        # Conv→BN→ReLU→Conv→BN + skip connection

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=10, num_res_blocks=6, filters=128):
        # Initial conv + N residual blocks + policy head + value head

    def forward(self, x):
        # x: [batch, 3, 10, 10]
        # returns: policy_logits [batch, 100], value [batch, 1]

    def predict(self, board_state: np.ndarray):
        # Single-position inference → (policy_probs, value_scalar)
```

**Test:** Forward pass on a random input tensor; verify output shapes.

---

### Phase 3: MCTS (`mcts.py`) (Day 1–2, ~4 hours)

**Goal:** Correct MCTS with PUCT selection, value backup, and tree reuse

```python
class MCTSNode:
    def __init__(self, prior: float):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.prior: float = prior

    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

class MCTS:
    def __init__(self, network, num_simulations=400, c_puct=2.5):
        ...

    def search(self, game: GomokuGame) -> np.ndarray:
        # Returns visit count distribution over 100 cells (the "improved policy")

    def _select(self, node, game) -> Tuple[MCTSNode, GomokuGame]:
        # Walk tree using PUCT until leaf

    def _expand_and_evaluate(self, node, game) -> float:
        # Call network, create child nodes with priors

    def _backup(self, path, value):
        # Update visit counts and value sums (negate value at each level)
```

**Critical detail:** When backing up values, **negate the value at each level** because the game is zero-sum — what's good for one player is bad for the other.

**Tree reuse:** After each real move, retain the subtree rooted at the chosen child. This is a significant speedup since partial MCTS work from the previous move remains valid.

---

### Phase 4: Self-Play (`self_play.py`) (Day 2, ~2 hours)

**Goal:** Generate training data via self-play

```python
def run_self_play_game(network, mcts_simulations=400) -> List[Tuple]:
    """
    Play one game, return list of:
    (board_state_tensor, mcts_policy, game_outcome_for_that_player)
    """
    game = GomokuGame()
    mcts = MCTS(network, mcts_simulations)
    game_history = []

    while not game.is_terminal()[0]:
        state = game.get_state_tensor()
        policy = mcts.search(game)  # MCTS visit distribution

        # Apply temperature to policy
        move = sample_move(policy, temperature)
        game_history.append((state, policy, game.current_player))
        game.make_move(move)

    # Assign outcomes retroactively
    winner = game.check_winner()
    training_examples = []
    for (state, policy, player) in game_history:
        outcome = +1 if player == winner else -1 if winner else 0
        training_examples.append((state, policy, outcome))

    return training_examples
```

**Parallelization:** Run multiple self-play games in parallel using `multiprocessing.Pool`. Each worker process loads the network weights, plays games on CPU, and returns examples. The GPU is used for training between self-play batches.

---

### Phase 5: Replay Buffer & Training (`trainer.py`) (Day 2, ~3 hours)

**Goal:** Stable training loop with experience replay

```python
class ReplayBuffer:
    def __init__(self, max_size=50_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, examples): ...
    def sample(self, batch_size) -> Tuple[Tensor, Tensor, Tensor]: ...

class Trainer:
    def __init__(self, network, config):
        self.optimizer = optim.Adam(network.parameters(), lr=config.LR, weight_decay=1e-4)
        self.scaler = GradScaler()  # Mixed precision

    def train_step(self, states, policies, outcomes):
        with autocast():
            pred_policies, pred_values = self.network(states)
            value_loss = F.mse_loss(pred_values.squeeze(), outcomes)
            policy_loss = -torch.mean(torch.sum(policies * F.log_softmax(pred_policies, dim=1), dim=1))
            loss = value_loss + policy_loss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item(), value_loss.item(), policy_loss.item()
```

---

### Phase 6: Evaluator (`evaluator.py`) (Day 2–3, ~2 hours)

**Goal:** Track training progress and guard against regression

The primary evaluation signal is win rate against a **fixed heuristic bot** (win-if-possible → block-opponent-win → play-toward-center). This gives a clean, monotonic progress curve because the benchmark never changes — unlike champion-vs-challenger where both sides are moving targets.

```python
def evaluate_vs_heuristic(network, config, device, num_games=20) -> Tuple:
    """
    Play num_games between neural net (MCTS) and heuristic bot.
    Alternate who plays Black (first-mover advantage is real).
    Returns (win_rate, wins, losses, draws) for the network.
    """

# Regression guard in training loop:
# If heuristic win rate drops >15% below the best seen so far,
# revert to pre-training weights to prevent a bad iteration from
# poisoning future self-play data.
if heuristic_wr < best_heuristic_wr - 0.15:
    champion.load_state_dict(pre_train_state)   # revert
else:
    best_heuristic_wr = max(best_heuristic_wr, heuristic_wr)
```

**Why a fixed heuristic instead of champion-vs-challenger?** In canonical AlphaZero, a new model must beat the current champion >55% to get promoted. That works well at scale, but in early/mid training on a single GPU, the heuristic benchmark is more informative: it gives an absolute measure of strength rather than a relative one between two potentially weak models. The regression guard provides the same safety net as the promotion gate — preventing bad iterations from cascading — without requiring a separate champion snapshot.

**Why alternate colors?** Black (first mover) has a theoretical advantage in Gomoku. Alternating colors ensures the evaluation is fair regardless of position.

### Snapshot Evaluation — Stronger Benchmark After Heuristic Saturates

The heuristic bot saturates at ~95% win rate early in training (by iteration 15 in 6×6 validation). Once saturated, it can no longer differentiate between a moderately strong and a very strong model.

**Solution:** When the heuristic win rate first reaches 80%, the training loop saves a **frozen snapshot** of the current champion as `snapshot_benchmark.pth`. From that point on, every evaluation round also plays against this snapshot. Because the snapshot is a real neural network (not a rule-based bot), it provides a much harder challenge that continues to be informative deep into training.

```python
def evaluate_vs_snapshot(challenger, snapshot, config, device, num_games=20):
    """Champion vs frozen earlier self — harder benchmark."""
    return evaluate_models(challenger, snapshot, config, device, num_games=num_games)
```

The snapshot win rate (`vs_snapshot_wr`) is logged to both CSV and TensorBoard alongside the heuristic win rate. When the model starts beating the snapshot at >80%, you know it has improved substantially beyond where it was when the snapshot was taken.

On `--resume`, the snapshot is automatically loaded from `checkpoints/snapshot_benchmark.pth` so evaluation continuity is preserved across training sessions.

---

### Phase 7: Play UI — Pygame (`play_gui.py`) (Day 3, ~3 hours)

Full details in Section 10 below.

---

### Phase 8: Logging & Monitoring (`utils.py`) (Ongoing)

**Goal:** Full visibility into training progress

**CSV logging** — append-only CSV for programmatic analysis and custom plots.

**TensorBoard** — real-time interactive dashboards during training:

```bash
# Launch during or after training
tensorboard --logdir runs
```

Metrics logged per iteration:
- `loss/total`, `loss/value`, `loss/policy` — training signal health
- `eval/elo`, `eval/vs_heuristic_wr`, `eval/vs_snapshot_wr` — strength progression
- `training/lr`, `training/buffer_size` — training state
- `params/*`, `grads/*` — weight and gradient histograms (every checkpoint interval)

**Policy loss monitoring** — A `PolicyLossMonitor` tracks whether policy loss has risen for 10+ consecutive iterations while Elo is stalled. If detected, a warning is printed to the console. This catches the "model is forgetting how to play" failure mode early — declining value loss can mask it because the network still evaluates positions well but the policy head is degrading.

Runs are auto-named by board config (e.g. `runs/6x6_win4/`), so different training modes appear as separate runs for side-by-side comparison. Config hyperparameters are recorded as text metadata in each run.

---

## 10. The Play UI

### Design

A self-contained Pygame desktop application. No server, no browser, no dependencies beyond Python + Pygame.

```
┌──────────────────────────────────────────┐
│         GOMOKU vs AlphaZero Bot          │
│                                          │
│   ┌─────────────────────────────────┐   │
│   │  · · · · · · · · · ·            │   │
│   │  · · · · · · · · · ·            │   │
│   │  · · ○ · · · · · · ·            │   │
│   │  · · · ● · · · · · ·            │   │
│   │  · · · · · · · · · ·            │   │
│   │  · · · · · · · · · ·            │   │
│   │  · · · · · · · · · ·            │   │
│   │  · · · · · · · · · ·            │   │
│   │  · · · · · · · · · ·            │   │
│   │  · · · · · · · · · ·            │   │
│   └─────────────────────────────────┘   │
│   [New Game]  [Undo]   Thinking... ■■□  │
│   You: ○ Black    Bot: ● White          │
└──────────────────────────────────────────┘
```

### Features

- **Click to place** your stone on any legal cell
- **Bot "thinking" indicator** — spinner or pulsing animation while MCTS runs
- **Win highlight** — the winning five-in-a-row glows/highlights
- **New Game button** — restart immediately
- **Undo** — take back your last move (the bot un-plays too)
- **Move heatmap (optional)** — show the bot's policy distribution as colored cell overlays, so you can see what it's considering
- **Bot MCTS simulations at play time**: 800 (more than training — it plays its strongest game)

### Key Code Sketch

```python
def draw_board(screen, game, bot_policy=None):
    # Draw grid lines
    # Draw stones (circles)
    # If bot_policy is set, draw heatmap overlay
    # Highlight winning line if game over

def main():
    net = PolicyValueNet()
    net.load_state_dict(torch.load("checkpoints/champion.pth"))
    mcts = MCTS(net, num_simulations=800)
    game = GomokuGame()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and human_turn:
                cell = pixel_to_cell(event.pos)
                if cell in game.get_legal_moves():
                    game.make_move(cell)
                    human_turn = False

        if not human_turn and not game.is_terminal()[0]:
            policy = mcts.search(game)       # This takes ~0.5–2 seconds
            move = np.argmax(policy)
            game.make_move(move)
            human_turn = True

        draw_board(screen, game, bot_policy=policy)
        pygame.display.flip()
```

---

## 11. On Rewards: Why Sparse is Better

This deserves its own section because it's counterintuitive.

**The tempting approach:** Give +0.5 for creating a four-in-a-row, +0.3 for blocking one, +0.1 for center placement...

**The problem:** You become the bottleneck. Every shaped reward encodes *your* understanding of strategy. AlphaZero's power is that it discovers strategy *better than you know it*. Shaped rewards can actually cause:

- **Sub-goal pursuit**: Bot creates threats instead of finishing them (maximizing reward, not winning)
- **Brittle play**: Bot has learned to mimic human heuristics rather than finding deeper patterns
- **Reduced generalization**: Over-tuned to the reward function, not the game

**The counterintuitive truth:** A sparse reward signal over millions of self-play games forces the network to internalize *why* four-in-a-rows are good — because they lead to wins. This understanding is deeper and more robust than any hand-crafted reward.

---

## 12. Common Pitfalls & How to Avoid Them

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Forgetting to negate value in MCTS backup** | Bot plays randomly or makes obviously bad moves | Negate value at each level during backup — the game alternates players |
| **Not alternating colors in evaluation** | Biased win rates; first mover always wins evaluation | Alternate who plays Black across evaluation games |
| **Replay buffer too small** | Training oscillates; bot forgets earlier lessons | Keep at least 50K positions; use a rolling window |
| **No Dirichlet noise** | Bot converges to one opening line; brittle play | Always add Dirichlet noise at root during self-play |
| **Temperature never drops to 0** | Bot plays randomly in late-game positions | Apply τ→0 after move 15 during self-play |
| **Training too frequently on too-small batches** | Overfitting to recent games | Maintain a large replay buffer; train on diverse samples |
| **Not clamping the board input to [0,1]** | Network gets arbitrary large inputs | Standardize input channels to binary (0.0 / 1.0) |
| **Single process for self-play** | GPU sits idle; training is bottlenecked by CPU | Parallelize self-play across CPU cores |

---

## 14. Lessons Learned

### The MCTS Selection Bug: Losses Drop, Bot Loses Every Game

**Symptom:** After 4+ training iterations, value loss and policy loss decreased steadily (1.4 → 0.74 policy loss), confirming the network was learning. Yet `vs_heuristic` stayed at 0.0% — the bot couldn't beat a trivial rule-based opponent.

**Root cause:** The PUCT selection formula in `mcts.py` was not negating the child's Q-value. Each node stores value from its own player's perspective. When a parent selects among children, it must pick the child that's *worst* for the child's player (= best for the parent). Without the negation, MCTS was actively selecting moves that were best for the opponent.

**Why losses still decreased:** Self-play generates training data where both sides use the same broken MCTS. The training signal was internally consistent — the network learned real board patterns. But at evaluation time, every ply chose the opponent's best move, so the bot lost every game.

**Fix:** One character — negate Q in `ucb_score`:
```python
# Before (wrong): self.q_value() + exploration
# After (correct): -self.q_value() + exploration
```

**Takeaway:** In AlphaZero implementations, declining loss curves prove the network is learning but say nothing about whether MCTS is using those learned values correctly. Always validate with an external benchmark (like the heuristic bot) early and often — a 0% win rate despite falling losses is the signature of a value-perspective bug in the search tree.

### Validation Training Analysis: Rising Policy Loss & Saturated Benchmarks

After 34 iterations of 6×6 validation training, four observations led to concrete improvements:

**1. Heuristic opponent saturated at iteration 15.** Win rate hit 95% and flatlined — the benchmark could no longer differentiate between "good" and "much better." The fix: when heuristic WR first reaches 80%, save a **frozen snapshot** of the current champion. All subsequent evaluations also play against this snapshot, providing a harder benchmark that stays informative much longer.

**2. Policy loss rose steadily after iteration 5.** It dropped fast (1.81 → 1.41) in the first 5 iterations, then climbed back to ~1.51 by iteration 34. Meanwhile Elo kept rising — a confusing signal. Some policy loss increase is natural in AlphaZero (stronger self-play produces harder positions with more diffuse MCTS visit distributions), but a persistent rise without Elo gains indicates the model is overshooting. The fix: **buffer-triggered LR decay** — when the replay buffer first reaches capacity, LR drops from 1e-3 to 3e-4, then to 1e-4 after 20 more iterations. This gives the optimizer finer-grained updates once the training distribution stabilizes.

**3. No early warning system for policy loss divergence.** A `PolicyLossMonitor` now tracks whether policy loss rises for 10+ consecutive iterations while Elo is stalled. If both conditions are true, a warning is printed. This catches the "forgetting how to play" failure mode before it wastes hours of training.

**4. Training was stopped too early.** At 34 iterations with linear Elo gain (~14.4 Elo per 5 iterations), the model was nowhere near convergence. Iteration counts were increased: validation from 80 → 150, full from 300 → 500. The convergence check (Elo plateau detector) will still terminate early if progress genuinely stalls — the higher limit just ensures training isn't cut short while the model is still improving.

**Summary of changes applied:**

| Change | File(s) | Config Fields |
|--------|---------|---------------|
| Snapshot eval benchmark | evaluator.py, utils.py, train.py | `SNAPSHOT_EVAL_GAMES`, `SNAPSHOT_HEURISTIC_WR_TRIGGER` |
| Buffer-triggered LR decay | trainer.py, train.py, config.py | `LR_BUFFER_FULL_DROP`, `LR_BUFFER_FULL_DECAY` |
| Policy loss monitor | utils.py, train.py | (10-iteration window, hardcoded) |
| Increased iterations | config.py | `NUM_ITERATIONS`: quick 50, validate 80, full 300 |

---

## 15. Suggested Additions for Your Medium Post

Here are angles and additions that will make your post stand out and be genuinely interesting to readers:

### Technical Additions Worth Including

**1. A Training Progress GIF**
Record the bot at iterations 5, 20, 50, and 150. Side-by-side comparisons of early random play vs strong play are the most compelling visual you can include.

**2. The "Aha Moment" Scatter Plot**
Plot value loss vs iteration. You'll typically see a sharp drop around iteration 10–20 when the bot "figures out" basic tactics. Readers love concrete evidence of emergent intelligence.

**3. MCTS Visualization**
One screenshot showing the heatmap overlay of the bot's policy distribution — what it's "thinking" about — is worth a thousand words about how MCTS works.

**4. Elo Curve**
Your rolling Elo chart from training logs is a clean, objective story of improvement that general readers can understand even without ML background.

### Narrative Angles That Work Well

**5. "The Blank Slate" Framing**
Open the post with the moment the bot plays its first game — completely randomly. Then contrast with what it looks like 5 days later. The journey from nothing to "it beat me" is inherently compelling.

**6. The Strategy It Invented**
Play a few games against the final bot and note tactics you didn't explicitly program — double-threat setups, defensive patterns, endgame sequences. The emergent strategy angle resonates with general audiences.

**7. What AlphaZero Can't Do (Yet)**
Honest limitations make a post credible: your 10×10 bot would likely lose to a specialized Gomoku program trained for much longer. The bot doesn't "understand" — it pattern-matches on value/policy distributions. This nuance impresses technical readers.

**8. The Hardware Story**
Frame the RTX 4070 as a democratization story — the same class of algorithm that required Google-scale compute in 2017 now runs on consumer hardware in a weekend. That's genuinely remarkable.

### Structural Suggestions for the Post

```
Intro:       The blank slate moment. What AlphaZero achieved.
Section 1:   Gomoku as a test bed — why this game, why 10×10
Section 2:   The AlphaZero algorithm (MCTS + ResNet, 3 paragraphs max)
Section 3:   Training journey with the Elo curve / loss curve images
Section 4:   Playing against the finished bot — what it does that surprised you
Section 5:   Code architecture (link to GitHub repo)
Conclusion:  What you'd do next (15×15? MuZero? Rust rewrite of MCTS?)
```

**Estimated reading time:** 8–12 minutes (ideal for Medium)

**Tags to use:** Machine Learning, Reinforcement Learning, Python, Deep Learning, Board Games

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **AlphaZero** | DeepMind's 2017 RL algorithm that mastered Go, chess, and shogi from scratch using self-play |
| **MCTS** | Monte Carlo Tree Search — an algorithm that explores future game states using simulation |
| **PUCT** | Predictor + Upper Confidence Bound, the selection formula used in AlphaZero's MCTS |
| **Policy network** | The part of the neural network that predicts which moves are worth exploring |
| **Value network** | The part of the neural network that predicts the expected outcome from a position |
| **Residual block** | A CNN building block with a skip connection that enables deep networks to train stably |
| **Self-play** | The bot plays games against copies of itself to generate training data |
| **Replay buffer** | A rolling window of past game positions used to sample training batches |
| **Dirichlet noise** | Random noise added to MCTS root priors to ensure exploration of diverse moves |
| **Temperature (τ)** | Controls how randomly moves are sampled from the MCTS policy distribution |
| **Elo rating** | A numerical skill rating system used to track bot improvement across training |
| **Sparse rewards** | Only rewarding win/loss/draw at the end of each game — no intermediate signals |
| **Shaped rewards** | Intermediate rewards for sub-goals like creating threats (not used in this project) |
| **Champion model** | The current best model, replaced only when a challenger wins >55% of evaluation games |
| **Gomoku** | Five-in-a-row board game; this project uses free-style rules on a 10×10 board |

---

*Project repository structure and implementation details are covered in the Implementation Phases section. All code targets Python 3.11+, PyTorch 2.x, and CUDA 12.x for RTX 4070 compatibility.*
