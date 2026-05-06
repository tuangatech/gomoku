"""
mcts.py — Monte Carlo Tree Search with PUCT selection.

Key design decisions:
  - Tree reuse: after each real move, the subtree rooted at the chosen child
    is retained, saving work from previous search.
  - Value is always from the perspective of the node's current player.
    We NEGATE the value at each backup step (zero-sum game).
  - Dirichlet noise is added only at the ROOT and only during self-play
    (not during evaluation or human play).
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple

from game import GomokuGame
from network import PolicyValueNet
from config import Config


# ── Tree node ─────────────────────────────────────────────────────────────────

class MCTSNode:
    """A single node in the search tree."""

    __slots__ = ["prior", "visit_count", "value_sum", "children", "is_expanded"]

    def __init__(self, prior: float):
        self.prior: float = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_expanded: bool = False

    def q_value(self) -> float:
        """Mean value — 0 if never visited (optimistic for unvisited nodes)."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, parent_visit_count: int, c_puct: float) -> float:
        """PUCT score used for child selection.
        Q is negated because the node stores value from its own player's
        perspective, but the parent wants to maximize from the parent's
        perspective (opponent of the child's player)."""
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return -self.q_value() + exploration

    def is_leaf(self) -> bool:
        return not self.is_expanded


# ── MCTS ──────────────────────────────────────────────────────────────────────

class MCTS:
    """
    AlphaZero-style MCTS.

    Usage:
        mcts = MCTS(network, config, device)
        policy = mcts.get_action_probs(game, temperature=1.0, add_noise=True)
        action = np.random.choice(len(policy), p=policy)
        mcts.update_with_move(action)  # reuse tree
    """

    def __init__(self, network: PolicyValueNet, config: Config, device):
        self.net = network
        self.cfg = config
        self.device = device
        self._root: Optional[MCTSNode] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def get_action_probs(
        self,
        game: GomokuGame,
        num_simulations: int,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        Run MCTS from the current game state.

        Returns a probability distribution over all ACTION_SIZE actions.
        Illegal moves have probability 0.

        Args:
            game:            current game state (not mutated)
            num_simulations: number of MCTS simulations to run
            temperature:     1.0 → proportional to visits; ~0 → greedy
            add_noise:       True during self-play to encourage exploration
        """
        # (Re)initialize root if needed
        if self._root is None:
            self._root = MCTSNode(prior=0.0)

        # Expand root if not yet expanded
        if self._root.is_leaf():
            self._expand(self._root, game)

        # Add Dirichlet noise at root to encourage exploration during self-play
        if add_noise:
            self._add_dirichlet_noise(self._root, game.get_legal_moves())

        # Run simulations
        for _ in range(num_simulations):
            self._simulate(self._root, game.clone())

        # Build action probability vector from visit counts
        visit_counts = np.zeros(self.cfg.ACTION_SIZE, dtype=np.float32)
        for action, child in self._root.children.items():
            visit_counts[action] = child.visit_count

        return self._visit_counts_to_probs(visit_counts, temperature)

    def update_with_move(self, action: int) -> None:
        """
        Advance the tree root to the child corresponding to `action`.
        Reuses the subtree — retains all MCTS work done below that node.
        Call this after every real move (both player and bot).
        """
        if self._root is not None and action in self._root.children:
            self._root = self._root.children[action]
        else:
            self._root = None   # tree miss — will reinitialize on next call

    def reset(self) -> None:
        """Discard the entire tree (call at start of each new game)."""
        self._root = None

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(self, root: MCTSNode, game: GomokuGame) -> None:
        """
        One full MCTS simulation: select → expand → evaluate → backup.

        Value perspective convention:
          - Each node stores value from the perspective of the player to move
            at that node (i.e., the node's "owner").
          - _expand() returns value from the leaf's current player perspective.
          - is_terminal() returns outcome from the player who just moved
            (NOT the player to move at the terminal node).
          - _backup() expects value for the DEEPEST child in the path,
            then negates at each level going up.
        """
        path: list[Tuple[MCTSNode, int]] = []
        node = root

        # ── SELECT: walk down tree using PUCT ────────────────────────────────
        while not node.is_leaf():
            done, outcome = game.is_terminal()
            if done:
                # outcome is from the perspective of the player who just moved
                # (the parent's player). The last child in path is the current
                # node, whose player is the opponent → negate.
                root.visit_count += 1
                self._backup(path, -outcome)
                return

            action, child = self._select_child(node)
            path.append((child, action))
            node = child
            game.make_move(action)

        # ── TERMINAL CHECK ────────────────────────────────────────────────────
        done, outcome = game.is_terminal()
        if done:
            # Same as above: outcome is from the player who just moved,
            # the leaf node's player is the opponent → negate.
            root.visit_count += 1
            self._backup(path, -outcome)
            return

        # ── EXPAND & EVALUATE ────────────────────────────────────────────────
        value = self._expand(node, game)

        # ── BACKUP ───────────────────────────────────────────────────────────
        # value is from the leaf's current player perspective.
        # The leaf node IS the deepest child in path → store directly.
        root.visit_count += 1
        self._backup(path, value)

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Return (action, child_node) with highest PUCT score."""
        best_score = -float("inf")
        best_action = -1
        best_child = None
        n_parent = node.visit_count

        for action, child in node.children.items():
            score = child.ucb_score(n_parent, self.cfg.C_PUCT)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node: MCTSNode, game: GomokuGame) -> float:
        """
        Expand `node` by calling the neural network to get priors and value.
        Creates child nodes for all legal moves.
        Returns the value estimate (from current player's perspective).
        """
        legal_moves = game.get_legal_moves()
        state = game.get_state_tensor()

        policy_probs, value = self.net.predict(state, legal_moves, self.device)

        for action in legal_moves:
            node.children[action] = MCTSNode(prior=float(policy_probs[action]))

        node.is_expanded = True
        return value   # in [-1, +1], from perspective of game.current_player

    def _backup(self, path: list, value: float) -> None:
        """
        Propagate value back up the path.
        Value is NEGATED at each level because the game alternates players.

        Root visit count is incremented by the caller (_simulate) since
        the root is not included in the path.
        """
        for node, _ in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value   # flip perspective at each level

    def _add_dirichlet_noise(self, node: MCTSNode, legal_moves: list) -> None:
        """Add Dirichlet noise to root node priors (self-play only)."""
        if not node.children:
            return
        alpha = self.cfg.DIRICHLET_ALPHA
        eps = self.cfg.DIRICHLET_EPSILON
        noise = np.random.dirichlet([alpha] * len(legal_moves))
        for i, action in enumerate(legal_moves):
            if action in node.children:
                child = node.children[action]
                child.prior = (1 - eps) * child.prior + eps * noise[i]

    @staticmethod
    def _visit_counts_to_probs(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
        """
        Convert visit counts to a probability distribution using temperature.
        temperature ≈ 1.0 → proportional to counts (exploratory)
        temperature → 0   → argmax (greedy / deterministic)
        """
        if temperature < 1e-3:
            # Greedy: put all mass on the most-visited action
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
            return probs

        mask = visit_counts > 0
        if not mask.any():
            return np.zeros_like(visit_counts)
        log_counts = np.full_like(visit_counts, -np.inf)
        log_counts[mask] = np.log(visit_counts[mask]) / temperature
        log_counts -= log_counts[mask].max()
        probs = np.exp(log_counts)
        probs /= probs.sum()
        return probs
