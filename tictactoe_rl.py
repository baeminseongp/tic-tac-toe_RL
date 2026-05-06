from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EMPTY = 0
X = 1
O = -1
Action = int
Board = Tuple[int, ...]


WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def empty_board() -> Board:
    return (EMPTY,) * 9


def legal_actions(board: Sequence[int]) -> List[Action]:
    return [i for i, cell in enumerate(board) if cell == EMPTY]


def apply_action(board: Sequence[int], action: Action, player: int) -> Board:
    if board[action] != EMPTY:
        raise ValueError(f"illegal action {action} on board {board}")
    next_board = list(board)
    next_board[action] = player
    return tuple(next_board)


def winner(board: Sequence[int]) -> int:
    for a, b, c in WIN_LINES:
        line_sum = board[a] + board[b] + board[c]
        if line_sum == 3:
            return X
        if line_sum == -3:
            return O
    return EMPTY


def is_draw(board: Sequence[int]) -> bool:
    return winner(board) == EMPTY and all(cell != EMPTY for cell in board)


def terminal_value(board: Sequence[int]) -> Optional[int]:
    won_by = winner(board)
    if won_by != EMPTY:
        return won_by
    if all(cell != EMPTY for cell in board):
        return 0
    return None


def encode_state(board: Sequence[int], player: int) -> str:
    symbols = {-1: "O", 0: ".", 1: "X"}
    return f"{''.join(symbols[cell] for cell in board)}:{'X' if player == X else 'O'}"


def decode_state(state: str) -> Tuple[Board, int]:
    raw_board, raw_player = state.split(":")
    values = {".": EMPTY, "X": X, "O": O}
    return tuple(values[ch] for ch in raw_board), X if raw_player == "X" else O


@dataclass
class Policy:
    logits: Dict[str, List[float]] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)

    def ensure_state(self, board: Sequence[int], player: int) -> str:
        state = encode_state(board, player)
        if state not in self.logits:
            self.logits[state] = [0.0] * 9
        return state

    def probabilities(self, board: Sequence[int], player: int, temperature: float = 1.0) -> List[float]:
        actions = legal_actions(board)
        if not actions:
            return [0.0] * 9

        state = self.ensure_state(board, player)
        state_logits = self.logits[state]
        temp = max(temperature, 1e-6)
        max_logit = max(state_logits[action] / temp for action in actions)
        weights = [0.0] * 9
        total = 0.0
        for action in actions:
            weight = math.exp((state_logits[action] / temp) - max_logit)
            weights[action] = weight
            total += weight
        return [weight / total for weight in weights]

    def choose_action(self, board: Sequence[int], player: int, temperature: float = 1.0) -> Action:
        probs = self.probabilities(board, player, temperature)
        threshold = self.rng.random()
        cumulative = 0.0
        for action, prob in enumerate(probs):
            cumulative += prob
            if threshold <= cumulative:
                return action
        return legal_actions(board)[-1]

    def greedy_action(self, board: Sequence[int], player: int) -> Action:
        probs = self.probabilities(board, player, temperature=0.05)
        return max(legal_actions(board), key=lambda action: probs[action])

    def update(self, trajectory: Iterable[Tuple[Board, int, Action]], winner_value: int, lr: float) -> None:
        for board, player, action in trajectory:
            reward = 0.0 if winner_value == 0 else (1.0 if winner_value == player else -1.0)
            probs = self.probabilities(board, player)
            state = self.ensure_state(board, player)
            for candidate in legal_actions(board):
                gradient = (1.0 if candidate == action else 0.0) - probs[candidate]
                self.logits[state][candidate] += lr * reward * gradient

    def save(self, path: Path, episodes: int, stats: Dict[str, int]) -> None:
        payload = {
            "algorithm": "tabular-reinforce-self-play",
            "episodes": episodes,
            "stats": stats,
            "logits": self.logits,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, seed: Optional[int] = None) -> "Policy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(logits=payload["logits"], rng=random.Random(seed))


def play_episode(policy: Policy, temperature: float) -> Tuple[List[Tuple[Board, int, Action]], int]:
    board = empty_board()
    player = X
    trajectory: List[Tuple[Board, int, Action]] = []

    while True:
        action = policy.choose_action(board, player, temperature)
        trajectory.append((board, player, action))
        board = apply_action(board, action, player)
        value = terminal_value(board)
        if value is not None:
            return trajectory, value
        player *= -1


def train(
    episodes: int,
    lr: float = 0.08,
    seed: int = 7,
    min_temperature: float = 0.2,
) -> Tuple[Policy, Dict[str, int]]:
    policy = Policy(rng=random.Random(seed))
    stats = {"x_win": 0, "o_win": 0, "draw": 0}

    for episode in range(episodes):
        progress = episode / max(episodes - 1, 1)
        temperature = max(min_temperature, 1.4 - 1.2 * progress)
        trajectory, winner_value = play_episode(policy, temperature)
        policy.update(trajectory, winner_value, lr)
        if winner_value == X:
            stats["x_win"] += 1
        elif winner_value == O:
            stats["o_win"] += 1
        else:
            stats["draw"] += 1

    return policy, stats


def evaluate(policy: Policy, games: int = 1000, seed: int = 11) -> Dict[str, int]:
    rng = random.Random(seed)
    stats = {"policy_win": 0, "random_win": 0, "draw": 0}

    for game in range(games):
        board = empty_board()
        policy_player = X if game % 2 == 0 else O
        player = X
        while True:
            if player == policy_player:
                action = policy.greedy_action(board, player)
            else:
                action = rng.choice(legal_actions(board))
            board = apply_action(board, action, player)
            value = terminal_value(board)
            if value is not None:
                if value == 0:
                    stats["draw"] += 1
                elif value == policy_player:
                    stats["policy_win"] += 1
                else:
                    stats["random_win"] += 1
                break
            player *= -1

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tic-tac-toe policy-gradient agent.")
    parser.add_argument("--episodes", type=int, default=80000)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("model.json"))
    return parser

