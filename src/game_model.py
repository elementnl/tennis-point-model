"""
Game-level model: predicts whether server holds or gets broken.
Analyzes momentum, clutch performance, and point importance.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class GameDataset(Dataset):
    """
    Dataset of individual service games.
    Each item: sequence of point scores within one game → did server hold?
    """

    def __init__(self, games: list[dict], max_points: int = 24):
        self.games = games
        self.max_points = max_points

    @classmethod
    def from_matches(cls, matches: list[dict], **kwargs):
        """Extract all service games from matches."""
        games = []

        for match in matches:
            current_game = []
            prev_game_score = (0, 0)
            current_server = None
            prev_set_score = (0, 0)

            for pt in match["points"]:
                game_score = (pt["game1"], pt["game2"])
                set_score = (pt["set1"], pt["set2"])

                # Skip if we're in a tiebreak (game score is 6-6 or higher for both)
                if pt["game1"] >= 6 and pt["game2"] >= 6:
                    continue

                # Check for new game (score or set changed)
                if (
                    game_score != prev_game_score or set_score != prev_set_score
                ) and current_game:
                    if current_server == 1:
                        held = (
                            game_score[0] > prev_game_score[0]
                            or set_score[0] > prev_set_score[0]
                        )
                    else:
                        held = (
                            game_score[1] > prev_game_score[1]
                            or set_score[1] > prev_set_score[1]
                        )

                    if len(current_game) >= 4:  # Minimum 4 points per game
                        games.append(
                            {
                                "points": current_game.copy(),
                                "held": held,
                                "server": current_server,
                            }
                        )
                    current_game = []

                current_game.append(
                    {
                        "server_pts": pt["point1"],
                        "receiver_pts": pt["point2"],
                        "server_won": pt["point_winner"] == pt["server"],
                    }
                )
                current_server = pt["server"]
                prev_game_score = game_score
                prev_set_score = set_score

        print(f"Extracted {len(games)} service games")
        return cls(games, **kwargs)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        points = game["points"]

        n_points = min(len(points), self.max_points)
        features = torch.zeros(self.max_points, 3)

        for i, pt in enumerate(points[:n_points]):
            features[i] = torch.tensor(
                [
                    pt["server_pts"] / 4,
                    pt["receiver_pts"] / 4,
                    float(pt["server_won"]),
                ]
            )

        return {
            "points": features,
            "held": torch.tensor(1 if game["held"] else 0, dtype=torch.long),
            "length": n_points,
        }


class GameModel(nn.Module):
    """Simple LSTM to predict game outcome from point sequence."""

    def __init__(self, input_dim=3, hidden_dim=32, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, lengths=None):
        out, _ = self.lstm(x)
        logits = self.head(out).squeeze(-1)
        return torch.sigmoid(logits)


def train_game_model(matches: list[dict], epochs: int = 20):
    """Train the game-level model."""
    all_games = GameDataset.from_matches(matches)

    n = len(all_games.games)
    train_games = all_games.games[: int(n * 0.8)]
    val_games = all_games.games[int(n * 0.8) :]

    train_ds = GameDataset(train_games)
    val_ds = GameDataset(val_games)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    def collate(batch):
        return {
            "points": torch.stack([b["points"] for b in batch]),
            "held": torch.stack([b["held"] for b in batch]),
            "length": torch.tensor([b["length"] for b in batch]),
        }

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate)

    model = GameModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            probs = model(batch["points"], batch["length"])
            target = batch["held"].float().unsqueeze(1).expand(-1, probs.size(1))
            mask = torch.arange(probs.size(1)).expand(len(batch["length"]), -1) < batch[
                "length"
            ].unsqueeze(1)
            loss = nn.functional.binary_cross_entropy(probs, target, reduction="none")
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                probs = model(batch["points"], batch["length"])
                final_probs = torch.stack(
                    [
                        probs[i, batch["length"][i] - 1]
                        for i in range(len(batch["length"]))
                    ]
                )
                preds = (final_probs > 0.5).long()
                val_correct += (preds == batch["held"]).sum().item()
                val_total += len(batch["held"])

        print(
            f"Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_acc={val_correct/val_total:.3f}"
        )

    return model, train_ds


def compute_hold_probability_by_score(model: GameModel, dataset: GameDataset) -> dict:
    """Compute P(server holds) at each point score state."""
    model.eval()
    score_probs = defaultdict(list)

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            probs = model(
                sample["points"].unsqueeze(0), torch.tensor([sample["length"]])
            )
            probs = probs[0, : sample["length"]].tolist()

            for j, pt in enumerate(dataset.games[i]["points"][: sample["length"]]):
                key = (pt["server_pts"], pt["receiver_pts"])
                score_probs[key].append(probs[j])

    return {k: sum(v) / len(v) for k, v in score_probs.items()}


def theoretical_hold_probability(p: float = 0.65) -> dict:
    """
    Calculate theoretical hold probability at each score,
    assuming each point is independent with server winning prob p.
    """
    # Deuce solution: P(hold from deuce) = p^2 / (p^2 + (1-p)^2)
    p_deuce = (p * p) / (p * p + (1 - p) * (1 - p))

    memo = {}

    def prob_hold(s, r):
        if (s, r) in memo:
            return memo[(s, r)]

        # Server won game
        if s >= 4 and s - r >= 2:
            return 1.0
        # Server broken
        if r >= 4 and r - s >= 2:
            return 0.0
        # Deuce or AD situation
        if s >= 3 and r >= 3:
            return p_deuce

        result = p * prob_hold(s + 1, r) + (1 - p) * prob_hold(s, r + 1)
        memo[(s, r)] = result
        return result

    # Calculate for standard scores (0-3 each)
    result = {}
    for s in range(4):
        for r in range(4):
            result[(s, r)] = prob_hold(s, r)

    return result


def compare_actual_vs_theoretical(actual_probs: dict, p: float = 0.65) -> dict:
    """
    Compare actual hold probabilities to theoretical expectation.
    Positive diff = outperformance, negative = underperformance.
    """
    theoretical = theoretical_hold_probability(p)

    comparison = {}
    for score, actual in actual_probs.items():
        if score in theoretical:
            expected = theoretical[score]
            diff = actual - expected
            comparison[score] = {
                "actual": actual,
                "expected": expected,
                "diff": diff,
            }

    return comparison


def analyze_momentum(games: list[dict]) -> dict:
    """Calculate P(win | won previous) vs P(win | lost previous)"""
    wins_after_win = 0
    total_after_win = 0
    wins_after_loss = 0
    total_after_loss = 0

    for game in games:
        points = game["points"]
        for i in range(1, len(points)):
            prev_won = points[i - 1]["server_won"]
            curr_won = points[i]["server_won"]

            if prev_won:
                total_after_win += 1
                if curr_won:
                    wins_after_win += 1
            else:
                total_after_loss += 1
                if curr_won:
                    wins_after_loss += 1

    p_win_after_win = wins_after_win / total_after_win if total_after_win > 0 else 0
    p_win_after_loss = wins_after_loss / total_after_loss if total_after_loss > 0 else 0

    return {
        "p_win_after_win": p_win_after_win,
        "p_win_after_loss": p_win_after_loss,
        "momentum_effect": p_win_after_win - p_win_after_loss,
        "n_after_win": total_after_win,
        "n_after_loss": total_after_loss,
    }


def analyze_first_point_impact(games: list[dict]) -> dict:
    """Calculate hold rate conditioned on first point outcome."""
    holds_won_first = 0
    total_won_first = 0
    holds_lost_first = 0
    total_lost_first = 0

    for game in games:
        if len(game["points"]) == 0:
            continue

        won_first = game["points"][0]["server_won"]
        held = game["held"]

        if won_first:
            total_won_first += 1
            if held:
                holds_won_first += 1
        else:
            total_lost_first += 1
            if held:
                holds_lost_first += 1

    p_hold_won = holds_won_first / total_won_first if total_won_first > 0 else 0
    p_hold_lost = holds_lost_first / total_lost_first if total_lost_first > 0 else 0

    return {
        "p_hold_if_won_first": p_hold_won,
        "p_hold_if_lost_first": p_hold_lost,
        "first_point_impact": p_hold_won - p_hold_lost,
        "n_won_first": total_won_first,
        "n_lost_first": total_lost_first,
    }


def compute_point_leverage(score_probs: dict) -> dict:
    """Calculate swing in hold probability from winning vs losing each point."""
    leverage = {}

    next_if_win = {
        (0, 0): (1, 0),
        (0, 1): (1, 1),
        (0, 2): (1, 2),
        (0, 3): (1, 3),
        (1, 0): (2, 0),
        (1, 1): (2, 1),
        (1, 2): (2, 2),
        (1, 3): (2, 3),
        (2, 0): (3, 0),
        (2, 1): (3, 1),
        (2, 2): (3, 2),
        (2, 3): (3, 3),
        (3, 0): "hold",
        (3, 1): "hold",
        (3, 2): "hold",
        (3, 3): (4, 3),
    }

    next_if_lose = {
        (0, 0): (0, 1),
        (0, 1): (0, 2),
        (0, 2): (0, 3),
        (0, 3): "break",
        (1, 0): (1, 1),
        (1, 1): (1, 2),
        (1, 2): (1, 3),
        (1, 3): "break",
        (2, 0): (2, 1),
        (2, 1): (2, 2),
        (2, 2): (2, 3),
        (2, 3): "break",
        (3, 0): (3, 1),
        (3, 1): (3, 2),
        (3, 2): (3, 3),
        (3, 3): (3, 4),
    }

    for score in next_if_win.keys():
        win_next = next_if_win[score]
        lose_next = next_if_lose[score]

        if win_next == "hold":
            prob_if_win = 1.0
        elif win_next in score_probs:
            prob_if_win = score_probs[win_next]
        else:
            continue

        if lose_next == "break":
            prob_if_lose = 0.0
        elif lose_next in score_probs:
            prob_if_lose = score_probs[lose_next]
        else:
            continue

        leverage[score] = {
            "prob_if_win": prob_if_win,
            "prob_if_lose": prob_if_lose,
            "swing": prob_if_win - prob_if_lose,
        }

    return leverage


if __name__ == "__main__":
    with open("data/processed/matches.json") as f:
        matches = json.load(f)

    print("Training game-level model...")
    model, dataset = train_game_model(matches, epochs=20)

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/game_model.pt")

    print("\nComputing hold probability by score state...")
    score_probs = compute_hold_probability_by_score(model, dataset)

    # ==========================================
    # ACTUAL vs THEORETICAL
    # ==========================================
    print("\n" + "=" * 60)
    print("ACTUAL vs THEORETICAL (Clutch vs Choke)")
    print("=" * 60)
    print("Positive = CLUTCH (beat expectations)")
    print("Negative = CHOKE (worse than expected)\n")

    comparison = compare_actual_vs_theoretical(score_probs, p=0.65)
    sorted_comp = sorted(
        comparison.items(), key=lambda x: abs(x[1]["diff"]), reverse=True
    )

    print(f"{'Score':<10} {'Actual':<10} {'Expected':<10} {'Diff':<10}")
    print("-" * 40)
    for score, data in sorted_comp:
        score_str = f"{['0','15','30','40'][score[0]]}-{['0','15','30','40'][score[1]]}"
        print(
            f"{score_str:<10} {data['actual']:.1%}      {data['expected']:.1%}       {data['diff']:+.1%}"
        )

    # ==========================================
    # MOMENTUM
    # ==========================================
    print("\n" + "=" * 60)
    print("MOMENTUM ANALYSIS")
    print("=" * 60)

    momentum = analyze_momentum(dataset.games)
    print(
        f"\nP(win | won previous):  {momentum['p_win_after_win']:.1%}  (n={momentum['n_after_win']})"
    )
    print(
        f"P(win | lost previous): {momentum['p_win_after_loss']:.1%}  (n={momentum['n_after_loss']})"
    )
    print(f"\nMomentum effect: {momentum['momentum_effect']:+.1%}")

    if abs(momentum["momentum_effect"]) < 0.02:
        print("→ Effect negligible")
    else:
        print(f"→ Effect detected")

    # ==========================================
    # FIRST POINT
    # ==========================================
    print("\n" + "=" * 60)
    print("FIRST POINT IMPACT")
    print("=" * 60)

    first_pt = analyze_first_point_impact(dataset.games)
    print(
        f"\nP(hold | won first):  {first_pt['p_hold_if_won_first']:.1%}  (n={first_pt['n_won_first']})"
    )
    print(
        f"P(hold | lost first): {first_pt['p_hold_if_lost_first']:.1%}  (n={first_pt['n_lost_first']})"
    )
    print(f"\nFirst point impact: {first_pt['first_point_impact']:+.1%}")

    # ==========================================
    # LEVERAGE
    # ==========================================
    print("\n" + "=" * 60)
    print("POINT LEVERAGE")
    print("=" * 60)

    leverage = compute_point_leverage(score_probs)
    sorted_lev = sorted(leverage.items(), key=lambda x: x[1]["swing"], reverse=True)

    print(f"\n{'Score':<10} {'Win→':<10} {'Lose→':<10} {'Swing':<10}")
    print("-" * 40)
    for score, data in sorted_lev:
        score_str = f"{['0','15','30','40'][score[0]]}-{['0','15','30','40'][score[1]]}"
        print(
            f"{score_str:<10} {data['prob_if_win']:.0%}        {data['prob_if_lose']:.0%}        {data['swing']:.0%}"
        )

    # ==========================================
    # SAVE ALL DATA
    # ==========================================
    with open("data/processed/hold_probability_by_score.json", "w") as f:
        json.dump({str(k): v for k, v in score_probs.items()}, f, indent=2)

    with open("data/processed/actual_vs_theoretical.json", "w") as f:
        json.dump({str(k): v for k, v in comparison.items()}, f, indent=2)

    with open("data/processed/momentum_analysis.json", "w") as f:
        json.dump(momentum, f, indent=2)

    with open("data/processed/first_point_analysis.json", "w") as f:
        json.dump(first_pt, f, indent=2)

    with open("data/processed/point_leverage.json", "w") as f:
        json.dump({str(k): v for k, v in leverage.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("Saved all analysis to data/processed/")
    print("=" * 60)
