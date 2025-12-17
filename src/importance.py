"""
This file computes point importance scores from the trained model
"""

import torch
import json
from pathlib import Path

from src.model import PointImportanceModel
from src.dataset import MatchDataset


def load_model(model_path: Path = Path("models/best_model.pt")) -> PointImportanceModel:
    """Load trained model."""
    model = PointImportanceModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def get_win_probability(model: PointImportanceModel, match: dict) -> list[float]:
    """
    Get win probability at each point in a match.

    Returns list of P(player 1 wins) after each point.
    """
    dataset = MatchDataset([match])
    sample = dataset[0]

    with torch.no_grad():
        probs = model(sample["points"].unsqueeze(0), torch.tensor([sample["length"]]))

    # Model outputs P(player 2 wins), so flip it
    probs = 1 - probs

    return probs[0, : sample["length"]].tolist()


def compute_importance_delta(probs: list[float]) -> list[float]:
    """
    Simple importance: |P(after) - P(before)|

    How much did each point shift the win probability?
    """
    importance = []
    prev = 0.5  # Starting probability

    for p in probs:
        importance.append(abs(p - prev))
        prev = p

    return importance


def analyze_match(model: PointImportanceModel, match: dict) -> dict:
    """
    Full analysis of a single match.
    """
    probs = get_win_probability(model, match)
    importance = compute_importance_delta(probs)

    # Find most important points
    top_indices = sorted(
        range(len(importance)), key=lambda i: importance[i], reverse=True
    )[:10]

    # Find the last time the probability crossed 0.5
    decision_point = 0
    for i, p in enumerate(probs):
        if i > 0:
            prev = probs[i - 1]
            if (prev < 0.5 and p >= 0.5) or (prev >= 0.5 and p < 0.5):
                decision_point = i

    return {
        "match_id": match["match_id"],
        "player1": match["player1"],
        "player2": match["player2"],
        "winner": match["winner"],
        "n_points": len(probs),
        "win_probability": probs,
        "importance": importance,
        "top_important_points": top_indices,
        "decision_point": decision_point,
        "final_prob": probs[-1],
    }


def aggregate_importance_by_score(
    matches: list[dict], model: PointImportanceModel
) -> dict:
    """
    Compute average importance for each point score (15-0, 30-30, etc.)
    Excludes (0,0) since that captures game transitions, not point importance.
    """
    from collections import defaultdict

    score_importance = defaultdict(list)

    for i, match in enumerate(matches):
        if i % 200 == 0:
            print(f"Processing {i}/{len(matches)}...")

        probs = get_win_probability(model, match)
        importance = compute_importance_delta(probs)

        for j, point in enumerate(match["points"]):
            if j >= len(importance):
                break

            if point["is_tiebreak"]:
                continue

            # Skip (0,0) - it captures game transitions, not point importance
            if point["point1"] == 0 and point["point2"] == 0:
                continue

            key = (point["point1"], point["point2"])
            score_importance[key].append(importance[j])

    return {str(k): sum(v) / len(v) for k, v in score_importance.items()}


if __name__ == "__main__":
    model = load_model()

    with open("data/processed/matches.json") as f:
        matches = json.load(f)

    print(f"Loaded {len(matches)} matches")

    # Analyze one famous match
    slam_finals = [
        m
        for m in matches
        if "Roland_Garros-F" in m["match_id"] or "Wimbledon-F" in m["match_id"]
    ]

    if slam_finals:
        match = slam_finals[0]
        print(f"\nAnalyzing: {match['player1']} vs {match['player2']}")

        analysis = analyze_match(model, match)

        print(f"Total points: {analysis['n_points']}")
        print(f"Decision point: {analysis['decision_point']}")
        print(f"Top 5 important points: {analysis['top_important_points'][:5]}")
        print(f"Final win prob: {analysis['final_prob']:.3f}")
        print(
            f"Winner: Player {analysis['winner']} ({match['player1'] if analysis['winner'] == 1 else match['player2']})"
        )

    # Aggregate importance by score state
    print("\nComputing score-state importance (this takes a minute)...")
    score_importance = aggregate_importance_by_score(
        matches[:1000], model
    )  # Use subset for speed

    print("\nAverage importance by point score:")
    for score in ["(0, 0)", "(1, 1)", "(2, 2)", "(3, 2)", "(2, 3)"]:
        if score in score_importance:
            print(f"  {score}: {score_importance[score]:.4f}")

    # Save for visualization
    with open("data/processed/score_importance.json", "w") as f:
        json.dump(score_importance, f, indent=2)

    print("\nSaved score importance to data/processed/score_importance.json")
