"""
Trace through a single match to understand the data flow.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/tennis_MatchChartingProject")


def main():
    matches = pd.read_csv(DATA_DIR / "charting-m-matches.csv")
    points = pd.read_csv(DATA_DIR / "charting-m-points-2020s.csv", low_memory=False)

    # Find a notable match
    slam_matches = matches[
        matches["Tournament"].str.contains(
            "Roland Garros|Wimbledon|US Open|Australian", na=False
        )
        & (matches["Round"] == "F")
    ]
    print(f"Grand Slam finals in dataset: {len(slam_matches)}")
    print(
        slam_matches[["match_id", "Player 1", "Player 2", "Tournament", "Date"]].head(
            10
        )
    )

    # Pick one
    match_id = slam_matches["match_id"].iloc[0]
    print(f"\n{'='*60}")
    print(f"Analyzing: {match_id}")
    print(f"{'='*60}")

    # Get match info
    match_info = matches[matches["match_id"] == match_id].iloc[0]
    print(f"\n{match_info['Player 1']} vs {match_info['Player 2']}")
    print(f"{match_info['Tournament']} {match_info['Round']}")
    print(f"Surface: {match_info['Surface']}, Best of: {match_info['Best of']}")

    # Get all points
    match_points = points[points["match_id"] == match_id].sort_values("Pt")
    print(f"\nTotal points: {len(match_points)}")

    # First 10 points
    print(f"\n--- First 10 points ---")
    print(
        match_points[
            ["Pt", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Svr", "PtWinner"]
        ].head(10)
    )

    # Last 10 points
    print(f"\n--- Last 10 points ---")
    print(
        match_points[
            ["Pt", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Svr", "PtWinner"]
        ].tail(10)
    )

    # Who won?
    final_point = match_points.iloc[-1]
    winner = final_point["PtWinner"]
    winner_name = match_info["Player 1"] if winner == 1 else match_info["Player 2"]
    print(f"\n--- Match Winner ---")
    print(f"Player {winner} ({winner_name})")
    print(f"Final score: {int(final_point['Set1'])}-{int(final_point['Set2'])} sets")

    # Check for any weird values
    print(f"\n--- Data Quality ---")
    print(f"Unique Pts values: {match_points['Pts'].unique()}")
    print(f"Svr values: {match_points['Svr'].unique()}")
    print(f"PtWinner values: {match_points['PtWinner'].unique()}")
    print(
        f"Any NaN in key columns: {match_points[['Set1','Set2','Gm1','Gm2','Svr','PtWinner']].isna().sum().to_dict()}"
    )


if __name__ == "__main__":
    main()
