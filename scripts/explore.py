"""
Explore the data - first look
Run from project root: python scripts/explore.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/tennis_MatchChartingProject")


def main():
    # Matches
    print("=" * 60)
    print("MATCHES")
    print("=" * 60)

    matches = pd.read_csv(DATA_DIR / "charting-m-matches.csv")
    print(f"Total matches: {len(matches)}")
    print(f"\nColumns:\n{matches.columns.tolist()}")
    print(f"\nFirst few rows:\n{matches.head()}")
    print(f"\nData types:\n{matches.dtypes}")

    # Points
    print("\n" + "=" * 60)
    print("POINTS")
    print("=" * 60)

    points = pd.read_csv(DATA_DIR / "charting-m-points-2020s.csv")
    print(f"Total points: {len(points)}")
    print(f"\nColumns:\n{points.columns.tolist()}")
    print(f"\nFirst few rows:\n{points.head()}")
    print(f"\nData types:\n{points.dtypes}")

    # Join
    print("\n" + "=" * 60)
    print("JOINING MATCHES + POINTS")
    print("=" * 60)

    common_cols = set(matches.columns) & set(points.columns)
    print(f"Common columns: {common_cols}")

    sample_match_id = points["match_id"].iloc[0]
    print(f"\nSample match_id: {sample_match_id}")

    match_info = matches[matches["match_id"] == sample_match_id]
    match_points = points[points["match_id"] == sample_match_id]

    print(f"\nMatch info:\n{match_info}")
    print(f"\nPoints in this match: {len(match_points)}")
    print(f"\nFirst 10 points:\n{match_points.head(10)}")


if __name__ == "__main__":
    main()
