"""Data loading and preprocessing."""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass

DATA_DIR = Path("data/raw/tennis_MatchChartingProject")


def parse_point_score(pts_str: str, is_tiebreak: bool) -> tuple[int, int]:
    """
    Parse point score string into numeric tuple.

    Regular game: '30-15' -> (2, 1)  # 0,15,30,40 -> 0,1,2,3
    Tiebreak: '5-3' -> (5, 3)  # actual points

    Returns (server_points, receiver_points)
    """
    if pd.isna(pts_str):
        return (0, 0)

    parts = pts_str.split("-")
    if len(parts) != 2:
        return (0, 0)
    left, right = parts[0], parts[1]

    if is_tiebreak:
        # Tiebreak: parse as integers
        try:
            return int(left), int(right)
        except ValueError:
            return (0, 0)
    else:
        # Map 0/15/30/40 to 0-3
        score_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
        left_score = score_map.get(left, 0)
        right_score = score_map.get(right, 0)
        return (left_score, right_score)


def load_points() -> pd.DataFrame:
    """Load and combine all points files."""
    files = [
        "charting-m-points-to-2009.csv",
        "charting-m-points-2010s.csv",
        "charting-m-points-2020s.csv",
    ]

    dfs = []
    for f in files:
        path = DATA_DIR / f
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            print(f"Loaded {f}: {len(df)} points")
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_matches() -> pd.DataFrame:
    """Load matches file."""
    df = pd.read_csv(DATA_DIR / "charting-m-matches.csv")
    print(f"Loaded matches: {len(df)}")
    return df


def determine_winner(match_points: pd.DataFrame) -> int:
    """Determine match winner from point data."""
    final_point = match_points.sort_values("Pt").iloc[-1]
    return int(final_point["PtWinner"])


def process_match(
    match_id: str, points_df: pd.DataFrame, matches_df: pd.DataFrame
) -> dict:
    """
    Process a single match into model-ready format.

    Returns dict with match info and processed point sequence.
    """
    match_points = points_df[points_df["match_id"] == match_id].sort_values("Pt")
    match_info = matches_df[matches_df["match_id"] == match_id].iloc[0]

    processed_points = []
    for _, row in match_points.iterrows():
        is_tb = row["TbSet"] == True or row["TbSet"] == "True"
        pt_score = parse_point_score(row["Pts"], is_tb)

        processed_points.append(
            {
                "set1": int(row["Set1"]),
                "set2": int(row["Set2"]),
                "game1": int(row["Gm1"]),
                "game2": int(row["Gm2"]) if pd.notna(row["Gm2"]) else 0,
                "point1": pt_score[0],
                "point2": pt_score[1],
                "server": int(row["Svr"]),
                "is_tiebreak": is_tb,
                "point_winner": int(row["PtWinner"]),
            }
        )

    return {
        "match_id": match_id,
        "player1": match_info["Player 1"],
        "player2": match_info["Player 2"],
        "best_of": int(match_info["Best of"]) if pd.notna(match_info["Best of"]) else 3,
        "surface": match_info["Surface"],
        "winner": determine_winner(match_points),
        "points": processed_points,
    }


def process_all_matches(save_path: Path = None) -> list[dict]:
    """Process all matches and optionally save to disk."""
    points_df = load_points()
    matches_df = load_matches()

    # Intersect match IDs from both sources
    match_ids = set(points_df["match_id"].unique()) & set(
        matches_df["match_id"].unique()
    )
    print(f"Matches with point data: {len(match_ids)}")

    processed = []
    for i, match_id in enumerate(match_ids):
        if i % 500 == 0:
            print(f"Processing {i}/{len(match_ids)}...")

        try:
            match_data = process_match(match_id, points_df, matches_df)
            if len(match_data["points"]) >= 50:  # Min 50 points
                processed.append(match_data)
        except Exception as e:
            print(f"Error processing {match_id}: {e}")

    print(f"Successfully processed: {len(processed)} matches")

    if save_path:
        import json

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(processed, f)
        print(f"Saved to {save_path}")

    return processed


if __name__ == "__main__":
    processed = process_all_matches(save_path=Path("data/processed/matches.json"))
