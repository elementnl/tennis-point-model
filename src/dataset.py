"""PyTorch Dataset for tennis match point sequences."""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MatchDataset(Dataset):
    """
    Dataset of tennis matches as point sequences.

    Each item returns:
    - points: (seq_len, n_features) tensor of point features
    - winner: 0 or 1 (player 1 or 2 won)
    - length: actual sequence length (before padding)
    """

    def __init__(
        self,
        matches: list[dict],
        max_len: int = 500,
    ):
        self.matches = matches
        self.max_len = max_len
        self.n_features = (
            9  # set1, set2, game1, game2, pt1, pt2, server, is_tb, best_of
        )

    @classmethod
    def from_json(cls, path: Path, **kwargs):
        """Load dataset from JSON file."""
        with open(path) as f:
            matches = json.load(f)
        return cls(matches, **kwargs)

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]
        points = match["points"]

        seq_len = min(len(points), self.max_len)
        features = torch.zeros(self.max_len, self.n_features)

        best_of = match.get("best_of", 3)

        for i, pt in enumerate(points[:seq_len]):
            features[i] = torch.tensor(
                [
                    pt["set1"] / 3,  # normalize by max sets
                    pt["set2"] / 3,
                    pt["game1"] / 7,  # normalize by ~max games
                    pt["game2"] / 7,
                    pt["point1"] / 4,  # normalize (0-4 for regular, higher for TB)
                    pt["point2"] / 4,
                    pt["server"] - 1,  # convert 1/2 to 0/1
                    float(pt["is_tiebreak"]),
                    (best_of - 3) / 2,  # 3 -> 0, 5 -> 1
                ]
            )

        winner = match["winner"] - 1  # Convert 1/2 to 0/1

        return {
            "points": features,
            "winner": torch.tensor(winner, dtype=torch.long),
            "length": seq_len,
            "match_id": match["match_id"],
        }


def create_splits(
    dataset_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, MatchDataset]:
    """Create train/val/test splits chronologically by match_id."""
    with open(dataset_path) as f:
        matches = json.load(f)

    matches.sort(key=lambda m: m["match_id"])

    n = len(matches)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": MatchDataset(matches[:train_end]),
        "val": MatchDataset(matches[train_end:val_end]),
        "test": MatchDataset(matches[val_end:]),
    }

    for name, ds in splits.items():
        print(f"{name}: {len(ds)} matches")

    return splits


if __name__ == "__main__":
    splits = create_splits(Path("data/processed/matches.json"))

    sample = splits["train"][0]
    print(f"\nSample match: {sample['match_id']}")
    print(f"Points shape: {sample['points'].shape}")
    print(f"Actual length: {sample['length']}")
    print(f"Winner: {sample['winner']}")
    print(f"First point features: {sample['points'][0]}")
