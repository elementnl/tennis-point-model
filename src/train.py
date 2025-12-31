"""Training loop for match prediction model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime

from src.model import PointImportanceModel
from src.dataset import create_splits


def train(
    data_path: Path = Path("data/processed/matches.json"),
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = None,
):
    if device is None:
        device = "cpu"
    print(f"Using device: {device}")

    splits = create_splits(data_path)

    def collate_fn(batch):
        return {
            "points": torch.stack([b["points"] for b in batch]),
            "winner": torch.stack([b["winner"] for b in batch]),
            "length": torch.tensor([b["length"] for b in batch]),
        }

    train_loader = DataLoader(
        splits["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        splits["val"],
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    model = PointImportanceModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_acc = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            points = batch["points"].to(device)
            winner = batch["winner"].to(device)
            lengths = batch["length"].to(device)

            optimizer.zero_grad()

            probs = model(points, lengths)

            # Expand winner label to all timesteps
            target = winner.float().unsqueeze(1).expand(-1, probs.size(1))

            # Mask out padding
            mask = torch.arange(probs.size(1), device=device).expand(
                len(lengths), -1
            ) < lengths.unsqueeze(1)

            loss = nn.functional.binary_cross_entropy(probs, target, reduction="none")
            loss = (loss * mask).sum() / mask.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            # Accuracy from final timestep prediction
            final_probs = torch.stack(
                [probs[i, lengths[i] - 1] for i in range(len(lengths))]
            )
            preds = (final_probs > 0.5).long()
            train_correct += (preds == winner).sum().item()
            train_total += len(winner)

        scheduler.step()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                points = batch["points"].to(device)
                winner = batch["winner"].to(device)
                lengths = batch["length"].to(device)

                probs = model(points, lengths)

                target = winner.float().unsqueeze(1).expand(-1, probs.size(1))
                mask = torch.arange(probs.size(1), device=device).expand(
                    len(lengths), -1
                ) < lengths.unsqueeze(1)

                loss = nn.functional.binary_cross_entropy(
                    probs, target, reduction="none"
                )
                loss = (loss * mask).sum() / mask.sum()

                val_loss += loss.item()

                final_probs = torch.stack(
                    [probs[i, lengths[i] - 1] for i in range(len(lengths))]
                )
                preds = (final_probs > 0.5).long()
                val_correct += (preds == winner).sum().item()
                val_total += len(winner)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Val Acc: {val_acc:.3f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "train_acc": train_acc,
                "val_loss": val_loss / len(val_loader),
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  → Saved new best model (val_acc: {val_acc:.3f})")

    torch.save(model.state_dict(), "models/final_model.pt")

    with open("models/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")

    return model, history


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    train()
