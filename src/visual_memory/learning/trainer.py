"""Triplet loss trainer for the projection head."""
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .projection_head import ProjectionHead
from .feedback_store import FeedbackStore


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    pos_dist = 1.0 - F.cosine_similarity(anchor, positive)
    neg_dist = 1.0 - F.cosine_similarity(anchor, negative)
    return F.relu(pos_dist - neg_dist + margin).mean()


class ProjectionTrainer:
    def __init__(self, head: ProjectionHead, lr: float = 1e-4):
        self.head = head
        self.optimizer = torch.optim.Adam(
            head.parameters(), lr=lr, weight_decay=1e-4
        )

    def train_step(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        weight: float = 1.0,
    ) -> float:
        self.head.train()
        self.optimizer.zero_grad()
        loss = triplet_loss(
            self.head(anchor),
            self.head(positive),
            self.head(negative),
        ) * weight
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(
        self,
        triplets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        epochs: int = 20,
    ) -> float:
        n = len(triplets)
        if n == 0:
            return 0.0

        # linear recency weights: oldest=0.5, newest=1.0
        weights = [0.5 + 0.5 * i / max(n - 1, 1) for i in range(n)]

        final_loss = 0.0
        for _ in range(epochs):
            total = 0.0
            for (anchor, positive, negative), w in zip(triplets, weights):
                total += self.train_step(anchor, positive, negative, weight=w)
            final_loss = total / n

        return final_loss

    def save(self, path: Path) -> None:
        self.head.save(path)


# ---- __main__ entry ----

def _parse_args():
    p = argparse.ArgumentParser(description="Train projection head from collected feedback")
    p.add_argument("--feedback-dir", default="feedback", type=Path)
    p.add_argument("--output", default="models/projection_head.pt", type=Path)
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    store = FeedbackStore(store_dir=args.feedback_dir)
    counts = store.count()
    print(f"Feedback: {counts['positives']} positives, {counts['negatives']} negatives, {counts['triplets']} triplets")

    triplets = store.load_triplets()
    if not triplets:
        print("No triplets available. Collect positive + negative feedback first.")
        raise SystemExit(1)

    head = ProjectionHead()
    trainer = ProjectionTrainer(head, lr=args.lr)
    final_loss = trainer.train(triplets, epochs=args.epochs)
    trainer.save(args.output)

    print(f"Training complete. Final loss: {final_loss:.4f}")
    print(f"Saved: {args.output}")
