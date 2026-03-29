"""Triplet loss trainer for the projection head."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from visual_memory.config import Settings
from visual_memory.database import DatabaseStore

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


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train projection head weights from SQLite feedback triplets."
    )
    parser.add_argument(
        "--db-path",
        default=settings.db_path,
        help=f"Path to SQLite DB (default: {settings.db_path})",
    )
    parser.add_argument(
        "--output",
        default=settings.projection_head_path,
        help=f"Output path for trained head weights (default: {settings.projection_head_path})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.projection_head_epochs,
        help=f"Training epochs (default: {settings.projection_head_epochs})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=settings.projection_head_dim,
        help=f"Projection head embedding dimension (default: {settings.projection_head_dim})",
    )
    parser.add_argument(
        "--feedback-dir",
        default=None,
        help="Deprecated legacy option. Feedback is loaded from --db-path.",
    )
    return parser


def main() -> int:
    settings = Settings()
    args = _build_parser(settings).parse_args()

    if args.feedback_dir:
        print("warning: --feedback-dir is deprecated and ignored; using --db-path")

    db = DatabaseStore(args.db_path)
    store = FeedbackStore(db)
    triplets = store.load_triplets()
    counts = store.count()

    if not triplets:
        print(
            f"no training data: positives={counts['positives']} "
            f"negatives={counts['negatives']} triplets={counts['triplets']}"
        )
        return 1

    head = ProjectionHead(dim=args.dim)
    trainer = ProjectionTrainer(head, lr=args.lr)
    final_loss = trainer.train(triplets, epochs=args.epochs)
    trainer.save(Path(args.output))

    print(
        f"trained projection head -> {args.output} "
        f"(triplets={counts['triplets']}, epochs={args.epochs}, loss={final_loss:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
