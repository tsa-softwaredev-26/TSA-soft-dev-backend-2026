"""Triplet loss trainer for the projection head."""
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


# Retraining is triggered via POST /retrain (HTTP API), not CLI.
# The API endpoint enforces minimum feedback requirements and handles
# DB access correctly. See visual_memory/api/routes/retrain.py.
