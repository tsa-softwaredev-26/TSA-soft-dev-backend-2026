"""SQLite-backed feedback store for triplet training data collection.

Delegates all persistence to DatabaseStore (feedback table).
The legacy flat-.pt-file interface is preserved so the rest of the codebase
can continue calling record_positive / record_negative / load_triplets / count
without changes.
"""
from typing import Dict, List, Tuple

import torch


class FeedbackStore:
    def __init__(self, db):
        """db: DatabaseStore instance."""
        self._db = db

    def record_positive(
        self,
        anchor_emb: torch.Tensor,
        query_emb: torch.Tensor,
        label: str,
    ) -> None:
        self._db.add_feedback(label, "positive", anchor_emb, query_emb)

    def record_negative(
        self,
        anchor_emb: torch.Tensor,
        query_emb: torch.Tensor,
        label: str,
    ) -> None:
        self._db.add_feedback(label, "negative", anchor_emb, query_emb)

    def load_triplets(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self._db.load_feedback_triplets()

    def count(self) -> Dict[str, int]:
        return self._db.count_feedback()
