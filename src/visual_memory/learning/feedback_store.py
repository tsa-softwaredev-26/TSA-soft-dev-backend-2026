"""File-based feedback store for triplet training data collection.

Flask endpoint contract (teammate: implement POST /feedback):
    body: {"scan_id": str, "label": str, "feedback": "correct" | "wrong"}
    ScanPipeline.run() must cache (anchor_emb, query_emb) by scan_id
    Flask handler calls:
        if feedback == "correct": store.record_positive(anchor, query, label)
        else:                     store.record_negative(anchor, query, label)
    ScanPipeline should expose: get_last_embeddings(scan_id) -> (anchor, query) | None

DB contract (awaiting teammate):
    Future: record_positive / record_negative will INSERT into user's .db file
    DB schema: (id, label, feedback_type, anchor_blob, query_blob, timestamp)
    load_triplets() will SELECT and pair from DB instead of reading .pt files
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import torch


class FeedbackStore:
    def __init__(self, store_dir: Path = Path("feedback")):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def record_positive(
        self,
        anchor_emb: torch.Tensor,
        query_emb: torch.Tensor,
        label: str,
    ) -> None:
        fname = f"{label}_pos_{uuid4().hex[:8]}.pt"
        torch.save(
            {
                "type": "positive",
                "anchor": anchor_emb.detach().cpu(),
                "query": query_emb.detach().cpu(),
                "label": label,
                "timestamp": time.time(),
            },
            self.store_dir / fname,
        )

    def record_negative(
        self,
        anchor_emb: torch.Tensor,
        query_emb: torch.Tensor,
        label: str,
    ) -> None:
        fname = f"{label}_neg_{uuid4().hex[:8]}.pt"
        torch.save(
            {
                "type": "negative",
                "anchor": anchor_emb.detach().cpu(),
                "query": query_emb.detach().cpu(),
                "label": label,
                "timestamp": time.time(),
            },
            self.store_dir / fname,
        )

    def load_triplets(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Pair each positive with each negative per label -> (anchor, positive, negative)."""
        by_label: Dict[str, Dict[str, list]] = {}

        for pt_file in sorted(self.store_dir.glob("*.pt")):
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            label = data["label"]
            kind = data["type"]
            if label not in by_label:
                by_label[label] = {"positive": [], "negative": []}
            by_label[label][kind].append((data["timestamp"], data["anchor"], data["query"]))

        triplets = []
        for label, groups in by_label.items():
            positives = sorted(groups["positive"], key=lambda x: x[0])
            negatives = sorted(groups["negative"], key=lambda x: x[0])
            if not positives or not negatives:
                continue
            for _, anc_p, q_p in positives:
                for _, anc_n, q_n in negatives:
                    # anchor = DB reference embedding, positive = correct query, negative = wrong query
                    triplets.append((anc_p, q_p, q_n))

        return triplets

    def count(self) -> Dict[str, int]:
        positives = 0
        negatives = 0
        for pt_file in self.store_dir.glob("*.pt"):
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            if data["type"] == "positive":
                positives += 1
            else:
                negatives += 1
        triplets = len(self.load_triplets())
        return {"positives": positives, "negatives": negatives, "triplets": triplets}
