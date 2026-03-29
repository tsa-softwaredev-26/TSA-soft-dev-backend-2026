"""CPU-only unit tests for the projection head triplet system. No model loading."""
import tempfile
from pathlib import Path

import torch

from visual_memory.learning.projection_head import ProjectionHead
from visual_memory.learning.trainer import triplet_loss, ProjectionTrainer
from visual_memory.learning.feedback_store import FeedbackStore


def _pass(name: str) -> None:
    print(f"PASS  {name}")


def _fail(name: str, reason: str) -> None:
    print(f"FAIL  {name}: {reason}")


# Test 1: ProjectionHead is identity at init

def test_identity_at_init():
    name = "ProjectionHead identity at init"
    head = ProjectionHead(dim=1536)
    head.eval()
    x = torch.randn(4, 1536)
    x_norm = torch.nn.functional.normalize(x, dim=1)
    with torch.no_grad():
        out = head(x_norm)
    sim = torch.nn.functional.cosine_similarity(x_norm, out).mean().item()
    if abs(sim - 1.0) < 1e-5:
        _pass(name)
    else:
        _fail(name, f"cosine sim={sim:.6f}, expected ~1.0")


# Test 2: triplet_loss = 0 when anchor==positive, anchor!=negative

def test_triplet_loss_zero():
    name = "triplet_loss=0 (anchor==positive, anchor!=negative)"
    a = torch.randn(1, 64)
    n = torch.randn(1, 64)
    loss = triplet_loss(a, a, n, margin=0.2).item()
    if loss == 0.0:
        _pass(name)
    else:
        _fail(name, f"loss={loss:.4f}, expected 0.0")


# Test 3: triplet_loss > 0 when anchor==negative (bad ordering)

def test_triplet_loss_nonzero():
    name = "triplet_loss>0 (anchor==negative)"
    a = torch.randn(1, 64)
    p = torch.randn(1, 64)
    loss = triplet_loss(a, p, a, margin=0.2).item()
    if loss > 0.0:
        _pass(name)
    else:
        _fail(name, f"loss={loss:.4f}, expected >0.0")


# Test 4: FeedbackStore roundtrip

def test_feedback_store_roundtrip():
    name = "FeedbackStore write + load_triplets roundtrip"
    with tempfile.TemporaryDirectory() as tmpdir:
        from visual_memory.database.store import DatabaseStore
        db = DatabaseStore(str(Path(tmpdir) / "test.db"))
        store = FeedbackStore(db)
        anc = torch.randn(1, 1536)
        q = torch.randn(1, 1536)
        store.record_positive(anc, q, label="wallet")
        store.record_positive(anc, q, label="wallet")
        store.record_negative(anc, q, label="wallet")
        store.record_negative(anc, q, label="wallet")
        triplets = store.load_triplets()
        counts = store.count()
        # 2 positives x 2 negatives = 4 triplets
        if len(triplets) == 4 and counts["positives"] == 2 and counts["negatives"] == 2:
            _pass(name)
        else:
            _fail(name, f"triplets={len(triplets)}, positives={counts['positives']}, negatives={counts['negatives']}")


# Test 5: ProjectionTrainer.train_step returns float, loss decreases

def test_trainer_loss_decreases():
    name = "ProjectionTrainer.train_step returns float and loss decreases"
    head = ProjectionHead(dim=32)
    trainer = ProjectionTrainer(head, lr=1e-2)

    # anchor and positive are identical; negative is very close to anchor
    # -> pos_dist=0, neg_dist~0, initial loss > 0 (margin=0.2 kicks in)
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(1, 32), dim=1)
    p = a.clone()
    n = torch.nn.functional.normalize(a + 0.01 * torch.randn(1, 32), dim=1)

    first_loss = trainer.train_step(a, p, n)
    if not isinstance(first_loss, float):
        _fail(name, f"train_step returned {type(first_loss)}, expected float")
        return

    for _ in range(9):
        last_loss = trainer.train_step(a, p, n)

    if last_loss < first_loss:
        _pass(name)
    else:
        _fail(name, f"loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}")


# Runner

if __name__ == "__main__":
    print("Running projection head tests (CPU-only, no model loading)")
    print()
    test_identity_at_init()
    test_triplet_loss_zero()
    test_triplet_loss_nonzero()
    test_feedback_store_roundtrip()
    test_trainer_loss_decreases()
    print()
    print("Done.")
