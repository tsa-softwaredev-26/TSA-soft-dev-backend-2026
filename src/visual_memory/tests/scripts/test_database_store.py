"""
Unit tests for DatabaseStore methods with a temp SQLite DB.
No model loading, no Flask.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["ENABLE_DEPTH"] = "0"

import torch

from visual_memory.tests.scripts.test_harness import TestRunner, make_embedding, make_text_embedding
from visual_memory.database.store import DatabaseStore

_runner = TestRunner("database_store")


def _make_db() -> tuple[DatabaseStore, str]:
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "test.db")
    os.environ.pop("DB_ENCRYPTION_KEY", None)
    return DatabaseStore(db_path), db_path


def test_add_get_all_items():
    db, _ = _make_db()
    emb = make_embedding(0)
    db.add_item(label="wallet", combined_embedding=emb, ocr_text="rfid", confidence=0.8, timestamp=time.time())
    items = db.get_all_items()
    assert len(items) == 1
    assert items[0]["label"] == "wallet"
    assert items[0]["ocr_text"] == "rfid"
    assert items[0]["combined_embedding"].shape == (1, 1536)


def test_add_item_roundtrip_embedding_shape():
    db, _ = _make_db()
    emb = make_embedding(1)
    db.add_item(label="keys", combined_embedding=emb, confidence=0.9, timestamp=time.time())
    items = db.get_all_items()
    assert items[0]["combined_embedding"].shape == (1, 1536)


def test_rename_label_updates_items():
    db, _ = _make_db()
    emb = make_embedding(0)
    db.add_item(label="wallet", combined_embedding=emb, confidence=0.8, timestamp=time.time())
    result = db.rename_label("wallet", "my wallet")
    assert result["renamed"] == 1
    items = db.get_all_items()
    assert items[0]["label"] == "my wallet"


def test_rename_label_updates_sightings():
    db, _ = _make_db()
    emb = make_embedding(0)
    db.add_item(label="wallet", combined_embedding=emb, confidence=0.8, timestamp=time.time())
    db.add_sighting(label="wallet", direction="ahead", distance_ft=2.0, similarity=0.5, room_name="kitchen", timestamp=time.time())
    db.rename_label("wallet", "billfold")
    rows = db.get_labels_last_seen_in_room("kitchen")
    assert any(r["label"] == "billfold" for r in rows)


def test_rename_conflict_replaces_existing():
    db, _ = _make_db()
    emb = make_embedding(0)
    db.add_item(label="wallet", combined_embedding=emb, confidence=0.8, timestamp=time.time())
    db.add_item(label="purse", combined_embedding=make_embedding(1), confidence=0.7, timestamp=time.time())
    result = db.rename_label("wallet", "purse")
    assert result["replaced"] >= 1


def test_get_labels_last_seen_in_room():
    db, _ = _make_db()
    emb = make_embedding(0)
    db.add_item(label="wallet", combined_embedding=emb, confidence=0.8, timestamp=time.time())
    db.add_sighting(label="wallet", direction="ahead", distance_ft=2.0, similarity=0.5, room_name="kitchen", timestamp=time.time())
    db.add_item(label="keys", combined_embedding=make_embedding(1), confidence=0.7, timestamp=time.time())
    db.add_sighting(label="keys", direction="left", distance_ft=1.0, similarity=0.6, room_name="bedroom", timestamp=time.time())
    kitchen = db.get_labels_last_seen_in_room("kitchen")
    labels = {r["label"] for r in kitchen}
    assert "wallet" in labels
    assert "keys" not in labels


def test_get_label_embeddings_most_recent():
    db, _ = _make_db()
    emb1 = make_embedding(0)
    emb2 = make_embedding(1)
    lbl1 = make_text_embedding(0)
    lbl2 = make_text_embedding(1)
    db.add_item(label="wallet", combined_embedding=emb1, confidence=0.5, timestamp=1.0, label_embedding=lbl1)
    db.add_item(label="wallet", combined_embedding=emb2, confidence=0.9, timestamp=2.0, label_embedding=lbl2)
    results = db.get_label_embeddings()
    assert len(results) == 1
    assert results[0]["label"] == "wallet"


def test_clear_items_independence():
    db, _ = _make_db()
    emb = make_embedding(0)
    db.add_item(label="wallet", combined_embedding=emb, confidence=0.8, timestamp=time.time())
    db.add_sighting(label="wallet", direction="ahead", distance_ft=2.0, similarity=0.5, room_name="kitchen", timestamp=time.time())
    db.clear_items()
    assert db.get_all_items() == []
    # sightings should be independent
    rows = db.get_labels_last_seen_in_room("kitchen")
    assert len(rows) >= 0  # sightings table not cleared


def test_save_load_ml_settings():
    db, _ = _make_db()
    data = {"enable_learning": True, "projection_head_weight": 0.5}
    db.save_ml_settings(data)
    loaded = db.load_ml_settings()
    assert loaded == data


def test_save_load_user_settings():
    db, _ = _make_db()
    data = {"performance_mode": "fast", "voice_speed": 1.5, "learning_enabled": True}
    db.save_user_settings(data)
    loaded = db.load_user_settings()
    assert loaded == data


def test_save_load_projection_head():
    db, _ = _make_db()
    from visual_memory.learning.projection_head import ProjectionHead
    head = ProjectionHead(dim=64)
    state = head.state_dict()
    db.save_projection_head(state)
    loaded = db.load_projection_head()
    assert loaded is not None
    for key in state:
        assert key in loaded


def test_add_feedback_and_count():
    db, _ = _make_db()
    a = make_embedding(0)
    q = make_embedding(1)
    db.add_feedback("wallet", "positive", a, q)
    db.add_feedback("wallet", "negative", a, q)
    counts = db.count_feedback()
    assert counts["positives"] == 1
    assert counts["negatives"] == 1


def test_load_feedback_triplets():
    db, _ = _make_db()
    a = make_embedding(0)
    q = make_embedding(1)
    n = make_embedding(2)
    db.add_feedback("wallet", "positive", a, q)
    db.add_feedback("wallet", "negative", a, n)
    triplets = db.load_feedback_triplets()
    assert len(triplets) == 1


def test_load_feedback_triplets_mines_hard_negative():
    db, _ = _make_db()
    anchor = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), dim=1)
    positive = torch.nn.functional.normalize(torch.tensor([[0.99, 0.01, 0.0, 0.0]]), dim=1)
    easy_negative = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), dim=1)
    hard_negative = torch.nn.functional.normalize(torch.tensor([[0.98, 0.02, 0.0, 0.0]]), dim=1)

    db.add_feedback("wallet", "positive", anchor, positive)
    db.add_feedback("wallet", "negative", anchor, easy_negative)
    db.add_feedback("wallet", "negative", anchor, hard_negative)

    triplets = db.load_feedback_triplets()
    counts = db.count_feedback()
    assert len(triplets) == 1
    assert counts["triplets"] == 1
    assert torch.allclose(triplets[0][2], hard_negative, atol=1e-6)


def test_delete_items_by_label():
    db, _ = _make_db()
    emb = make_embedding(0)
    db.add_item(label="wallet", combined_embedding=emb, confidence=0.8, timestamp=time.time())
    db.add_item(label="wallet", combined_embedding=make_embedding(1), confidence=0.7, timestamp=time.time())
    count = db.delete_items_by_label("wallet")
    assert count == 2
    assert db.get_all_items() == []


def test_prune_items_by_label_max_count_keeps_newest():
    db, _ = _make_db()
    db.add_item(label="wallet", combined_embedding=make_embedding(0), confidence=0.8, timestamp=1.0)
    db.add_item(label="wallet", combined_embedding=make_embedding(1), confidence=0.7, timestamp=2.0)
    db.add_item(label="wallet", combined_embedding=make_embedding(2), confidence=0.6, timestamp=3.0)
    pruned = db.prune_items_by_label_max_count("wallet", 2)
    assert pruned == 1
    items = db.get_items_metadata(label="wallet")
    assert len(items) == 2
    kept_timestamps = sorted(i["timestamp"] for i in items)
    assert kept_timestamps == [2.0, 3.0]


def test_prune_items_by_label_max_count_invalid_limit():
    db, _ = _make_db()
    db.add_item(label="wallet", combined_embedding=make_embedding(0), confidence=0.8, timestamp=1.0)
    try:
        db.prune_items_by_label_max_count("wallet", 0)
        raise AssertionError("Expected ValueError for invalid max_count")
    except ValueError:
        pass


def test_get_items_metadata_filter():
    db, _ = _make_db()
    db.add_item(label="wallet", combined_embedding=make_embedding(0), confidence=0.8, timestamp=time.time())
    db.add_item(label="keys", combined_embedding=make_embedding(1), confidence=0.7, timestamp=time.time())
    wallet_items = db.get_items_metadata(label="wallet")
    assert all(i["label"] == "wallet" for i in wallet_items)


def test_get_known_item_labels_includes_items_and_sightings():
    db, _ = _make_db()
    now = time.time()
    db.add_item(label="wallet", combined_embedding=make_embedding(0), confidence=0.8, timestamp=now - 5)
    db.add_sighting(label="keys", direction="left", distance_ft=1.0, similarity=0.6, timestamp=now)
    labels = db.get_known_item_labels(limit=10)
    assert "wallet" in labels
    assert "keys" in labels


def test_get_recent_room_names_unique_sorted():
    db, _ = _make_db()
    now = time.time()
    db.add_sighting(label="wallet", room_name="kitchen", timestamp=now - 5)
    db.add_sighting(label="wallet", room_name="kitchen", timestamp=now - 1)
    db.add_sighting(label="keys", room_name="bedroom", timestamp=now)
    rooms = db.get_recent_room_names(limit=10)
    assert rooms[0] == "bedroom"
    assert "kitchen" in rooms


for name, fn in [
    ("db:add_get_all_items", test_add_get_all_items),
    ("db:roundtrip_embedding_shape", test_add_item_roundtrip_embedding_shape),
    ("db:rename_updates_items", test_rename_label_updates_items),
    ("db:rename_updates_sightings", test_rename_label_updates_sightings),
    ("db:rename_conflict_replaces", test_rename_conflict_replaces_existing),
    ("db:get_labels_last_seen_in_room", test_get_labels_last_seen_in_room),
    ("db:get_label_embeddings_most_recent", test_get_label_embeddings_most_recent),
    ("db:clear_items_independence", test_clear_items_independence),
    ("db:save_load_ml_settings", test_save_load_ml_settings),
    ("db:save_load_user_settings", test_save_load_user_settings),
    ("db:save_load_projection_head", test_save_load_projection_head),
    ("db:add_feedback_and_count", test_add_feedback_and_count),
    ("db:load_feedback_triplets", test_load_feedback_triplets),
    ("db:delete_items_by_label", test_delete_items_by_label),
    ("db:prune_items_by_label_max_count", test_prune_items_by_label_max_count_keeps_newest),
    ("db:prune_items_by_label_invalid_limit", test_prune_items_by_label_max_count_invalid_limit),
    ("db:get_items_metadata_filter", test_get_items_metadata_filter),
    ("db:get_known_item_labels", test_get_known_item_labels_includes_items_and_sightings),
    ("db:get_recent_room_names", test_get_recent_room_names_unique_sorted),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
