"""
Integration tests for Ask Mode: POST /ask, POST /item/ask, GET /find
(narration, OCR fallback, room query).

These tests use the Flask test client and a real temporary SQLite database.
No ML models are loaded - embeddings are pre-seeded directly into the DB
so tests are fast (< 1 second each).

Run:
    python -m visual_memory.tests.scripts.test_ask_mode
    VERBOSE=1 python -m visual_memory.tests.scripts.test_ask_mode
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

VERBOSE = os.environ.get("VERBOSE", "0") == "1"

# Suppress model-loading noise
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

# Disable models not needed for these tests
os.environ["ENABLE_DEPTH"] = "0"

import torch

_G = "\033[32m"
_R = "\033[31m"
_B = "\033[1m"
_Y = "\033[33m"
_X = "\033[0m"

_results: list[tuple[str, bool, str]] = []


def _pass(tag: str, detail: str) -> None:
    _results.append((tag, True, detail))
    print(f"  {_G}PASS{_X}  {detail}")


def _fail(tag: str, detail: str) -> None:
    _results.append((tag, False, detail))
    print(f"  {_R}FAIL{_X}  {detail}")


def _section(title: str) -> None:
    print(f"\n{_B}{title}{_X}")


def _dump(data: dict) -> None:
    if VERBOSE:
        print(f"     json: {json.dumps(data, indent=2)}")


# helpers

def _make_embedding(seed: int = 0) -> torch.Tensor:
    """Return a deterministic normalized 1536-dim embedding."""
    torch.manual_seed(seed)
    e = torch.randn(1, 1536)
    return torch.nn.functional.normalize(e, dim=1)


def _make_text_embedding(seed: int = 0) -> torch.Tensor:
    """Return a deterministic normalized 512-dim text embedding."""
    torch.manual_seed(seed)
    e = torch.randn(1, 512)
    return torch.nn.functional.normalize(e, dim=1)


def _seed_db(db, label: str, ocr_text: str = "", room_name: str | None = None, emb_seed: int = 0):
    """Insert a taught item and a sighting into the DB."""
    emb = _make_embedding(emb_seed)
    label_emb = _make_text_embedding(emb_seed)
    db.add_item(
        label=label,
        combined_embedding=emb,
        ocr_text=ocr_text,
        confidence=0.8,
        timestamp=time.time(),
        label_embedding=label_emb,
    )
    db.add_sighting(
        label=label,
        direction="to your left",
        distance_ft=2.5,
        similarity=0.7,
        room_name=room_name,
        timestamp=time.time(),
    )


# Test setup

_tmp = tempfile.mkdtemp()
_db_path = str(Path(_tmp) / "test_ask.db")
os.environ["DB_PATH"] = _db_path

from visual_memory.database.store import DatabaseStore
from visual_memory.api.routes.find import build_narration, _normalize_room


# Section 1: build_narration helper

_section("[1] build_narration - unit tests")

s_full = {
    "label": "wallet",
    "room_name": "kitchen",
    "direction": "to your left",
    "distance_ft": 2.5,
    "last_seen": "5 minutes ago",
}
n = build_narration("wallet", s_full)
if "kitchen" in n and "to your left" in n and "2.5" in n and "5 minutes ago" in n:
    _pass("narration:full", f"'{n}'")
else:
    _fail("narration:full", f"missing fields in: '{n}'")

s_no_room = {
    "label": "wallet",
    "room_name": None,
    "direction": "ahead",
    "distance_ft": None,
    "last_seen": "just now",
}
n2 = build_narration("wallet", s_no_room)
if "ahead" in n2 and "just now" in n2:
    _pass("narration:no_room", f"'{n2}'")
else:
    _fail("narration:no_room", f"unexpected: '{n2}'")

s_empty = {"label": "wallet", "room_name": None, "direction": None, "distance_ft": None, "last_seen": "3 days ago"}
n3 = build_narration("wallet", s_empty)
if "3 days ago" in n3:
    _pass("narration:minimal", f"'{n3}'")
else:
    _fail("narration:minimal", f"unexpected: '{n3}'")


# Section 2: room normalization

_section("[2] _normalize_room - unit tests")

cases = [
    ("In The Kitchen", "kitchen"),
    ("the bedroom", "bedroom"),
    ("my living room", "living room"),
    ("kitchen", "kitchen"),
    ("In my Office", "office"),
]
for raw, expected in cases:
    got = _normalize_room(raw)
    if got == expected:
        _pass("room_normalize", f"'{raw}' -> '{got}'")
    else:
        _fail("room_normalize", f"'{raw}' -> '{got}' (expected '{expected}')")


# Section 3: DB helpers

_section("[3] DatabaseStore - get_items_with_ocr, get_labels_last_seen_in_room")

db = DatabaseStore(_db_path)

_seed_db(db, "wallet", ocr_text="RFID Blocking", room_name="kitchen", emb_seed=1)
_seed_db(db, "keys", ocr_text="", room_name="bedroom", emb_seed=2)
_seed_db(db, "receipt", ocr_text="Office Chair $299", room_name="kitchen", emb_seed=3)

ocr_items = db.get_items_with_ocr()
ocr_labels = {i["label"] for i in ocr_items}
if "wallet" in ocr_labels and "receipt" in ocr_labels and "keys" not in ocr_labels:
    _pass("db:get_items_with_ocr", f"returned {ocr_labels}")
else:
    _fail("db:get_items_with_ocr", f"unexpected: {ocr_labels}")

kitchen_items = db.get_labels_last_seen_in_room("kitchen")
kitchen_labels = {r["label"] for r in kitchen_items}
if "wallet" in kitchen_labels and "receipt" in kitchen_labels and "keys" not in kitchen_labels:
    _pass("db:get_labels_last_seen_in_room", f"kitchen -> {kitchen_labels}")
else:
    _fail("db:get_labels_last_seen_in_room", f"unexpected: {kitchen_labels}")

bedroom_items = db.get_labels_last_seen_in_room("bedroom")
if len(bedroom_items) == 1 and bedroom_items[0]["label"] == "keys":
    _pass("db:room_bedroom", "keys only in bedroom")
else:
    _fail("db:room_bedroom", f"unexpected: {[r['label'] for r in bedroom_items]}")


# Section 4: Flask test client - GET /find

_section("[4] GET /find - narration field, room query, exact match")

# Inject singletons directly into the pipelines module so routes use our test DB.
# Variable names must match pipelines.py exactly: _database, _scan_pipeline, _settings.
import visual_memory.api.pipelines as _pm
from visual_memory.config.settings import Settings as _Settings

_test_settings = _Settings()
_test_settings.db_path = _db_path
_test_settings.enable_depth = False
_test_settings.enable_ocr = False
_test_settings.enable_learning = False

class _StubTextEmbedder:
    def embed_text(self, text: str) -> torch.Tensor:
        torch.manual_seed(hash(text) % (2**32))
        e = torch.randn(1, 512)
        return torch.nn.functional.normalize(e, dim=1)

    def batch_embed_text(self, texts):
        return torch.cat([self.embed_text(t) for t in texts])

class _StubScanPipeline:
    text_embedder = _StubTextEmbedder()
    database_embeddings = []

    def reload_database(self):
        pass

# Inject using the correct variable names from pipelines.py
_pm._settings = _test_settings
_pm._database = db
_pm._scan_pipeline = _StubScanPipeline()

# Minimal Flask app without warm_all
from flask import Flask, jsonify, request
from visual_memory.api.routes.find import find_bp
from visual_memory.api.routes.ask import ask_bp
from visual_memory.api.routes.item_ask import item_ask_bp

test_app = Flask(__name__)
test_app.register_blueprint(find_bp)
test_app.register_blueprint(ask_bp)
test_app.register_blueprint(item_ask_bp)

client = test_app.test_client()

# exact label match
resp = client.get("/find?label=wallet")
data = resp.get_json()
_dump(data)
if data.get("found") and "narration" in data and data["narration"]:
    _pass("find:exact_label", f"found=True, narration present")
else:
    _fail("find:exact_label", f"unexpected: {data}")

# room query
resp = client.get("/find?room=kitchen")
data = resp.get_json()
_dump(data)
room_labels = {r["label"] for r in data.get("results", [])}
if "wallet" in room_labels and "receipt" in room_labels:
    _pass("find:room_query", f"kitchen -> {room_labels}")
else:
    _fail("find:room_query", f"unexpected: {room_labels}")

# room normalization via query param
resp = client.get("/find?room=In%20The%20Kitchen")
data = resp.get_json()
if data.get("room") == "kitchen":
    _pass("find:room_normalize_param", "normalized correctly")
else:
    _fail("find:room_normalize_param", f"unexpected room: {data.get('room')}")

# label not found
resp = client.get("/find?label=nonexistent_object_xyz")
data = resp.get_json()
if not data.get("found"):
    _pass("find:not_found", "found=False as expected")
else:
    _fail("find:not_found", f"unexpected: {data}")


# Section 5: POST /ask

_section("[5] POST /ask - exact match, not found, narration")

# Re-seed in case section 4 (find) queries affected state (they shouldn't, but be safe)
_seed_db(db, "wallet", ocr_text="RFID Blocking", room_name="kitchen", emb_seed=1)

resp = client.post("/ask", json={"query": "wallet"})
data = resp.get_json()
_dump(data)
if data.get("found") and "narration" in data and data.get("matched_label") == "wallet":
    _pass("ask:exact_match", f"matched_label=wallet, narration present")
else:
    _fail("ask:exact_match", f"unexpected: {data}")

resp = client.post("/ask", json={"query": "this object does not exist anywhere xyz"})
data = resp.get_json()
_dump(data)
if not data.get("found") and "narration" in data:
    _pass("ask:not_found", "found=False with narration")
else:
    _fail("ask:not_found", f"unexpected: {data}")

# missing query
resp = client.post("/ask", json={})
if resp.status_code == 400:
    _pass("ask:missing_query", "400 on empty query")
else:
    _fail("ask:missing_query", f"expected 400, got {resp.status_code}")


# Section 6: POST /item/ask

_section("[6] POST /item/ask - read_ocr, rename, find, no-op rename, describe deferred")

# Re-seed fresh state - rename tests mutate the DB so order matters without this
db.clear_items()
db.clear_sightings()
_seed_db(db, "wallet", ocr_text="RFID Blocking", room_name="kitchen", emb_seed=1)
_seed_db(db, "keys", ocr_text="", room_name="bedroom", emb_seed=2)
_pm._database = db

# read_ocr - wallet has "RFID Blocking"
resp = client.post("/item/ask", json={
    "scan_id": "test-scan-001",
    "label": "wallet",
    "query": "read the text in this",
})
data = resp.get_json()
_dump(data)
if data.get("action") == "read_ocr" and "RFID Blocking" in data.get("ocr_text", ""):
    _pass("item_ask:read_ocr", f"ocr_text='{data['ocr_text']}'")
else:
    _fail("item_ask:read_ocr", f"unexpected: {data}")

# read_ocr - keys has no OCR text
resp = client.post("/item/ask", json={
    "scan_id": "test-scan-001",
    "label": "keys",
    "query": "what does it say",
})
data = resp.get_json()
_dump(data)
if data.get("action") == "read_ocr" and data.get("ocr_text") == "":
    _pass("item_ask:read_ocr_empty", "empty OCR handled gracefully")
else:
    _fail("item_ask:read_ocr_empty", f"unexpected: {data}")

# find intent
resp = client.post("/item/ask", json={
    "scan_id": "test-scan-001",
    "label": "wallet",
    "query": "where is this normally",
})
data = resp.get_json()
_dump(data)
if data.get("action") == "find" and data.get("found"):
    _pass("item_ask:find", f"narration='{data.get('narration', '')}'")
else:
    _fail("item_ask:find", f"unexpected: {data}")

# rename no-op (same label)
resp = client.post("/item/ask", json={
    "scan_id": "test-scan-001",
    "label": "wallet",
    "query": "rename this to wallet",
})
data = resp.get_json()
_dump(data)
if data.get("action") == "rename" and data.get("unchanged"):
    _pass("item_ask:rename_noop", "same-label rename is a no-op")
else:
    _fail("item_ask:rename_noop", f"unexpected: {data}")

# describe deferred
resp = client.post("/item/ask", json={
    "scan_id": "test-scan-001",
    "label": "wallet",
    "query": "describe this for me",
})
data = resp.get_json()
_dump(data)
if data.get("action") == "describe" and data.get("deferred"):
    _pass("item_ask:describe_deferred", "describe returns deferred=true")
else:
    _fail("item_ask:describe_deferred", f"unexpected: {data}")

# missing fields
for missing in [
    {"scan_id": "x", "label": "wallet"},         # no query
    {"scan_id": "x", "query": "read this"},       # no label
    {"label": "wallet", "query": "read this"},    # no scan_id
]:
    resp = client.post("/item/ask", json=missing)
    if resp.status_code == 400:
        _pass("item_ask:missing_field", f"400 for body={missing}")
    else:
        _fail("item_ask:missing_field", f"expected 400, got {resp.status_code} for body={missing}")


# Summary

passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
total = len(_results)

print(f"\n{_B}Results: {passed}/{total} passed{_X}")
if failed:
    print(f"{_R}Failed:{_X}")
    for tag, ok, detail in _results:
        if not ok:
            print(f"  {_R}[FAIL]{_X} [{tag}] {detail}")
else:
    print(f"{_G}All tests passed.{_X}")
print()

sys.exit(0 if failed == 0 else 1)
