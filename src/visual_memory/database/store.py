from __future__ import annotations
import io
import sqlite3
import time
from pathlib import Path
from typing import Optional

import torch


class DatabaseStore:
    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._create_tables()

    # ---- schema ----

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id               INTEGER PRIMARY KEY,
                label            TEXT    NOT NULL,
                combined_embedding BLOB  NOT NULL,
                ocr_text         TEXT,
                image_path       TEXT,
                confidence       REAL,
                timestamp        REAL
            );
            CREATE TABLE IF NOT EXISTS user_state (
                id               INTEGER PRIMARY KEY CHECK(id = 1),
                projection_head  BLOB
            );
        """)
        self._conn.commit()

    # ---- serialization helpers ----

    def _tensor_to_blob(self, tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        return buf.getvalue()

    def _blob_to_tensor(self, blob: bytes) -> torch.Tensor:
        buf = io.BytesIO(blob)
        return torch.load(buf, map_location="cpu", weights_only=True)

    # ---- items table ----

    def add_item(
        self,
        label: str,
        combined_embedding: torch.Tensor,
        ocr_text: str = "",
        image_path: str = "",
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> int:
        if timestamp is None:
            timestamp = time.time()
        blob = self._tensor_to_blob(combined_embedding)
        cur = self._conn.execute(
            "INSERT INTO items "
            "(label, combined_embedding, ocr_text, image_path, confidence, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (label, blob, ocr_text or "", image_path or "", confidence, timestamp),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_all_items(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, combined_embedding, ocr_text FROM items"
        ).fetchall()
        items = []
        for row_id, label, blob, ocr_text in rows:
            items.append({
                "id": row_id,
                "label": label,
                "combined_embedding": self._blob_to_tensor(blob),
                "ocr_text": ocr_text or "",
            })
        return items

    # ---- user_state table ----

    def save_projection_head(self, state_dict: dict) -> None:
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        blob = buf.getvalue()
        self._conn.execute(
            "INSERT OR REPLACE INTO user_state (id, projection_head) VALUES (1, ?)",
            (blob,),
        )
        self._conn.commit()

    def load_projection_head(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT projection_head FROM user_state WHERE id = 1"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        buf = io.BytesIO(row[0])
        return torch.load(buf, map_location="cpu", weights_only=True)

    # ---- lifecycle ----

    def close(self):
        self._conn.close()
