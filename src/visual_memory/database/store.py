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
            -- id is the user ID; hardcoded to 1 for single-user demo.
            -- Multi-user: remove CHECK constraint and pass uid into all user_state methods.
            CREATE TABLE IF NOT EXISTS user_state (
                id               INTEGER PRIMARY KEY CHECK(id = 1),
                projection_head  BLOB,
                user_settings    TEXT
            );
            CREATE TABLE IF NOT EXISTS sightings (
                id           INTEGER PRIMARY KEY,
                label        TEXT    NOT NULL,
                timestamp    REAL    NOT NULL,
                direction    TEXT,
                distance_ft  REAL,
                similarity   REAL
            );
            CREATE INDEX IF NOT EXISTS sightings_label_ts
                ON sightings (label, timestamp DESC);
        """)
        # migrate existing DBs that predate the user_settings column
        try:
            self._conn.execute("ALTER TABLE user_state ADD COLUMN user_settings TEXT")
            self._conn.commit()
        except Exception:
            pass  # column already exists
        # migrate existing sightings tables that predate crop_path
        try:
            self._conn.execute("ALTER TABLE sightings ADD COLUMN crop_path TEXT")
            self._conn.commit()
        except Exception:
            pass  # column already exists

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

    def save_user_settings(self, data: dict) -> None:
        import json
        self._conn.execute(
            "INSERT INTO user_state (id, user_settings) VALUES (1, ?)"
            " ON CONFLICT(id) DO UPDATE SET user_settings = excluded.user_settings",
            (json.dumps(data),),
        )
        self._conn.commit()

    def load_user_settings(self) -> Optional[dict]:
        import json
        row = self._conn.execute(
            "SELECT user_settings FROM user_state WHERE id = 1"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    # ---- sightings table ----

    def add_sighting(
        self,
        label: str,
        direction: Optional[str] = None,
        distance_ft: Optional[float] = None,
        similarity: Optional[float] = None,
        crop_path: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        if timestamp is None:
            timestamp = time.time()
        cur = self._conn.execute(
            "INSERT INTO sightings (label, timestamp, direction, distance_ft, similarity, crop_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (label, timestamp, direction, distance_ft, similarity, crop_path),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_last_sighting(self, label: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, label, timestamp, direction, distance_ft, similarity, crop_path "
            "FROM sightings WHERE label = ? ORDER BY timestamp DESC LIMIT 1",
            (label,),
        ).fetchone()
        if row is None:
            return None
        return self._sighting_row_to_dict(row)

    def get_sightings(self, label: Optional[str] = None, limit: int = 20) -> list[dict]:
        if label is not None:
            rows = self._conn.execute(
                "SELECT id, label, timestamp, direction, distance_ft, similarity, crop_path "
                "FROM sightings WHERE label = ? ORDER BY timestamp DESC LIMIT ?",
                (label, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, label, timestamp, direction, distance_ft, similarity, crop_path "
                "FROM sightings ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._sighting_row_to_dict(r) for r in rows]

    def get_known_labels(self) -> list[str]:
        """Return labels that have at least one sighting, most recently seen first."""
        rows = self._conn.execute(
            "SELECT label FROM sightings GROUP BY label ORDER BY MAX(timestamp) DESC"
        ).fetchall()
        return [r[0] for r in rows]

    def delete_sighting(self, sighting_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM sightings WHERE id = ?", (sighting_id,))
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _sighting_row_to_dict(row: tuple) -> dict:
        sid, label, timestamp, direction, distance_ft, similarity, crop_path = row
        return {
            "id": sid,
            "label": label,
            "timestamp": timestamp,
            "direction": direction,
            "distance_ft": distance_ft,
            "similarity": similarity,
            "crop_path": crop_path,
        }

    # ---- lifecycle ----

    def close(self):
        self._conn.close()
