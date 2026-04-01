from __future__ import annotations
import io
import json
import os
import time
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Optional

import torch

try:
    from pysqlcipher3 import dbapi2 as sqlite3
    _SQLCIPHER_AVAILABLE = True
except ImportError:
    import sqlite3
    _SQLCIPHER_AVAILABLE = False

try:
    from visual_memory.utils import get_logger
    _log = get_logger(__name__)
except Exception:
    _log = None


def _synchronized(method):
    @wraps(method)
    def _wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return _wrapped


class DatabaseStore:
    def __init__(self, db_path: str | Path):
        self._lock = RLock()
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._set_permissions(self._path.parent, 0o700)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._configure_encryption()
        self._create_tables()
        self._set_permissions(self._path, 0o600)

    @_synchronized
    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id               INTEGER PRIMARY KEY,
                label            TEXT    NOT NULL,
                combined_embedding BLOB  NOT NULL,
                ocr_text         TEXT,
                image_path       TEXT,
                confidence       REAL,
                timestamp        REAL,
                visual_attributes TEXT
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
            CREATE TABLE IF NOT EXISTS feedback (
                id            INTEGER PRIMARY KEY,
                user_id       INTEGER NOT NULL DEFAULT 1,
                label         TEXT    NOT NULL,
                feedback_type TEXT    NOT NULL,
                anchor_blob   BLOB    NOT NULL,
                query_blob    BLOB    NOT NULL,
                timestamp     REAL    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS feedback_label
                ON feedback (label, feedback_type);
        """)
        self._apply_optional_migration("ALTER TABLE user_state ADD COLUMN user_settings TEXT")
        self._apply_optional_migration("ALTER TABLE sightings ADD COLUMN crop_path TEXT")
        self._apply_optional_migration("ALTER TABLE sightings ADD COLUMN room_name TEXT")
        self._apply_optional_migration("ALTER TABLE items ADD COLUMN label_embedding BLOB")
        self._apply_optional_migration("ALTER TABLE items ADD COLUMN ocr_embedding BLOB")
        self._apply_optional_migration("ALTER TABLE items ADD COLUMN visual_attributes TEXT")
        self._apply_optional_migration("ALTER TABLE user_state ADD COLUMN ml_settings TEXT")

    @_synchronized
    def _apply_optional_migration(self, statement: str) -> None:
        try:
            self._conn.execute(statement)
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                return
            if _log is not None:
                _log.error({"event": "db_migration_failed", "statement": statement, "error": str(exc)})
            raise
        except sqlite3.DatabaseError as exc:
            if _log is not None:
                _log.error({"event": "db_migration_failed", "statement": statement, "error": str(exc)})
            raise

    @_synchronized
    def _configure_encryption(self) -> None:
        key = os.environ.get("DB_ENCRYPTION_KEY", "").strip()
        if not key:
            return
        if not _SQLCIPHER_AVAILABLE:
            pyver = os.sys.version_info
            if pyver.major == 3 and pyver.minor >= 13:
                if _log is not None:
                    _log.warning({
                        "event": "db_encryption_skipped",
                        "reason": "sqlcipher_unavailable_py313",
                        "python": f"{pyver.major}.{pyver.minor}",
                    })
                return
            raise RuntimeError(
                "DB_ENCRYPTION_KEY is set but SQLCipher is unavailable. "
                "Install core dependencies with pysqlcipher3 support."
            )

        escaped = key.replace("'", "''")
        self._conn.execute(f"PRAGMA key = '{escaped}'")
        self._conn.execute("PRAGMA kdf_iter = 500000")
        try:
            self._conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(
                "Failed to unlock encrypted database. "
                "Check DB_ENCRYPTION_KEY or migrate to a fresh encrypted database."
            ) from exc

    @staticmethod
    def _set_permissions(path: Path, mode: int) -> None:
        try:
            path.chmod(mode)
        except OSError:
            return

    def _tensor_to_blob(self, tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)
        return buf.getvalue()

    def _blob_to_tensor(self, blob: bytes) -> torch.Tensor:
        buf = io.BytesIO(blob)
        return torch.load(buf, map_location="cpu", weights_only=True)

    @_synchronized
    def add_item(
        self,
        label: str,
        combined_embedding: torch.Tensor,
        ocr_text: str = "",
        image_path: str = "",
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
        label_embedding: Optional[torch.Tensor] = None,
        ocr_embedding: Optional[torch.Tensor] = None,
        visual_attributes: Optional[dict] = None,
    ) -> int:
        if timestamp is None:
            timestamp = time.time()
        blob = self._tensor_to_blob(combined_embedding)
        label_blob = self._tensor_to_blob(label_embedding) if label_embedding is not None else None
        ocr_blob = self._tensor_to_blob(ocr_embedding) if ocr_embedding is not None else None
        cur = self._conn.execute(
            "INSERT INTO items "
            "(label, combined_embedding, ocr_text, image_path, confidence, timestamp, label_embedding, ocr_embedding, visual_attributes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                label,
                blob,
                ocr_text or "",
                image_path or "",
                confidence,
                timestamp,
                label_blob,
                ocr_blob,
                json.dumps(visual_attributes or {}),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    @_synchronized
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

    @_synchronized
    def get_items_metadata(self, label: Optional[str] = None) -> list[dict]:
        """Return item rows without embedding blobs - for listing and confirmation UI."""
        if label is not None:
            rows = self._conn.execute(
                "SELECT id, label, confidence, ocr_text, image_path, visual_attributes, timestamp FROM items "
                "WHERE label = ? ORDER BY timestamp DESC",
                (label,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, label, confidence, ocr_text, image_path, visual_attributes, timestamp FROM items "
                "ORDER BY timestamp DESC"
            ).fetchall()
        result = []
        for row_id, lbl, confidence, ocr_text, image_path, visual_attributes, timestamp in rows:
            try:
                attrs = json.loads(visual_attributes) if visual_attributes else {}
            except (json.JSONDecodeError, TypeError):
                attrs = {}
            result.append({
                "id": row_id,
                "label": lbl,
                "confidence": round(confidence, 4) if confidence else None,
                "ocr_text": ocr_text or "",
                "image_path": image_path or "",
                "visual_attributes": attrs,
                "timestamp": timestamp,
            })
        return result

    @_synchronized
    def get_label_embeddings(self) -> list[dict]:
        """Return one label embedding per unique label (most recent row per label)."""
        rows = self._conn.execute(
            "SELECT label, label_embedding FROM items WHERE id IN "
            "(SELECT MAX(id) FROM items GROUP BY label) AND label_embedding IS NOT NULL"
        ).fetchall()
        result = []
        for label, blob in rows:
            if blob:
                result.append({"label": label, "embedding": self._blob_to_tensor(blob)})
        return result

    @_synchronized
    def get_items_with_ocr(self) -> list[dict]:
        """Return one row per unique label where OCR text is non-empty (most recent teach per label).

        Each dict includes `ocr_embedding` (Tensor) when the item was taught with
        pre-embedded OCR, or None for items taught before this feature was added.
        """
        rows = self._conn.execute(
            "SELECT label, ocr_text, ocr_embedding FROM items WHERE id IN "
            "(SELECT MAX(id) FROM items GROUP BY label) AND ocr_text IS NOT NULL AND ocr_text != ''"
        ).fetchall()
        result = []
        for label, ocr_text, ocr_blob in rows:
            result.append({
                "label": label,
                "ocr_text": ocr_text,
                "ocr_embedding": self._blob_to_tensor(ocr_blob) if ocr_blob else None,
            })
        return result

    @_synchronized
    def get_labels_last_seen_in_room(self, room_name: str) -> list[dict]:
        """Return the most recent sighting per label last seen in the given room."""
        rows = self._conn.execute(
            "SELECT id, label, timestamp, direction, distance_ft, similarity, crop_path, room_name "
            "FROM sightings WHERE room_name = ? AND id IN "
            "(SELECT MAX(id) FROM sightings WHERE room_name = ? GROUP BY label) "
            "ORDER BY timestamp DESC",
            (room_name, room_name),
        ).fetchall()
        return [self._sighting_row_to_dict(r) for r in rows]

    @_synchronized
    def rename_label(self, old_label: str, new_label: str) -> dict:
        """
        Rename all items and sightings from old_label to new_label.

        Deletes any existing items with new_label first (caller must handle
        conflict checking / force logic before calling this).

        Returns {"renamed": int, "replaced": int}.
        """
        cur = self._conn.execute("DELETE FROM items WHERE label = ?", (new_label,))
        replaced = cur.rowcount
        cur = self._conn.execute(
            "UPDATE items SET label = ? WHERE label = ?", (new_label, old_label)
        )
        renamed = cur.rowcount
        # Keep sighting history intact under the new name
        self._conn.execute(
            "UPDATE sightings SET label = ? WHERE label = ?", (new_label, old_label)
        )
        self._conn.commit()
        return {"renamed": renamed, "replaced": replaced}

    @_synchronized
    def get_label_avg_confidence(self, label: str) -> Optional[float]:
        """Average detection confidence for all previous teaches of this label. None if no history."""
        row = self._conn.execute(
            "SELECT AVG(confidence) FROM items WHERE label = ? AND confidence > 0",
            (label,),
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
        return None

    @_synchronized
    def delete_item(self, item_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        self._conn.commit()
        return cur.rowcount > 0

    @_synchronized
    def delete_items_by_label(self, label: str) -> int:
        """Delete all items with the given label. Returns number of deleted rows."""
        cur = self._conn.execute("DELETE FROM items WHERE label = ?", (label,))
        self._conn.commit()
        return cur.rowcount

    @_synchronized
    def save_projection_head(self, state_dict: dict) -> None:
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        blob = buf.getvalue()
        self._conn.execute(
            "INSERT OR REPLACE INTO user_state (id, projection_head) VALUES (1, ?)",
            (blob,),
        )
        self._conn.commit()

    @_synchronized
    def load_projection_head(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT projection_head FROM user_state WHERE id = 1"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        buf = io.BytesIO(row[0])
        return torch.load(buf, map_location="cpu", weights_only=True)

    @_synchronized
    def save_user_settings(self, data: dict) -> None:
        self._conn.execute(
            "INSERT INTO user_state (id, user_settings) VALUES (1, ?)"
            " ON CONFLICT(id) DO UPDATE SET user_settings = excluded.user_settings",
            (json.dumps(data),),
        )
        self._conn.commit()

    @_synchronized
    def load_user_settings(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT user_settings FROM user_state WHERE id = 1"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    @_synchronized
    def save_ml_settings(self, data: dict) -> None:
        self._conn.execute(
            "INSERT INTO user_state (id, ml_settings) VALUES (1, ?)"
            " ON CONFLICT(id) DO UPDATE SET ml_settings = excluded.ml_settings",
            (json.dumps(data),),
        )
        self._conn.commit()

    @_synchronized
    def load_ml_settings(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT ml_settings FROM user_state WHERE id = 1"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    @_synchronized
    def add_feedback(
        self,
        label: str,
        feedback_type: str,
        anchor_emb: torch.Tensor,
        query_emb: torch.Tensor,
        timestamp: Optional[float] = None,
    ) -> int:
        if timestamp is None:
            timestamp = time.time()
        cur = self._conn.execute(
            "INSERT INTO feedback (label, feedback_type, anchor_blob, query_blob, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                label,
                feedback_type,
                self._tensor_to_blob(anchor_emb),
                self._tensor_to_blob(query_emb),
                timestamp,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    @_synchronized
    def load_feedback_triplets(self) -> list:
        """Pair each positive with each negative per label -> (anchor, positive, negative)."""
        rows = self._conn.execute(
            "SELECT label, feedback_type, anchor_blob, query_blob, timestamp "
            "FROM feedback ORDER BY timestamp ASC"
        ).fetchall()

        by_label: dict = {}
        for label, ftype, anchor_blob, query_blob, ts in rows:
            if label not in by_label:
                by_label[label] = {"positive": [], "negative": []}
            anchor = self._blob_to_tensor(anchor_blob)
            query = self._blob_to_tensor(query_blob)
            by_label[label][ftype].append((ts, anchor, query))

        triplets = []
        for groups in by_label.values():
            positives = groups["positive"]
            negatives = groups["negative"]
            if not positives or not negatives:
                continue
            for _, anc_p, q_p in positives:
                for _, _anc_n, q_n in negatives:
                    triplets.append((anc_p, q_p, q_n))
        return triplets

    @_synchronized
    def count_feedback(self) -> dict:
        row = self._conn.execute(
            "SELECT "
            "  SUM(CASE WHEN feedback_type = 'positive' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN feedback_type = 'negative' THEN 1 ELSE 0 END) "
            "FROM feedback"
        ).fetchone()
        if row is None:
            positives = 0
            negatives = 0
        else:
            positives = row[0] or 0
            negatives = row[1] or 0
        # triplets = cartesian product of pos x neg per label
        label_rows = self._conn.execute(
            "SELECT label, "
            "  SUM(CASE WHEN feedback_type = 'positive' THEN 1 ELSE 0 END) AS p, "
            "  SUM(CASE WHEN feedback_type = 'negative' THEN 1 ELSE 0 END) AS n "
            "FROM feedback GROUP BY label"
        ).fetchall()
        triplets = sum(p * n for _, p, n in label_rows)
        return {"positives": positives, "negatives": negatives, "triplets": triplets}

    @_synchronized
    def add_sighting(
        self,
        label: str,
        direction: Optional[str] = None,
        distance_ft: Optional[float] = None,
        similarity: Optional[float] = None,
        crop_path: Optional[str] = None,
        room_name: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        if timestamp is None:
            timestamp = time.time()
        cur = self._conn.execute(
            "INSERT INTO sightings "
            "(label, timestamp, direction, distance_ft, similarity, crop_path, room_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (label, timestamp, direction, distance_ft, similarity, crop_path, room_name),
        )
        self._conn.commit()
        return cur.lastrowid

    @_synchronized
    def get_last_sighting(self, label: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, label, timestamp, direction, distance_ft, similarity, crop_path, room_name "
            "FROM sightings WHERE label = ? ORDER BY timestamp DESC LIMIT 1",
            (label,),
        ).fetchone()
        if row is None:
            return None
        return self._sighting_row_to_dict(row)

    @_synchronized
    def get_sightings(
        self,
        label: Optional[str] = None,
        limit: int = 20,
        since: Optional[float] = None,
        before: Optional[float] = None,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list = []
        if label is not None:
            conditions.append("label = ?")
            params.append(label)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        if before is not None:
            conditions.append("timestamp <= ?")
            params.append(before)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT id, label, timestamp, direction, distance_ft, similarity, crop_path, room_name "
            f"FROM sightings {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._sighting_row_to_dict(r) for r in rows]

    @_synchronized
    def get_known_labels(self) -> list[str]:
        """Return labels that have at least one sighting, most recently seen first."""
        rows = self._conn.execute(
            "SELECT label FROM sightings GROUP BY label ORDER BY MAX(timestamp) DESC"
        ).fetchall()
        return [r[0] for r in rows]

    @_synchronized
    def get_known_item_labels(self, limit: int = 64) -> list[str]:
        """Return known labels from items and sightings, newest first."""
        rows = self._conn.execute(
            "SELECT label FROM ("
            "  SELECT label, timestamp AS ts FROM items "
            "  UNION ALL "
            "  SELECT label, timestamp AS ts FROM sightings"
            ") "
            "GROUP BY label ORDER BY MAX(ts) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [str(r[0]) for r in rows if r and r[0]]

    @_synchronized
    def get_recent_room_names(self, limit: int = 24) -> list[str]:
        """Return recently used room names from sightings, newest first."""
        rows = self._conn.execute(
            "SELECT room_name FROM sightings "
            "WHERE room_name IS NOT NULL AND room_name != '' "
            "GROUP BY room_name ORDER BY MAX(timestamp) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [str(r[0]) for r in rows if r and r[0]]

    @_synchronized
    def count_sightings(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM sightings").fetchone()
        return row[0] if row else 0

    @_synchronized
    def clear_items(self) -> int:
        cur = self._conn.execute("DELETE FROM items")
        self._conn.commit()
        return cur.rowcount

    @_synchronized
    def clear_sightings(self) -> int:
        cur = self._conn.execute("DELETE FROM sightings")
        self._conn.commit()
        return cur.rowcount

    @_synchronized
    def clear_feedback(self) -> int:
        cur = self._conn.execute("DELETE FROM feedback")
        self._conn.commit()
        return cur.rowcount

    @_synchronized
    def clear_projection_head(self) -> None:
        self._conn.execute(
            "INSERT INTO user_state (id, projection_head) VALUES (1, NULL)"
            " ON CONFLICT(id) DO UPDATE SET projection_head = NULL"
        )
        self._conn.commit()

    @_synchronized
    def reset_ml_settings(self) -> None:
        self._conn.execute(
            "INSERT INTO user_state (id, ml_settings) VALUES (1, NULL)"
            " ON CONFLICT(id) DO UPDATE SET ml_settings = NULL"
        )
        self._conn.commit()

    @_synchronized
    def reset_user_settings_db(self) -> None:
        self._conn.execute(
            "INSERT INTO user_state (id, user_settings) VALUES (1, NULL)"
            " ON CONFLICT(id) DO UPDATE SET user_settings = NULL"
        )
        self._conn.commit()

    @_synchronized
    def delete_sighting(self, sighting_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM sightings WHERE id = ?", (sighting_id,))
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _sighting_row_to_dict(row: tuple) -> dict:
        sid, label, timestamp, direction, distance_ft, similarity, crop_path, room_name = row
        return {
            "id": sid,
            "label": label,
            "timestamp": timestamp,
            "direction": direction,
            "distance_ft": distance_ft,
            "similarity": similarity,
            "crop_path": crop_path,
            "room_name": room_name,
        }

    @_synchronized
    def close(self):
        self._conn.close()
