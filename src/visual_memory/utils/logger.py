"""
Structured JSON logging for VisualMemory.

Usage:
    from visual_memory.utils import get_logger
    log = get_logger(__name__)
    log.info({"event": "similarity_check", "score": 0.24, "threshold": 0.3})

Log file: logs/app.log (project root), JSON Lines format.
"""
import json
import logging
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parents[3] / "logs" / "app.log"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = record.msg if isinstance(record.msg, dict) else {"message": record.msg}
        return json.dumps({
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            **payload,
        })


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.DEBUG)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_mark() -> int:
    """Return current line count in app.log (0 if file absent). Use as a section bookmark."""
    if not _LOG_PATH.exists():
        return 0
    return len(_LOG_PATH.read_text(encoding="utf-8").splitlines())


def tail_logs(event: str | None = None, n: int = 50, since_line: int = 0) -> list[dict]:
    """
    Read the last n lines from app.log, optionally filtered by event type.
    Returns list of parsed JSON dicts.
    """
    if not _LOG_PATH.exists():
        return []
    all_lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    lines = all_lines[since_line:][-n:]
    records = []
    for line in lines:
        try:
            r = json.loads(line)
            if event is None or r.get("event") == event:
                records.append(r)
        except json.JSONDecodeError:
            pass
    return records
