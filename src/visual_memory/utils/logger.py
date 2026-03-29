"""Structured JSON logging for VisualMemory.

Two log files under logs/:
  app.log       - all levels, rotated daily, 7-day retention
  important.log - WARNING+ only, rotated weekly, 60-day retention
  crash.log     - faulthandler + unhandled exceptions, never auto-deleted

Usage:
    from visual_memory.utils import get_logger
    log = get_logger(__name__)
    log.info({"event": "similarity_check", "score": 0.24, "threshold": 0.3})
    log.info({"event": "pipeline_done", "tag": LogTag.PERF, "duration_ms": 1200})

Call setup_crash_handler() once at process startup (done in create_app).
"""
import datetime
import json
import sys
from pathlib import Path

from loguru import logger as _logger

_LOG_DIR  = Path(__file__).resolve().parents[3] / "logs"
_LOG_PATH = _LOG_DIR / "app.log"

_logger.remove()  # remove default stderr sink


class LogTag:
    PERF      = "perf"
    DETECTION = "detection"
    LEARNING  = "learning"
    API       = "api"
    OCR       = "ocr"
    VRAM      = "vram"


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
        except ImportError:
            pass
        try:
            import torch
            if isinstance(obj, torch.Tensor): return obj.tolist()
        except ImportError:
            pass
        return str(obj)


def _configure_sinks() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    # enqueue=False: sync writes. This server runs a single gunicorn worker (GPU model
    # footprint makes multi-worker impractical), so sync is safe and ensures pre-crash
    # context is flushed. Change to enqueue=True if multi-worker is ever needed.
    #
    # format="{message}\n" is intentional: JSON is built inside _BoundLogger._emit
    # and passed as the full message string. Using a callable format here causes
    # loguru to apply format_map() on the returned JSON string, breaking on {keys}.
    _logger.add(
        str(_LOG_PATH),
        format="{message}\n",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        enqueue=False,
    )
    _logger.add(
        str(_LOG_DIR / "important.log"),
        format="{message}\n",
        level="WARNING",
        rotation="1 week",
        retention="60 days",
        enqueue=False,
    )


_configure_sinks()


class _BoundLogger:
    """Thin wrapper around a loguru bound logger.

    Accepts dicts as log messages (preserving the existing call-site convention):
        log.info({"event": "scan_complete", "duration_ms": 1234})

    Dict fields are serialized to JSON and written as a single line. The ts,
    level, and module fields are injected automatically - call sites do not need
    to supply them.

    String messages are also accepted:
        log.warning("something unexpected happened")
    """

    __slots__ = ("_log", "_module_name")

    def __init__(self, name: str):
        self._module_name = name
        self._log = _logger.bind(module=name)

    def _emit(self, level: str, msg) -> None:
        if isinstance(msg, dict):
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            data: dict = {
                "ts":     ts,
                "level":  level.upper(),
                "module": self._module_name,
            }
            for k, v in msg.items():
                data[k] = v
            line = json.dumps(data, cls=_SafeEncoder)
            getattr(self._log, level)(line)
        else:
            getattr(self._log, level)(str(msg))

    def debug(self, msg)     -> None: self._emit("debug",    msg)
    def info(self, msg)      -> None: self._emit("info",     msg)
    def warning(self, msg)   -> None: self._emit("warning",  msg)
    def error(self, msg)     -> None: self._emit("error",    msg)
    def critical(self, msg)  -> None: self._emit("critical", msg)

    def exception(self, msg) -> None:
        """Log at ERROR level and attach the current exception traceback."""
        import traceback as _tb
        exc_info = sys.exc_info()
        if isinstance(msg, dict):
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            data: dict = {
                "ts":     ts,
                "level":  "ERROR",
                "module": self._module_name,
            }
            for k, v in msg.items():
                data[k] = v
            if exc_info[0] is not None:
                data["exception"] = "".join(_tb.format_exception(*exc_info))
            line = json.dumps(data, cls=_SafeEncoder)
            self._log.error(line)
        else:
            if exc_info[0] is not None:
                self._log.opt(exception=exc_info).error(str(msg))
            else:
                self._log.error(str(msg))


def get_logger(name: str) -> _BoundLogger:
    return _BoundLogger(name)


def setup_crash_handler() -> None:
    """Install faulthandler (C crashes) and sys.excepthook (Python exceptions).

    Writes to logs/crash.log. Call once at process startup.
    """
    import faulthandler
    crash_path = _LOG_DIR / "crash.log"
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    # faulthandler handles SIGSEGV, SIGBUS, stack overflow, etc.
    faulthandler.enable(file=open(crash_path, "a"), all_threads=True)

    _orig_hook = sys.excepthook

    def _excepthook(exc_type, exc_val, exc_tb):
        import traceback
        with open(crash_path, "a") as f:
            f.write(f"\n--- CRASH {datetime.datetime.now().isoformat()} ---\n")
            traceback.print_exception(exc_type, exc_val, exc_tb, file=f)
        _logger.opt(exception=(exc_type, exc_val, exc_tb)).critical(
            "Unhandled exception"
        )
        _orig_hook(exc_type, exc_val, exc_tb)

    sys.excepthook = _excepthook


def log_mark() -> int:
    """Return current line count in app.log (0 if absent). Use as a section bookmark."""
    if not _LOG_PATH.exists():
        return 0
    return len(_LOG_PATH.read_text(encoding="utf-8").splitlines())


def tail_logs(event: str | None = None, n: int = 50, since_line: int = 0) -> list[dict]:
    """Read last n lines from app.log, optionally filtered by event type."""
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
