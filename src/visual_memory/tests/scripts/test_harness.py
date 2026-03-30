"""
Shared test utilities for the visual_memory test suite.

Single import for all test files. Provides:
- TestRunner: collects results, prints summary, handles exit codes
- make_test_app: creates a minimal Flask test client with stubs, no model loading
- Stub classes for all pipeline components
- Assertion helpers

Environment variables:
    TEST_VERBOSITY: 0=quiet, 1=normal (default), 2=verbose, 3=debug
    VERBOSE=1: maps to verbosity level 2 (backward compat)
    TEST_FORMAT=json: machine-readable output
    TEST_BASE_URL: when set, returns HTTP client instead of Flask test client
    API_KEY: used with TEST_BASE_URL for auth header

Usage:
    from visual_memory.tests.scripts.test_harness import TestRunner, make_test_app
"""
from __future__ import annotations

import io
import json
import os
import sys
import types
import tempfile
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from visual_memory.learning.projection_head import ProjectionHead

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ["ENABLE_DEPTH"] = "0"

try:
    import loguru  # type: ignore
except ImportError:
    class _DummyLoguruLogger:
        def remove(self, *args, **kwargs):
            return None

        def add(self, *args, **kwargs):
            return None

        def bind(self, **kwargs):
            return self

        def opt(self, **kwargs):
            return self

        def debug(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def critical(self, *args, **kwargs):
            return None

    sys.modules["loguru"] = types.SimpleNamespace(logger=_DummyLoguruLogger())

_VERBOSE_ENV = os.environ.get("VERBOSE", "0")
_verbosity_default = 2 if _VERBOSE_ENV == "1" else 1
VERBOSITY = int(os.environ.get("TEST_VERBOSITY", str(_verbosity_default)))
TEST_FORMAT = os.environ.get("TEST_FORMAT", "")
TEST_BASE_URL = os.environ.get("TEST_BASE_URL", "").rstrip("/")


# Output helpers

_G = "\033[32m"
_R = "\033[31m"
_B = "\033[1m"
_X = "\033[0m"


# Stubs

class StubTextEmbedder:
    """Deterministic text embedder - no model loading."""

    def embed_text(self, text: str) -> torch.Tensor:
        torch.manual_seed(hash(text) % (2**32))
        e = torch.randn(1, 512)
        return F.normalize(e, dim=1)

    def batch_embed_text(self, texts):
        return torch.cat([self.embed_text(t) for t in texts])


class StubScanPipeline:
    """Minimal ScanPipeline stub for API integration tests."""

    text_embedder = StubTextEmbedder()
    database_embeddings: list = []
    _enable_learning: bool = True
    _head_weight: float = 1.0
    _head_ramp_at: int = 50
    _triplet_count: int = 0
    _head_trained: bool = False

    def __init__(self):
        self._cached_embeddings: dict = {}
        self._cached_crops: dict = {}
        self._head = ProjectionHead(dim=1536)

    def reload_database(self) -> None:
        pass

    def set_enable_learning(self, v: bool) -> None:
        self._enable_learning = v

    def set_head_weight(self, w: float, r: int) -> None:
        self._head_weight = w
        self._head_ramp_at = r


    def _apply_head(self, emb: torch.Tensor) -> torch.Tensor:
        if not self._head_trained or not self._enable_learning:
            return emb
        alpha = min(
            self._head_weight,
            self._head_weight * self._triplet_count / max(self._head_ramp_at, 1),
        )
        if alpha <= 0.0:
            return emb
        with torch.no_grad():
            projected = self._head.project(emb)
        if alpha >= 1.0:
            return projected
        blended = (1.0 - alpha) * emb + alpha * projected
        return F.normalize(blended, dim=1)

    def reload_head(self) -> bool:
        import visual_memory.api.pipelines as _pm
        if _pm._database is None:
            self._head_trained = False
            return False
        state = _pm._database.load_projection_head()
        if state is None:
            self._head_trained = False
            return False
        self._head.load_state_dict(state)
        self._head_trained = True
        return True

    def get_cached_embeddings(self, scan_id: str, label: str):
        """Return (anchor, query) tensors or None. Configurable per test."""
        key = (scan_id, label)
        return self._cached_embeddings.get(key)

    def get_cached_crop(self, scan_id: str, index: int) -> Optional[Image.Image]:
        """Return cached PIL Image or None."""
        key = (scan_id, index)
        return self._cached_crops.get(key)

    def seed_embeddings(self, scan_id: str, label: str, seed: int = 0) -> None:
        torch.manual_seed(seed)
        a = F.normalize(torch.randn(1, 1536), dim=1)
        q = F.normalize(torch.randn(1, 1536), dim=1)
        self._cached_embeddings[(scan_id, label)] = (a, q)

    def seed_crop(self, scan_id: str, index: int, size: Tuple[int, int] = (64, 64)) -> None:
        self._cached_crops[(scan_id, index)] = Image.new("RGB", size, color=(128, 128, 128))

    def run(self, image, scan_id: str = "", focal_length_px: float = 0.0) -> dict:
        return {"matches": [], "scan_id": scan_id, "is_dark": False, "darkness_level": 10.0}


class StubRememberPipeline:
    """Minimal RememberPipeline stub for API integration tests."""

    def __init__(self, success: bool = True, result: Optional[dict] = None):
        self._success = success
        self._result = result or {
            "label": "wallet",
            "confidence": 0.75,
            "detection_quality": "high",
            "detection_hint": "",
            "blur_score": 150.0,
            "is_blurry": False,
            "second_pass": False,
            "second_pass_prompt": None,
            "box": [10, 10, 100, 100],
            "ocr_text": "",
            "ocr_confidence": 0.0,
        }

    def run(self, path, prompt: str) -> dict:
        if self._success:
            return {"success": True, "message": f"Saved: {prompt}", "result": dict(self._result)}
        return {"success": False, "message": "No object detected.", "result": None}

    def detect_score(self, path, prompt: str) -> dict:
        return {
            "detected": self._success,
            "score": 0.75 if self._success else 0.0,
            "blur_score": 150.0,
            "second_pass_prompt": None,
        }


class StubFeedbackStore:
    """In-memory feedback store for tests."""

    def __init__(self):
        self._positives: list = []
        self._negatives: list = []

    def record_positive(self, anchor, query, label: str) -> None:
        self._positives.append((anchor, query, label))

    def record_negative(self, anchor, query, label: str) -> None:
        self._negatives.append((anchor, query, label))

    def load_triplets(self) -> list:
        triplets = []
        for a, p, _ in self._positives:
            for a2, n, _ in self._negatives:
                triplets.append((a, p, n))
        return triplets

    def count(self) -> dict:
        pos = len(self._positives)
        neg = len(self._negatives)
        return {
            "positives": pos,
            "negatives": neg,
            "triplets": pos * neg,
        }


# Embedding helpers

def make_embedding(seed: int = 0, dim: int = 1536) -> torch.Tensor:
    """Return a deterministic normalized embedding."""
    torch.manual_seed(seed)
    e = torch.randn(1, dim)
    return F.normalize(e, dim=1)


def make_text_embedding(seed: int = 0) -> torch.Tensor:
    return make_embedding(seed, dim=512)


# App factory

def make_test_app(blueprints: list, db_path: str = None, seed_fn: Callable = None):
    """
    Create a minimal Flask test client with stubs injected.

    Does NOT call warm_all() - no model loading.

    Returns (client, db, cleanup_fn).

    When TEST_BASE_URL is set, returns an HTTP client wrapper instead.
    Tests that seed the DB print SKIP and do not run in remote mode.
    """
    if TEST_BASE_URL:
        return _make_remote_client(), None, lambda: None

    tmp_dir = tempfile.mkdtemp()
    if db_path is None:
        db_path = str(Path(tmp_dir) / "test.db")

    from visual_memory.database.store import DatabaseStore
    from visual_memory.config.settings import Settings
    from visual_memory.learning.feedback_store import FeedbackStore
    import visual_memory.api.pipelines as _pm

    db = DatabaseStore(db_path)

    settings = Settings()
    settings.db_path = db_path
    settings.enable_depth = False
    settings.enable_ocr = False
    settings.enable_learning = True

    scan_stub = StubScanPipeline()
    remember_stub = StubRememberPipeline()
    feedback_stub = FeedbackStore(db)
    from visual_memory.config.user_settings import UserSettings
    user_settings = UserSettings()

    _pm._settings = settings
    _pm._database = db
    _pm._scan_pipeline = scan_stub
    _pm._remember_pipeline = remember_stub
    _pm._feedback_store = feedback_stub
    _pm._user_settings = user_settings

    if seed_fn is not None:
        seed_fn(db)

    from flask import Flask
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    for bp in blueprints:
        app.register_blueprint(bp)

    client = app.test_client()

    def cleanup():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # reset pipelines module so next test gets a clean slate
        _pm._database = None
        _pm._scan_pipeline = None
        _pm._remember_pipeline = None
        _pm._feedback_store = None
        _pm._user_settings = None
        _pm._settings = Settings()

    return client, db, cleanup


class _RemoteClient:
    """HTTP wrapper that mimics the Flask test client interface."""

    def __init__(self, base_url: str, api_key: str = ""):
        self._base_url = base_url
        self._api_key = api_key

    def _headers(self, extra: dict = None) -> dict:
        h = {}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        if extra:
            h.update(extra)
        return h

    def get(self, path: str, **kwargs) -> "_RemoteResponse":
        import urllib.request
        url = self._base_url + path
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req) as resp:
                return _RemoteResponse(resp.status, resp.read())
        except Exception as exc:
            return _RemoteResponse(500, str(exc).encode())

    def post(self, path: str, json: dict = None, data=None, content_type: str = None, **kwargs) -> "_RemoteResponse":
        import urllib.request
        url = self._base_url + path
        if json is not None:
            body = __import__("json").dumps(json).encode()
            hdrs = self._headers({"Content-Type": "application/json"})
        else:
            body = data or b""
            hdrs = self._headers({"Content-Type": content_type or "application/octet-stream"})
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return _RemoteResponse(resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            return _RemoteResponse(exc.code, exc.read())
        except Exception as exc:
            return _RemoteResponse(500, str(exc).encode())

    def patch(self, path: str, json: dict = None, **kwargs) -> "_RemoteResponse":
        import urllib.request
        url = self._base_url + path
        body = __import__("json").dumps(json or {}).encode()
        hdrs = self._headers({"Content-Type": "application/json"})
        req = urllib.request.Request(url, data=body, headers=hdrs, method="PATCH")
        try:
            with urllib.request.urlopen(req) as resp:
                return _RemoteResponse(resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            return _RemoteResponse(exc.code, exc.read())
        except Exception as exc:
            return _RemoteResponse(500, str(exc).encode())

    def delete(self, path: str, **kwargs) -> "_RemoteResponse":
        import urllib.request
        url = self._base_url + path
        req = urllib.request.Request(url, headers=self._headers(), method="DELETE")
        try:
            with urllib.request.urlopen(req) as resp:
                return _RemoteResponse(resp.status, resp.read())
        except urllib.error.HTTPError as exc:
            return _RemoteResponse(exc.code, exc.read())
        except Exception as exc:
            return _RemoteResponse(500, str(exc).encode())


class _RemoteResponse:
    def __init__(self, status_code: int, data: bytes):
        self.status_code = status_code
        self._data = data

    def get_json(self):
        try:
            return json.loads(self._data.decode())
        except Exception:
            return None

    @property
    def data(self) -> bytes:
        return self._data


def _make_remote_client() -> _RemoteClient:
    api_key = os.environ.get("API_KEY", "")
    return _RemoteClient(TEST_BASE_URL, api_key)


# TestRunner

class TestRunner:
    """Collects test results, prints summary, returns exit code."""

    def __init__(self, suite_name: str = ""):
        self.suite_name = suite_name
        self._results: list[tuple[str, bool, float, str]] = []

    def run(self, name: str, fn: Callable) -> bool:
        t0 = time.monotonic()
        try:
            fn()
            duration = time.monotonic() - t0
            self._record(name, True, duration, "")
            if VERBOSITY >= 1:
                if TEST_FORMAT == "json":
                    print(json.dumps({"test": name, "status": "pass", "duration": round(duration, 3)}))
                else:
                    print(f"  [PASS]  {name}  ({duration*1000:.0f}ms)")
            return True
        except AssertionError as exc:
            duration = time.monotonic() - t0
            detail = str(exc)
            self._record(name, False, duration, detail)
            if VERBOSITY >= 1:
                if TEST_FORMAT == "json":
                    print(json.dumps({"test": name, "status": "fail", "detail": detail, "duration": round(duration, 3)}))
                else:
                    print(f"  [FAIL]  {name}  ({duration*1000:.0f}ms)")
                    if VERBOSITY >= 2 and detail:
                        print(f"          {detail}")
            return False
        except Exception as exc:
            duration = time.monotonic() - t0
            detail = f"{type(exc).__name__}: {exc}"
            self._record(name, False, duration, detail)
            if VERBOSITY >= 1:
                if TEST_FORMAT == "json":
                    print(json.dumps({"test": name, "status": "error", "detail": detail, "duration": round(duration, 3)}))
                else:
                    print(f"  [FAIL]  {name}  (exception)")
                    if VERBOSITY >= 2:
                        print(f"          {detail}")
            return False

    def _record(self, name: str, passed: bool, duration: float, detail: str) -> None:
        self._results.append((name, passed, duration, detail))

    def summary(self) -> int:
        passed = sum(1 for _, ok, _, _ in self._results if ok)
        failed = len(self._results) - passed
        total = len(self._results)
        total_ms = sum(d for _, _, d, _ in self._results) * 1000

        if TEST_FORMAT == "json":
            print(json.dumps({
                "suite": self.suite_name,
                "passed": passed,
                "failed": failed,
                "total": total,
                "duration_ms": round(total_ms),
            }))
        else:
            label = f" [{self.suite_name}]" if self.suite_name else ""
            print(f"\n{_B}Results{label}: {passed}/{total} passed  ({total_ms:.0f}ms){_X}")
            if failed:
                print(f"{_R}Failures:{_X}")
                for name, ok, _, detail in self._results:
                    if not ok:
                        print(f"  [FAIL] {name}")
                        if detail and VERBOSITY >= 1:
                            print(f"         {detail}")
            else:
                print(f"{_G}All tests passed.{_X}")

        return 0 if failed == 0 else 1


# Assertion helpers

def assert_status(resp, expected: int) -> None:
    code = resp.status_code
    assert code == expected, f"expected HTTP {expected}, got {code}: {resp.get_json()}"


def assert_json_field(resp, field: str, expected: Any) -> None:
    data = resp.get_json()
    assert data is not None, "response body is not JSON"
    assert field in data, f"field '{field}' missing from response: {data}"
    assert data[field] == expected, f"field '{field}': expected {expected!r}, got {data[field]!r}"


def assert_narration_present(resp) -> None:
    data = resp.get_json()
    assert data is not None, "response body is not JSON"
    narration = data.get("narration")
    assert narration and isinstance(narration, str), f"narration missing or empty: {data}"


# DB seeding helpers

def seed_item(db, label: str, ocr_text: str = "", room_name: str = None, emb_seed: int = 0) -> None:
    """Insert a taught item and a sighting into a DatabaseStore."""
    emb = make_embedding(emb_seed)
    label_emb = make_text_embedding(emb_seed)
    db.add_item(
        label=label,
        combined_embedding=emb,
        ocr_text=ocr_text,
        confidence=0.8,
        timestamp=time.time(),
        label_embedding=label_emb,
    )
    if room_name is not None:
        db.add_sighting(
            label=label,
            direction="to your left",
            distance_ft=2.5,
            similarity=0.7,
            room_name=room_name,
            timestamp=time.time(),
        )
