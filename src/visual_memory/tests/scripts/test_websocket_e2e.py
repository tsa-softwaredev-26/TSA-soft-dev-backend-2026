"""
End to end WebSocket validation against a live server using python-socketio client.

Usage:
  TEST_BASE_URL=https://nre5bjw44wddpu2zjg4fe4iehq.srv.us \
  API_KEY=... \
  python3 -m visual_memory.tests.scripts.test_websocket_e2e

Optional env knobs:
  WS_E2E_ALLOW_WIPE=0|1         default 0
  WS_E2E_WIPE_TARGET=all        debug wipe target
  WS_E2E_CONNECT_TIMEOUT=12
  WS_E2E_EVENT_TIMEOUT=45
  WS_E2E_MAX_CONNECT_LATENCY=2.0
  WS_E2E_RAPID_FIRE_COUNT=10
  WS_E2E_RAPID_FIRE_INTERVAL=0.08
  WS_E2E_SCAN_IMAGE=path
  WS_E2E_TEACH_IMAGE=path
  WS_E2E_DATASET=path
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from visual_memory.tests.scripts.test_harness import TestRunner
from visual_memory.tests.scripts.voice_eval_common import resolve_audio_path

try:
    import socketio
except ImportError as exc:
    print('ERROR: missing dependency python-socketio[client]. Install with: pip install "python-socketio[client]"')
    raise SystemExit(2) from exc

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_BASE_URL = "https://nre5bjw44wddpu2zjg4fe4iehq.srv.us"


class WsRecorder:
    def __init__(self, base_url: str, api_key: str, connect_timeout: float):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.client = socketio.Client(reconnection=False, logger=False, engineio_logger=False, ssl_verify=True)

        self.client.on("connect", self._make_handler("connect"))
        self.client.on("disconnect", self._make_handler("disconnect"))
        self.client.on("tts", self._make_handler("tts"))
        self.client.on("transcription", self._make_handler("transcription"))
        self.client.on("action_result", self._make_handler("action_result"))
        self.client.on("control", self._make_handler("control"))
        self.client.on("error", self._make_handler("error"))

    def _make_handler(self, name: str):
        def _handler(data=None):
            with self._lock:
                self.events.append(
                    {
                        "name": name,
                        "data": data,
                        "ts": time.monotonic(),
                    }
                )

        return _handler

    def connect(self) -> float:
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        t0 = time.monotonic()
        self.client.connect(
            self.base_url,
            headers=headers,
            transports=["websocket", "polling"],
            socketio_path="socket.io",
            wait_timeout=self.connect_timeout,
        )
        return time.monotonic() - t0

    def disconnect(self) -> None:
        if self.client.connected:
            self.client.disconnect()

    def mark(self) -> int:
        with self._lock:
            return len(self.events)

    def wait_event(self, name: str, start: int, timeout: float, predicate=None) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() <= deadline:
            with self._lock:
                for event in self.events[start:]:
                    if event["name"] != name:
                        continue
                    if predicate is None or predicate(event["data"]):
                        return event
            time.sleep(0.05)
        return None

    def collect_since(self, start: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events[start:])

    def emit_audio(self, audio_bytes: bytes, image_bytes: bytes | None = None, focal_length_px: float | None = None) -> None:
        payload: dict[str, Any] = {
            "audio": base64.b64encode(audio_bytes).decode("ascii"),
        }
        if image_bytes is not None:
            payload["image"] = base64.b64encode(image_bytes).decode("ascii")
        if focal_length_px is not None:
            payload["focal_length_px"] = focal_length_px
        self.client.emit("audio", payload)

    def emit_raw_audio_payload(self, payload: Any) -> None:
        self.client.emit("audio", payload)

    def emit_navigate(self, direction: str) -> None:
        self.client.emit("navigate", {"direction": direction})


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _load_dataset(dataset_path: Path) -> dict[str, dict]:
    payload = json.loads(dataset_path.read_text())
    out: dict[str, dict] = {}
    for case in payload.get("cases", []):
        out[str(case.get("id", "")).lower()] = case
    return out


def _load_audio_case(case_map: dict[str, dict], case_id: str) -> bytes:
    case = case_map.get(case_id.lower())
    assert case is not None, f"missing dataset case: {case_id}"
    audio_path = resolve_audio_path(case, _REPO_ROOT)
    assert audio_path is not None and audio_path.exists(), f"missing audio for case: {case_id}"
    return audio_path.read_bytes()


def _load_file(path_value: str, fallback_rel: str) -> bytes:
    path = Path(path_value.strip()) if path_value.strip() else (_REPO_ROOT / fallback_rel)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    assert path.exists(), f"file not found: {path}"
    return path.read_bytes()


def _post_json(base_url: str, api_key: str, route: str, payload: dict, timeout: float) -> tuple[int, dict | None]:
    url = f"{base_url.rstrip('/')}{route}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"raw": body}
    except urllib.error.URLError as exc:
        return 599, {"error": str(exc)}


def _assert_event_response(events: list[dict[str, Any]], expected: set[str], context: str) -> None:
    seen = {e["name"] for e in events}
    assert seen.intersection(expected), f"{context}: expected one of {sorted(expected)}, got {sorted(seen)}"


def _wait_any_event(ws: WsRecorder, start: int, timeout: float, names: set[str]) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() <= deadline:
        events = ws.collect_since(start)
        for event in events:
            if event["name"] in names:
                return event
        time.sleep(0.05)
    return None


def _test_runtime_worker_config() -> None:
    wsgi_path = _REPO_ROOT / "services" / "core" / "wsgi.py"
    service_path = _REPO_ROOT / "deploy" / "spaitra-core.service"
    app_path = _REPO_ROOT / "src" / "visual_memory" / "api" / "app.py"

    wsgi_text = wsgi_path.read_text()
    service_text = service_path.read_text()
    app_text = app_path.read_text()

    worker = "geventwebsocket.gunicorn.workers.GeventWebSocketWorker"
    mismatches: list[str] = []

    if worker not in wsgi_text:
        mismatches.append("services/core/wsgi.py missing GeventWebSocketWorker hint")
    if worker not in service_text:
        mismatches.append("deploy/spaitra-core.service missing GeventWebSocketWorker")
    if 'async_mode="gevent"' not in app_text:
        mismatches.append('src/visual_memory/api/app.py missing async_mode="gevent"')

    assert not mismatches, " ; ".join(mismatches)


def _test_connect_and_welcome(cfg: dict, assets: dict) -> None:
    ws = WsRecorder(cfg["base_url"], cfg["api_key"], cfg["connect_timeout"])
    try:
        latency = ws.connect()
        first_tts = ws.wait_event("tts", start=0, timeout=cfg["event_timeout"])
        assert first_tts is not None, "no tts event after connect"
        payload = first_tts["data"] or {}
        next_state = payload.get("next_state")
        assert next_state in {"onboarding_teach", "idle"}, f"unexpected next_state={next_state!r}"
        assert latency <= cfg["max_connect_latency"], (
            f"connect latency {latency:.2f}s exceeds {cfg['max_connect_latency']:.2f}s"
        )
    finally:
        ws.disconnect()


def _test_onboarding_flow_skeleton(cfg: dict, assets: dict) -> None:
    if cfg["allow_wipe"]:
        status, body = _post_json(
            cfg["base_url"],
            cfg["api_key"],
            "/debug/wipe",
            {"confirm": True, "target": cfg["wipe_target"]},
            timeout=max(cfg["event_timeout"], 20.0),
        )
        assert status == 200, f"debug wipe failed: status={status}, body={body}"

    ws = WsRecorder(cfg["base_url"], cfg["api_key"], cfg["connect_timeout"])
    try:
        ws.connect()
        first_tts = ws.wait_event("tts", start=0, timeout=cfg["event_timeout"])
        assert first_tts is not None, "missing welcome tts"

        marker = ws.mark()
        ws.emit_audio(assets["audio_teach_1"], image_bytes=assets["teach_image"])
        assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "teach_1 missing transcription"
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"action_result", "control", "error", "tts"}) is not None
        teach_events = ws.collect_since(marker)
        _assert_event_response(teach_events, {"action_result", "control", "error", "tts"}, "teach_1")

        latest_tts = [e for e in teach_events if e["name"] == "tts"]
        if latest_tts and (latest_tts[-1].get("data") or {}).get("next_state") == "awaiting_location":
            marker = ws.mark()
            ws.emit_audio(assets["audio_room_1"])
            assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "room_1 missing transcription"
            assert _wait_any_event(ws, marker, cfg["event_timeout"], {"tts", "action_result", "error"}) is not None
            room_events = ws.collect_since(marker)
            _assert_event_response(room_events, {"tts", "action_result", "error"}, "room_1")

        marker = ws.mark()
        ws.emit_audio(assets["audio_teach_2"], image_bytes=assets["teach_image"])
        assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "teach_2 missing transcription"
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"action_result", "control", "error", "tts"}) is not None
        teach2_events = ws.collect_since(marker)
        _assert_event_response(teach2_events, {"action_result", "control", "error", "tts"}, "teach_2")

        latest_tts = [e for e in teach2_events if e["name"] == "tts"]
        if latest_tts and (latest_tts[-1].get("data") or {}).get("next_state") == "awaiting_location":
            marker = ws.mark()
            ws.emit_audio(assets["audio_room_2"])
            assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "room_2 missing transcription"
            assert _wait_any_event(ws, marker, cfg["event_timeout"], {"tts", "action_result", "error"}) is not None
            room2_events = ws.collect_since(marker)
            _assert_event_response(room2_events, {"tts", "action_result", "error"}, "room_2")

        marker = ws.mark()
        ws.emit_audio(assets["audio_scan"], image_bytes=assets["scan_image"])
        assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "scan missing transcription"
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"action_result", "control", "error", "tts"}) is not None
        scan_events = ws.collect_since(marker)
        _assert_event_response(scan_events, {"action_result", "control", "error", "tts"}, "scan")

        marker = ws.mark()
        ws.emit_navigate("next")
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"tts", "action_result", "error"}) is not None
        nav_next = ws.collect_since(marker)
        _assert_event_response(nav_next, {"tts", "action_result", "error"}, "navigate_next")

        marker = ws.mark()
        ws.emit_navigate("prev")
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"tts", "action_result", "error"}) is not None
        nav_prev = ws.collect_since(marker)
        _assert_event_response(nav_prev, {"tts", "action_result", "error"}, "navigate_prev")

        marker = ws.mark()
        ws.emit_audio(assets["audio_ask"])
        assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "ask missing transcription"
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"action_result", "tts", "error"}) is not None
        ask_events = ws.collect_since(marker)
        _assert_event_response(ask_events, {"action_result", "tts", "error"}, "ask")
    finally:
        ws.disconnect()


def _test_teach_scan_ask_cycle(cfg: dict, assets: dict) -> None:
    ws = WsRecorder(cfg["base_url"], cfg["api_key"], cfg["connect_timeout"])
    try:
        ws.connect()
        ws.wait_event("tts", 0, cfg["event_timeout"])

        marker = ws.mark()
        ws.emit_audio(assets["audio_scan"], image_bytes=assets["scan_image"])
        assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "cycle scan missing transcription"
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"action_result", "control", "error", "tts"}) is not None
        scan_events = ws.collect_since(marker)

        scan_result = None
        for event in scan_events:
            if event["name"] == "action_result":
                payload = event.get("data") or {}
                if payload.get("type") == "scan":
                    scan_result = payload.get("data") or {}
                    break

        if scan_result:
            scan_id = scan_result.get("scan_id", "")
            matches = scan_result.get("matches") or []
            if scan_id and matches:
                label = (matches[0] or {}).get("label", "")
                if label:
                    status, body = _post_json(
                        cfg["base_url"],
                        cfg["api_key"],
                        "/feedback",
                        {"scan_id": scan_id, "label": label, "feedback": "correct"},
                        timeout=max(cfg["event_timeout"], 20.0),
                    )
                    assert status == 200, f"feedback failed: status={status}, body={body}"

        marker = ws.mark()
        ws.emit_audio(assets["audio_ask"])
        assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "cycle ask missing transcription"
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"action_result", "tts", "error"}) is not None
        ask_events = ws.collect_since(marker)
        _assert_event_response(ask_events, {"action_result", "tts", "error"}, "cycle ask")
    finally:
        ws.disconnect()


def _test_error_handling(cfg: dict, assets: dict) -> None:
    ws = WsRecorder(cfg["base_url"], cfg["api_key"], cfg["connect_timeout"])
    try:
        ws.connect()
        ws.wait_event("tts", 0, cfg["event_timeout"])

        marker = ws.mark()
        ws.emit_audio(assets["audio_scan"])
        assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "missing-image scan missing transcription"
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"control", "tts", "error"}) is not None
        miss_events = ws.collect_since(marker)
        _assert_event_response(miss_events, {"control", "tts", "error"}, "missing_image")

        marker = ws.mark()
        ws.emit_raw_audio_payload({"audio": base64.b64encode(assets["audio_scan"]).decode("ascii"), "image": "not-base64!!"})
        err_event = ws.wait_event("error", marker, cfg["event_timeout"])
        assert err_event is not None, "missing bad_payload error for invalid image"
        code = (err_event.get("data") or {}).get("code")
        assert code == "bad_payload", f"expected bad_payload, got {code!r}"

        marker = ws.mark()
        ws.emit_raw_audio_payload({"audio": ""})
        empty_err = ws.wait_event("error", marker, cfg["event_timeout"])
        assert empty_err is not None, "missing error for empty audio"
        empty_code = (empty_err.get("data") or {}).get("code")
        assert empty_code in {"transcription_failed", "bad_payload"}, f"unexpected empty-audio code {empty_code!r}"

        marker = ws.mark()
        ws.emit_navigate("sideways")
        assert _wait_any_event(ws, marker, cfg["event_timeout"], {"tts", "action_result", "error"}) is not None
        nav_events = ws.collect_since(marker)
        _assert_event_response(nav_events, {"tts", "action_result", "error"}, "invalid navigate")
    finally:
        ws.disconnect()


def _test_stress_rapid_fire(cfg: dict, assets: dict) -> None:
    ws = WsRecorder(cfg["base_url"], cfg["api_key"], cfg["connect_timeout"])
    try:
        ws.connect()
        ws.wait_event("tts", 0, cfg["event_timeout"])

        marker = ws.mark()
        for _ in range(cfg["rapid_fire_count"]):
            ws.emit_audio(assets["audio_scan"])
            time.sleep(cfg["rapid_fire_interval"])

        deadline = time.monotonic() + cfg["event_timeout"]
        while time.monotonic() <= deadline:
            events = ws.collect_since(marker)
            terminal = [e for e in events if e["name"] in {"transcription", "error"}]
            if len(terminal) >= cfg["rapid_fire_count"]:
                break
            time.sleep(0.1)

        events = ws.collect_since(marker)
        terminal = [e for e in events if e["name"] in {"transcription", "error"}]
        assert len(terminal) >= cfg["rapid_fire_count"], (
            f"rapid fire responses {len(terminal)} < sent {cfg['rapid_fire_count']}"
        )
    finally:
        ws.disconnect()


def _test_stress_reconnect(cfg: dict, assets: dict) -> None:
    ws = WsRecorder(cfg["base_url"], cfg["api_key"], cfg["connect_timeout"])
    try:
        ws.connect()
        assert ws.wait_event("tts", 0, cfg["event_timeout"]) is not None, "first connect missing welcome"

        marker = ws.mark()
        ws.emit_audio(assets["audio_ask"])
        assert ws.wait_event("transcription", marker, cfg["event_timeout"]) is not None, "first session command failed"

        ws.disconnect()
        time.sleep(0.25)

        reconnect_marker = ws.mark()
        ws.connect()
        reconnect_tts = ws.wait_event("tts", reconnect_marker, cfg["event_timeout"])
        assert reconnect_tts is not None, "reconnect missing welcome tts"
        state = (reconnect_tts.get("data") or {}).get("next_state")
        assert state in {"idle", "onboarding_teach"}, f"unexpected reconnect state {state!r}"
    finally:
        ws.disconnect()


def _test_stress_invalid_payloads(cfg: dict, assets: dict) -> None:
    ws = WsRecorder(cfg["base_url"], cfg["api_key"], cfg["connect_timeout"])
    try:
        ws.connect()
        ws.wait_event("tts", 0, cfg["event_timeout"])

        payloads = [
            "invalid-json-object",
            {},
            {"audio": "%%%%"},
            {"audio": base64.b64encode(assets["audio_scan"]).decode("ascii"), "image": "%%%%"},
            {"audio": base64.b64encode(assets["audio_scan"]).decode("ascii"), "image": base64.b64encode(b"not-an-image").decode("ascii")},
        ]

        marker = ws.mark()
        for payload in payloads:
            ws.emit_raw_audio_payload(payload)
            time.sleep(0.05)

        deadline = time.monotonic() + cfg["event_timeout"]
        while time.monotonic() <= deadline:
            events = ws.collect_since(marker)
            errors = [e for e in events if e["name"] == "error"]
            if len(errors) >= len(payloads):
                break
            time.sleep(0.1)

        events = ws.collect_since(marker)
        errors = [e for e in events if e["name"] == "error"]
        assert len(errors) >= len(payloads), f"invalid payload errors {len(errors)} < sent {len(payloads)}"
    finally:
        ws.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="WebSocket E2E validation runner")
    parser.add_argument("--base-url", default=os.environ.get("TEST_BASE_URL", _DEFAULT_BASE_URL).strip())
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "").strip())
    parser.add_argument(
        "--dataset",
        default=os.environ.get("WS_E2E_DATASET", "src/visual_memory/tests/input_data/voice_eval_dataset.json"),
    )
    args = parser.parse_args()

    if not args.base_url:
        print("ERROR: base URL required (set --base-url or TEST_BASE_URL)")
        return 2
    if not args.api_key:
        print("ERROR: API key required (set --api-key or API_KEY)")
        return 2

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    case_map = _load_dataset(dataset_path)

    assets = {
        "audio_teach_1": _load_audio_case(case_map, os.environ.get("WS_E2E_AUDIO_TEACH_1", "v061")),
        "audio_room_1": _load_audio_case(case_map, os.environ.get("WS_E2E_AUDIO_ROOM_1", "v076")),
        "audio_teach_2": _load_audio_case(case_map, os.environ.get("WS_E2E_AUDIO_TEACH_2", "v066")),
        "audio_room_2": _load_audio_case(case_map, os.environ.get("WS_E2E_AUDIO_ROOM_2", "v077")),
        "audio_scan": _load_audio_case(case_map, os.environ.get("WS_E2E_AUDIO_SCAN", "v078")),
        "audio_ask": _load_audio_case(case_map, os.environ.get("WS_E2E_AUDIO_ASK", "v001")),
        "teach_image": _load_file(
            os.environ.get("WS_E2E_TEACH_IMAGE", ""),
            "src/visual_memory/tests/input_images/wallet_1ft_table.jpg",
        ),
        "scan_image": _load_file(
            os.environ.get("WS_E2E_SCAN_IMAGE", ""),
            "src/visual_memory/tests/input_images/wallet_3ft_table.jpg",
        ),
    }

    cfg = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "allow_wipe": _env_bool("WS_E2E_ALLOW_WIPE", False),
        "wipe_target": os.environ.get("WS_E2E_WIPE_TARGET", "all"),
        "connect_timeout": _env_float("WS_E2E_CONNECT_TIMEOUT", 12.0),
        "event_timeout": _env_float("WS_E2E_EVENT_TIMEOUT", 45.0),
        "max_connect_latency": _env_float("WS_E2E_MAX_CONNECT_LATENCY", 2.0),
        "rapid_fire_count": _env_int("WS_E2E_RAPID_FIRE_COUNT", 10),
        "rapid_fire_interval": _env_float("WS_E2E_RAPID_FIRE_INTERVAL", 0.08),
    }

    runner = TestRunner("websocket_e2e")

    tests = [
        ("ws_config:worker_alignment", lambda: _test_runtime_worker_config()),
        ("ws_e2e:connect_and_welcome", lambda: _test_connect_and_welcome(cfg, assets)),
        ("ws_e2e:onboarding_flow_skeleton", lambda: _test_onboarding_flow_skeleton(cfg, assets)),
        ("ws_e2e:teach_scan_ask_cycle", lambda: _test_teach_scan_ask_cycle(cfg, assets)),
        ("ws_e2e:error_handling", lambda: _test_error_handling(cfg, assets)),
        ("ws_e2e:stress_rapid_fire", lambda: _test_stress_rapid_fire(cfg, assets)),
        ("ws_e2e:stress_reconnect", lambda: _test_stress_reconnect(cfg, assets)),
        ("ws_e2e:stress_invalid_payloads", lambda: _test_stress_invalid_payloads(cfg, assets)),
    ]

    for name, fn in tests:
        runner.run(name, fn)

    return runner.summary()


if __name__ == "__main__":
    raise SystemExit(main())
