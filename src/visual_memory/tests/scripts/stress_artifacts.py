"""Helpers for writing stress and pentest artifacts."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[4]
_ARTIFACT_DIR = _ROOT / "benchmarks" / "baslines" / "stress_tests"
_RECOMMENDATIONS = _ARTIFACT_DIR / "recommendations.txt"
_FAILURE_MODES = _ARTIFACT_DIR / "failure_modes_report.json"


def artifact_dir() -> Path:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return _ARTIFACT_DIR


def write_json_report(filename: str, payload: dict) -> Path:
    path = artifact_dir() / filename
    data = dict(payload)
    data.setdefault("generated_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def append_recommendations(lines: Iterable[str]) -> Path:
    path = artifact_dir() / _RECOMMENDATIONS.name
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    prefix = "" if not existing else "\n"
    block = "\n".join(lines).strip()
    path.write_text(existing + prefix + block + "\n", encoding="utf-8")
    return path


def record_failure_modes(cases: list[dict]) -> Path:
    path = artifact_dir() / _FAILURE_MODES.name
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []
    if not isinstance(existing, list):
        existing = []
    existing.extend(cases)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


def percentile_ms(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(float(ordered[idx]), 2)
