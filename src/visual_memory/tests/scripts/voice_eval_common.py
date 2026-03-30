from __future__ import annotations

import json
from pathlib import Path


def load_voice_cases(dataset_path: str, target: str, core_only: bool = False) -> list[dict]:
    payload = json.loads(Path(dataset_path).read_text())
    cases = payload.get("cases", [])
    out = []
    for case in cases:
        include_for = case.get("include_for", [])
        if target not in include_for:
            continue
        if core_only and case.get("priority") != "core":
            continue
        out.append(case)
    return out


def resolve_audio_path(case: dict, repo_root: Path) -> Path | None:
    declared = str(case.get("audio_path", "")).strip()
    if declared:
        p = Path(declared)
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            return p
        parent = p.parent
        stem = p.stem.lower()
        if parent.exists():
            for cand in parent.iterdir():
                if cand.is_file() and cand.stem.lower() == stem:
                    return cand

    base = repo_root / "src" / "visual_memory" / "tests" / "input_audio" / case["id"]
    search_dir = base.parent
    stem = base.stem.lower()
    if search_dir.exists():
        for cand in search_dir.iterdir():
            if cand.is_file() and cand.stem.lower() == stem:
                return cand
    return None
