"""
Whisper context-bias evaluator using unified voice_eval_dataset.json.

Quick run:
  TEST_BASE_URL=... API_KEY=... python -m visual_memory.tests.scripts.test_whisper_context_bias_eval \
    --dataset src/visual_memory/tests/input_data/voice_eval_dataset.json --core-only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from visual_memory.tests.scripts.voice_eval_common import load_voice_cases, resolve_audio_path


def _post_audio(base_url: str, api_key: str, audio_path: Path, context: int) -> tuple[int, dict | None]:
    import urllib.request
    import urllib.error

    data = audio_path.read_bytes()
    req = urllib.request.Request(
        f"{base_url}/transcribe?context={context}",
        data=data,
        method="POST",
        headers={
            "X-API-Key": api_key,
            "Content-Type": "audio/m4a" if audio_path.suffix.lower() == ".m4a" else "audio/ogg",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"raw": body}
    except Exception as exc:
        return 599, {"error": str(exc)}


def _contains_all(text: str, tokens: list[str]) -> bool:
    low = text.lower()
    return all(tok.lower() in low for tok in tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper context bias evaluator")
    parser.add_argument("--dataset", default="src/visual_memory/tests/input_data/voice_eval_dataset.json")
    parser.add_argument("--base-url", default=os.environ.get("TEST_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()

    if not args.base_url:
        print("SKIP: base URL not configured (set --base-url or TEST_BASE_URL)")
        return 0
    if not args.api_key:
        print("SKIP: API key not configured (set --api-key or API_KEY)")
        return 0

    repo_root = Path(__file__).resolve().parents[4]
    cases = load_voice_cases(args.dataset, target="whisper", core_only=args.core_only)
    if not cases:
        print("ERROR: no whisper cases in dataset")
        return 2

    rows = []
    missing_audio = []
    wins_context_1 = 0
    wins_context_0 = 0
    ties = 0

    for case in cases:
        audio_path = resolve_audio_path(case, repo_root)
        if audio_path is None:
            missing_audio.append(case["id"])
            continue

        tokens = list(case.get("expected_text_contains", []))
        code0, out0 = _post_audio(args.base_url, args.api_key, audio_path, context=0)
        code1, out1 = _post_audio(args.base_url, args.api_key, audio_path, context=1)

        text0 = (out0 or {}).get("text", "") if code0 == 200 else ""
        text1 = (out1 or {}).get("text", "") if code1 == 200 else ""
        pass0 = code0 == 200 and _contains_all(text0, tokens)
        pass1 = code1 == 200 and _contains_all(text1, tokens)

        if pass1 and not pass0:
            wins_context_1 += 1
        elif pass0 and not pass1:
            wins_context_0 += 1
        else:
            ties += 1

        rows.append(
            {
                "id": case["id"],
                "feature": case.get("feature"),
                "priority": case.get("priority"),
                "audio_path": str(audio_path),
                "expected_text_contains": tokens,
                "meta": {
                    "noise": case.get("noise"),
                    "accent": case.get("accent"),
                    "speed": case.get("speed"),
                },
                "context0": {"status": code0, "text": text0, "pass": pass0},
                "context1": {"status": code1, "text": text1, "pass": pass1},
            }
        )

    summary = {
        "dataset": args.dataset,
        "core_only": bool(args.core_only),
        "eligible_cases": len(cases),
        "executed_cases": len(rows),
        "missing_audio_count": len(missing_audio),
        "missing_audio_ids": missing_audio,
        "context0_passed": sum(1 for r in rows if r["context0"]["pass"]),
        "context1_passed": sum(1 for r in rows if r["context1"]["pass"]),
        "wins_context1": wins_context_1,
        "wins_context0": wins_context_0,
        "ties": ties,
    }

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "base_url": args.base_url,
        "summary": summary,
        "cases": rows,
    }

    logs_dir = repo_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_path = logs_dir / "whisper_context_bias_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(summary))
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
