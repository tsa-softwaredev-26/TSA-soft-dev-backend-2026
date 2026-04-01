"""
Teach a full image database, then replay every audio case and analyze results.

Usage:
  TEST_BASE_URL=https://<srv-url> API_KEY=<key> \
  python -m visual_memory.tests.scripts.test_full_db_audio_query_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path

from visual_memory.tests.scripts.voice_eval_common import resolve_audio_path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_BASE_URL = "https://nre5bjw44wddpu2zjg4fe4iehq.srv.us"

_PLACEHOLDER_API_KEYS = {
    "placeholder",
    "your_api_key",
    "your-api-key",
    "api_key",
    "api-key",
    "changeme",
    "dummy",
    "test",
    "none",
    "null",
}


def _is_placeholder_api_key(api_key: str) -> bool:
    normalized = api_key.strip().lower()
    if not normalized:
        return True
    if normalized in _PLACEHOLDER_API_KEYS:
        return True
    if normalized.startswith("<") and normalized.endswith(">"):
        return True
    return False


def _probe_api_key(base_url: str, api_key: str, timeout: float) -> tuple[str, str]:
    if _is_placeholder_api_key(api_key):
        return "skip", "API key is missing or placeholder; skipping live full DB audio eval."

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/items",
        method="GET",
        headers={"X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status in (200, 204):
                return "ok", "api key accepted"
            return "warn", f"auth preflight returned status={resp.status}; continuing"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return "skip", f"API key rejected by auth preflight (status={exc.code}); skipping."
        return "error", f"auth preflight failed with HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return "error", f"auth preflight failed: {exc}"


def _guess_audio_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".m4a":
        return "audio/m4a"
    if ext == ".ogg":
        return "audio/ogg"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".webm":
        return "audio/webm"
    return "application/octet-stream"


def _post_json(base_url: str, api_key: str, route: str, payload: dict, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{route}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"raw": body}
    except Exception as exc:
        return 599, {"error": str(exc)}


def _post_multipart_remember(
    base_url: str,
    api_key: str,
    image_bytes: bytes,
    filename: str,
    prompt: str,
    timeout: float,
) -> tuple[int, dict]:
    boundary = uuid.uuid4().hex
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        f"{prompt}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/remember",
        data=payload,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-API-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"raw": body}
    except Exception as exc:
        return 599, {"error": str(exc)}


def _post_voice_audio(base_url: str, api_key: str, audio_path: Path, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/voice?context=1",
        data=audio_path.read_bytes(),
        method="POST",
        headers={
            "X-API-Key": api_key,
            "Content-Type": _guess_audio_content_type(audio_path),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"raw": body}
    except Exception as exc:
        return 599, {"error": str(exc)}


def _contains_all(text: str, tokens: list[str]) -> bool:
    low = (text or "").lower()
    return all(str(token).lower() in low for token in tokens)


def _resolve_image_path(images_dir: Path, image_name: str) -> Path | None:
    direct = images_dir / image_name
    if direct.exists():
        return direct
    stem = Path(image_name).stem.lower()
    for cand in images_dir.iterdir():
        if cand.is_file() and cand.stem.lower() == stem:
            return cand
    return None


def _load_dataset_cases(dataset_path: Path) -> list[dict]:
    payload = json.loads(dataset_path.read_text())
    return list(payload.get("cases", []))


def _load_teach_rows(csv_path: Path, teach_limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        raw_lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
        if not raw_lines:
            return rows
        reader = csv.DictReader(raw_lines)
        for row in reader:
            if not row.get("image") or not row.get("label"):
                continue
            rows.append({"image": row["image"].strip(), "label": row["label"].strip()})
    if teach_limit > 0:
        return rows[:teach_limit]
    return rows


def _maybe_transcode_audio(audio_path: Path, tmp_dir: Path) -> tuple[Path, bool]:
    """
    Convert m4a inputs to wav when ffmpeg is available.

    This avoids remote decoder mismatches while keeping source fixtures unchanged.
    """
    if audio_path.suffix.lower() != ".m4a":
        return audio_path, False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return audio_path, False
    out_path = tmp_dir / f"{audio_path.stem}.ogg"
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-acodec",
            "libopus",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        return audio_path, False
    return out_path, True


def _write_markdown(path: Path, summary: dict, teach_rows: list[dict], query_rows: list[dict]) -> None:
    failed_teach = [r for r in teach_rows if not r.get("success")]
    failed_queries = [r for r in query_rows if not r.get("token_pass") or r.get("status") != 200]
    lines = [
        "# Full DB Audio Query Evaluation",
        "",
        f"- Date: `{summary['date']}`",
        f"- Base URL: `{summary['base_url']}`",
        f"- Taught rows: `{summary['teach_rows_success']}/{summary['teach_rows_attempted']}`",
        f"- Unique taught labels: `{summary['teach_unique_labels']}`",
        f"- Audio executed: `{summary['query_executed_cases']}/{summary['query_total_cases']}`",
        f"- Voice status 200: `{summary['query_status_200_count']}`",
        f"- Token pass: `{summary['query_token_pass_count']}`",
        f"- Token pass rate: `{summary['query_token_pass_rate']}`",
        "",
        "## Top teach failures",
        "",
    ]
    if failed_teach:
        for row in failed_teach[:15]:
            lines.append(
                f"- `{row.get('image')}` label=`{row.get('label')}` "
                f"status=`{row.get('status')}` error=`{row.get('error')}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Top query misses", ""])
    if failed_queries:
        for row in failed_queries[:20]:
            lines.append(
                f"- `{row.get('id')}` feature=`{row.get('feature')}` status=`{row.get('status')}` "
                f"token_pass=`{row.get('token_pass')}` request_type=`{row.get('request_type')}` "
                f"transcription=`{(row.get('transcription') or '')[:100]}`"
            )
    else:
        lines.append("- None")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Teach full DB, then run all audio query cases.")
    parser.add_argument("--dataset", default="src/visual_memory/tests/input_data/voice_eval_dataset.json")
    parser.add_argument("--teach-csv", default="benchmarks/dataset.csv")
    parser.add_argument("--images-dir", default="benchmarks/images")
    parser.add_argument("--base-url", default=os.environ.get("TEST_BASE_URL", _DEFAULT_BASE_URL).strip())
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "").strip())
    parser.add_argument("--no-wipe", action="store_true")
    parser.add_argument("--teach-limit", type=int, default=0, help="For smoke runs; 0 means all teach rows.")
    parser.add_argument("--max-cases", type=int, default=0, help="For smoke runs; 0 means all audio cases.")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output", default="logs/full_db_audio_query_report.json")
    parser.add_argument("--markdown", default="logs/full_db_audio_query_report.md")
    args = parser.parse_args()

    if not args.base_url:
        print("ERROR: base URL required (set --base-url or TEST_BASE_URL)")
        return 2

    preflight_status, preflight_message = _probe_api_key(args.base_url, args.api_key, timeout=10.0)
    if preflight_status == "skip":
        print(f"[SKIP] full_db_audio_query_eval: {preflight_message}")
        return 0
    if preflight_status == "warn":
        print(f"WARN: full_db_audio_query_eval: {preflight_message}")
    if preflight_status == "error":
        print(f"ERROR: full_db_audio_query_eval: {preflight_message}")
        return 2

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = _REPO_ROOT / dataset_path
    teach_csv = Path(args.teach_csv)
    if not teach_csv.is_absolute():
        teach_csv = _REPO_ROOT / teach_csv
    images_dir = Path(args.images_dir)
    if not images_dir.is_absolute():
        images_dir = _REPO_ROOT / images_dir

    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        return 2
    if not teach_csv.exists():
        print(f"ERROR: teach csv not found: {teach_csv}")
        return 2
    if not images_dir.exists():
        print(f"ERROR: images dir not found: {images_dir}")
        return 2

    if not args.no_wipe:
        status, body = _post_json(args.base_url, args.api_key, "/debug/wipe", {"confirm": True, "target": "all"}, args.timeout)
        if status != 200:
            print(f"WARN: debug wipe failed status={status} body={body}")

    teach_input_rows = _load_teach_rows(teach_csv, teach_limit=args.teach_limit)
    teach_rows: list[dict] = []
    for row in teach_input_rows:
        image_path = _resolve_image_path(images_dir, row["image"])
        if image_path is None:
            teach_rows.append({
                "image": row["image"],
                "label": row["label"],
                "status": 404,
                "success": False,
                "error": "image_not_found",
            })
            continue

        status, body = _post_multipart_remember(
            args.base_url,
            args.api_key,
            image_path.read_bytes(),
            image_path.name,
            row["label"],
            timeout=args.timeout,
        )
        success = bool(status == 200 and isinstance(body, dict) and body.get("success"))
        teach_rows.append(
            {
                "image": row["image"],
                "label": row["label"],
                "resolved_image_path": str(image_path),
                "status": status,
                "success": success,
                "error": (body or {}).get("error"),
                "message": (body or {}).get("message"),
            }
        )

    all_cases = _load_dataset_cases(dataset_path)
    if args.max_cases > 0:
        all_cases = all_cases[: args.max_cases]

    transcode_count = 0
    query_rows: list[dict] = []
    tmp_audio_dir = Path(tempfile.mkdtemp(prefix="full-db-audio-eval-"))
    try:
        for case in all_cases:
            case_id = str(case.get("id", ""))
            audio_path = resolve_audio_path(case, _REPO_ROOT)
            if audio_path is None or not audio_path.exists():
                query_rows.append(
                    {
                        "id": case_id,
                        "feature": case.get("feature"),
                        "priority": case.get("priority"),
                        "audio_path": None,
                        "status": 404,
                        "token_pass": False,
                        "error": "audio_not_found",
                    }
                )
                continue

            request_audio_path, transcoded = _maybe_transcode_audio(audio_path, tmp_audio_dir)
            if transcoded:
                transcode_count += 1

            status, body = _post_voice_audio(args.base_url, args.api_key, request_audio_path, timeout=args.timeout)
            transcription = ""
            if isinstance(body, dict):
                transcription = (body.get("transcription") or "").strip()
                if not transcription:
                    transcription = ((body.get("transcription_meta") or {}).get("text") or "").strip()
            expected_tokens = list(case.get("expected_text_contains", []))
            token_pass = status == 200 and _contains_all(transcription, expected_tokens)

            result = body.get("result", {}) if isinstance(body, dict) else {}
            query_rows.append(
                {
                    "id": case_id,
                    "feature": case.get("feature"),
                    "priority": case.get("priority"),
                    "audio_path": str(audio_path),
                    "request_audio_path": str(request_audio_path),
                    "transcoded_for_request": transcoded,
                    "status": status,
                    "request_type": body.get("request_type") if isinstance(body, dict) else None,
                    "expected_intent": case.get("expected_intent"),
                    "transcription": transcription,
                    "expected_text_contains": expected_tokens,
                    "token_pass": token_pass,
                    "found": result.get("found") if isinstance(result, dict) else None,
                    "matched_label": result.get("matched_label") if isinstance(result, dict) else None,
                    "error": body.get("error") if isinstance(body, dict) else str(body),
                }
            )
    finally:
        shutil.rmtree(tmp_audio_dir, ignore_errors=True)

    teach_success = [r for r in teach_rows if r.get("success")]
    executed_queries = [r for r in query_rows if r.get("status") != 404]
    status_200 = [r for r in query_rows if r.get("status") == 200]
    token_passed = [r for r in query_rows if r.get("token_pass")]

    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "base_url": args.base_url,
        "dataset": str(dataset_path),
        "teach_csv": str(teach_csv),
        "images_dir": str(images_dir),
        "teach_rows_attempted": len(teach_rows),
        "teach_rows_success": len(teach_success),
        "teach_unique_labels": len({r.get("label") for r in teach_success}),
        "query_total_cases": len(query_rows),
        "query_executed_cases": len(executed_queries),
        "query_status_200_count": len(status_200),
        "query_missing_audio_count": sum(1 for r in query_rows if r.get("status") == 404),
        "query_transcoded_count": transcode_count,
        "query_token_pass_count": len(token_passed),
        "query_token_pass_rate": round(len(token_passed) / len(query_rows), 3) if query_rows else 0.0,
    }

    report = {
        "summary": summary,
        "teach_rows": teach_rows,
        "query_rows": query_rows,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    md_path = Path(args.markdown)
    if not md_path.is_absolute():
        md_path = _REPO_ROOT / md_path
    _write_markdown(md_path, summary, teach_rows, query_rows)

    print(json.dumps(summary))
    print(f"report={out_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
