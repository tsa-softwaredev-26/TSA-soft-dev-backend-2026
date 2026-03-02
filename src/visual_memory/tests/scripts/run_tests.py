"""
Integration test runner for the visual_memory pipelines.

Flags (env vars, set before running):
    DEPTH=1    load and test DepthEstimator (default off - loads 2GB model)
    VERBOSE=1  show OCR text diff and log entries

Run:
    python -m visual_memory.tests.scripts.run_tests
    DEPTH=1 python -m visual_memory.tests.scripts.run_tests
    VERBOSE=1 python -m visual_memory.tests.scripts.run_tests

Tests:
    1. RememberPipeline: wallet_1ft_table.jpg + "small rectangular wallet"
    2. ScanPipeline: wallet_3ft_table.jpg
    3. DepthEstimator: reused from scan (skipped unless DEPTH=1)
    4. TextRecognition: magnesium.heic - TextRecognizer + CLIPTextEmbedder
"""

from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path

# Set pipeline flags before any project imports so Settings instances read them
DEPTH   = os.environ.get("DEPTH",   "0") == "1"
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

if not DEPTH:
    os.environ["ENABLE_DEPTH"] = "0"

# Suppress model-loading noise
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

import pillow_heif
pillow_heif.register_heif_opener()

from visual_memory.utils import load_image, get_logger, log_mark, tail_logs
from visual_memory.config import Settings

_log = get_logger(__name__)

# Paths (cwd-independent)
SCRIPTS_DIR = Path(__file__).resolve().parent   # tests/scripts/
TESTS_DIR   = SCRIPTS_DIR.parent                # tests/
INPUT_DIR   = TESTS_DIR / "input_images"
DEMO_DB     = TESTS_DIR / "demo_database"
TEXT_DEMO   = TESTS_DIR / "text_demo"

WALLET_1FT = INPUT_DIR / "wallet_1ft_table.jpg"
WALLET_3FT = INPUT_DIR / "wallet_3ft_table.jpg"
MAGNESIUM  = INPUT_DIR / "magnesium.heic"

FOCAL_PX = 3094.0

_G = "\033[32m"
_R = "\033[31m"
_B = "\033[1m"
_Y = "\033[33m"
_X = "\033[0m"

_results: list[tuple[str, bool, str]] = []


def _load_ground_truth(image_stem: str) -> str:
    gt_file = TEXT_DEMO / "ground_truth" / f"{image_stem}.txt"
    return gt_file.read_text().strip() if gt_file.exists() else None


def _show_text_diff(recognized: str, ground_truth: str) -> None:
    if not ground_truth:
        return
    print(f"     {_Y}Ground Truth:{_X} {ground_truth[:100]}{'...' if len(ground_truth) > 100 else ''}")
    print(f"     {_Y}Recognized  :{_X} {recognized[:100]}{'...' if len(recognized) > 100 else ''}")
    if recognized.lower() == ground_truth.lower():
        print(f"     {_G}Match: EXACT{_X}")
    else:
        gt_words  = set(ground_truth.lower().split())
        rec_words = set(recognized.lower().split())
        overlap = len(gt_words & rec_words) / len(gt_words | rec_words) if gt_words or rec_words else 0
        print(f"     {_Y}Match: {overlap*100:.1f}% word overlap{_X}")


def _pass(tag: str, detail: str) -> None:
    _results.append((tag, True, detail))
    print(f"  {_G}PASS{_X}  {detail}")


def _fail(tag: str, detail: str) -> None:
    _results.append((tag, False, detail))
    print(f"  {_R}FAIL{_X}  {detail}")


def _skip(tag: str, detail: str) -> None:
    print(f"  {_Y}SKIP{_X}  {detail}")


def _section(title: str) -> None:
    print(f"\n{_B}{title}{_X}")


# Test 1 - RememberPipeline
_section("[1] remember_mode - wallet_1ft_table.jpg")

_mark1 = log_mark()
from visual_memory.pipelines.remember_mode.pipeline import RememberPipeline

remember = RememberPipeline()
r1 = remember.run(image_path=WALLET_1FT, prompt="small rectangular wallet")

print("     json:", json.dumps(r1))

has_box = bool(r1.get("result") and "box" in r1["result"])
if r1["success"] and has_box:
    res = r1["result"]
    _pass("remember:detect", f"label={res['label']}  conf={res['confidence']:.3f}")
else:
    _fail("remember:detect", f"success={r1['success']}  result={r1['result']}")

if VERBOSE:
    for _e in tail_logs(since_line=_mark1): print(f"       {json.dumps(_e)}")


# Test 2 - ScanPipeline
_section("[2] scan_mode - wallet_3ft_table.jpg")

_mark2 = log_mark()
from visual_memory.pipelines.scan_mode.pipeline import ScanPipeline

try:
    scan = ScanPipeline(database_dir=DEMO_DB, focal_length_px=FOCAL_PX)
    r2 = scan.run(load_image(str(WALLET_3FT)))
    print("     json:", json.dumps(r2))

    if r2["count"] >= 1:
        _pass("scan:match", f"count={r2['count']}")
    else:
        _fail("scan:match", "count=0")

except Exception as exc:
    _fail("scan:match", str(exc))

if VERBOSE:
    for _e in tail_logs(since_line=_mark2): print(f"       {json.dumps(_e)}")


# Test 3 - DepthEstimator
_section("[3] estimator")

_mark3 = log_mark()
if not DEPTH:
    _skip("estimator:load", "DEPTH=0 (set DEPTH=1 to enable)")
else:
    try:
        estimator = scan.estimator
        if estimator is not None:
            _pass("estimator:load", "reused from scan")
        else:
            _fail("estimator:load", "scan.estimator is None")
    except Exception as exc:
        _fail("estimator:load", str(exc))

if VERBOSE:
    for _e in tail_logs(since_line=_mark3): print(f"       {json.dumps(_e)}")


# Test 4 - Text Recognition + Embedding
_section("[4] text_recognition + embedding - magnesium.heic")

_mark4 = log_mark()
if scan.text_recognizer is None:
    _skip("text_recognition:recognizer", "ENABLE_OCR=0")
    _skip("text_recognition:embedder",   "ENABLE_OCR=0")
else:
    try:
        recognizer = scan.text_recognizer
        embedder   = scan.text_embedder

        r4_img = load_image(str(MAGNESIUM))
        ocr = recognizer.recognize(r4_img)

        ocr_display = {"text": ocr["text"][:80], "confidence": round(ocr["confidence"], 3), "segments": len(ocr["segments"])}
        print("     ocr:", json.dumps(ocr_display))

        if isinstance(ocr, dict) and "text" in ocr and "confidence" in ocr and "segments" in ocr:
            _pass("text_recognition:recognizer", f"engine=paddle  segments={len(ocr['segments'])}  conf={ocr['confidence']:.3f}")
        else:
            _fail("text_recognition:recognizer", "unexpected result structure")

        if VERBOSE:
            gt = _load_ground_truth("magnesium")
            if gt:
                _show_text_diff(ocr["text"], gt)

        sample_text = ocr["text"] if ocr["text"] else "test document"
        emb = embedder.embed_text(sample_text)
        if emb.shape == (1, 512):
            _pass("text_recognition:embedder", f"shape={list(emb.shape)}")
        else:
            _fail("text_recognition:embedder", f"unexpected shape={list(emb.shape)}")

    except Exception as exc:
        _fail("text_recognition:recognizer", str(exc))
        _fail("text_recognition:embedder", "skipped due to earlier error")

if VERBOSE:
    for _e in tail_logs(since_line=_mark4): print(f"       {json.dumps(_e)}")


# Summary
n_pass = sum(1 for _, ok, _ in _results if ok)
n_fail = len(_results) - n_pass

print()
print(f"{_B}Results:{_X} {n_pass} passed / {n_fail} failed / {len(_results)} total")

for tag, ok, detail in _results:
    icon = f"{_G}+{_X}" if ok else f"{_R}x{_X}"
    print(f"  {icon}  {tag:<30}  {detail}")

sys.exit(0 if n_fail == 0 else 1)
