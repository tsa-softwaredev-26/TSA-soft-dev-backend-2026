"""Targeted tests for item VLM and OCR question intents."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["ENABLE_DEPTH"] = "0"

from visual_memory.api.routes.find import find_bp
from visual_memory.api.routes.ask import ask_bp
from visual_memory.api.routes.item_ask import item_ask_bp, _keyword_intent
from visual_memory.tests.scripts.test_harness import TestRunner, make_embedding, make_test_app
import visual_memory.api.pipelines as _pm
import visual_memory.api.routes.item_ask as _item_ask

_runner = TestRunner("vlm_qa_improvements")


def _seed(db):
    emb = make_embedding(101)
    test_image = str(Path(__file__).resolve().parents[1] / "input_images" / "wallet_1ft_table.jpg")
    db.add_item(
        label="wallet",
        combined_embedding=emb,
        ocr_text="Receipt Total 9.95 Date 03/31/2026",
        image_path=test_image,
        confidence=0.9,
    )
    db.add_sighting(
        label="wallet",
        direction="to your left",
        distance_ft=2.0,
        similarity=0.8,
        room_name="kitchen",
    )
    db.add_item(
        label="keys",
        combined_embedding=make_embedding(102),
        ocr_text="",
        image_path="",
        confidence=0.8,
    )


client, db, cleanup = make_test_app([find_bp, ask_bp, item_ask_bp], seed_fn=_seed)


class _StubVLM:
    def __init__(self, mode: str = "ok"):
        self.mode = mode

    def answer(self, image_path: str, question: str, timeout: float) -> str:
        if self.mode == "timeout":
            raise TimeoutError("timed out")
        if self.mode == "missing":
            raise FileNotFoundError(image_path)
        if self.mode == "error":
            raise RuntimeError("boom")
        return f"stub answer for: {question}"


_ORIG_GET_VLM = _item_ask.get_vlm_pipeline


def _set_vlm(mode: str = "ok"):
    _item_ask.get_vlm_pipeline = lambda: _StubVLM(mode)


def _restore_vlm():
    _item_ask.get_vlm_pipeline = _ORIG_GET_VLM


def test_keyword_intent_question_patterns():
    assert _keyword_intent("what color is this") == "question"
    assert _keyword_intent("how many objects are there") == "question"
    assert _keyword_intent("can you see a logo") == "question"


def test_keyword_intent_ocr_question_patterns():
    assert _keyword_intent("who is written on this receipt") == "ocr_question"
    assert _keyword_intent("what date is written on this document") == "ocr_question"
    assert _keyword_intent("how much is written on the text") == "ocr_question"


def test_question_intent_success():
    _set_vlm("ok")
    resp = client.post("/item/ask", json={"scan_id": "scan-1", "label": "wallet", "query": "what color is this"})
    _restore_vlm()
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("action") == "question"
    assert data.get("method") == "vlm"
    assert "stub answer" in data.get("answer", "")


def test_question_intent_timeout_path():
    _set_vlm("timeout")
    resp = client.post("/item/ask", json={"scan_id": "scan-1", "label": "wallet", "query": "how many items are there"})
    _restore_vlm()
    assert resp.status_code == 504
    data = resp.get_json()
    assert data.get("method") == "vlm_timeout"


def test_question_intent_missing_image_path_in_db():
    _set_vlm("ok")
    resp = client.post("/item/ask", json={"scan_id": "scan-1", "label": "keys", "query": "is there text"})
    _restore_vlm()
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("action") == "question"
    assert data.get("answer") == ""
    assert "image" in data.get("narration", "").lower()


def test_question_intent_missing_image_file_on_disk():
    _set_vlm("missing")
    resp = client.post("/item/ask", json={"scan_id": "scan-1", "label": "wallet", "query": "does it have a logo"})
    _restore_vlm()
    assert resp.status_code == 404
    data = resp.get_json()
    assert data.get("action") == "question"


def test_ocr_question_amount_and_date():
    resp_amount = client.post("/item/ask", json={"scan_id": "scan-1", "label": "wallet", "query": "how much is written on the receipt text"})
    assert resp_amount.status_code == 200
    amount = resp_amount.get_json()
    assert amount.get("action") == "ocr_question"
    assert "9.95" in amount.get("answer", "")

    resp_date = client.post("/item/ask", json={"scan_id": "scan-1", "label": "wallet", "query": "what date is written on this receipt"})
    assert resp_date.status_code == 200
    date = resp_date.get_json()
    assert date.get("action") == "ocr_question"
    assert date.get("answer") == "03/31/2026"


def test_ocr_question_no_text():
    resp = client.post("/item/ask", json={"scan_id": "scan-1", "label": "keys", "query": "when was this document written"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("action") == "ocr_question"
    assert data.get("ocr_text") == ""


def test_vlm_pipeline_answer_timeout_and_errors():
    from visual_memory.engine.vlm.pipeline import VLMPipeline

    pipeline = VLMPipeline()
    image_path = Path(__file__).resolve().parents[1] / "input_images" / "wallet_1ft_table.jpg"
    assert image_path.exists()

    def _slow(_p: Path, _q: str) -> str:
        import time as _time
        _time.sleep(0.2)
        return "late"

    orig = pipeline._infer
    pipeline._infer = _slow
    try:
        raised = False
        try:
            pipeline.answer(str(image_path), "what color", timeout=0.01)
        except TimeoutError:
            raised = True
        assert raised is True
    finally:
        pipeline._infer = orig

    raised_empty = False
    pipeline._infer = lambda _p, _q: ""
    try:
        try:
            pipeline.answer(str(image_path), "what color", timeout=1.0)
        except RuntimeError:
            raised_empty = True
        assert raised_empty is True
    finally:
        pipeline._infer = orig


for name, fn in [
    ("vlmqa:intent_question_pattern", test_keyword_intent_question_patterns),
    ("vlmqa:intent_ocr_question_pattern", test_keyword_intent_ocr_question_patterns),
    ("vlmqa:question_success", test_question_intent_success),
    ("vlmqa:question_timeout", test_question_intent_timeout_path),
    ("vlmqa:question_missing_image_path", test_question_intent_missing_image_path_in_db),
    ("vlmqa:question_missing_image_file", test_question_intent_missing_image_file_on_disk),
    ("vlmqa:ocr_question_amount_date", test_ocr_question_amount_and_date),
    ("vlmqa:ocr_question_no_text", test_ocr_question_no_text),
    ("vlmqa:pipeline_answer_timeout_errors", test_vlm_pipeline_answer_timeout_and_errors),
]:
    _runner.run(name, fn)

cleanup()
_pm._database = None
_pm._scan_pipeline = None
_pm._remember_pipeline = None
_pm._feedback_store = None
_pm._user_settings = None
sys.exit(_runner.summary())
