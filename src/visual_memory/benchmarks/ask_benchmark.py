"""Ask benchmark runner built on top of the benchmark dataset.

Part 1:
- Build a benchmark memory DB from the normal benchmark teach images
  (`*_1ft_bright_clean.*`), using detected crops and combined embeddings.

Part 2:
- Run 120 realistic natural-language ask queries directly through /ask logic
  (`process_ask_query`) without any Whisper/audio layer.
- Write a CSV with query and result metadata for manual review.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from visual_memory.api import pipelines as api_pipelines
from visual_memory.api.routes.ask import process_ask_query
from visual_memory.config import Settings
from visual_memory.database import DatabaseStore
from visual_memory.engine.embedding import make_combined_embedding
from visual_memory.engine.model_registry import registry
from visual_memory.engine.text_recognition import TextRecognizer
from visual_memory.pipelines.scan_mode.pipeline import ScanPipeline
from visual_memory.utils.image_utils import crop_object, load_image
from visual_memory.utils.quality_utils import estimate_text_likelihood
from visual_memory.utils.voice_state_context import build_state_contract

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"
_SECOND_PASS_TEMPLATES = [
    "a {prompt}",
    "{prompt} object",
    "close up of a {prompt}",
]


class _LightScanPipeline:
    """Minimal scan pipeline shim for ask benchmark when YOLOE assets are unavailable."""

    def __init__(self) -> None:
        self.text_embedder = registry.text_embedder

    def reload_database(self) -> None:
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask benchmark over benchmark memory DB")
    p.add_argument("--dataset", type=Path, default=_BENCHMARKS_DIR / "dataset.csv")
    p.add_argument("--images", type=Path, default=_BENCHMARKS_DIR / "images")
    p.add_argument("--db-path", type=Path, default=_BENCHMARKS_DIR / "ask_benchmark_memory.db")
    p.add_argument("--output", type=Path, default=_BENCHMARKS_DIR / "ask_results.csv")
    p.add_argument(
        "--query-set",
        choices=["core_120", "ux_blind_120", "both_240"],
        default="core_120",
        help="Which query suite to run",
    )
    return p.parse_args()


def _load_dataset(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        filtered = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.DictReader(filtered)
        for row in reader:
            rows.append(
                {
                    "image": row["image"],
                    "label": row["label"],
                    "distance_ft": float(row["ground_truth_distance_ft"]),
                    "dino_prompt": row["dino_prompt"],
                }
            )
    return rows


def _direction_from_box(box: list[float], image_width: int) -> str:
    cx = (box[0] + box[2]) / 2.0
    nx = (cx / max(float(image_width), 1.0)) * 2.0 - 1.0
    if nx < -0.5:
        return "to your left"
    if nx < -0.15:
        return "slightly left"
    if nx < 0.15:
        return "ahead"
    if nx < 0.5:
        return "slightly right"
    return "to your right"


def _detect_with_fallback(image, prompt: str):
    detection = registry.gdino_detector.detect(image, prompt)
    if detection is not None:
        return detection, None
    for template in _SECOND_PASS_TEMPLATES:
        alt = template.format(prompt=prompt)
        detection = registry.gdino_detector.detect(image, alt)
        if detection is not None:
            return detection, alt
    return None, None


def _teach_benchmark_db(
    rows: list[dict],
    images_dir: Path,
    db: DatabaseStore,
    settings: Settings,
) -> list[dict]:
    teach_rows = [r for r in rows if "_1ft_bright_clean." in r["image"]]
    text_embedder = registry.text_embedder if settings.enable_ocr else None
    ocr_client = TextRecognizer() if settings.enable_ocr else None
    taught = []
    now = time.time()

    for idx, row in enumerate(teach_rows):
        image_path = images_dir / row["image"]
        if not image_path.exists():
            taught.append(
                {
                    "label": row["label"],
                    "image": row["image"],
                    "success": False,
                    "reason": "image_not_found",
                    "second_pass_prompt": None,
                }
            )
            continue

        image = load_image(str(image_path))
        detection, second_pass_prompt = _detect_with_fallback(image, row["dino_prompt"])
        if detection is None:
            taught.append(
                {
                    "label": row["label"],
                    "image": row["image"],
                    "success": False,
                    "reason": "no_detection",
                    "second_pass_prompt": None,
                }
            )
            continue

        crop = crop_object(image, detection["box"])
        image_embedding = registry.img_embedder.embed(crop)

        text_likelihood = estimate_text_likelihood(crop)
        ocr_ran = (
            ocr_client is not None
            and text_likelihood >= settings.ocr_text_likelihood_threshold
        )

        ocr_text = ""
        ocr_confidence = 0.0
        text_embedding = None
        if ocr_ran:
            ocr_result = ocr_client.recognize(crop)
            ocr_text = str(ocr_result.get("text", "") or "").strip()
            ocr_confidence = float(ocr_result.get("confidence", 0.0) or 0.0)
            if ocr_text and text_embedder is not None:
                text_embedding = text_embedder.embed_text(ocr_text)

        combined_embedding = make_combined_embedding(image_embedding, text_embedding)

        label = row["label"]
        label_embedding = text_embedder.embed_text(label) if text_embedder is not None else None

        db.delete_items_by_label(label)
        db.add_item(
            label=label,
            combined_embedding=combined_embedding,
            ocr_text=ocr_text,
            image_path=str(image_path),
            confidence=float(detection["score"]),
            timestamp=now + idx,
            label_embedding=label_embedding,
            ocr_embedding=text_embedding,
            visual_attributes={},
        )
        db.add_sighting(
            label=label,
            direction=_direction_from_box(detection["box"], image.size[0]),
            distance_ft=row["distance_ft"],
            similarity=1.0,
            crop_path=str(image_path),
            room_name="benchmark_room",
            timestamp=now + idx,
        )
        taught.append(
            {
                "label": label,
                "image": row["image"],
                "success": True,
                "reason": "",
                "second_pass_prompt": second_pass_prompt,
            }
        )

    return taught


def _build_queries_core_120() -> list[dict]:
    queries = {
        "wallet_zipper": [
            "Where is my zip up wallet?",
            "Can you find my zipper wallet for me?",
            "uh where did I leave my zip wallet",
            "What color is my zip up wallet?",
            "Can you describe what my zip up wallet looks like?",
            "Where was the zipper wallet seen last?",
            "I need my zip wallet, where is it right now?",
            "Is my zipper wallet to the left or right?",
            "Could you tell me where my zip up wallet is please?",
            "um I cannot find my zip wallet, can you help",
            "Which way should I turn to get to my zipper wallet?",
            "Tell me the last place you saw my zip up wallet.",
        ],
        "wallet_trifold": [
            "Where is my trifold wallet?",
            "Find my trifold wallet please.",
            "uh where did I put the trifold wallet",
            "Can you describe my trifold wallet?",
            "What color is the trifold wallet?",
            "Last time you saw my trifold wallet, where was it?",
            "Is my trifold wallet ahead of me?",
            "Can you tell me if my trifold wallet is on my left or right?",
            "I need the trifold wallet now, where should I look?",
            "um can you help me locate my trifold wallet",
            "Where did I leave that trifold wallet earlier?",
            "Please find the trifold wallet for me.",
        ],
        "receipt_salon": [
            "Where did I leave my salon receipt?",
            "Read the text on my receipt from the salon.",
            "uh can you find the receipt about the salon appointment",
            "Do you see the salon receipt anywhere?",
            "I need the salon receipt, where is it?",
            "Tell me where my salon receipt was last seen.",
            "Can you read what the salon receipt says?",
            "Where is that paper receipt from the salon?",
            "um where did my salon receipt go",
            "Please help me locate the receipt for the salon.",
            "Could you find my salon paper receipt?",
            "What does my salon receipt say?",
        ],
        "receipt_eye_doctor": [
            "Where is my eye doctor receipt?",
            "Can you read the receipt from my eye doctor visit?",
            "uh I need the eye doctor receipt, where did I put it",
            "Find the eye doctor paper receipt.",
            "Tell me what the eye doctor receipt says.",
            "Where was my eye doctor receipt seen last?",
            "Do you still have my receipt from the eye doctor?",
            "um can you locate the eye doctor receipt for me",
            "Read the text on the eye doctor receipt.",
            "Where did I leave that eye clinic receipt?",
            "Can you help me find the eye doctor paper?",
            "What does the eye doctor receipt say?",
        ],
        "keys_house": [
            "Where are my house keys?",
            "Find my house keys please.",
            "uh where did I put my house keys",
            "Are my house keys to the left or right?",
            "Tell me where my house keys were last seen.",
            "I need my house keys now, where are they?",
            "Can you locate the keys for my house?",
            "um help me find my house keys",
            "Which direction are my house keys from me?",
            "Do you see my house keys?",
            "Where did I leave my house key set?",
            "Please guide me to my house keys.",
        ],
        "keys_safe": [
            "Where are my safe keys?",
            "Find the keys for my safe.",
            "uh I cannot find my safe keys",
            "Tell me where I left the safe keys.",
            "Are my safe keys ahead of me?",
            "Which side are my safe keys on?",
            "Can you help locate the safe key set?",
            "um where did my safe keys go",
            "Do you see the keys to my safe?",
            "I need the safe keys right now, where are they?",
            "Last place my safe keys were seen?",
            "Please find my safe keys for me.",
        ],
        "magnesium_bottle": [
            "Where is my magnesium bottle?",
            "Find my magnesium pill bottle.",
            "uh where did I leave the magnesium bottle",
            "Is my magnesium bottle nearby?",
            "Can you describe where my magnesium bottle is?",
            "Tell me the direction to my magnesium bottle.",
            "I need my magnesium pills, where is that bottle?",
            "um can you locate my magnesium bottle",
            "Where was the magnesium bottle last seen?",
            "Please help me find the magnesium pill bottle.",
            "Do you see my magnesium supplement bottle?",
            "Where is the bottle of magnesium I taught you?",
        ],
        "water_bottle": [
            "Where is my water bottle?",
            "Find my water bottle please.",
            "uh where did I put my water bottle",
            "Is my water bottle to the left or right?",
            "Can you tell me where my water bottle is?",
            "I need a drink, where is my water bottle?",
            "um help me locate the water bottle",
            "Where was my water bottle last seen?",
            "Please find my water bottle.",
            "Do you see the bottle of water?",
            "Which way should I move to reach my water bottle?",
            "Can you locate my water bottle for me?",
        ],
        "sunglasses_sun": [
            "Where are my sunglasses?",
            "Find my sunglasses please.",
            "uh where did I leave my sunglasses",
            "Can you tell me where my sun glasses are?",
            "Are my sunglasses on my left or right?",
            "Where were my sunglasses last seen?",
            "um can you help me find my sunglasses",
            "I need my sunglasses now, where are they?",
            "Please locate my sunglasses.",
            "Do you see my sun shades?",
            "Which direction are my sunglasses from me?",
            "Tell me where I put my sunglasses.",
        ],
        "glasses_prescription": [
            "Where are my prescription glasses?",
            "Find my glasses please.",
            "uh where did I leave my prescription glasses",
            "Can you locate my eyeglasses?",
            "Are my glasses ahead of me?",
            "Tell me where my prescription glasses were last seen.",
            "um help me find my glasses",
            "I need my prescription glasses now, where are they?",
            "Please find my eyeglasses.",
            "Do you see my regular glasses?",
            "Which direction should I turn for my glasses?",
            "Can you tell me where my prescription glasses are?",
        ],
    }
    out = []
    for expected_label, items in queries.items():
        for text in items:
            out.append({"query_set": "core_120", "query": text, "expected_label": expected_label})
    if len(out) != 120:
        raise ValueError(f"query set size mismatch: expected 120, got {len(out)}")
    return out


def _build_queries_ux_blind_120() -> list[dict]:
    queries = {
        "wallet_zipper": [
            "where is my zipper wallet",
            "can you find the zipper wallet for me",
            "is the wallet with the zipper near me",
            "where did i leave my zipper wallet",
            "um where is my zip wallet right now",
            "is the zipper wallet to my left or right",
            "can you point me toward the zipper wallet",
            "how far ahead is the zipper wallet",
            "do you see the zipper wallet on the table",
            "help me locate my zipper wallet please",
            "what does my zipper wallet look like",
            "describe the zipper wallet quickly",
        ],
        "wallet_trifold": [
            "where is my trifold wallet",
            "can you help me find the trifold wallet",
            "did i put the trifold wallet nearby",
            "um is the trifold wallet to my right",
            "is my trifold wallet ahead of me",
            "point me to the trifold wallet",
            "where did the trifold wallet go",
            "can you locate the trifold wallet now",
            "describe my trifold wallet",
            "what color is the trifold wallet",
            "does the trifold wallet look open or closed",
            "is the trifold wallet next to anything",
        ],
        "receipt_salon": [
            "where is the salon receipt",
            "can you find my salon receipt",
            "read the salon receipt to me",
            "what does the salon receipt say",
            "uh can you read the total on the salon receipt",
            "tell me the date on the salon receipt",
            "is the salon receipt to my left",
            "is the salon receipt in front of me",
            "can you read the line items on the salon receipt",
            "how much did i tip on the salon receipt",
            "please read the salon receipt from top to bottom",
            "can you check if the salon receipt mentions tax",
        ],
        "receipt_eye_doctor": [
            "where is the eye doctor receipt",
            "can you find my eye doctor receipt",
            "read the eye doctor receipt out loud",
            "what does the eye doctor receipt say",
            "tell me the total on the eye doctor receipt",
            "can you read the date on the eye doctor receipt",
            "is the eye doctor receipt on my right",
            "is the eye doctor receipt ahead of me",
            "read the charges on the eye doctor receipt",
            "does the eye doctor receipt mention insurance",
            "please read the eye doctor receipt line by line",
            "uh what is the amount due on the eye doctor receipt",
        ],
        "keys_house": [
            "where are my house keys",
            "can you help me find the house keys",
            "did i leave the house keys nearby",
            "are the house keys to my left",
            "are the house keys to my right",
            "are the house keys in front of me",
            "point me toward the house keys",
            "how far away are the house keys",
            "do you see the house keys on the counter",
            "um can you locate my house keys now",
            "are the house keys next to my wallet",
            "where did i set down the house keys",
        ],
        "keys_safe": [
            "where are my safe keys",
            "can you find the safe keys for me",
            "are the safe keys close by",
            "are the safe keys on my left side",
            "are the safe keys on my right side",
            "are the safe keys straight ahead",
            "point me to the safe keys please",
            "how far ahead are the safe keys",
            "do you see the safe keys on a shelf",
            "um where did i put the safe keys",
            "are the safe keys near the house keys",
            "can you help me locate the safe keys quickly",
        ],
        "magnesium_bottle": [
            "where is the magnesium bottle",
            "can you find my magnesium bottle",
            "is the magnesium bottle nearby",
            "is the magnesium bottle to my left",
            "is the magnesium bottle to my right",
            "is the magnesium bottle ahead of me",
            "point me toward the magnesium bottle",
            "how far is the magnesium bottle",
            "do you see the magnesium bottle on the table",
            "um can you locate the magnesium bottle now",
            "is the magnesium bottle next to the water bottle",
            "where did i leave the magnesium bottle",
        ],
        "water_bottle": [
            "where is my water bottle",
            "can you help me find the water bottle",
            "is the water bottle close to me",
            "is the water bottle on my left",
            "is the water bottle on my right",
            "is the water bottle straight ahead",
            "point me to the water bottle please",
            "how far away is the water bottle",
            "do you see the water bottle by the chair",
            "um where did i put my water bottle",
            "is the water bottle near the magnesium bottle",
            "can you locate the water bottle quickly",
        ],
        "sunglasses_sun": [
            "where are my sunglasses",
            "can you find the sun sunglasses",
            "are the sunglasses to my left",
            "are the sunglasses to my right",
            "are the sunglasses in front of me",
            "point me toward the sunglasses",
            "how far ahead are my sunglasses",
            "do you see the sunglasses on the desk",
            "what do my sunglasses look like",
            "describe the sunglasses for me",
            "um are the sunglasses dark tinted",
            "can you help me locate my sunglasses now",
        ],
        "glasses_prescription": [
            "where are my prescription glasses",
            "can you find my prescription glasses",
            "are the prescription glasses to my left",
            "are the prescription glasses to my right",
            "are the prescription glasses ahead of me",
            "point me toward the prescription glasses",
            "how far away are the prescription glasses",
            "do you see the prescription glasses on the nightstand",
            "what do my prescription glasses look like",
            "describe the prescription glasses please",
            "uh are the prescription glasses full frame",
            "can you locate my prescription glasses now",
        ],
    }
    out = []
    for expected_label, items in queries.items():
        for text in items:
            out.append({"query_set": "ux_blind_120", "query": text, "expected_label": expected_label})
    if len(out) != 120:
        raise ValueError(f"query set size mismatch: expected 120, got {len(out)}")
    return out


def _build_queries(query_set: str) -> list[dict]:
    if query_set == "core_120":
        return _build_queries_core_120()
    if query_set == "ux_blind_120":
        return _build_queries_ux_blind_120()
    if query_set == "both_240":
        return _build_queries_core_120() + _build_queries_ux_blind_120()
    raise ValueError(f"unsupported query_set: {query_set}")


def _configure_benchmark_singletons(db_path: Path) -> tuple[DatabaseStore, list[str]]:
    settings = api_pipelines.get_settings()
    settings.db_path = str(db_path)
    settings.llm_query_fallback_enabled = True
    db = DatabaseStore(db_path)
    api_pipelines._database = db
    try:
        api_pipelines._scan_pipeline = ScanPipeline(focal_length_px=0, db_path=db_path)
        api_pipelines._scan_pipeline.reload_database()
    except FileNotFoundError:
        api_pipelines._scan_pipeline = _LightScanPipeline()
    known_labels = db.get_known_labels()
    return db, known_labels


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.db_path.exists():
        args.db_path.unlink()

    settings = api_pipelines.get_settings()
    rows = _load_dataset(args.dataset)
    db, _ = _configure_benchmark_singletons(args.db_path)
    teach_summary = _teach_benchmark_db(rows, args.images, db, settings)
    api_pipelines._scan_pipeline.reload_database()
    known_labels = db.get_known_labels()

    state_contract = build_state_contract(
        mode="idle",
        context={"known_labels": known_labels},
    )
    queries = _build_queries(args.query_set)

    fieldnames = [
        "query_set",
        "query_id",
        "query",
        "expected_label",
        "status_code",
        "latency_ms",
        "found",
        "matched_label",
        "matched_by",
        "match_expected",
        "search_term",
        "ollama_used",
        "strategies_tried",
        "narration",
        "error",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(queries, start=1):
            t0 = time.perf_counter()
            result, status = process_ask_query(item["query"], state_context=state_contract)
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            matched_label = str(result.get("matched_label", "") or "")
            writer.writerow(
                {
                    "query_set": item["query_set"],
                    "query_id": idx,
                    "query": item["query"],
                    "expected_label": item["expected_label"],
                    "status_code": status,
                    "latency_ms": lat_ms,
                    "found": bool(result.get("found", False)),
                    "matched_label": matched_label,
                    "matched_by": str(result.get("matched_by", "") or ""),
                    "match_expected": matched_label == item["expected_label"],
                    "search_term": str(result.get("search_term", "") or ""),
                    "ollama_used": bool(result.get("ollama_used", False)),
                    "strategies_tried": json.dumps(result.get("strategies_tried", [])),
                    "narration": str(result.get("narration", "") or ""),
                    "error": str(result.get("error", "") or ""),
                }
            )

    taught_ok = sum(1 for r in teach_summary if r["success"])
    print(f"Taught items: {taught_ok}/{len(teach_summary)}")
    print(f"Query set: {args.query_set}, rows: {len(queries)}")
    print(f"Known labels: {', '.join(known_labels)}")
    print(f"Ask benchmark CSV written: {args.output}")


if __name__ == "__main__":
    main()
