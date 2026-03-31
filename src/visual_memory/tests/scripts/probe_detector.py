"""
probe_detector.py
Clean GroundingDINO / YOLOE probe tool.

Modes:
  Compact:
    python -m visual_memory.tests.scripts.probe_detector img.jpg --prompt "wallet"

  Batch:
    python -m visual_memory.tests.scripts.probe_detector --batch wallet

  Open visualization:
    add --open
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
import pillow_heif
pillow_heif.register_heif_opener()

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from visual_memory.config import Settings
from visual_memory.utils import load_image

# PATHS (cwd-independent)
SCRIPTS_DIR = Path(__file__).resolve().parent   # tests/scripts/
TESTS_DIR = SCRIPTS_DIR.parent                  # tests/
INPUT_DIR = TESTS_DIR / "input_images"
PROJECT_ROOT = TESTS_DIR.parents[2]             # project root
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# YOLOE
def run_yoloe(image_path: Path, settings: Settings):
    model_path = Path(__file__).resolve().parents[2] / "engine" / "object_detection" / "yoloe-26l-seg-pf.pt"

    if not model_path.exists():
        print(f"ERROR: YOLOE model not found at {model_path}", file=sys.stderr)
        return 0.0, False

    from ultralytics import YOLOE

    model = YOLOE(model_path)
    image = load_image(str(image_path))
    results = model.predict(image, conf=0.01, verbose=False)

    boxes = results[0].boxes
    if len(boxes) == 0:
        return 0.0, False

    scores = sorted(boxes.conf.tolist(), reverse=True)
    max_score = scores[0]
    detected = max_score >= settings.yoloe_confidence
    return max_score, detected


# GroundingDINO
def run_grounding_dino(
    image_path: Path,
    prompt: str,
    settings: Settings,
    processor,
    model,
    device: str,
    open_image: bool = False,
):
    image = load_image(str(image_path))
    inputs = processor(images=image, text=[[prompt]], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=settings.box_threshold,
        text_threshold=settings.text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results.get("boxes", [])
    scores = results.get("scores", [])

    detected = len(boxes) > 0
    max_score = max(scores).item() if detected else 0.0

    print(f"\nImage: {image_path.name}")
    print(f"Prompt: '{prompt}'")
    print(f"Box threshold: {settings.box_threshold}")
    print(f"Detected: {detected}")
    print(f"Max score: {max_score:.4f}")

    if open_image and detected:
        _open_visualization(image, boxes, scores, settings.box_threshold)

    return max_score, detected


# Visualization (bbx)
def _open_visualization(image, boxes, scores, threshold):
    import subprocess
    import tempfile
    from PIL import ImageDraw

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        color = "#00ff00" if score >= threshold else "#ff8800"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False, dir="/tmp"
    ) as f:
        tmp_path = Path(f.name)

    img.save(tmp_path)
    subprocess.run(["open", str(tmp_path)])


# Batch mode
BATCH_PRESETS = {
    obj: [
        f"{obj}_{dist}_{bg}.jpg"
        for dist in ("1ft", "3ft", "6ft")
        for bg in ("floor", "table")
    ]
    for obj in ("wallet", "airpods", "mouse")
}

BATCH_PRESETS["all"] = (
    BATCH_PRESETS["wallet"]
    + BATCH_PRESETS["airpods"]
    + BATCH_PRESETS["mouse"]
)


def batch_run(preset: str, prompt: str, settings: Settings, processor, model, device):
    print(f"\nBatch preset: {preset}")
    print("=" * 50)

    for filename in BATCH_PRESETS[preset]:
        path = INPUT_DIR / filename
        if not path.exists():
            print(f"Missing: {filename}")
            continue

        _, detected = run_grounding_dino(
            path, prompt, settings, processor, model, device
        )

        print(f"{filename:30} -> {detected}")

    print("=" * 50)


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--detector", choices=["grounding_dino", "yoloe"],
                        default="grounding_dino")
    parser.add_argument("--batch", choices=["wallet", "airpods", "mouse", "all"])
    parser.add_argument("--open", action="store_true")

    args = parser.parse_args()

    settings = Settings()
    device = get_device()

    if args.batch:
        if not args.prompt:
            args.prompt = args.batch

        processor = AutoProcessor.from_pretrained(
            settings.grounding_dino_model
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            settings.grounding_dino_model
        ).to(device)
        model.eval()

        batch_run(args.batch, args.prompt, settings, processor, model, device)
        return

    if not args.image:
        parser.error("Provide image or --batch")

    if args.detector == "yoloe":
        score, detected = run_yoloe(args.image, settings)
        print(f"YOLOE -> score={score:.4f}, detected={detected}")
        return

    if not args.prompt:
        parser.error("--prompt required for GroundingDINO")

    processor = AutoProcessor.from_pretrained(
        settings.grounding_dino_model
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        settings.grounding_dino_model
    ).to(device)
    model.eval()

    run_grounding_dino(
        args.image,
        args.prompt,
        settings,
        processor,
        model,
        device,
        open_image=args.open,
    )


if __name__ == "__main__":
    main()