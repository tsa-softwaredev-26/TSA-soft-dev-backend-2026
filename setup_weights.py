"""
Run once after `pip install -e .` to download all model weights.

    python setup_weights.py

Downloads:
  - depth_pro.pt (~2GB)           -> checkpoints/
  - yoloe-26l-seg-pf.pt (~80MB)   -> src/visual_memory/engine/object_detection/
  - CLIP text encoder (~180MB)     -> HuggingFace cache
  - DINOv3 ViT-S/16 (~1.2GB)      -> HuggingFace cache  [requires HF login]
  - GroundingDINO base (~900MB)    -> HuggingFace cache  [requires HF login]

Gated models require accepting the license on huggingface.co and running:
    hf auth login
"""
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
YOLOE_DIR = PROJECT_ROOT / "src" / "visual_memory" / "engine" / "object_detection"

DEPTH_PRO_REPO = "apple/DepthPro"
DEPTH_PRO_FILE = "depth_pro.pt"

YOLOE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg-pf.pt"
)
YOLOE_FILE = "yoloe-26l-seg-pf.pt"

# Must match settings.py
CLIP_MODEL = "openai/clip-vit-base-patch32"
DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
GDINO_MODEL = "IDEA-Research/grounding-dino-base"


def _check_hf_auth():
    try:
        from huggingface_hub import get_token
        return bool(get_token())
    except Exception:
        return False


def _download_depth_pro():
    dest = CHECKPOINT_DIR / DEPTH_PRO_FILE
    if dest.exists():
        print("  already present, skipping.")
        return
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print("  Downloading (~2GB)...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not found. Run `pip install -e .` first.")
        sys.exit(1)
    hf_hub_download(repo_id=DEPTH_PRO_REPO, filename=DEPTH_PRO_FILE, local_dir=str(CHECKPOINT_DIR))
    print(f"  -> {dest}")


def _download_yoloe():
    dest = YOLOE_DIR / YOLOE_FILE
    if dest.exists():
        print("  already present, skipping.")
        return

    def _progress(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            print(f"\r  {pct}%", end="", flush=True)

    print("  Downloading (~80MB)...")
    urllib.request.urlretrieve(YOLOE_URL, dest, reporthook=_progress)
    print(f"\r  -> {dest}")


def _download_hf_model(repo_id: str, size_hint: str, gated: bool = False):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not found. Run `pip install -e .` first.")
        sys.exit(1)
    print(f"  Downloading ({size_hint})...")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            ignore_patterns=["*.msgpack", "flax_*", "tf_*"],
        )
        print(f"  -> {path}")
    except Exception as exc:
        msg = str(exc)
        if gated and ("401" in msg or "403" in msg or "gated" in msg.lower() or "access" in msg.lower()):
            print(
                f"\nERROR: {repo_id} is a gated model.\n"
                f"  1. Accept the license at https://huggingface.co/{repo_id}\n"
                f"  2. Run: hf auth login\n"
                f"  3. Re-run this script.\n"
            )
        else:
            print(f"ERROR downloading {repo_id}: {exc}")
        sys.exit(1)


def main():
    print("=== Spaitra weight setup ===\n")

    if not _check_hf_auth():
        print(
            "WARNING: No HuggingFace token found. Gated models will fail.\n"
            "  Run `hf auth login` before this script.\n"
        )

    print("[1/5] Depth Pro checkpoint")
    _download_depth_pro()

    print("[2/5] YOLOE detection weights")
    _download_yoloe()

    print("[3/5] CLIP text encoder")
    _download_hf_model(CLIP_MODEL, "~180MB")

    print("[4/5] DINOv3 image embedder  [gated]")
    _download_hf_model(DINOV3_MODEL, "~1.2GB", gated=True)

    print("[5/5] GroundingDINO detector  [gated]")
    _download_hf_model(GDINO_MODEL, "~900MB", gated=True)

    print("\nAll weights downloaded. First startup will be fast.")


if __name__ == "__main__":
    main()
