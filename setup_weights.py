"""
Run once after `pip install -e .` to download the Depth Pro checkpoint.

    python setup_weights.py

Downloads depth_pro.pt (~2GB) from HuggingFace into checkpoints/ at the project root.
Works on Windows, macOS, and Linux — no bash or symlinks required.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "depth_pro.pt"


def main():
    if CHECKPOINT_FILE.exists():
        print(f"Checkpoint already present at {CHECKPOINT_FILE}, skipping download.")
        return

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("Downloading Depth Pro weights (~2GB) from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not found. Run `pip install -e .` first.")
        return

    hf_hub_download(
        repo_id="apple/DepthPro",
        filename="depth_pro.pt",
        local_dir=str(CHECKPOINT_DIR),
    )
    print(f"\nSetup complete. Checkpoint saved to {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
