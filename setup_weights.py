"""
Run once after `pip install -e .` to set up the depth-pro checkpoint symlink.

    python setup.py
"""
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_SRC = PROJECT_ROOT / "depth_demo" / "ml-depth-pro" / "checkpoints"
CHECKPOINT_LINK = PROJECT_ROOT / "checkpoints"

def main():
    # Download depth-pro weights if not already present
    if not CHECKPOINT_SRC.exists():
        print("Downloading Depth Pro weights (~2GB)...")
        script = PROJECT_ROOT / "depth_demo" / "ml-depth-pro" / "get_pretrained_models.sh"
        if not script.exists():
            print("ERROR: ml-depth-pro not found. Run `pip install -e .` first.")
            return
        subprocess.run(["bash", str(script)], cwd=CHECKPOINT_SRC.parent, check=True)
    else:
        print("Checkpoint already exists, skipping download.")

    # Create symlink at project root so depth-pro finds ./checkpoints/depth_pro.pt
    if CHECKPOINT_LINK.exists() or CHECKPOINT_LINK.is_symlink():
        print(f"Symlink already exists at {CHECKPOINT_LINK}, skipping.")
    else:
        CHECKPOINT_LINK.symlink_to(CHECKPOINT_SRC)
        print(f"Symlink created: {CHECKPOINT_LINK} -> {CHECKPOINT_SRC}")

    print("\nSetup complete. You can now run depth_test.py from any directory.")

if __name__ == "__main__":
    main()
