"""
Benchmark: DINOv3 image similarity + CLIP text similarity + combined.

Run:
    python -m visual_memory.tests.scripts.benchmark_embedder

Sections:
    A - Intra-class / inter-class DINOv3 image similarity (physical objects)
    B - CLIPText similarity on ground-truth texts (text_demo)
    C - Combined (DINOv3 + CLIPText) similarity matrix (text_demo)

Output: markdown tables for ARCHITECTURE.md.
"""
from __future__ import annotations

import itertools
import logging
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
TESTS_DIR   = SCRIPTS_DIR.parent
INPUT_DIR   = TESTS_DIR / "input_images"
DEMO_DB     = TESTS_DIR / "demo_database"
TEXT_DEMO   = TESTS_DIR / "text_demo"
GT_DIR      = TEXT_DEMO / "ground_truth"

OBJECT_GROUPS: dict[str, list[str]] = {
    "wallet":  ["wallet_1ft_table.jpg", "wallet_3ft_table.jpg", "wallet_6ft_table.jpg"],
    "airpods": ["airpods_1ft_table.jpg", "airpods_3ft_table.jpg", "airpods_6ft_table.jpg"],
    "mouse":   ["mouse_1ft_table.jpg",   "mouse_3ft_table.jpg",   "mouse_6ft_table.jpg"],
}

TEXT_STEMS     = ["marker", "pen", "pencil", "typed"]
ALL_TEXT_STEMS = ["marker", "pen", "pencil", "typed", "random_printed_notes", "malarkey"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=1).item())


def _load_image(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    return Image.open(path).convert("RGB")


def _load_gt(stem: str) -> str | None:
    p = GT_DIR / f"{stem}.txt"
    return p.read_text().strip() if p.exists() else None


# ---------------------------------------------------------------------------
# Section A - DINOv3 intra/inter class + scan match test
# ---------------------------------------------------------------------------
def section_a(img_embedder):
    print("\n" + "=" * 60)
    print("SECTION A - Intra-class vs Inter-class (DINOv3 image)")
    print("=" * 60)

    embs: dict[str, list[torch.Tensor]] = {}
    missing: list[str] = []

    for label, files in OBJECT_GROUPS.items():
        embs[label] = []
        for fname in files:
            img = _load_image(INPUT_DIR / fname)
            if img is None:
                missing.append(fname)
                continue
            embs[label].append(img_embedder.embed(img))

    if missing:
        print(f"  [warn] Missing images: {missing}")

    labels = list(embs.keys())
    intra_sims, inter_sims = [], []

    for label in labels:
        vecs = embs[label]
        for i, j in itertools.combinations(range(len(vecs)), 2):
            intra_sims.append(_cosine(vecs[i], vecs[j]))

    for l1, l2 in itertools.combinations(labels, 2):
        for v1 in embs[l1]:
            for v2 in embs[l2]:
                inter_sims.append(_cosine(v1, v2))

    intra_avg = sum(intra_sims) / len(intra_sims) if intra_sims else 0.0
    inter_avg = sum(inter_sims) / len(inter_sims) if inter_sims else 0.0
    ratio     = intra_avg / inter_avg if inter_avg else float("inf")

    print()
    print("| Embedder | Intra-class | Inter-class | Ratio |")
    print("|----------|-------------|-------------|-------|")
    print(f"| DINOv3   | {intra_avg:.4f}      | {inter_avg:.4f}      | {ratio:.3f} |")

    print()
    print("Scan match test - cropped_wallet.png vs wallet_Nft images")
    ref = _load_image(DEMO_DB / "cropped_wallet.png")
    if ref is None:
        print("  [skip] cropped_wallet.png not found")
        return

    ref_emb = img_embedder.embed(ref)

    print()
    print("| Image                   | DINOv3 sim |")
    print("|-------------------------|------------|")
    for fname in OBJECT_GROUPS["wallet"]:
        img = _load_image(INPUT_DIR / fname)
        if img is None:
            print(f"| {fname:<23} | (missing)  |")
            continue
        sim = _cosine(ref_emb, img_embedder.embed(img))
        print(f"| {fname:<23} | {sim:.4f}     |")


# ---------------------------------------------------------------------------
# Section B - CLIPText similarity on ground-truth text
# ---------------------------------------------------------------------------
def section_b(txt_embedder):
    print("\n" + "=" * 60)
    print("SECTION B - CLIPText similarity on ground-truth text")
    print("=" * 60)

    txt_embs: dict[str, torch.Tensor] = {}

    for stem in TEXT_STEMS:
        gt = _load_gt(stem)
        if gt is None:
            continue
        txt_embs[stem] = txt_embedder.embed_text(gt)

    rp_gt = _load_gt("random_printed_notes")
    if rp_gt:
        txt_embs["random_printed_notes"] = txt_embedder.embed_text(rp_gt)

    stems = list(txt_embs.keys())
    if not stems:
        print("  [skip] No ground truth found")
        return

    print()
    pad = max(len(s) for s in stems) + 1
    print("| " + " " * pad + " | " + " | ".join(f"{s[:10]:<10}" for s in stems) + " |")
    print("|" + "-" * (pad + 2) + "|" + ("-----------|" * len(stems)))

    matched, mismatched = [], []
    for s1 in TEXT_STEMS:
        if s1 not in txt_embs:
            continue
        cells = []
        for s2 in stems:
            sim = _cosine(txt_embs[s1], txt_embs[s2])
            mark = " *" if s1 == s2 else "  "
            cells.append(f"{sim:.4f}{mark}")
            if s1 == s2:
                matched.append(sim)
            elif s2 in TEXT_STEMS:
                mismatched.append(sim)
        print(f"| {s1:<{pad}} | " + " | ".join(cells) + " |")

    avg_m  = sum(matched)    / len(matched)    if matched    else 0.0
    avg_mm = sum(mismatched) / len(mismatched) if mismatched else 0.0
    print()
    print(f"Avg self-sim (diagonal)  : {avg_m:.4f}")
    print(f"Avg cross-sim (off-diag) : {avg_mm:.4f}")
    print(f"Gap                      : {avg_m - avg_mm:.4f}")


# ---------------------------------------------------------------------------
# Section C - Combined (DINOv3 + CLIPText) similarity matrix
# ---------------------------------------------------------------------------
def section_c(img_embedder, txt_embedder):
    print("\n" + "=" * 60)
    print("SECTION C - Combined embedding similarity matrix (text_demo)")
    print("=" * 60)

    from visual_memory.engine.embedding.embed_combined import make_combined_embedding

    comb_embs: dict[str, torch.Tensor] = {}
    for stem in ALL_TEXT_STEMS:
        img = _load_image(TEXT_DEMO / f"{stem}.jpeg")
        if img is None:
            continue
        ie = img_embedder.embed(img)
        gt = _load_gt(stem)
        te = txt_embedder.embed_text(gt) if gt else None
        comb_embs[stem] = make_combined_embedding(ie, te)

    stems = list(comb_embs.keys())
    if not stems:
        print("  [skip] No text_demo images found")
        return

    pad = max(len(s) for s in stems) + 1
    print()
    print("| " + " " * pad + " | " + " | ".join(f"{s[:8]:<8}" for s in stems) + " |")
    print("|" + "-" * (pad + 2) + "|" + ("---------|" * len(stems)))

    off_diag = []
    for s1 in stems:
        cells = []
        for s2 in stems:
            sim = _cosine(comb_embs[s1], comb_embs[s2])
            cells.append(f"{sim:.4f}")
            if s1 != s2:
                off_diag.append(sim)
        print(f"| {s1:<{pad}} | " + " | ".join(cells) + " |")

    if off_diag:
        print()
        print(f"Off-diagonal range : {min(off_diag):.4f} - {max(off_diag):.4f}")
        print(f"Off-diagonal mean  : {sum(off_diag)/len(off_diag):.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    from visual_memory.engine.embedding import ImageEmbedder, CLIPTextEmbedder

    print("Loading ImageEmbedder (DINOv3)...")
    img_embedder = ImageEmbedder()
    print(f"  device: {img_embedder.device}")

    print("Loading CLIPTextEmbedder...")
    txt_embedder = CLIPTextEmbedder()
    print(f"  device: {txt_embedder.device}")

    section_a(img_embedder)
    section_b(txt_embedder)
    section_c(img_embedder, txt_embedder)

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
