"""
Benchmark: DINOv3 vs CLIP image embeddings + CLIP cross-modal alignment.

Run:
    python -m visual_memory.tests.scripts.benchmark_embedder

Sections:
    A — Intra-class / inter-class similarity (DINOv3 vs CLIP image)
    B — CLIP text↔image cross-modal alignment (text_demo images)
    C — Combined embedding similarity matrix (text_demo, ground-truth text)

Output: markdown tables suitable for ARCHITECTURE.md.
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent   # tests/scripts/
TESTS_DIR   = SCRIPTS_DIR.parent                # tests/
INPUT_DIR   = TESTS_DIR / "input_images"
DEMO_DB     = TESTS_DIR / "demo_database"
TEXT_DEMO   = TESTS_DIR / "text_demo"
GT_DIR      = TEXT_DEMO / "ground_truth"

OBJECT_GROUPS: dict[str, list[str]] = {
    "wallet":  ["wallet_1ft_table.jpg", "wallet_3ft_table.jpg", "wallet_6ft_table.jpg"],
    "airpods": ["airpods_1ft_table.jpg", "airpods_3ft_table.jpg", "airpods_6ft_table.jpg"],
    "mouse":   ["mouse_1ft_table.jpg",  "mouse_3ft_table.jpg",  "mouse_6ft_table.jpg"],
}

TEXT_IMAGES = ["marker", "pen", "pencil", "typed"]  # malarkey + random_printed_notes excluded from Section B/C (no matching GT)
ALL_TEXT_IMAGES = ["marker", "pen", "pencil", "typed", "random_printed_notes"]  # malarkey has no OCR text expected

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_clip(device: torch.device):
    print(f"Loading CLIP ({CLIP_MODEL_NAME}) on {device}...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor


def _load_dino(device: torch.device):
    """DINOv3 no longer in engine — skip gracefully."""
    return None


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _clip_embed_image(model, processor, img: Image.Image, device: torch.device) -> torch.Tensor:
    """Return L2-normalized CLIP image embedding, shape (1, 512)."""
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        vision_out = model.vision_model(pixel_values=pixel_values)
        features   = model.visual_projection(vision_out.pooler_output)
    return F.normalize(features, dim=-1).detach().cpu()


def _clip_embed_text(model, processor, text: str, device: torch.device) -> torch.Tensor:
    """Return L2-normalized CLIP text embedding, shape (1, 512)."""
    enc = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.inference_mode():
        text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        features = model.text_projection(text_out.pooler_output)
    return F.normalize(features, dim=-1).detach().cpu()


def _dino_embed_image(embedder, img: Image.Image) -> torch.Tensor:
    """DINOv3 removed — return None."""
    return None


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
# Section A — Intra/inter class similarity
# ---------------------------------------------------------------------------
def section_a(clip_model, clip_proc, device):
    print("\n" + "=" * 60)
    print("SECTION A — Intra-class vs Inter-class similarity (CLIP)")
    print("=" * 60)

    clip_embs: dict[str, list[torch.Tensor]] = {}
    missing: list[str] = []

    for label, files in OBJECT_GROUPS.items():
        clip_embs[label] = []
        for fname in files:
            img = _load_image(INPUT_DIR / fname)
            if img is None:
                missing.append(fname)
                continue
            clip_embs[label].append(_clip_embed_image(clip_model, clip_proc, img, device))

    if missing:
        print(f"  [warn] Missing images (skipped): {missing}")

    def _compute_stats(embs: dict[str, list[torch.Tensor]]):
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
        return intra_avg, inter_avg, ratio

    c_intra, c_inter, c_ratio = _compute_stats(clip_embs)

    print()
    print("| Embedder | Intra-class sim | Inter-class sim | Discrimination ratio |")
    print("|----------|-----------------|-----------------|----------------------|")
    print(f"| CLIP     | {c_intra:.4f}          | {c_inter:.4f}          | {c_ratio:.3f}                |")

    # Scan matching test: cropped_wallet.png as reference
    print()
    print("### Scan match test — cropped_wallet.png vs wallet_{1ft,3ft,6ft}")
    ref_img = _load_image(DEMO_DB / "cropped_wallet.png")
    if ref_img is None:
        print("  [skip] cropped_wallet.png not found in demo_database/")
    else:
        clip_ref = _clip_embed_image(clip_model, clip_proc, ref_img, device)

        print()
        print("| Image                   | CLIP sim |")
        print("|-------------------------|----------|")
        for fname in OBJECT_GROUPS["wallet"]:
            img = _load_image(INPUT_DIR / fname)
            if img is None:
                print(f"| {fname:<23} | (missing) |")
                continue
            csim = _cosine(clip_ref, _clip_embed_image(clip_model, clip_proc, img, device))
            print(f"| {fname:<23} | {csim:.4f}   |")


# ---------------------------------------------------------------------------
# Section B — CLIP text↔image cross-modal alignment
# ---------------------------------------------------------------------------
def section_b(clip_model, clip_proc, device):
    print("\n" + "=" * 60)
    print("SECTION B — CLIP text↔image cross-modal alignment")
    print("=" * 60)

    img_embs: dict[str, torch.Tensor] = {}
    txt_embs: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for stem in TEXT_IMAGES:
        img = _load_image(TEXT_DEMO / f"{stem}.jpeg")
        gt  = _load_gt(stem)
        if img is None or gt is None:
            skipped.append(stem)
            continue
        img_embs[stem] = _clip_embed_image(clip_model, clip_proc, img, device)
        txt_embs[stem] = _clip_embed_text(clip_model, clip_proc, gt, device)

    if skipped:
        print(f"  [skip] {skipped}")

    stems = list(img_embs.keys())
    if not stems:
        print("  [skip] No valid images found")
        return

    # Build full similarity matrix
    print()
    header = "| img \\ text   | " + " | ".join(f"{s:<10}" for s in stems) + " |"
    print(header)
    sep = "|" + "---|" * (len(stems) + 1)
    print(sep)

    matched_sims, mismatched_sims = [], []

    for img_stem in stems:
        row = f"| {img_stem:<12} | "
        cells = []
        for txt_stem in stems:
            sim = _cosine(img_embs[img_stem], txt_embs[txt_stem])
            marker = " *" if img_stem == txt_stem else "  "
            cells.append(f"{sim:.4f}{marker}")
            if img_stem == txt_stem:
                matched_sims.append(sim)
            else:
                mismatched_sims.append(sim)
        print(row + " | ".join(cells) + " |")

    avg_matched    = sum(matched_sims) / len(matched_sims) if matched_sims else 0
    avg_mismatched = sum(mismatched_sims) / len(mismatched_sims) if mismatched_sims else 0

    print()
    print(f"Avg matched (diagonal) sim   : {avg_matched:.4f}")
    print(f"Avg mismatched sim           : {avg_mismatched:.4f}")
    print(f"Cross-modal gap              : {avg_matched - avg_mismatched:.4f}")
    print("(* = matched pair; higher diagonal = stronger alignment)")


# ---------------------------------------------------------------------------
# Section C — Combined embedding similarity matrix
# ---------------------------------------------------------------------------
def section_c(clip_model, clip_proc, device):
    print("\n" + "=" * 60)
    print("SECTION C — Combined embedding similarity matrix (text_demo)")
    print("=" * 60)

    from visual_memory.engine.embedding.embed_combined import make_combined_embedding

    # Use ALL_TEXT_IMAGES + malarkey (no GT → zero text slot)
    all_stems = ALL_TEXT_IMAGES + ["malarkey"]
    img_embs : dict[str, torch.Tensor] = {}
    comb_embs: dict[str, torch.Tensor] = {}
    img_only_embs: dict[str, torch.Tensor] = {}

    for stem in all_stems:
        img = _load_image(TEXT_DEMO / f"{stem}.jpeg")
        if img is None:
            continue
        ie = _clip_embed_image(clip_model, clip_proc, img, device)
        gt = _load_gt(stem)
        te = _clip_embed_text(clip_model, clip_proc, gt, device) if gt else None
        img_embs[stem]      = ie
        comb_embs[stem]     = make_combined_embedding(ie, te)
        img_only_embs[stem] = make_combined_embedding(ie, None)

    stems = list(comb_embs.keys())
    if not stems:
        print("  [skip] No text_demo images found")
        return

    def _print_matrix(embs: dict[str, torch.Tensor], title: str):
        print(f"\n**{title}**")
        pad = max(len(s) for s in stems) + 2
        header = "| " + " " * (pad - 2) + " | " + " | ".join(f"{s:<{pad}}" for s in stems) + " |"
        print(header)
        print("|" + "-" * pad + "|" + ("---|" * len(stems)))
        for s1 in stems:
            cells = []
            for s2 in stems:
                sim = _cosine(embs[s1], embs[s2])
                cells.append(f"{sim:.4f}")
            print(f"| {s1:<{pad-1}}| " + " | ".join(cells) + " |")

    _print_matrix(comb_embs,     "Combined embedding (image + GT text)")
    _print_matrix(img_only_embs, "Image-only combined (zero text slot)")

    # Off-diagonal stats for combined
    off_diag = [_cosine(comb_embs[s1], comb_embs[s2])
                for s1, s2 in itertools.combinations(stems, 2)]
    if off_diag:
        print()
        print(f"Combined off-diagonal range : {min(off_diag):.4f} – {max(off_diag):.4f}")
        print(f"Combined off-diagonal mean  : {sum(off_diag)/len(off_diag):.4f}")
        print()
        print("Interpretation:")
        print("  - 'marker', 'pen', 'pencil', 'typed' share near-identical GT text → high sim expected")
        print("  - 'random_printed_notes' has different text → should score lower against GT-text docs")
        print("  - 'malarkey' (no GT text) → zero text slot, image-only similarity")
        print()
        print("Threshold guidance:")
        print("  - For document items (all text), set threshold ~0.05 below min(GT-matched sims)")
        print("  - For physical objects (no text), use image-only combined similarity baseline")
        print("  - Current text_similarity_threshold=0.3 may need retuning based on above matrix")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    device = _get_device()
    print(f"Device: {device}")

    clip_model, clip_proc = _load_clip(device)

    section_a(clip_model, clip_proc, device)
    section_b(clip_model, clip_proc, device)
    section_c(clip_model, clip_proc, device)

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
