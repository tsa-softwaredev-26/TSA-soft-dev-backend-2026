# Benchmark Image Capture Guide

120 images total: 10 objects x 12 conditions (3 distances x 2 lighting x 2 backgrounds).

---

## Objects

| ID | Description |
|----|-------------|
| wallet_a | First wallet |
| wallet_b | Second wallet (different color or style) |
| book_a | First book (front cover facing camera) |
| book_b | Second book (different cover) |
| sunglasses_a | First sunglasses (laid flat on surface) |
| sunglasses_b | Second sunglasses |
| receipt_a | First receipt - run redact_receipt.py before benchmark |
| receipt_b | Second receipt - run redact_receipt.py before benchmark |
| keys_a | First set of keys (spread naturally, not piled) |
| keys_b | Second set of keys |

---

## Conditions (12 per object)

| Distance | Lighting | Background | Filename suffix |
|----------|----------|------------|-----------------|
| 1 ft | bright | clean | 1ft_bright_clean |
| 1 ft | bright | messy | 1ft_bright_messy |
| 1 ft | dim | clean | 1ft_dim_clean |
| 1 ft | dim | messy | 1ft_dim_messy |
| 3 ft | bright | clean | 3ft_bright_clean |
| 3 ft | bright | messy | 3ft_bright_messy |
| 3 ft | dim | clean | 3ft_dim_clean |
| 3 ft | dim | messy | 3ft_dim_messy |
| 6 ft | bright | clean | 6ft_bright_clean |
| 6 ft | bright | messy | 6ft_bright_messy |
| 6 ft | dim | clean | 6ft_dim_clean |
| 6 ft | dim | messy | 6ft_dim_messy |

---

## Naming Convention

```
{object_id}_{distance}ft_{lighting}_{background}.jpg
```

Examples:
```
wallet_a_1ft_bright_clean.jpg
book_b_6ft_dim_messy.jpg
receipt_a_3ft_bright_clean.jpg
keys_b_1ft_dim_messy.jpg
```

All images go in `benchmarks/images/`.

---

## Setup Definitions

### Distance
Measure from camera lens to the nearest edge of the object.
- 1 ft = ~30 cm
- 3 ft = ~91 cm
- 6 ft = ~183 cm

Use a tape measure. Mark the floor with tape for repeatability.

### Lighting
- **Bright**: Overhead lights on + window light. Daytime indoors. Aim for even, shadow-free illumination.
- **Dim**: Close the blinds, turn off overhead lights. Use a single lamp 4-5 ft away, or rely on ambient glow from a TV/monitor. The object should still be visible but noticeably underexposed.

### Background
- **Clean**: Bare table, plain floor, or single-color sheet. No other objects in the background.
- **Messy**: Desk cluttered with papers, books, or household items. Background should visually compete with the subject.

---

## Camera Settings

- Use the rear camera, no digital zoom.
- Portrait or landscape - pick one and stay consistent across all shots.
- Let auto-focus lock before shooting. Tap the object on screen to focus.
- Keep the object centered in frame.
- No flash (it defeats dim lighting conditions).

---

## Shooting Workflow

Shoot all 12 conditions for one object before moving to the next. This keeps setup time low.

Recommended order per object:
1. Set up bright + clean background. Shoot 1ft, 3ft, 6ft.
2. Switch to bright + messy background. Shoot 1ft, 3ft, 6ft.
3. Dim the lights. Shoot dim + clean 1ft, 3ft, 6ft.
4. Switch to dim + messy background. Shoot dim + messy 1ft, 3ft, 6ft.

Rename images on your phone or computer to match the naming convention above, then drop into `benchmarks/images/`.

---

## Object-Specific Notes

**Wallets** - Open vs closed does not matter, but keep the same state across all shots for a given wallet.

**Books** - Front cover facing the camera. Title visible if possible.

**Sunglasses** - Lay flat on the surface, temples folded. Both lenses visible.

**Receipts** - Lay completely flat, no folds or curls. Full text should be visible at 1ft.
Run the redaction script before your benchmark run:
```
python -m visual_memory.benchmarks.redact_receipt a --images benchmarks/images
python -m visual_memory.benchmarks.redact_receipt b --images benchmarks/images
```
The script redacts sensitive text in all 12 images for each receipt, prompts for ground truth text,
and outputs OCR accuracy immediately.

**Keys** - Spread the keys naturally (fan shape or loose cluster). Avoid piling them.

---

## Checklist

Before running `full_benchmark.py`:

- [ ] All 120 images present in `benchmarks/images/`
- [ ] Images named exactly per convention (use `python -m visual_memory.benchmarks.check_dataset` to verify)
- [ ] receipt_a redacted (all 12 images)
- [ ] receipt_b redacted (all 12 images)
- [ ] `benchmarks/ground_truth/receipt_a.txt` exists
- [ ] `benchmarks/ground_truth/receipt_b.txt` exists

Quick image count check:
```bash
ls benchmarks/images/*.jpg | wc -l   # should be 120 (or more if you have extra)
```

---

## Quick Verify

After placing images, confirm dataset parses cleanly:
```bash
python -m visual_memory.benchmarks.check_dataset
```

This lists any missing images and confirms all 120 entries are resolvable.
