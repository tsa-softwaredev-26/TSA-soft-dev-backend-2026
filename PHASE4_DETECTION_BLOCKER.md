# Detection Threshold Optimization - Hardware Blocker

## Status: BLOCKED

**Issue:** GroundingDINO grid search requires more VRAM than available (1060 6GB).

**Attempts:**
- Shell 132: Default run -> OOM killed (exit 137)
- Shell 155: batch_size=1 -> Failed (exit 1)
- Shell 162: fp16 autocast -> Failed (exit 1)

**Current detection thresholds:** (from settings.py)
- box_threshold: 0.30
- text_threshold: 0.25

**Recommendation:**
Keep current defaults. Detection rate on Phase 3 was 100% at 1ft bright, which is acceptable for production use.

**Future work:**
Could optimize on hardware with >8GB VRAM or via CPU-based sharded approach (runtime: several hours).
