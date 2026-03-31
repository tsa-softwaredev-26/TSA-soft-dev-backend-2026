# Phase 6: Degradation Curves & Tuning

## Overview
Phase 6 analyzes how visual memory system performance degrades as image quality decreases.
Tests include blur, compression, and noise variants to understand real-world robustness.

## Files

- `src/visual_memory/benchmarks/create_degraded.py` - Generate degraded image variants
- `src/visual_memory/benchmarks/degradation_curves.py` - Analyze performance degradation
- `src/visual_memory/benchmarks/optimize_detection_threshold.py` - Tune GroundingDINO threshold
- `src/visual_memory/benchmarks/optimize_similarity_threshold.py` - Tune similarity threshold
- `benchmarks/dataset_degraded.csv` - Mapping of degraded images
- `benchmarks/degraded/` - Directory containing degraded image variants

## User Preference for Tuning

**Priority: Low False Positives for Text Detection**

When tuning detection thresholds, prioritize precision over recall:
- **Favor lower false-positive rate** for text detection
- Accept missing some low-signal text to avoid spurious detections
- This reduces alert fatigue in real deployment scenarios

**Implementation:**
- Set `detection_threshold` (GroundingDINO) conservatively
- Prefer `higher precision @ lower recall` over balanced F1-score
- Log recall trade-off in optimization results

## Workflow

1. **Generate degraded variants:**
   ```
   python -m visual_memory.benchmarks.create_degraded \
       --dataset benchmarks/dataset.csv \
       --images benchmarks/images
   ```

2. **Analyze degradation curves:**
   ```
   python -m visual_memory.benchmarks.degradation_curves \
       --dataset benchmarks/dataset_degraded.csv \
       --images benchmarks/degraded
   ```

3. **Optimize thresholds on degraded data:**
   ```
   python -m visual_memory.benchmarks.optimize_detection_threshold \
       --dataset benchmarks/dataset_degraded.csv \
       --images benchmarks/degraded
   
   python -m visual_memory.benchmarks.optimize_similarity_threshold \
       --dataset benchmarks/dataset_degraded.csv \
       --images benchmarks/degraded
   ```

## Degradation Types

- **Blur**: Gaussian blur (kernel size 1-5) simulates motion/focus blur
- **Compression**: JPEG compression (quality 30-90) simulates network/storage compression
- **Noise**: Additive Gaussian noise (std 0.01-0.1) simulates sensor noise

## Outputs

- `benchmarks/degradation_curves.json` - Degradation analysis and curve parameters
- `benchmarks/results_degraded.json` - Threshold optimization results on degraded data
