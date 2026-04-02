"""
Text-likelihood FP tuning - optimized to minimize disk I/O.
Pre-compute scores on all images once, then sweep thresholds.
"""
import sys
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visual_memory.utils.quality_utils import estimate_text_likelihood
from visual_memory.utils.image_utils import load_image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def main():
    images_dir = PROJECT_ROOT / "benchmarks" / "images"
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return 1
    
    print(f"Loading and scoring images from {images_dir}...")
    
    text_scores = {}  # receipt files
    textless_scores = {}  # non-receipt files
    
    # First pass: compute all scores
    for img_file in sorted(images_dir.glob("*")):
        if not img_file.is_file():
            continue
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".heic", ".heif"]:
            continue
        
        try:
            img = load_image(img_file)
            if img is None:
                print(f"  Skip (invalid): {img_file.name}")
                continue
            
            score = estimate_text_likelihood(img)
            
            if "receipt" in img_file.name.lower():
                text_scores[img_file.name] = score
            else:
                textless_scores[img_file.name] = score
            
            print(f"  {img_file.name}: {score:.4f} ({'TEXT' if 'receipt' in img_file.name.lower() else 'TEXTLESS'})")
        except Exception as e:
            print(f"  Error: {img_file.name}: {e}")
    
    print(f"\nScores computed: {len(text_scores)} text, {len(textless_scores)} textless")
    
    # Second pass: sweep thresholds
    thresholds = np.arange(0.0, 1.05, 0.05)
    results = []
    
    print("\n" + "=" * 95)
    print(f"{'Threshold':>10} | {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} | "
          f"{'Precision':>9} {'Recall':>9} {'Specificity':>11} {'FP_Rate':>9}")
    print("-" * 95)
    
    for threshold in thresholds:
        tp = sum(1 for s in text_scores.values() if s >= threshold)
        fn = len(text_scores) - tp
        fp = sum(1 for s in textless_scores.values() if s >= threshold)
        tn = len(textless_scores) - fp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        specificity = tn / (fp + tn) if (fp + tn) > 0 else 1.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        result = {
            "threshold": float(threshold),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "fp_rate": fp_rate,
        }
        results.append(result)
        
        print(
            f"{threshold:10.2f} | "
            f"{tp:4d} {fp:4d} {tn:4d} {fn:4d} | "
            f"{precision:9.4f} {recall:9.4f} {specificity:11.4f} {fp_rate:9.4f}"
        )
    
    print("=" * 95)
    
    # Find best threshold: minimize FP while keeping recall reasonable
    best = min(results, key=lambda r: (r["FP"], -r["recall"]))
    
    print(f"\nBEST THRESHOLD: {best['threshold']:.2f}")
    print(f"  Confusion Matrix: TP={best['TP']}, FP={best['FP']}, TN={best['TN']}, FN={best['FN']}")
    print(f"  Metrics: Precision={best['precision']:.4f}, Recall={best['recall']:.4f}")
    print(f"  FP_Rate={best['fp_rate']:.4f} (Specificity={best['specificity']:.4f})")
    print("  Minimizes false positives on textless images")
    
    # Save detailed results
    output_file = PROJECT_ROOT / "benchmarks" / "text_likelihood_sweep.json"
    with open(output_file, "w") as f:
        json.dump({
            "text_image_scores": text_scores,
            "textless_image_scores": textless_scores,
            "results": results,
            "best_threshold": best["threshold"],
            "best_metrics": best,
        }, f, indent=2)
    
    print(f"\n→ Results saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
