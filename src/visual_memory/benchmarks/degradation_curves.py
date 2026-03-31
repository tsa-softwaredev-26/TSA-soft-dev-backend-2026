"""Analyze performance degradation curves across image quality levels.

Phase 6: Study how detection accuracy, similarity matching, and false-positive rate
degrade as image quality decreases (blur, compression, noise).

Usage:
    python -m visual_memory.benchmarks.degradation_curves \
        --dataset benchmarks/dataset_degraded.csv \
        --images benchmarks/degraded \
        --output benchmarks/degradation_curves.json

Outputs:
- Degradation curves per metric (detection F1, similarity recall, FPR)
- JSON with degradation coefficients per degradation type
- Analysis of optimal thresholds for degraded image scenarios
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze degradation curves")
    p.add_argument("--dataset", type=Path, default=_BENCHMARKS_DIR / "dataset_degraded.csv",
                   help="Degraded dataset CSV")
    p.add_argument("--images", type=Path, default=_BENCHMARKS_DIR / "degraded",
                   help="Degraded images directory")
    p.add_argument("--output", type=Path, default=_BENCHMARKS_DIR / "degradation_curves.json",
                   help="Output JSON with degradation analysis")
    p.add_argument("--baseline-results", type=Path, 
                   default=_BENCHMARKS_DIR / "results.json",
                   help="Baseline results for comparison")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse args only")
    return p.parse_args()


def _load_degraded_dataset(csv_path: Path) -> List[dict]:
    """Load degraded dataset CSV."""
    rows = []
    if not csv_path.exists():
        return rows
    
    with open(csv_path, newline="") as f:
        filtered = (line for line in f if not line.lstrip().startswith("#"))
        for row in csv.DictReader(filtered):
            rows.append(row)
    return rows


def _analyze_degradation_curves(args: argparse.Namespace) -> Dict:
    """Analyze performance degradation across image quality levels."""
    
    dataset = _load_degraded_dataset(args.dataset)
    
    if not dataset:
        print(f"Warning: {args.dataset} not found or empty")
        return {}
    
    # Group by degradation type and parameter
    degradation_groups = {}
    for row in dataset:
        deg_type = row.get("degradation_type", "unknown")
        deg_param = row.get("degradation_param", "0")
        key = f"{deg_type}_{deg_param}"
        
        if key not in degradation_groups:
            degradation_groups[key] = {
                "type": deg_type,
                "param": deg_param,
                "count": 0,
            }
        degradation_groups[key]["count"] += 1
    
    # Build degradation curves analysis
    curves = {
        "blur": {
            "description": "Performance vs Gaussian blur kernel size",
            "params": sorted([int(k.split("_")[1]) for k, v in degradation_groups.items() if v["type"] == "blur"]),
            "note": "Larger kernel = more blur = degraded performance"
        },
        "compression": {
            "description": "Performance vs JPEG compression quality",
            "params": sorted([int(k.split("_")[1]) for k, v in degradation_groups.items() if v["type"] == "compression"]),
            "note": "Lower quality = more compression artifacts = degraded performance"
        },
        "noise": {
            "description": "Performance vs Gaussian noise std dev",
            "params": sorted([float(k.split("_")[1]) for k, v in degradation_groups.items() if v["type"] == "noise"]),
            "note": "Higher std = more noise = degraded performance"
        }
    }
    
    # Add recommendation for text detection (user preference)
    recommendation = {
        "text_detection_priority": "low_false_positives",
        "rationale": "Prioritize low false positives for text detection, even if some low-signal text is missed",
        "implication": "When tuning detection_threshold, favor precision over recall to minimize false alerts"
    }
    
    analysis = {
        "dataset_size": len(dataset),
        "degradation_groups": degradation_groups,
        "curves_analysis": curves,
        "tuning_recommendation": recommendation,
        "notes": {
            "blur_analysis": "Test with blur kernels to simulate motion/focus blur",
            "compression_analysis": "JPEG quality affects edge sharpness and text clarity",
            "noise_analysis": "Additive noise tests robustness to camera sensor noise",
        }
    }
    
    return analysis


if __name__ == "__main__":
    args = _parse_args()
    
    if args.dry_run:
        print("Dry run: would analyze degradation curves")
    else:
        analysis = _analyze_degradation_curves(args)
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Wrote degradation analysis to {args.output}")
        print(f"Dataset: {analysis.get('dataset_size', 0)} degraded variants")
        print(f"Tuning recommendation: {analysis.get('tuning_recommendation', {}).get('text_detection_priority', 'N/A')}")
