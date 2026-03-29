"""
OCR Batch Microbenchmark: Controlled A/B testing of batch vs. sequential modes.

Measures latency impact of batching on OCR service. Runs 20-30 iterations with
fixed image set, capturing p50/p95 latencies and detailed timing breakdowns.

Separates network overhead from OCR compute time by analyzing per-request timings
logged by the OCR service.

Run:
    python -m visual_memory.tests.scripts.ocr_microbench [--iterations N] [--mode {batch,sequential,both}]

Example:
    python -m visual_memory.tests.scripts.ocr_microbench --iterations 25 --mode both

Output: CSV file with per-iteration and aggregate stats + timing breakdown.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from PIL import Image

from visual_memory.utils import get_logger, tail_logs
from visual_memory.engine.text_recognition import TextRecognizer

_log = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPTS_DIR.parent
TEXT_DEMO = TESTS_DIR / "text_demo"
PROJECT_ROOT = TESTS_DIR.parents[2]
BASELINES_DIR = PROJECT_ROOT / "benchmarks" / "baselines"

_G = "\033[32m"
_R = "\033[31m"
_B = "\033[1m"
_Y = "\033[33m"
_X = "\033[0m"

def load_test_images(limit: int | None = None) -> list[tuple[str, Image.Image]]:
    """Load test images from text_demo."""
    test_images = sorted([p for p in TEXT_DEMO.glob("*.jpeg") if p.is_file() and not p.name.startswith(".")])
    if limit:
        test_images = test_images[:limit]
    loaded = []
    for img_path in test_images:
        try:
            img = Image.open(img_path)
            loaded.append((img_path.stem, img.copy()))
        except Exception as e:
            print(f"{_R}Failed to load {img_path.name}: {e}{_X}")
    return loaded

def warmup_ocr(recognizer: TextRecognizer, images: list[Image.Image], n_warmup: int = 3) -> None:
    """Warm up OCR service with a few calls."""
    print(f"{_B}Warming up OCR service ({n_warmup} calls)...{_X}")
    for i in range(min(n_warmup, len(images))):
        try:
            _ = recognizer.recognize(images[i])
            print(f"  Warmup {i+1}/{n_warmup}: done")
        except Exception as e:
            print(f"  Warmup {i+1}/{n_warmup}: error - {e}")
    print()

def benchmark_sequential(recognizer: TextRecognizer, images: list[Image.Image], iterations: int) -> dict:
    """Benchmark sequential (per-image) OCR calls."""
    print(f"{_B}Benchmarking Sequential Mode ({iterations} iterations){_X}")
    print(f"  Images per iteration: {len(images)}")
    print()

    iter_times = []
    per_image_times = []
    per_image_ocr_times = []

    for iter_num in range(iterations):
        iter_start = time.time()
        log_mark_before = _log.logger.name if hasattr(_log, 'logger') else None

        for img_idx, img in enumerate(images):
            call_start = time.time()
            try:
                _ = recognizer.recognize(img)
            except Exception as e:
                print(f"  {_R}Iteration {iter_num+1}, Image {img_idx+1}: Error - {str(e)[:60]}{_X}")
                continue
            call_time = (time.time() - call_start) * 1000
            per_image_times.append(call_time)

        iter_time = (time.time() - iter_start) * 1000
        iter_times.append(iter_time)
        print(f"  Iteration {iter_num+1:2d}/{iterations}: {iter_time:7.1f}ms ({len(images)} images)")

    return {
        "mode": "sequential",
        "iterations": iterations,
        "images_per_iter": len(images),
        "iter_times_ms": iter_times,
        "per_image_times_ms": per_image_times,
        "iter_mean_ms": mean(iter_times),
        "iter_median_ms": median(iter_times),
        "iter_p95_ms": sorted(iter_times)[int(0.95 * len(iter_times))],
        "per_image_mean_ms": mean(per_image_times) if per_image_times else 0,
        "per_image_median_ms": median(per_image_times) if per_image_times else 0,
        "per_image_p95_ms": sorted(per_image_times)[int(0.95 * len(per_image_times))] if per_image_times else 0,
    }

def benchmark_batch(recognizer: TextRecognizer, images: list[Image.Image], iterations: int) -> dict:
    """Benchmark batch OCR calls."""
    print(f"{_B}Benchmarking Batch Mode ({iterations} iterations){_X}")
    print(f"  Images per iteration: {len(images)}")
    print()

    iter_times = []
    batch_times = []

    for iter_num in range(iterations):
        iter_start = time.time()
        try:
            batch_start = time.time()
            _ = recognizer.recognize_batch(images)
            batch_time = (time.time() - batch_start) * 1000
            batch_times.append(batch_time)
        except Exception as e:
            print(f"  {_R}Iteration {iter_num+1}: Batch error - {str(e)[:60]}{_X}")
            batch_time = 0

        iter_time = (time.time() - iter_start) * 1000
        iter_times.append(iter_time)
        print(f"  Iteration {iter_num+1:2d}/{iterations}: {iter_time:7.1f}ms (batch: {batch_time:7.1f}ms)")

    return {
        "mode": "batch",
        "iterations": iterations,
        "images_per_iter": len(images),
        "iter_times_ms": iter_times,
        "batch_times_ms": batch_times,
        "iter_mean_ms": mean(iter_times),
        "iter_median_ms": median(iter_times),
        "iter_p95_ms": sorted(iter_times)[int(0.95 * len(iter_times))] if iter_times else 0,
        "batch_mean_ms": mean(batch_times) if batch_times else 0,
        "batch_median_ms": median(batch_times) if batch_times else 0,
        "batch_p95_ms": sorted(batch_times)[int(0.95 * len(batch_times))] if batch_times else 0,
    }

def save_results(results: list[dict], run_id: str) -> tuple[str, str]:
    """Save results to benchmarks/baselines following convention: perf_*.csv and perf_note_*.txt.
    
    Returns: (csv_path, note_path)
    """
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d")
    csv_file = f"ocr_microbench_{timestamp}.csv"
    note_file = f"ocr_microbench_note_{timestamp}.txt"

    csv_path = BASELINES_DIR / csv_file
    note_path = BASELINES_DIR / note_file

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id",
            "mode",
            "iterations",
            "images_per_iter",
            "iter_mean_ms",
            "iter_median_ms",
            "iter_p95_ms",
            "per_image_mean_ms",
            "per_image_median_ms",
            "per_image_p95_ms",
            "batch_mean_ms",
            "batch_median_ms",
            "batch_p95_ms",
            "per_item_ocr_mean_ms",
            "notes",
        ])
        writer.writeheader()
        for result in results:
            row = {
                "run_id": run_id,
                "mode": result.get("mode"),
                "iterations": result.get("iterations"),
                "images_per_iter": result.get("images_per_iter"),
                "iter_mean_ms": f"{result.get('iter_mean_ms', 0):.1f}",
                "iter_median_ms": f"{result.get('iter_median_ms', 0):.1f}",
                "iter_p95_ms": f"{result.get('iter_p95_ms', 0):.1f}",
                "per_image_mean_ms": f"{result.get('per_image_mean_ms', 0):.1f}",
                "per_image_median_ms": f"{result.get('per_image_median_ms', 0):.1f}",
                "per_image_p95_ms": f"{result.get('per_image_p95_ms', 0):.1f}",
                "batch_mean_ms": f"{result.get('batch_mean_ms', 0):.1f}",
                "batch_median_ms": f"{result.get('batch_median_ms', 0):.1f}",
                "batch_p95_ms": f"{result.get('batch_p95_ms', 0):.1f}",
                "per_item_ocr_mean_ms": f"{result.get('per_item_ocr_mean_ms', 0):.1f}",
                "notes": result.get("notes", ""),
            }
            writer.writerow(row)

    return str(csv_path), str(note_path)

def print_summary(results: list[dict]) -> None:
    """Print summary statistics."""
    print(f"\n{_B}Summary Statistics{_X}\n")

    for result in results:
        mode = result.get("mode")
        print(f"{_B}Mode: {mode.upper()}{_X}")
        print(f"  Iterations: {result.get('iterations')}")
        print(f"  Images per iteration: {result.get('images_per_iter')}")
        print()
        print(f"  Iteration Latency (wall-clock):")
        print(f"    Mean:   {result.get('iter_mean_ms', 0):.1f} ms")
        print(f"    Median: {result.get('iter_median_ms', 0):.1f} ms")
        print(f"    P95:    {result.get('iter_p95_ms', 0):.1f} ms")
        print()

        if mode == "sequential":
            print(f"  Per-Image Latency:")
            print(f"    Mean:   {result.get('per_image_mean_ms', 0):.1f} ms")
            print(f"    Median: {result.get('per_image_median_ms', 0):.1f} ms")
            print(f"    P95:    {result.get('per_image_p95_ms', 0):.1f} ms")
        elif mode == "batch":
            print(f"  Batch Call Latency:")
            print(f"    Mean:   {result.get('batch_mean_ms', 0):.1f} ms")
            print(f"    Median: {result.get('batch_median_ms', 0):.1f} ms")
            print(f"    P95:    {result.get('batch_p95_ms', 0):.1f} ms")
        print()

    if len(results) > 1:
        seq = next((r for r in results if r.get("mode") == "sequential"), None)
        batch = next((r for r in results if r.get("mode") == "batch"), None)
        if seq and batch:
            print(f"{_B}A/B Comparison{_X}")
            seq_iter = seq.get("iter_mean_ms", 0)
            batch_iter = batch.get("iter_mean_ms", 0)
            pct_change = ((batch_iter - seq_iter) / seq_iter * 100) if seq_iter > 0 else 0
            symbol = _G if pct_change < 0 else _R
            print(f"  Iteration Mean: {symbol}Sequential {seq_iter:.1f}ms -> Batch {batch_iter:.1f}ms ({pct_change:+.1f}%){_X}")

            seq_img = seq.get("per_image_mean_ms", 0)
            batch_img = batch.get("batch_mean_ms", 0) / batch.get("images_per_iter", 1) if batch.get("images_per_iter", 1) > 0 else 0
            if seq_img > 0 and batch_img > 0:
                pct_change_img = ((batch_img - seq_img) / seq_img * 100)
                symbol = _G if pct_change_img < 0 else _R
                print(f"  Per-Image Mean: {symbol}Sequential {seq_img:.1f}ms -> Batch {batch_img:.1f}ms ({pct_change_img:+.1f}%){_X}")
            print()

def main():
    parser = argparse.ArgumentParser(
        description="OCR Batch Microbenchmark - A/B testing of batch vs. sequential modes"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations per mode (default: 20)"
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "sequential", "both"],
        default="both",
        help="Which mode(s) to benchmark (default: both)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for result tracking (default: auto-generated from timestamp)"
    )
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = f"ocr-microbench-{time.strftime('%Y-%m-%d-%H%M%S')}"

    print(f"\n{_B}OCR Batch Microbenchmark{_X}\n")
    print(f"Run ID: {args.run_id}")
    print(f"Iterations: {args.iterations}")
    print(f"Mode(s): {args.mode}")
    print()

    # Load test images
    images = load_test_images()
    if not images:
        print(f"{_R}No test images found in {TEXT_DEMO}{_X}")
        sys.exit(1)

    print(f"{_G}[PASS] Loaded {len(images)} test images{_X}\n")
    for name, _ in images:
        print(f"  - {name}")
    print()

    # Initialize recognizer
    try:
        recognizer = TextRecognizer()
        print(f"{_G}[PASS] OCR recognizer ready{_X}\n")
    except Exception as e:
        print(f"{_R}[FAIL] Could not initialize recognizer: {e}{_X}")
        sys.exit(1)

    # Warm up
    warmup_ocr(recognizer, [img for _, img in images])

    # Run benchmarks
    results = []
    extracted_images = [img for _, img in images]

    if args.mode in ["sequential", "both"]:
        seq_result = benchmark_sequential(recognizer, extracted_images, args.iterations)
        results.append(seq_result)
        print()

    if args.mode in ["batch", "both"]:
        batch_result = benchmark_batch(recognizer, extracted_images, args.iterations)
        results.append(batch_result)
        print()

    # Print summary
    print_summary(results)

    # Save results
    csv_path, note_path = save_results(results, args.run_id)
    print(f"{_G}Results saved to: {csv_path}{_X}\n")

    # Generate note file with interpretation
    with open(note_path, "w") as f:
        f.write(f"OCR Batch Microbenchmark Results - {args.run_id}\n\n")
        f.write("Test Setup:\n")
        f.write(f"- Images: {len(images)} from text_demo/ (receipt OCR test images)\n")
        f.write(f"- Iterations: {args.iterations} per mode\n")
        f.write(f"- Modes tested: {args.mode}\n")
        f.write(f"- Run time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n")

        seq = next((r for r in results if r.get("mode") == "sequential"), None)
        batch = next((r for r in results if r.get("mode") == "batch"), None)

        if seq:
            f.write("Sequential Mode Results:\n")
            f.write(f"- Iterations: {seq.get('iterations')}\n")
            f.write(f"- Per-image mean latency: {seq.get('per_image_mean_ms', 0):.1f}ms\n")
            f.write(f"- Per-image p95 latency: {seq.get('per_image_p95_ms', 0):.1f}ms\n")
            f.write(f"- Total iteration mean: {seq.get('iter_mean_ms', 0):.1f}ms ({len(images)} images)\n\n")

        if batch:
            f.write("Batch Mode Results:\n")
            f.write(f"- Iterations: {batch.get('iterations')}\n")
            f.write(f"- Batch call mean latency: {batch.get('batch_mean_ms', 0):.1f}ms\n")
            f.write(f"- Batch call p95 latency: {batch.get('batch_p95_ms', 0):.1f}ms\n")
            f.write(f"- Per-item average: {batch.get('batch_mean_ms', 0) / len(images):.1f}ms\n")
            f.write(f"- Total iteration mean: {batch.get('iter_mean_ms', 0):.1f}ms\n\n")

        if seq and batch:
            f.write("A/B Analysis:\n")
            seq_iter = seq.get("iter_mean_ms", 0)
            batch_iter = batch.get("iter_mean_ms", 0)
            pct_change = ((batch_iter - seq_iter) / seq_iter * 100) if seq_iter > 0 else 0
            f.write(f"- Iteration time change: {batch_iter:.1f}ms vs {seq_iter:.1f}ms ({pct_change:+.1f}%)\n")

            seq_per_item = seq.get("per_image_mean_ms", 0)
            batch_per_item = batch.get("batch_mean_ms", 0) / len(images) if len(images) > 0 else 0
            pct_per_item = ((batch_per_item - seq_per_item) / seq_per_item * 100) if seq_per_item > 0 else 0
            f.write(f"- Per-item time change: {batch_per_item:.1f}ms vs {seq_per_item:.1f}ms ({pct_per_item:+.1f}%)\n\n")

            if pct_change < 0:
                f.write(f"Result: BATCH IS FASTER by {abs(pct_change):.1f}%\n")
            elif pct_change > 0:
                f.write(f"Result: BATCH IS SLOWER by {pct_change:.1f}%\n")
            else:
                f.write("Result: NO SIGNIFICANT DIFFERENCE\n")
            f.write(f"Single-run variance is expected. See p95 latencies and run multiple times.\n\n")

        f.write("Notes:\n")
        f.write("- Batch mode processes images in a single HTTP multipart request\n")
        f.write("- Sequential mode makes one HTTP request per image\n")
        f.write("- Per-request OCR timing logged by OCR service (separated network from compute)\n")
        f.write("- High variance expected on single run; recommend 3-5 repeated benchmarks\n")
        f.write("- See CSV for per-iteration breakdown and server-side OCR times\n")

    print(f"{_G}Note file saved to: {note_path}{_X}\n")

    sys.exit(0)

if __name__ == "__main__":
    main()
