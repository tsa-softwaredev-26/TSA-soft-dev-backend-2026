"""
Accuracy validation suite against a ground-truth JSON manifest.

NOT part of the default run_all suite - run explicitly after capturing benchmark images.

Usage:
    python -m visual_memory.tests.scripts.test_benchmark_validation \
        --manifest benchmarks/ground_truth.json \
        --base-url http://localhost:5050

Manifest format:
    {
      "name": "120-image-v1",
      "created": "2026-03-28",
      "images_dir": "benchmarks/images",
      "teach_items": [
        {"image": "wallet_1ft_table.jpg", "label": "wallet"}
      ],
      "scan_tests": [
        {
          "image": "wallet_3ft_table.jpg",
          "expected_matches": [{"label": "wallet", "min_similarity": 0.3}],
          "expected_no_match": ["phone"]
        }
      ]
    }
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path


def _load_image_bytes(path: Path) -> bytes:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _post_multipart(base_url: str, path: str, image_bytes: bytes, fields: dict) -> dict:
    import urllib.request
    import uuid

    boundary = uuid.uuid4().hex
    body_parts = []
    for k, v in fields.items():
        body_parts.append(
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
            f'{v}\r\n'
        )
    body_parts.append(
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="image"; filename="frame.jpg"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    )
    body_prefix = "".join(body_parts).encode()
    body_suffix = f'\r\n--{boundary}--\r\n'.encode()
    body = body_prefix + image_bytes + body_suffix

    req = urllib.request.Request(
        base_url + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    import urllib.error
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read())
    except Exception as exc:
        return {"error": str(exc)}


def run_validation(manifest_path: str, base_url: str) -> int:
    with open(manifest_path) as f:
        manifest = json.load(f)

    images_dir = Path(manifest.get("images_dir", "benchmarks/images"))
    teach_items = manifest.get("teach_items", [])
    scan_tests = manifest.get("scan_tests", [])

    print(f"\nManifest: {manifest.get('name', '?')}  ({len(teach_items)} teaches, {len(scan_tests)} scans)\n")

    # Teach phase
    print("Teaching items...")
    for item in teach_items:
        img_path = images_dir / item["image"]
        if not img_path.exists():
            print(f"  [SKIP] image not found: {img_path}")
            continue
        img_bytes = _load_image_bytes(img_path)
        result = _post_multipart(base_url, "/remember", img_bytes, {"prompt": item["label"]})
        status = "[PASS]" if result.get("success") else "[FAIL]"
        print(f"  {status}  teach {item['label']} from {item['image']}")

    # Scan phase
    print("\nValidating scan accuracy...")
    passed = 0
    failed = 0

    for test in scan_tests:
        img_path = images_dir / test["image"]
        if not img_path.exists():
            print(f"  [SKIP] image not found: {img_path}")
            continue

        img_bytes = _load_image_bytes(img_path)
        result = _post_multipart(base_url, "/scan", img_bytes, {})
        matches = result.get("matches", [])
        matched_labels = {m["label"]: m.get("similarity", 0.0) for m in matches}

        test_pass = True
        for expected in test.get("expected_matches", []):
            exp_label = expected["label"]
            min_sim = expected.get("min_similarity", 0.0)
            if exp_label not in matched_labels:
                print(f"  [FAIL]  {test['image']}: expected {exp_label!r} not found")
                test_pass = False
            elif matched_labels[exp_label] < min_sim:
                print(f"  [FAIL]  {test['image']}: {exp_label!r} sim={matched_labels[exp_label]:.3f} < {min_sim}")
                test_pass = False

        for no_match_label in test.get("expected_no_match", []):
            if no_match_label in matched_labels:
                print(f"  [FAIL]  {test['image']}: unexpected match {no_match_label!r}")
                test_pass = False

        if test_pass:
            print(f"  [PASS]  {test['image']}")
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\nResults: {passed}/{total} scan tests passed")
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-url", default="http://localhost:5050")
    args = parser.parse_args()
    sys.exit(run_validation(args.manifest, args.base_url))


if __name__ == "__main__":
    main()
