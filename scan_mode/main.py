"""
Scan Mode — full pipeline: detect all objects, match against database, narrate matches.

GPU SERVER MIGRATION TODO:
- device='cuda' in YoloeDetector, ImageEmbedder, DepthEstimator
- Embeddings cache (.npz) to avoid recomputing database each run
- Batch processing in embedder + detector
- HNSW index (hnswlib/FAISS) for similarity search
"""

import os
from pathlib import Path
from embed_image import ImageEmbedder
from object_detection import YoloeDetector
from depth_estimation import DepthEstimator
from utils import crop_object, find_match, load_image, load_folder_images, deduplicate_matches

CURRENT_DIR = Path(__file__).parent

SIMILARITY_THRESHOLD = 0.3

DEMO_DB_DIR      = CURRENT_DIR / "demo_database"
QUERY_IMAGE_PATH = CURRENT_DIR / "input_images" / "cropped_wallet2.png"

# TODO: replace with focal length passed from Android camera API
# f_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
# iPhone 15 Plus reference: f_mm=6.24, sensor_width=8.64mm
FOCAL_LENGTH_PX = 3094.0


def embed_database_images(embedder, images):
    embeddings = []
    for file_path, img in images:
        embedding = embedder.embed(img)
        embeddings.append((file_path, embedding))
    return embeddings


def main():
    print("Initializing models...")
    embedder  = ImageEmbedder()
    detector  = YoloeDetector()
    estimator = DepthEstimator(focal_length_px=FOCAL_LENGTH_PX)  

    print(f"Loading database from '{DEMO_DB_DIR}'...")
    database_images = load_folder_images(DEMO_DB_DIR)
    if not database_images:
        print("Error: No images in demo_database/")
        return

    print(f"Embedding {len(database_images)} database images...")
    database_embeddings = embed_database_images(embedder, database_images)

    if not os.path.exists(QUERY_IMAGE_PATH):
        print(f"Error: Query image not found at {QUERY_IMAGE_PATH}")
        return

    print(f"Processing query image: {QUERY_IMAGE_PATH}\n")
    query_image = load_image(str(QUERY_IMAGE_PATH))
    boxes, scores = detector.detect_all(str(QUERY_IMAGE_PATH))

    if not boxes:
        print("No objects detected.")
        return

    # Pass 1: find all matches before running depth (depth is expensive)
    print(f"Detected {len(boxes)} objects. Matching against database...")
    matches = []
    for box, score in zip(boxes, scores):
        cropped   = crop_object(query_image, box)
        embedding = embedder.embed(cropped)
        match_path, similarity = find_match(embedding, database_embeddings, SIMILARITY_THRESHOLD)
        if match_path:
            matches.append({"box": box, "label": Path(match_path).stem, "similarity": similarity})

    if not matches:
        print("No confident matches found.")
        return

    # Deduplicate matches - remove overlapping boxes that match the same object
    print(f"Deduplicating {len(matches)} matches...")
    matches = deduplicate_matches(matches, iou_threshold=0.5)
    print(f"After deduplication: {len(matches)} unique match(es).")

    if not matches:
        print("No unique matches after deduplication.")
        return

    #Pass 2: run depth once, reuse for all matches
    print(f"Found {len(matches)} match(es). Running depth estimation...")
    depth_map = estimator.estimate(query_image)

    print()
    print(f"Found {len(matches)} unique match(es).")
    print("=" * 50)
    print("FINAL MATCHES:")
    for m in matches:
        print(f"  {m['label']}: similarity={m['similarity']:.3f}, box={m['box']}")
    
    narrations = []
    for m in matches:
        distance_ft = estimator.get_depth_at_bbox(depth_map, m["box"])
        direction   = estimator.get_direction(m["box"], query_image.width)
        narration   = estimator.build_narration(m["label"], direction, distance_ft, m["similarity"])

        if narration:
            narrations.append(narration)
            print(f"  match={m['label']}  sim={m['similarity']:.3f}  dist={distance_ft:.2f}ft  dir={direction}")
            print(f"  → {narration}\n")

    if not narrations:
        print("No confident matches found.")
    else:
        print("=" * 50)
        print("NARRATION OUTPUT:")
        for n in narrations:
            print(f"  {n}")


if __name__ == "__main__":
    main()
