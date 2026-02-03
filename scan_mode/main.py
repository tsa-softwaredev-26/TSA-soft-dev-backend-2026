"""
Main script for visual memory demo.

This script demonstrates the end-to-end workflow:
1. Loads and embeds all images from demo_database/
2. Runs YOLOE detection on the query image (hardcoded path)
3. Crops detected objects
4. Embeds each cropped object
5. Performs linear search through database embeddings using cosine similarity
6. Prints the closest match or None if no match is found

# GPU SERVER MIGRATION TODO:
# 1. Add device='cuda' parameter to both YoloeDetector and ImageEmbedder __init__
# 2. Move models to device in __init__
# 3. Add embeddings cache (embeddings.npz) to avoid recomputing database on each run
# 4. Add batch processing for embedding multiple images at once (critical for GPU efficiency)
# 5. Replace linear search with HNSW index (w/ hnswlib, FAISS, or Milvus) ~ log(N) search time

TODO NEXT:
add images of cropped objects to demo_database/ 
"""
import os
from image_similarity.embed_image import ImageEmbedder
from object_detection.detect_objects import YoloeDetector
from utils import crop_object, find_closest_match, load_image, load_folder_images
from pathlib import Path
CURRENT_DIR = Path(__file__).parent

SIMILARITY_THRESHOLD = 0.3  # Tunable threshold for similarity matching (0-1, higher is more similar)

# Configuration for demo
DEMO_DB_DIR = CURRENT_DIR / "demo_database"
QUERY_IMAGE_PATH = CURRENT_DIR / "input_images" / "cropped_wallet2.png"


def embed_database_images(embedder, images):
    """    
    Args:
        embedder: ImageEmbedder instance
        images: List of (file_path, PIL Image) tuples
        
    Returns:
        List of (file_path, embedding) tuples
    """
    embeddings = []
    for file_path, img in images:
        embedding = embedder.embed(img)
        embeddings.append((file_path, embedding))
    return embeddings


def main():
    """Main workflow pipeline."""
    print("Initializing models...")
    # Initialize models
    embedder = ImageEmbedder()
    detector = YoloeDetector()
    
    print(f"Loading database images from '{DEMO_DB_DIR}'...")
    # Load and embed database images
    database_images = load_folder_images(DEMO_DB_DIR)
    
    if not database_images:
        print("Error: No images found in database. Please add images to demo_database/")
        return
    
    print(f"Embedding {len(database_images)} database images...")
    database_embeddings = embed_database_images(embedder, database_images)
    print(f"Database ready with {len(database_embeddings)} embeddings.")
    
    if not os.path.exists(QUERY_IMAGE_PATH):
        print(f"Error: Query image '{QUERY_IMAGE_PATH}' not found.")
        return
    
    print(f"Processing query image: {QUERY_IMAGE_PATH}")
    
    boxes, scores = detector.detect_all(QUERY_IMAGE_PATH)
    print(f"Detected {len(boxes)} objects")
    
    if not boxes:
        print("No objects detected in query image.")
        return
    
    query_image = load_image(QUERY_IMAGE_PATH)
    
    # Process each detected object
    print("\nProcessing detected objects...")
    best_path = None
    best_similarity = -1.0
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        cropped = crop_object(query_image, box)
        embedding = embedder.embed(cropped)
        match_path, match_similarity = find_closest_match(embedding, database_embeddings, SIMILARITY_THRESHOLD)

        if match_similarity > best_similarity:
            best_similarity = match_similarity
            best_path = match_path

    if best_path:
        print(f"\nClosest match: {best_path}")
        print(f"     Similarity: {best_similarity:.4f}")


if __name__ == "__main__":
    main() 
    