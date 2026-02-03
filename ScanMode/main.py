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
from PIL import Image
from ImageSimilarity.embed_image import ImageEmbedder
from ObjectDetection.detect_objects import YoloeDetector
from utils import crop_object, find_closest_match


# Configuration constants
DEMO_DB_DIR = "demo_database"
QUERY_IMAGE_PATH = "/Users/joeroche/Developer/VisualMemory/ScanMode/input_images/cropped_wallet2.png"
SIMILARITY_THRESHOLD = 0.3  # Tunable threshold for similarity matching (0-1, higher is more similar)


def load_image(file_path):
    """
    Load a single image from a given path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        PIL Image object or None if loading fails
    """
    if not os.path.isfile(file_path):
        print(f"Warning: File '{file_path}' does not exist.")
        return None
    
    try:
        img = Image.open(file_path).convert("RGB")
        return img
    except Exception as e:
        print(f"Warning: Could not load '{file_path}': {e}")
        return None


def load_folder_images(folder_path):
    """
    Load all images from a folder.
    
    Args:
        folder_path: Folder containing images        
    Returns:
        List of tuples: (file_path, PIL Image)
    """

    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' not found.")
        return []

    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        _, ext = os.path.splitext(filename.lower())
        img = load_image(file_path)
        if img is not None:
            images.append((file_path, img))
    return images


def embed_database_images(embedder, images):
    """
    Embed all database images.
    
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
    
    # Open query image for cropping
    query_image = Image.open(QUERY_IMAGE_PATH)
    
    # Process each detected object
    print("\nProcessing detected objects...")
    best_path = None
    best_similarity = -1.0
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        cropped = crop_object(query_image, box)
        embedding = embedder.embed(cropped)
        match_path, match_similarity = find_closest_match(embedding, database_embeddings, SIMILARITY_THRESHOLD)

        # Update best match
        if match_similarity > best_similarity:
            best_similarity = match_similarity
            best_path = match_path

    if best_path:
        print(f"  -> Closest match: {best_path}")
        print(f"     Similarity: {best_similarity:.4f}")


if __name__ == "__main__":
    main() 
    