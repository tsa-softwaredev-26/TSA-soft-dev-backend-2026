"""
Main script for visual memory demo.

This script demonstrates the end-to-end workflow:
1. Loads and embeds all images from demo_database/
2. Runs YOLOE detection on the query image (hardcoded path)
3. Crops detected objects
4. Embeds each cropped object
5. Performs linear search through database embeddings using cosine similarity
6. Prints the closest match or None if no match is found
"""

import os
from PIL import Image
from ImageSimilarity.embed_image import ImageEmbedder
from ObjectDetection.detect_objects import YoloeDetector, crop_object


# Configuration constants
DEMO_DB_DIR = "demo_database"
QUERY_IMAGE_PATH = "ObjectDetection/images/messy-desk.webp"
SIMILARITY_THRESHOLD = 0.5  # Tunable threshold for similarity matching


def load_database_images(folder_path):
    """
    Load all images from the database folder.
    
    Args:
        folder_path: Path to folder containing database images
        
    Returns:
        List of tuples (image_path, PIL Image)
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    images = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Database folder '{folder_path}' not found.")
        return images
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename.lower())
            if ext in image_extensions:
                try:
                    img = Image.open(file_path)
                    images.append((file_path, img))
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
    
    return images


def embed_database_images(embedder, images):
    """
    Embed all database images.
    
    Args:
        embedder: ImageEmbedder instance
        images: List of (image_path, PIL Image) tuples
        
    Returns:
        List of (image_path, embedding_tensor) tuples
        
    Note:
        TODO: For future server version, implement batch processing to embed 
        all database images in batches for efficiency.
    """
    embeddings = []
    for image_path, image in images:
        embedding = embedder.embed(image)
        embeddings.append((image_path, embedding))
    return embeddings


def find_closest_match(query_embedding, database_embeddings, threshold):
    """
    Find the closest matching image in the database using cosine similarity.
    
    Args:
        query_embedding: Embedding tensor for the query image
        database_embeddings: List of (image_path, embedding_tensor) tuples
        threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (best_match_path, best_similarity) or (None, None) if no match found
        
    Note:
        TODO: Replace linear search with HNSW index (via hnswlib, FAISS, or Milvus) 
        for fast ANN search at scale.
    """
    best_similarity = -1.0
    best_match_path = None
    
    for db_path, db_embedding in database_embeddings:
        similarity = ImageEmbedder.cosine_similarity(query_embedding, db_embedding)
        similarity_value = similarity.item()
        
        if similarity_value > best_similarity:
            best_similarity = similarity_value
            best_match_path = db_path
    
    if best_similarity < threshold:
        return None, None
    
    return best_match_path, best_similarity


def main():
    """Main workflow pipeline."""
    print("Initializing models...")
    # Initialize models
    embedder = ImageEmbedder()
    detector = YoloeDetector()
    
    print(f"Loading database images from '{DEMO_DB_DIR}'...")
    # Load and embed database images
    database_images = load_database_images(DEMO_DB_DIR)
    
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
    
    # Run YOLOE detection on query image
    boxes, scores, class_ids = detector.detect(QUERY_IMAGE_PATH)
    print(f"Detected {len(boxes)} objects")
    
    if not boxes:
        print("No objects detected in query image.")
        return
    
    # Open query image for cropping
    query_image = Image.open(QUERY_IMAGE_PATH)
    
    # Process each detected object
    for idx, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box
        print(f"\nObject {idx + 1}: Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
              f"Score={score:.3f}, Class={class_id}")
        
        # Crop the detected object
        cropped_object = crop_object(query_image, x1, y1, x2, y2)
        
        # Embed the cropped object
        object_embedding = embedder.embed(cropped_object)
        
        # Find closest match in database
        match_path, match_similarity = find_closest_match(
            object_embedding, 
            database_embeddings, 
            SIMILARITY_THRESHOLD
        )
        
        if match_path:
            print(f"  -> Closest match: {match_path} (similarity: {match_similarity:.4f})")
        else:
            print(f"  -> No match found (best similarity: {match_similarity:.4f} < threshold: {SIMILARITY_THRESHOLD})")


if __name__ == "__main__":
    main()
