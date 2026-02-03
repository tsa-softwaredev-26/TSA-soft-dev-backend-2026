"""
Utility functions for image processing and similarity matching.
"""

import torch
import torch.nn as nn


def crop_object(image, box):
    """
    Crop an object from an image given a bounding box.
    
    Args:
        image: PIL Image object
        box: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.width), min(y2, image.height)
    return image.crop((x1, y1, x2, y2))


def cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two 2D embeddings.
    """
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
    return cos_sim(embedding1, embedding2)

def find_closest_match(query_embedding, database_embeddings, threshold):
    """
    Find the closest matching image in the database using linear search.
    
    Args:
        query_embedding: Query embedding tensor
        database_embeddings: List of (image_path, embedding) tuples
        threshold: Minimum similarity threshold to consider a match
        
    Returns:
        Tuple of (best_path, best_similarity) or (None, None) if no match above threshold
        
    # TODO: Replace linear search with HNSW index (via hnswlib, FAISS, or Milvus)
    # for fast approximate nearest neighbor search at scale
    """
    best_similarity = -1.0
    best_path = None

    for db_path, db_embedding in database_embeddings:
        sim = cosine_similarity(query_embedding, db_embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_path = db_path

    if best_similarity < threshold:
        return None, 0

    return best_path, best_similarity.item()

