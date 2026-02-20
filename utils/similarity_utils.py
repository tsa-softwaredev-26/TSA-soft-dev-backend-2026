"""Utility functions for similarity matching and search."""
from typing import Optional, Tuple, List
import torch
import torch.nn as nn


def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two 2D embeddings.
    
    Args:
        embedding1: First embedding tensor of shape (1, dim)
        embedding2: Second embedding tensor of shape (1, dim)
        
    Returns:
        Cosine similarity score as tensor
    """
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
    return cos_sim(embedding1, embedding2)


def find_closest_match(
    query_embedding: torch.Tensor, 
    database_embeddings: List[Tuple[str, torch.Tensor]], 
    threshold: float
) -> Tuple[Optional[str], float]:
    """
    Find the closest matching image in the database using linear search.
    
    Args:
        query_embedding: Query embedding tensor of shape (1, dim)
        database_embeddings: List of (image_path, embedding) tuples
        threshold: Minimum similarity threshold to consider a match (0-1)
        
    Returns:
        Tuple of (best_path, best_similarity) or (None, 0.0) if no match above threshold
        
    Note:
        TODO: Batch similarity computations.
        Faster search like HNSW would only slightly improve speed since database will never be >5k images
    """
    if not database_embeddings:
        return None, 0.0
    
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    best_similarity = -1.0
    best_path = None

    for db_path, db_embedding in database_embeddings:
        sim = cosine_similarity(query_embedding, db_embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_path = db_path

    if best_similarity < threshold:
        return None, 0.0

    return best_path, best_similarity.item()
