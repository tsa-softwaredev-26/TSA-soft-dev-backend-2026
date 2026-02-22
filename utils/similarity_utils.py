"""Utility functions for similarity matching and search."""
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
from pathlib import Path


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


def find_match(
    query_embedding: torch.Tensor, 
    database_embeddings: List[Tuple[str, torch.Tensor]], 
    threshold: float
) -> Tuple[Optional[str], float]:
    """
    Find a matching image in the database using linear search.
    
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
        sim_val = sim.item()
        if sim > best_similarity:
            if sim >= threshold:
                print(f"{Path(db_path).stem}: {sim_val:.3f}")
            best_similarity = sim
            best_path = db_path

    if best_similarity < threshold:
        return None, 0.0

    return best_path, best_similarity.item()


def iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def deduplicate_matches(matches: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Remove duplicate matches that refer to the same physical object.
    
    Strategy:
    1. Sort matches by similarity (descending) - keep best matches first
    2. For each match, check against already-kept matches:
       a. If they match the SAME label:
          - Check if one box is contained within the other (>80% overlap)
          - OR if they have ANY IoU overlap (IoU > 0.1)
          → If either is true, skip (same physical object)
       b. If different labels but high overlap (IoU > iou_threshold), skip (overlapping objects)
    3. Otherwise, keep it
    
    This prevents multiple narrations of the same object when:
    - Detector creates multiple overlapping boxes for one object
    - Detector creates one huge box and one precise box for the same object
    - Multiple similar boxes all match to the same database image
    
    Args:
        matches: List of match dictionaries with keys 'box', 'label', 'similarity'
        iou_threshold: Maximum allowed IoU for different objects (default: 0.5)
        
    Returns:
        Deduplicated list of matches, sorted by similarity
    """
    if not matches:
        return []
    
    # Sort by similarity descending - keep best matches first
    sorted_matches = sorted(matches, key=lambda m: m["similarity"], reverse=True)
    
    kept_matches = []
    
    for match in sorted_matches:
        # Check if this box overlaps significantly with any already-kept box
        is_duplicate = False
        for kept_match in kept_matches:
            overlap_iou = iou(match["box"], kept_match["box"])
            
            # If they match the same database object, check for containment or overlap
            if match["label"] == kept_match["label"]:
                # Calculate intersection area
                x1_inter = max(match["box"][0], kept_match["box"][0])
                y1_inter = max(match["box"][1], kept_match["box"][1])
                x2_inter = min(match["box"][2], kept_match["box"][2])
                y2_inter = min(match["box"][3], kept_match["box"][3])
                
                if x2_inter > x1_inter and y2_inter > y1_inter:
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    
                    # Calculate areas of both boxes
                    match_area = (match["box"][2] - match["box"][0]) * (match["box"][3] - match["box"][1])
                    kept_area = (kept_match["box"][2] - kept_match["box"][0]) * (kept_match["box"][3] - kept_match["box"][1])
                    
                    # Check if one box is mostly contained in the other (>80% containment)
                    match_contained = inter_area / match_area > 0.8 if match_area > 0 else False
                    kept_contained = inter_area / kept_area > 0.8 if kept_area > 0 else False
                    
                    if match_contained or kept_contained or overlap_iou > 0.1:
                        is_duplicate = True
                        break
            
    
        
        if not is_duplicate:
            kept_matches.append(match)
    
    return kept_matches
