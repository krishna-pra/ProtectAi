# backend/search.py

import os
from typing import List, Dict
from imagehash import hex_to_hash
from backend.fingerprint import generate_fingerprint

# Optional: import your database interface if available
# from backend import db  


def search_image_in_dataset(image_path: str, dataset_folder: str = "dataset") -> Dict:
    """
    Search for exact fingerprint matches in a dataset folder.

    Args:
        image_path (str): Path to the query image.
        dataset_folder (str): Path to dataset folder containing images.

    Returns:
        dict: Query filename and list of matches (or error message).
    """
    query_fingerprint = generate_fingerprint(image_path)
    if "Error" in query_fingerprint:
        return {"error": query_fingerprint}

    if not os.path.exists(dataset_folder):
        return {"error": f"Dataset folder '{dataset_folder}' not found."}

    matches: List[str] = []
    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        if os.path.isfile(file_path):
            dataset_fingerprint = generate_fingerprint(file_path)
            if dataset_fingerprint == query_fingerprint:
                matches.append(file)

    return {
        "query": os.path.basename(image_path),
        "matches": matches if matches else ["No matches found"]
    }


def search_similar(hash_value: str, threshold: int = 5, all_fingerprints: List[Dict] = None) -> List[Dict]:
    """
    Search for similar fingerprints using Hamming distance.

    Args:
        hash_value (str): Query image fingerprint (SHA256 hex).
        threshold (int): Maximum Hamming distance allowed (lower = stricter match).
        all_fingerprints (List[Dict], optional): List of dicts with {'filename', 'hash_value'}.

    Returns:
        List[Dict]: List of matches with filename, hash, and distance.
    """
    if all_fingerprints is None:
        # If db is available, replace this line with db.get_fingerprints()
        all_fingerprints = []

    target_hash = hex_to_hash(hash_value)
    results: List[Dict] = []

    for fp in all_fingerprints:
        db_hash = hex_to_hash(fp["hash_value"])
        distance = target_hash - db_hash  # Hamming distance
        if distance <= threshold:
            results.append({
                "filename": fp["filename"],
                "hash": fp["hash_value"],
                "distance": distance
            })

    return results