# backend/alerts.py

import os
from datetime import datetime
from typing import List, Optional

# Ensure the alerts folder exists
ALERTS_FOLDER = "alerts"
os.makedirs(ALERTS_FOLDER, exist_ok=True)


def generate_alert(
    image_file: str, 
    prediction: str, 
    confidence: float, 
    matches: Optional[List[str]] = None
) -> str:
    """
    Generate a user-friendly alert report and save it to the alerts/ folder.

    Args:
        image_file (str): Name or path of the image analyzed.
        prediction (str): Deepfake prediction result.
        confidence (float): Confidence of the prediction (0.0 - 1.0).
        matches (List[str], optional): List of matched images or URLs.

    Returns:
        str: Path to the saved alert file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alert_filename = f"alert_{timestamp}.txt"
    alert_path = os.path.join(ALERTS_FOLDER, alert_filename)

    matches_text = ", ".join(matches) if matches else "None"

    with open(alert_path, "w") as f:
        f.write("==== ProtectAI Alert ====\n")
        f.write(f"Image: {image_file}\n")
        f.write(f"Deepfake Prediction: {prediction} (Confidence: {confidence:.2f})\n")
        f.write(f"Matches Found: {matches_text}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=========================\n")

    return alert_path


def generate_takedown_request(
    image_file: str, 
    prediction: str, 
    matches: Optional[List[str]] = None
) -> str:
    """
    Generate a takedown request template for reporting misuse and save it.

    Args:
        image_file (str): Name or path of the image misused.
        prediction (str): Deepfake prediction result.
        matches (List[str], optional): List of matched images or URLs.

    Returns:
        str: Path to the saved takedown request file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    takedown_filename = f"takedown_{timestamp}.txt"
    takedown_path = os.path.join(ALERTS_FOLDER, takedown_filename)

    matches_text = ", ".join(matches) if matches else "None"

    with open(takedown_path, "w") as f:
        f.write("To: Abuse/Privacy Team\n")
        f.write("Subject: Takedown Request - Misuse of Personal Image\n\n")
        f.write("Dear Team,\n\n")
        f.write(
            f"I am writing to request the immediate removal of content that misuses my personal image ({image_file}).\n"
        )
        f.write(f"Detection Result: {prediction}\n")
        if matches:
            f.write(f"Matched Files: {matches_text}\n")
        f.write(
            "\nThis violates my privacy rights. Please take urgent action.\n\nSincerely,\nUser\n"
        )
        f.write(f"Timestamp: {timestamp}\n")

    return takedown_path
