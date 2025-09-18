# backend/main.py

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# backend/main.py

from .fingerprint import generate_fingerprint
from .search import search_image_in_dataset
from .deepfake_detector import detect_deepfake
from .alerts import generate_alert, generate_takedown_request
from . import db

# --------------------------
# Configuration & logging
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("protectai")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
DATASET_FOLDER = PROJECT_ROOT / "dataset"

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="ProtectAI", description="Agentic AI Security Assistant")


@app.on_event("startup")
def startup_event():
    """
    Ensure necessary folders and DB exist at startup.
    """
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "alerts").mkdir(parents=True, exist_ok=True)  # alerts module expects this
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    db.init_db()  # create tables if not present
    logger.info(f"Startup complete. Uploads: {UPLOAD_FOLDER}, Dataset: {DATASET_FOLDER}")


@app.get("/")
def home():
    return {"message": "Welcome to ProtectAI Security Assistant 🚀"}


def _safe_filename(original_filename: str) -> str:
    """
    Produce a safe, unique filename to avoid collisions and directory traversal.
    """
    base = os.path.basename(original_filename)
    unique = f"{uuid.uuid4().hex}_{base}"
    return unique


def _save_upload_to_disk(upload_file: UploadFile, dest_path: Path) -> None:
    """
    Save UploadFile to dest_path using stream (avoids loading entire file into memory).
    """
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        # ensure the SpooledTemporaryFile is closed
        try:
            upload_file.file.close()
        except Exception:
            pass


@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image → fingerprint it → search dataset → run deepfake detection
    → generate alert & takedown if needed → return structured result.
    """
    # Basic file validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    safe_name = _safe_filename(file.filename)
    file_location = UPLOAD_FOLDER / safe_name

    # Save file to disk
    try:
        _save_upload_to_disk(file, file_location)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # STEP 1: Generate fingerprint and store in DB
    fingerprint = generate_fingerprint(str(file_location))
    if isinstance(fingerprint, str) and fingerprint.startswith("Error"):
        # fingerprint function returned an error string
        logger.error("Fingerprinting error: %s", fingerprint)
        raise HTTPException(status_code=500, detail=fingerprint)

    try:
        db.save_fingerprint(safe_name, fingerprint)
    except Exception as e:
        logger.exception("Failed to save fingerprint to DB")

    # STEP 2: Search dataset (local folder)
    try:
        search_result = search_image_in_dataset(str(file_location), dataset_folder=str(DATASET_FOLDER))
    except Exception as e:
        logger.exception("Dataset search failed")
        search_result = {"error": f"Dataset search failed: {str(e)}"}

    # Normalize matches to a list (empty if none)
    matches = search_result.get("matches") if isinstance(search_result, dict) else None
    if isinstance(matches, list) and matches == ["No matches found"]:
        matches_list: List[str] = []
    elif isinstance(matches, list):
        matches_list = matches
    else:
        matches_list = []

    # STEP 3: Run deepfake detection
    try:
        detection_result = detect_deepfake(str(file_location))
    except Exception as e:
        logger.exception("Deepfake detection failed")
        return JSONResponse(status_code=500, content={"error": f"Deepfake detection failed: {str(e)}"})

    # If detector returned an error
    if isinstance(detection_result, dict) and "error" in detection_result:
        logger.error("Detector error: %s", detection_result["error"])
        return JSONResponse(status_code=500, content=detection_result)

    # STEP 4: Generate alert & takedown request (only if suspicious)
    # We consider suspicious if label says Fake or confidence is high enough.
    prediction = detection_result.get("prediction")
    confidence = detection_result.get("confidence")

    try:
        alert_path = generate_alert(
            image_file=safe_name,
            prediction=prediction,
            confidence=float(confidence) if confidence is not None else 0.0,
            matches=matches_list
        )
        takedown_path = generate_takedown_request(
            image_file=safe_name,
            prediction=prediction,
            matches=matches_list
        )
    except Exception as e:
        logger.exception("Failed to generate alert/takedown")
        alert_path = None
        takedown_path = None

    response = {
        "file": safe_name,
        "fingerprint": fingerprint,
        "search_result": search_result,
        "deepfake_result": detection_result,
        "alert_file": alert_path,
        "takedown_request": takedown_path,
    }

    return JSONResponse(status_code=200, content=response)