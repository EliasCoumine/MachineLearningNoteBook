"""
Flask API – Aerial House Segmentation
Lab 2 · CEG4195/SEG4180 Applied ML · uOttawa

Secrets are loaded from .env via python-dotenv (never hard-coded).
Exposes:
  GET  /          – health check
  POST /segment   – binary house-segmentation mask for a given aerial image
"""

import os
import io
import base64
import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ── 1. Secrets injection ──────────────────────────────────────────────────────
# load_dotenv reads the .env file (if present) and injects values into the
# process environment.  Container deployments can skip the file and pass vars
# directly via -e / --env-file flags.
load_dotenv()

LOG_LEVEL  = os.getenv("LOG_LEVEL",   "INFO")
MODEL_PATH = os.getenv("MODEL_PATH",  "/app/checkpoints/best_model.pth")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
API_KEY    = os.getenv("API_KEY",     "")          # empty → auth disabled

# ── 2. Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── 3. Model loading ──────────────────────────────────────────────────────────
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms as T

app = Flask(__name__)

def _load_model(path: str, device: torch.device):
    """Load a trained UNet checkpoint; fall back to random weights for demo."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # weights come from the checkpoint
        in_channels=3,
        classes=1,
    )
    if os.path.isfile(path):
        logger.info("Loading checkpoint from %s", path)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    else:
        logger.warning(
            "Checkpoint not found at %s — using untrained weights (demo mode).", path
        )
    model.to(device).eval()
    return model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)

logger.info("Loading segmentation model …")
try:
    model = _load_model(MODEL_PATH, DEVICE)
    logger.info("Model ready.")
except Exception as exc:
    logger.error("Model load failed: %s", exc)
    model = None

# Pre-processing transform (must match training pipeline)
_preprocess = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── 4. Helper utilities ───────────────────────────────────────────────────────

def _require_api_key():
    """Return an error response if API_KEY is set and not provided correctly."""
    if not API_KEY:
        return None
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _image_from_request() -> Image.Image:
    """Accept an image as multipart file OR as base64-encoded JSON field."""
    if "file" in request.files:
        return Image.open(request.files["file"].stream).convert("RGB")
    data = request.get_json(silent=True) or {}
    if "image_b64" in data:
        raw = base64.b64decode(data["image_b64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError("No image supplied. Send 'file' (multipart) or 'image_b64' (JSON).")


def _mask_to_b64(mask: np.ndarray) -> str:
    """Convert a binary uint8 mask (H×W) to a base64-encoded PNG string."""
    img = Image.fromarray((mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _compute_metrics(pred: np.ndarray, gt: np.ndarray | None):
    """Return IoU and Dice if ground truth is supplied, else None."""
    if gt is None:
        return None
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    inter  = (pred_b & gt_b).sum()
    union  = (pred_b | gt_b).sum()
    iou   = float(inter / union) if union > 0 else 1.0
    dice  = float(2 * inter / (pred_b.sum() + gt_b.sum())) \
            if (pred_b.sum() + gt_b.sum()) > 0 else 1.0
    return {"iou": round(iou, 4), "dice": round(dice, 4)}


# ── 5. Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    """Health-check / info endpoint."""
    return jsonify({
        "status":    "running",
        "model":     "UNet (ResNet-34 encoder) – Aerial House Segmentation",
        "device":    str(DEVICE),
        "endpoints": {
            "segment": "POST /segment",
        },
    })


@app.route("/segment", methods=["POST"])
def segment():
    """
    Segment houses in an aerial image.

    Request (multipart/form-data):
        file        – aerial image (JPEG / PNG)
        mask_b64    – optional ground-truth mask (PNG, base64) for metric computation

    Request (application/json):
        image_b64   – base64-encoded aerial image
        mask_b64    – optional ground-truth mask (base64) for metric computation

    Response (JSON):
        mask_b64    – predicted binary mask as base64 PNG
        metrics     – {iou, dice} if ground truth was provided, else null
    """
    # Auth guard
    err = _require_api_key()
    if err:
        return err

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        image = _image_from_request()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error("Image decode error: %s", exc)
        return jsonify({"error": "Could not decode image"}), 400

    # Optional ground-truth mask
    gt_mask = None
    data = request.get_json(silent=True) or {}
    raw_gt = request.form.get("mask_b64") or data.get("mask_b64")
    if raw_gt:
        try:
            gt_bytes = base64.b64decode(raw_gt)
            gt_img   = Image.open(io.BytesIO(gt_bytes)).convert("L")
            gt_arr   = np.array(gt_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
            gt_mask  = (gt_arr > 127).astype(np.uint8)
        except Exception as exc:
            logger.warning("Could not decode ground-truth mask: %s", exc)

    # Inference
    tensor = _preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)                          # (1, 1, H, W)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()

    pred_mask = (prob > 0.5).astype(np.uint8)

    metrics = _compute_metrics(pred_mask, gt_mask)
    logger.info(
        "Segmented image %s → house pixels: %d  metrics: %s",
        getattr(image, "filename", "upload"),
        pred_mask.sum(),
        metrics,
    )

    return jsonify({
        "mask_b64": _mask_to_b64(pred_mask),
        "metrics":  metrics,
        "input_size":  list(image.size),
        "output_size": [IMAGE_SIZE, IMAGE_SIZE],
    })


# ── 6. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
