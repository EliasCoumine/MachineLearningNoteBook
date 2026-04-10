"""
Pixel Mask Generation – Week 7 Code
CEG4195/SEG4180 · uOttawa Winter 2026

Generates binary segmentation masks from aerial imagery where:
  pixel = 1  →  house / building footprint
  pixel = 0  →  background (roads, vegetation, water, …)

Two approaches are implemented:

  1. HSV colour-thresholding  – fast, zero external models, works when
     rooftops have a distinct colour palette (e.g. red/orange tiles,
     grey concrete).

  2. SAM-assisted generation  – uses Meta's Segment Anything Model to
     produce high-quality masks with minimal supervision.  Requires
     segment-anything and a SAM checkpoint (ViT-B or ViT-H).

Usage
-----
    python pixel_mask_generation.py \
        --input_dir  data/raw/images \
        --output_dir data/raw/masks  \
        --method     threshold        # or 'sam'
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Colour-threshold approach (no external model required)
# ─────────────────────────────────────────────────────────────────────────────

def _threshold_mask(
    img_bgr: np.ndarray,
    min_area: int = 200,
) -> np.ndarray:
    """
    Generate a binary building mask using HSV colour thresholding.

    The heuristic targets typical rooftop colours found in aerial imagery:
      - Grey / dark-grey  (concrete rooftops)
      - Red / terracotta  (tile rooftops)
      - White / beige     (flat rooftops)

    Parameters
    ----------
    img_bgr   : BGR image array (H × W × 3, uint8).
    min_area  : Morphological opening removes connected components smaller
                than this (pixels) to clean up noise.

    Returns
    -------
    mask : Binary uint8 array (H × W), 1 = building, 0 = background.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # -- Grey / dark-grey rooftops (low saturation)
    grey_mask = cv2.inRange(
        hsv,
        np.array([0,   0,  50]),    # lower: any hue, low sat, moderate val
        np.array([180, 50, 210]),   # upper: any hue, low sat, high val
    )

    # -- Red / terracotta tile rooftops (two hue ranges in HSV wrap-around)
    red_lo = cv2.inRange(hsv, np.array([0,  60,  60]), np.array([15, 255, 220]))
    red_hi = cv2.inRange(hsv, np.array([160, 60, 60]), np.array([180, 255, 220]))
    red_mask = cv2.bitwise_or(red_lo, red_hi)

    # Combine all building-colour candidates
    combined = cv2.bitwise_or(grey_mask, red_mask)

    # Morphological cleanup: close small holes, remove tiny blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed,   cv2.MORPH_OPEN,  kernel, iterations=1)

    # Remove connected components below min_area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    clean = np.zeros_like(opened)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255

    return (clean > 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SAM-assisted approach  (Segment Anything Model – Meta AI)
# ─────────────────────────────────────────────────────────────────────────────

def _sam_mask(
    img_rgb: np.ndarray,
    sam_checkpoint: str,
    model_type: str = "vit_b",
    iou_thresh: float = 0.88,
    stability_thresh: float = 0.92,
    min_area: int = 200,
) -> np.ndarray:
    """
    Generate a binary building mask using SAM's automatic mask generator.

    SAM (Segment Anything Model, Kirillov et al. 2023) segments *everything*
    in an image.  We filter the resulting proposals to keep only those with
    high IoU / stability scores and a rectangular aspect ratio, which is
    characteristic of building footprints.

    Parameters
    ----------
    img_rgb          : RGB image array (H × W × 3, uint8).
    sam_checkpoint   : Path to the downloaded SAM checkpoint (.pth file).
                       Download ViT-B:  sam_vit_b_01ec64.pth (~375 MB)
                       Download ViT-H:  sam_vit_h_4b8939.pth (~2.4 GB)
    model_type       : 'vit_b' | 'vit_l' | 'vit_h'
    iou_thresh       : Discard proposals with predicted IoU below this.
    stability_thresh : Discard proposals with stability score below this.
    min_area         : Discard tiny proposals (pixels).

    Returns
    -------
    mask : Binary uint8 array (H × W), 1 = building, 0 = background.
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "segment-anything is not installed.\n"
            "Install it with:  pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=iou_thresh,
        stability_score_thresh=stability_thresh,
        min_mask_region_area=min_area,
    )

    masks = generator.generate(img_rgb)

    H, W = img_rgb.shape[:2]
    building_mask = np.zeros((H, W), dtype=np.uint8)

    for m in masks:
        seg        = m["segmentation"]          # bool (H × W)
        area       = m["area"]
        bbox       = m["bbox"]                  # [x, y, w, h]
        pred_iou   = m["predicted_iou"]
        stability  = m["stability_score"]

        if area < min_area:
            continue
        if pred_iou < iou_thresh or stability < stability_thresh:
            continue

        # Prefer roughly rectangular regions (buildings tend to be compact)
        bw, bh = bbox[2], bbox[3]
        if bw == 0 or bh == 0:
            continue
        aspect = max(bw, bh) / min(bw, bh)
        if aspect > 6:          # very elongated → likely a road or shadow
            continue

        building_mask[seg] = 1

    return building_mask


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_mask(
    image_path: str,
    method: str = "threshold",
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_b",
) -> np.ndarray:
    """
    Generate a binary pixel mask for a single aerial image.

    Parameters
    ----------
    image_path     : Path to the input aerial image.
    method         : 'threshold' (default) or 'sam'.
    sam_checkpoint : Required when method='sam'.
    sam_model_type : SAM model variant to use.

    Returns
    -------
    mask : Binary uint8 array (H × W).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if method == "threshold":
        return _threshold_mask(img_bgr)

    if method == "sam":
        if sam_checkpoint is None:
            raise ValueError("sam_checkpoint must be provided when method='sam'.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return _sam_mask(img_rgb, sam_checkpoint, sam_model_type)

    raise ValueError(f"Unknown method '{method}'. Choose 'threshold' or 'sam'.")


def process_directory(
    input_dir: str,
    output_dir: str,
    method: str = "threshold",
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_b",
    extensions: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> None:
    """
    Run mask generation on every image in input_dir and save PNG masks to
    output_dir, preserving filenames.
    """
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in in_path.iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images.  Method: {method}")

    for i, img_path in enumerate(image_files, start=1):
        mask = generate_mask(
            str(img_path),
            method=method,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
        )
        # Save as PNG with same stem
        out_file = out_path / (img_path.stem + "_mask.png")
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(out_file)
        print(f"  [{i:4d}/{len(image_files)}]  {img_path.name}  →  {out_file.name}")

    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate binary pixel masks from aerial imagery (Week 7)."
    )
    parser.add_argument("--input_dir",       required=True,  help="Directory of input aerial images.")
    parser.add_argument("--output_dir",      required=True,  help="Directory to save binary masks.")
    parser.add_argument("--method",          default="threshold",
                        choices=["threshold", "sam"],
                        help="Mask generation method (default: threshold).")
    parser.add_argument("--sam_checkpoint",  default=None,   help="Path to SAM .pth checkpoint (required for --method sam).")
    parser.add_argument("--sam_model_type",  default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model variant (default: vit_b).")
    args = parser.parse_args()

    process_directory(
        input_dir      = args.input_dir,
        output_dir     = args.output_dir,
        method         = args.method,
        sam_checkpoint = args.sam_checkpoint,
        sam_model_type = args.sam_model_type,
    )
