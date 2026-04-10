"""
Evaluation Script – Aerial House Segmentation
CEG4195/SEG4180 · uOttawa Winter 2026

Loads the best checkpoint and evaluates it on the held-out test set.
Produces:
  • Per-image and aggregate IoU / Dice scores printed to stdout.
  • Visualisation grid (image | ground-truth | prediction) saved as PNG.

Usage
─────
    python model/evaluate.py \
        --data_dir   data                          \
        --ckpt_path  checkpoints/best_model.pth    \
        --img_size   256                           \
        --batch_size 8                             \
        --out_dir    results
"""

import argparse
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from model.unet import build_transfer_unet
from model.train import AerialDataset


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection-over-Union for two binary arrays."""
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union > 0 else 1.0


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice / F1 score for two binary arrays."""
    inter  = (pred & gt).sum()
    total  = pred.sum() + gt.sum()
    return float(2 * inter / total) if total > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation and convert to uint8 HWC."""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def save_prediction_grid(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    ious: list,
    dices: list,
    save_path: str,
    n_cols: int = 4,
) -> None:
    """
    Save a grid of (image | ground-truth mask | predicted mask) rows.

    Parameters
    ----------
    imgs, masks, preds : Batch tensors (B, C, H, W) or (B, 1, H, W).
    ious, dices        : Per-sample metric lists.
    save_path          : Output PNG path.
    n_cols             : Number of samples per row (each sample = 3 panels).
    """
    B      = imgs.shape[0]
    n_rows = (B + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(n_cols * 4, n_rows * 9))
    axes = np.array(axes).reshape(n_rows * 3, n_cols)

    for i in range(B):
        row_base = (i // n_cols) * 3
        col      =  i  % n_cols

        ax_img   = axes[row_base,     col]
        ax_gt    = axes[row_base + 1, col]
        ax_pred  = axes[row_base + 2, col]

        ax_img.imshow(_denorm(imgs[i]))
        ax_img.set_title("Image", fontsize=9)
        ax_img.axis("off")

        gt_arr = masks[i, 0].cpu().numpy()
        ax_gt.imshow(gt_arr, cmap="gray", vmin=0, vmax=1)
        ax_gt.set_title("Ground Truth", fontsize=9)
        ax_gt.axis("off")

        pr_arr = preds[i, 0].cpu().numpy()
        ax_pred.imshow(pr_arr, cmap="gray", vmin=0, vmax=1)
        ax_pred.set_title(
            f"Pred  IoU={ious[i]:.3f}  Dice={dices[i]:.3f}", fontsize=8
        )
        ax_pred.axis("off")

    # Hide unused axes
    for i in range(B, n_rows * n_cols):
        row_base = (i // n_cols) * 3
        col      =  i  % n_cols
        for r in range(3):
            axes[row_base + r, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()
    print(f"Prediction grid saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    test_ds     = AerialDataset(f"{args.data_dir}/test", args.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)
    print(f"Test samples: {len(test_ds)}")

    # Model
    model = build_transfer_unet(encoder="resnet34", pretrained=False).to(device)
    ckpt  = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_iou, all_dice = [], []
    first_batch_done  = False

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(test_loader):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()

            # Per-sample metrics
            batch_ious, batch_dices = [], []
            for i in range(imgs.shape[0]):
                p = preds[i, 0].cpu().numpy().astype(bool)
                g = masks[i, 0].cpu().numpy().astype(bool)
                batch_ious.append(iou_score(p, g))
                batch_dices.append(dice_score(p, g))

            all_iou.extend(batch_ious)
            all_dice.extend(batch_dices)

            # Save prediction grid for the first two batches
            if not first_batch_done or batch_idx == 1:
                grid_path = out_dir / f"predictions_batch{batch_idx:02d}.png"
                save_prediction_grid(
                    imgs.cpu(), masks.cpu(), preds.cpu(),
                    batch_ious, batch_dices,
                    str(grid_path),
                )
                first_batch_done = True

    # Aggregate results
    mean_iou  = float(np.mean(all_iou))
    mean_dice = float(np.mean(all_dice))
    std_iou   = float(np.std(all_iou))
    std_dice  = float(np.std(all_dice))

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print(f"  Samples : {len(all_iou)}")
    print(f"  IoU     : {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"  Dice    : {mean_dice:.4f} ± {std_dice:.4f}")
    print("=" * 50)

    # Save IoU distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(all_iou,  bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(mean_iou, color="red", linestyle="--", label=f"Mean={mean_iou:.3f}")
    axes[0].set_title("IoU Distribution (Test Set)")
    axes[0].set_xlabel("IoU"); axes[0].legend()

    axes[1].hist(all_dice, bins=20, color="seagreen",  edgecolor="white")
    axes[1].axvline(mean_dice, color="red", linestyle="--", label=f"Mean={mean_dice:.3f}")
    axes[1].set_title("Dice Score Distribution (Test Set)")
    axes[1].set_xlabel("Dice Score"); axes[1].legend()

    plt.tight_layout()
    dist_path = out_dir / "metric_distributions.png"
    plt.savefig(dist_path, dpi=120)
    plt.close()
    print(f"Metric distributions saved → {dist_path}")

    return {"iou": mean_iou, "dice": mean_dice}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on test set.")
    parser.add_argument("--data_dir",   default="data",                          help="Root data directory.")
    parser.add_argument("--ckpt_path",  default="checkpoints/best_model.pth",    help="Path to trained checkpoint.")
    parser.add_argument("--img_size",   type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_dir",    default="results",                        help="Directory for output visualisations.")
    main(parser.parse_args())
