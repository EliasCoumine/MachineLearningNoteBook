"""
Training Script – Aerial House Segmentation
CEG4195/SEG4180 · uOttawa Winter 2026

Trains a UNet (ResNet-34 encoder, transfer learning) on binary aerial
building segmentation data prepared by dataset/prepare_dataset.py.

Key features
────────────
  • Combined BCE + Dice loss for class-imbalanced masks.
  • IoU and Dice tracked at every epoch (train + val).
  • Early stopping to avoid overfitting.
  • Checkpoint saved whenever validation IoU improves.

Usage
─────
    python model/train.py \
        --data_dir   data          \
        --ckpt_dir   checkpoints   \
        --epochs     30            \
        --batch_size 8             \
        --lr         1e-4          \
        --img_size   256

    # Resume from checkpoint:
    python model/train.py --resume checkpoints/best_model.pth ...
"""

import argparse
import os
import time
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.unet import build_transfer_unet


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AerialDataset(Dataset):
    """
    Loads image/mask pairs from the directory layout produced by
    dataset/prepare_dataset.py.

    Split directory:
        <split>/images/<id>.png
        <split>/masks/<id>.png
    """

    def __init__(self, split_dir: str, img_size: int = 256, augment: bool = False):
        self.img_dir  = Path(split_dir) / "images"
        self.mask_dir = Path(split_dir) / "masks"
        self.ids      = sorted(p.stem for p in self.img_dir.glob("*.png"))
        self.img_size = img_size

        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15, p=0.4),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.1, hue=0.05, p=0.4),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img  = np.array(Image.open(self.img_dir  / f"{name}.png").convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / f"{name}.png").convert("L"))

        # Binary mask: 0 or 1
        mask = (mask > 127).astype(np.float32)

        out  = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)   # (3,H,W), (1,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.

    BCE handles per-pixel accuracy; Dice optimises the overlap metric directly,
    which is more stable when building pixels are rare (class imbalance).
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth     = smooth
        self.bce        = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        prob   = torch.sigmoid(logits)
        inter  = (prob * targets).sum(dim=(2, 3))
        union  = prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice   = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss = dice.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Compute IoU and Dice score for a batch."""
    prob  = torch.sigmoid(logits).detach().cpu()
    pred  = (prob > threshold).float()
    tgt   = targets.detach().cpu()

    inter = (pred * tgt).sum(dim=(1, 2, 3))
    union = (pred + tgt).clamp(0, 1).sum(dim=(1, 2, 3))
    sum_  = pred.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))

    iou  = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    dice = ((2 * inter + 1e-6) / (sum_ + 1e-6)).mean().item()
    return {"iou": iou, "dice": dice}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        m = compute_metrics(logits, masks)
        total_loss += loss.item()
        total_iou  += m["iou"]
        total_dice += m["dice"]

    n = len(loader)
    return {"loss": total_loss / n, "iou": total_iou / n, "dice": total_dice / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks)

        m = compute_metrics(logits, masks)
        total_loss += loss.item()
        total_iou  += m["iou"]
        total_dice += m["dice"]

    n = len(loader)
    return {"loss": total_loss / n, "iou": total_iou / n, "dice": total_dice / n}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history: dict, save_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, title in zip(
        axes,
        ["loss", "iou", "dice"],
        ["Loss", "IoU", "Dice Score"],
    ):
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"],   label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved training curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets & loaders
    train_ds = AerialDataset(f"{args.data_dir}/train", args.img_size, augment=True)
    val_ds   = AerialDataset(f"{args.data_dir}/val",   args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    # Model – transfer learning with pretrained ResNet-34 encoder
    model = build_transfer_unet(
        encoder="resnet34",
        pretrained=True,
        in_channels=3,
        classes=1,
    ).to(device)

    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {k: [] for k in
               ["train_loss", "train_iou", "train_dice",
                "val_loss",   "val_iou",   "val_dice"]}

    best_iou    = 0.0
    no_improve  = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        for k in ["loss", "iou", "dice"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(va[k])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"[{elapsed:.0f}s]  "
            f"train loss={tr['loss']:.4f} iou={tr['iou']:.4f} dice={tr['dice']:.4f}  "
            f"val   loss={va['loss']:.4f} iou={va['iou']:.4f} dice={va['dice']:.4f}"
        )

        # Save best checkpoint
        if va["iou"] > best_iou:
            best_iou   = va["iou"]
            no_improve = 0
            ckpt_path  = ckpt_dir / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best val IoU={best_iou:.4f}  saved → {ckpt_path}")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    plot_curves(history, str(ckpt_dir / "training_curves.png"))
    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for aerial house segmentation.")
    parser.add_argument("--data_dir",   default="data",        help="Root data directory.")
    parser.add_argument("--ckpt_dir",   default="checkpoints", help="Checkpoint output directory.")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--img_size",   type=int,   default=256)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--patience",   type=int,   default=7,
                        help="Early-stopping patience (epochs).")
    parser.add_argument("--resume",     default=None,
                        help="Path to checkpoint to resume from.")
    main(parser.parse_args())
