"""Train the U-Net on EchoNet-Dynamic labeled frames with BCE + (1 - Dice) loss."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import set_seed
from .segment import UNet, choose_device, frame_to_tensor


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class TrainConfig:
    """Training settings. Kept small for CPU."""

    seed: int = 42
    features: tuple[int, ...] = (32, 64, 128, 256)
    target_size: tuple[int, int] = (112, 112)
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 5
    time_budget_minutes: float = 90.0
    augment: bool = True
    train_video_limit: int | None = None
    resume: bool = False
    resume_lr: float = 3e-4


def load_cache_entries(split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load all (frame, mask) pairs from the cache for one split."""
    entries: list[tuple[np.ndarray, np.ndarray]] = []
    for path in sorted(CACHE_DIR.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if str(data["split"]) != split:
            continue
        frames, masks = data["frames"], data["masks"]
        for frame, mask in zip(frames, masks):
            entries.append((frame, mask))
    return entries


def _augment(frame: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Apply small brightness, contrast, and shift jitter; never flip."""
    frame = frame.astype(np.float32)
    frame = frame * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    tx = rng.randint(-4, 4)
    ty = rng.randint(-4, 4)
    if tx or ty:
        h, w = frame.shape
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        frame = cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return frame, mask


class LVFrameDataset(Dataset):
    """Dataset of grayscale frames + binary masks."""

    def __init__(self, entries: list[tuple[np.ndarray, np.ndarray]], target_size: tuple[int, int], augment: bool, seed: int):
        self.entries = entries
        self.target_size = target_size
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        frame, mask = self.entries[index]
        if self.augment:
            frame, mask = _augment(frame, mask, self._rng)
        tensor = frame_to_tensor(frame, self.target_size).squeeze(0)
        resized_mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((resized_mask > 0).astype(np.float32)).unsqueeze(0)
        return tensor, mask_tensor


def dice_score(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice over the batch."""
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (denom + eps)).mean()


def hard_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Dice after thresholding; used for validation reporting."""
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    intersection = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return (2 * intersection + eps) / (denom + eps)


def train(cfg: TrainConfig) -> dict:
    """Train the U-Net and save the best weights by VAL Dice."""
    set_seed(cfg.seed)
    device = choose_device(None)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_entries = load_cache_entries("TRAIN")
    val_entries = load_cache_entries("VAL")
    if cfg.train_video_limit:
        train_entries = train_entries[: cfg.train_video_limit * 2]
    print(f"TRAIN frames={len(train_entries)} VAL frames={len(val_entries)} device={device}", flush=True)

    train_dataset = LVFrameDataset(train_entries, cfg.target_size, cfg.augment, cfg.seed)
    val_dataset = LVFrameDataset(val_entries, cfg.target_size, False, cfg.seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=1, out_channels=1, features=list(cfg.features)).to(device)
    weights_path = MODELS_DIR / "unet_lv.pth"
    sidecar_path = MODELS_DIR / "unet_lv.json"
    log_path = RESULTS_DIR / "training_log.csv"
    best_val_dice = -1.0
    best_epoch = -1
    start_epoch = 1
    if cfg.resume:
        if not weights_path.exists() or not sidecar_path.exists():
            raise FileNotFoundError("--resume needs models/unet_lv.pth and models/unet_lv.json")
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        best_val_dice = float(sidecar.get("best_val_dice", -1.0))
        best_epoch = int(sidecar.get("best_epoch", -1))
        if log_path.exists():
            with open(log_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                start_epoch = int(rows[-1]["epoch"]) + 1
        lr = cfg.resume_lr
        print(f"RESUME from epoch {start_epoch}, best_val_dice={best_val_dice:.4f} (epoch {best_epoch}), lr={lr}", flush=True)
    else:
        lr = cfg.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    epochs_since_improve = 0
    if not cfg.resume or not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dice", "elapsed_minutes"])

    budget_seconds = cfg.time_budget_minutes * 60
    start_time = time.time()
    stop_reason = "max_epochs"

    for epoch in range(start_epoch, start_epoch + cfg.max_epochs):
        model.train()
        train_losses = []
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            prob = torch.sigmoid(logits)
            loss = bce(logits, targets) + (1.0 - dice_score(prob, targets))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if time.time() - start_time > budget_seconds:
                stop_reason = "time_budget"
                break
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses, val_dices = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                prob = torch.sigmoid(logits)
                val_losses.append((bce(logits, targets) + (1.0 - dice_score(prob, targets))).item())
                val_dices.append(hard_dice(logits, targets))
        val_loss = float(np.mean(val_losses))
        val_dice = float(np.mean(val_dices))
        elapsed_min = (time.time() - start_time) / 60.0

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6), round(val_loss, 6), round(val_dice, 6), round(elapsed_min, 2)])
        print(f"epoch {epoch} train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_dice {val_dice:.4f} time {elapsed_min:.1f}m", flush=True)

        if val_dice > best_val_dice + 1e-4:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), weights_path)
            sidecar_path.write_text(
                json.dumps(
                    {
                        "features": list(cfg.features),
                        "target_size": list(cfg.target_size),
                        "best_val_dice": best_val_dice,
                        "best_epoch": best_epoch,
                        "seed": cfg.seed,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.patience:
                stop_reason = "early_stop"
                break
        if time.time() - start_time > budget_seconds:
            stop_reason = "time_budget"
            break

    plot_log(log_path, RESULTS_DIR / "training_curve.png")
    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest() if weights_path.exists() else None
    return {
        "best_val_dice": best_val_dice,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "elapsed_minutes": round((time.time() - start_time) / 60.0, 2),
        "weights_sha256": sha,
        "train_frames": len(train_entries),
        "val_frames": len(val_entries),
    }


def plot_log(log_path: Path, out_path: Path) -> None:
    """Plot train/val loss and VAL Dice vs epoch."""
    rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    val_dice = [float(r["val_dice"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss, label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")
    ax2.plot(epochs, val_dice, color="green")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("VAL Dice"); ax2.set_title("Validation Dice")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-min", type=float, default=90.0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume from models/unet_lv.pth with lr=3e-4")
    args = parser.parse_args()
    cfg = TrainConfig(
        time_budget_minutes=args.budget_min,
        train_video_limit=args.train_limit,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        resume=args.resume,
    )
    summary = train(cfg)
    (RESULTS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
