"""Render 6 cached frames with their mask contour overlaid for a sanity check."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "results" / "mask_check"


def _entries_for(split: str) -> list[Path]:
    paths = []
    for path in sorted(CACHE.glob("*.npz")):
        meta = np.load(path, allow_pickle=True)
        if str(meta["split"]) == split:
            paths.append(path)
        meta.close()
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    for split in ("TRAIN", "VAL"):
        pool = _entries_for(split)
        sample = rng.sample(pool, k=3)
        for path in sample:
            data = np.load(path, allow_pickle=True)
            frames, masks = data["frames"], data["masks"]
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            for ax, frame, mask in zip(axes, frames, masks):
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contour, -1, (0, 255, 0), 1)
                ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{split} {path.stem}")
            fig.tight_layout()
            out = OUT / f"{split}_{path.stem}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    main()
