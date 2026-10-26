"""Phase 3 extras: scatter + Bland-Altman from the full 1277 rescore, replot
the 3 old worst with new pairs (f=0.3), and list the 5 new worst with causes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.contour_tracking import track_contours
from src.ef_calculation import compute_lv_areas, estimate_ejection_fraction
from src.extract_frames import extract_frames, get_video_metadata
from src.preprocess import preprocess_frames
from src.run import read_sidecar
from src.segment import load_model, segment_frames


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EVAL_DIR = ROOT / "data" / "test_eval"
WEIGHTS = ROOT / "models" / "unet_lv.pth"
FROZEN_F = 0.3


def _plot_area(areas, result, out_path: Path) -> None:
    frames = np.arange(len(areas))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frames, areas, color="lightsteelblue", label="raw area")
    ax.plot(frames, result["smoothed_areas"], color="steelblue", label="smoothed")
    for pair in result["pairs"]:
        ax.axvline(pair["ed_frame_index"], color="green", linestyle="--")
        ax.axvline(pair["es_frame_index"], color="red", linestyle="--")
    ax.set_xlabel("frame"); ax.set_ylabel("area (pixel^2)"); ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter(label: np.ndarray, pred: np.ndarray, name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(label.min(), pred.min()) - 2
    hi = max(label.max(), pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--")
    ax.scatter(label, pred, alpha=0.4, s=10)
    ax.set_xlabel("EchoNet EF label (%)")
    ax.set_ylabel(f"Predicted {name} (%)")
    ax.set_title(f"{name} vs EF (rescore, f=0.3)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ba(label: np.ndarray, pred: np.ndarray, name: str, path: Path) -> None:
    mean = (label + pred) / 2.0
    diff = pred - label
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean, diff, alpha=0.4, s=10)
    ax.axhline(bias, color="red")
    ax.axhline(bias + 1.96 * sd, color="orange", linestyle="--")
    ax.axhline(bias - 1.96 * sd, color="orange", linestyle="--")
    ax.set_xlabel("(label + prediction) / 2 (%)")
    ax.set_ylabel("prediction - label (%)")
    ax.set_title(f"Bland-Altman: {name} (rescore)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _diagnose(areas, result, old_fac_err: float | None, new_fac_err: float) -> str:
    """Short cause string. If old error >> new error, it's a pairing fix."""
    arr = np.asarray(areas, dtype=float)
    if len(arr) == 0:
        return "no frames extracted"
    zero_frac = float(np.mean(arr == 0))
    jumps = np.abs(np.diff(arr)) / np.maximum(arr[:-1], 1.0) if len(arr) > 1 else np.array([])
    jump_frac = float(np.mean(jumps > 0.5)) if len(jumps) else 0.0
    if old_fac_err is not None and (old_fac_err - new_fac_err) >= 20.0:
        return "spurious ED peaks (no prominence limit)"
    if result.get("used_single_pair_fallback"):
        return f"single ED-ES pair fallback; empty_frac={zero_frac:.2f}"
    if zero_frac > 0.1:
        return f"segmentation drops out on {zero_frac*100:.0f}% of frames"
    if jump_frac > 0.1:
        return f"unstable contour: {jump_frac*100:.0f}% of frames show >50% area jump"
    if result["quality_flags"]["number_of_beats"] == 1:
        return "only one beat detected; limited averaging"
    return "cause unknown"


def main() -> None:
    cfg = PipelineConfig(denoise=False)
    features, target_size = read_sidecar(WEIGHTS, cfg)
    model = load_model(str(WEIGHTS), device=None, widths=features)

    preds = pd.read_csv(RESULTS_DIR / "predictions_rescore.csv")

    label = preds["ef_label"].to_numpy()
    for col, name in [("area_ratio_pct", "FAC"), ("area_power_pct", "FAC_power"), ("method_of_disks_pct", "Disks")]:
        pred = preds[col].to_numpy()
        m = np.isfinite(label) & np.isfinite(pred)
        _scatter(label[m], pred[m], name, RESULTS_DIR / f"scatter_{name}_rescore.png")
        _ba(label[m], pred[m], name, RESULTS_DIR / f"bland_altman_{name}_rescore.png")

    # Replot the 3 old worst with new pairs under the frozen f.
    first_run_worst = json.loads((RESULTS_DIR / "worst_videos.json").read_text(encoding="utf-8"))
    worst_dir = RESULTS_DIR / "worst_videos_rescore"
    worst_dir.mkdir(parents=True, exist_ok=True)
    old_worst_updated = []
    for entry in first_run_worst:
        name = entry["file_name"]
        video = EVAL_DIR / f"{name}.avi"
        if not video.exists():
            old_worst_updated.append({**entry, "new_fac_pred_pct": None, "new_abs_error_pp": None, "cause_after_fix": "video not in test_eval cache"})
            continue
        frames = extract_frames(str(video))
        meta = get_video_metadata(str(video))
        processed = preprocess_frames(frames, denoise=False)
        masks = segment_frames(model, processed, cfg.threshold, None, cfg.batch_size, target_size)
        contours = track_contours(masks)
        areas = compute_lv_areas(contours)
        res = estimate_ejection_fraction(areas, masks, meta["fps"], cfg.min_beat_seconds, min_prominence_fraction=FROZEN_F)
        fac_new = res["area_ratio"]["median_pct"]
        err_new = abs((fac_new or float("nan")) - entry["ef_label"]) if fac_new is not None else float("nan")
        _plot_area(areas, res, worst_dir / f"old_rank{entry['rank']}_{name}.png")
        old_worst_updated.append({
            **entry,
            "new_fac_pred_pct": None if fac_new is None else round(float(fac_new), 4),
            "new_abs_error_pp": None if not np.isfinite(err_new) else round(float(err_new), 4),
            "cause_after_fix": _diagnose(areas, res, entry.get("abs_error_pp"), float(err_new) if np.isfinite(err_new) else entry.get("abs_error_pp", 0.0)),
        })
    (RESULTS_DIR / "worst_videos.json").write_text(json.dumps(old_worst_updated, indent=2), encoding="utf-8")

    # New 5 worst from the full 1277 rescore (ranked by |FAC - label|).
    err = np.abs(preds["area_ratio_pct"].to_numpy() - preds["ef_label"].to_numpy())
    order = np.argsort(np.where(np.isfinite(err), err, -1))[::-1][:5]
    new_dir = RESULTS_DIR / "worst5_rescore"
    new_dir.mkdir(parents=True, exist_ok=True)
    entries_new = []
    for rank, idx in enumerate(order, start=1):
        row = preds.iloc[int(idx)]
        name = str(row["file_name"])
        video = EVAL_DIR / f"{name}.avi"
        frames = extract_frames(str(video))
        meta = get_video_metadata(str(video))
        processed = preprocess_frames(frames, denoise=False)
        masks = segment_frames(model, processed, cfg.threshold, None, cfg.batch_size, target_size)
        contours = track_contours(masks)
        areas = compute_lv_areas(contours)
        res = estimate_ejection_fraction(areas, masks, meta["fps"], cfg.min_beat_seconds, min_prominence_fraction=FROZEN_F)
        _plot_area(areas, res, new_dir / f"rank{rank}_{name}.png")
        entries_new.append({
            "rank": int(rank),
            "file_name": name,
            "ef_label": float(row["ef_label"]),
            "fac_pred_pct": float(row["area_ratio_pct"]) if np.isfinite(row["area_ratio_pct"]) else None,
            "abs_error_pp": float(err[int(idx)]) if np.isfinite(err[int(idx)]) else None,
            "n_beats": int(row["n_beats"]) if "n_beats" in row else None,
            "rejected_ed_peaks": int(row["rejected_ed_peaks"]) if "rejected_ed_peaks" in row else None,
            "dice_mean": float(row["dice_mean"]) if np.isfinite(row["dice_mean"]) else None,
            "cause": _diagnose(areas, res, None, float(err[int(idx)]) if np.isfinite(err[int(idx)]) else 0.0),
        })
    (RESULTS_DIR / "worst_videos_rescore.json").write_text(json.dumps(entries_new, indent=2), encoding="utf-8")
    print(json.dumps({"old_updated": old_worst_updated, "new_worst_5": entries_new}, indent=2))


if __name__ == "__main__":
    main()
