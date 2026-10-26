"""Evaluate the trained U-Net on 50 TEST videos; score EF estimators and Dice."""
from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig, set_seed
from .contour_tracking import track_contours
from .dataset import ZIP_PATH, ZIP_ROOT, build_mask, group_tracings, read_frames
from .ef_calculation import compute_lv_areas, estimate_ejection_fraction, method_of_disks
from .extract_frames import extract_frames, get_video_metadata
from .preprocess import preprocess_frames
from .run import read_sidecar
from .segment import load_model, segment_frames


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings."""

    seed: int = 42
    n_videos: int = 50
    weights_path: Path = ROOT / "models" / "unet_lv.pth"


def pick_test_videos(cfg: EvalConfig) -> pd.DataFrame:
    """Select the TEST video rows reproducibly."""
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(f"{ZIP_ROOT}/FileList.csv") as f:
            file_list = pd.read_csv(f)
    test = file_list[file_list["Split"] == "TEST"].sort_values("FileName").reset_index(drop=True)
    sample = test.sample(n=cfg.n_videos, random_state=cfg.seed).reset_index(drop=True)
    return sample[["FileName", "EF", "ESV", "EDV", "FPS", "NumberOfFrames"]]


def extract_videos(selected: pd.DataFrame) -> None:
    """Extract the chosen TEST videos from the zip into data/eval/."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in selected["FileName"]:
            dst = EVAL_DIR / f"{name}.avi"
            if dst.exists():
                continue
            with z.open(f"{ZIP_ROOT}/Videos/{name}.avi") as src, open(dst, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)


def _tracings_lookup() -> dict[str, dict[int, np.ndarray]]:
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(f"{ZIP_ROOT}/VolumeTracings.csv") as f:
            tracings = pd.read_csv(f)
    return group_tracings(tracings)


def _dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Return Dice between two binary masks."""
    pred_bool = pred > 0
    truth_bool = truth > 0
    inter = np.logical_and(pred_bool, truth_bool).sum()
    denom = pred_bool.sum() + truth_bool.sum()
    return 0.0 if denom == 0 else float(2 * inter / denom)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _summary(predictions: np.ndarray, labels: np.ndarray) -> dict:
    diff = predictions - labels
    mask = np.isfinite(diff)
    if mask.sum() == 0:
        return {"mae_pp": None, "rmse_pp": None, "bias_pp": None, "pearson_r": None, "n": 0}
    return {
        "mae_pp": round(float(np.mean(np.abs(diff[mask]))), 3),
        "rmse_pp": round(float(math.sqrt(np.mean(diff[mask] ** 2))), 3),
        "bias_pp": round(float(np.mean(diff[mask])), 3),
        "pearson_r": round(_safe_corr(predictions, labels), 3),
        "n": int(mask.sum()),
    }


def evaluate(cfg: EvalConfig | None = None) -> dict:
    """Run the full 50-video evaluation; save predictions, metrics, and plots."""
    cfg = cfg or EvalConfig()
    set_seed(cfg.seed)
    selected = pick_test_videos(cfg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected[["FileName"]].to_csv(RESULTS_DIR / "eval_selection.csv", index=False)

    extract_videos(selected)
    tracings = _tracings_lookup()

    features, target_size = read_sidecar(cfg.weights_path, PipelineConfig())
    rows_denoise_off = _score_all(selected, tracings, features, target_size, denoise=False, label="base")
    rows_denoise_on = _score_all(selected, tracings, features, target_size, denoise=True, label="denoised")

    preds = pd.DataFrame(rows_denoise_off)
    preds_dn = pd.DataFrame(rows_denoise_on)
    preds.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    preds_dn.to_csv(RESULTS_DIR / "predictions_denoised.csv", index=False)

    metrics = _assemble_metrics(preds, preds_dn)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _plot_scatter_and_bland_altman(preds, "base")
    _plot_worst_three(preds)

    return metrics


def _score_all(
    selected: pd.DataFrame,
    tracings: dict[str, dict[int, np.ndarray]],
    features: tuple[int, ...],
    target_size: tuple[int, int],
    denoise: bool,
    label: str,
) -> list[dict]:
    """Score every selected video with the given denoise setting."""
    cfg = PipelineConfig(denoise=denoise)
    device = None
    model = load_model(str(EvalConfig().weights_path), device=device, widths=features)
    rows = []
    for row in selected.itertuples(index=False):
        video_path = EVAL_DIR / f"{row.FileName}.avi"
        try:
            frames = extract_frames(str(video_path))
            metadata = get_video_metadata(str(video_path))
            processed = preprocess_frames(frames, denoise=denoise)
            masks = segment_frames(model, processed, cfg.threshold, device, cfg.batch_size, target_size)
            contours = track_contours(masks)
            result = estimate_ejection_fraction(compute_lv_areas(contours), masks, metadata["fps"], cfg.min_beat_seconds)
            area_med = result["area_ratio"]["median_pct"]
            power_med = result["area_power_volume_proxy"]["median_pct"]
            disk_med = result["method_of_disks"]["median_pct"]
            failed = False
        except Exception as exc:
            area_med = power_med = disk_med = float("nan")
            masks = []
            failed = True
            result = {"error": str(exc)}
        gt_tracings = tracings.get(f"{row.FileName}.avi", {})
        dices = _compute_label_dice(video_path, gt_tracings, masks)
        rows.append(
            {
                "file_name": row.FileName,
                "ef_label": float(row.EF),
                "area_ratio_pct": area_med,
                "area_power_pct": power_med,
                "method_of_disks_pct": disk_med,
                "dice_mean": dices["mean"],
                "dice_count": dices["n"],
                "failed": int(failed),
                "label": label,
            }
        )
    return rows


def _compute_label_dice(video_path: Path, gt_tracings: dict[int, np.ndarray], masks: list[np.ndarray]) -> dict:
    """Compute Dice at labeled ED and ES frames."""
    if not gt_tracings or not masks:
        return {"mean": float("nan"), "n": 0}
    gray_by_index = read_frames(video_path, set(gt_tracings.keys()))
    scores: list[float] = []
    for frame_index, chords in gt_tracings.items():
        if frame_index >= len(masks):
            continue
        gt_mask = build_mask(chords, shape=masks[frame_index].shape)
        scores.append(_dice(masks[frame_index], gt_mask))
    return {"mean": float(np.mean(scores)) if scores else float("nan"), "n": len(scores)}


def _assemble_metrics(preds: pd.DataFrame, preds_dn: pd.DataFrame) -> dict:
    label = preds["ef_label"].to_numpy()
    dices = preds["dice_mean"].to_numpy()
    metrics = {
        "n_videos": int(len(preds)),
        "n_failed_base": int(preds["failed"].sum()),
        "n_failed_denoised": int(preds_dn["failed"].sum()),
        "test_dice": {
            "mean": round(float(np.nanmean(dices)), 4),
            "median": round(float(np.nanmedian(dices)), 4),
            "n": int(np.sum(np.isfinite(dices))),
        },
        "estimators_base": {
            "area_ratio": _summary(preds["area_ratio_pct"].to_numpy(), label),
            "area_power_volume_proxy": _summary(preds["area_power_pct"].to_numpy(), label),
            "method_of_disks": _summary(preds["method_of_disks_pct"].to_numpy(), label),
        },
        "estimators_denoised": {
            "area_ratio": _summary(preds_dn["area_ratio_pct"].to_numpy(), label),
            "area_power_volume_proxy": _summary(preds_dn["area_power_pct"].to_numpy(), label),
            "method_of_disks": _summary(preds_dn["method_of_disks_pct"].to_numpy(), label),
        },
    }
    return metrics


def _plot_scatter_and_bland_altman(preds: pd.DataFrame, suffix: str) -> None:
    """Scatter and Bland-Altman plot for each estimator."""
    label = preds["ef_label"].to_numpy()
    cols = [("area_ratio_pct", "FAC"), ("area_power_pct", "FAC^(3/2)"), ("method_of_disks_pct", "Disks")]
    for col, name in cols:
        pred = preds[col].to_numpy()
        mask = np.isfinite(pred) & np.isfinite(label)
        if mask.sum() == 0:
            continue
        _save_scatter(label[mask], pred[mask], name, suffix)
        _save_bland_altman(label[mask], pred[mask], name, suffix)


def _save_scatter(label: np.ndarray, pred: np.ndarray, name: str, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(label.min(), pred.min()) - 2
    hi = max(label.max(), pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--")
    ax.scatter(label, pred, alpha=0.7)
    ax.set_xlabel("EchoNet EF label (%)")
    ax.set_ylabel(f"Predicted {name} (%)")
    ax.set_title(f"{name} vs EF ({suffix})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"scatter_{name.replace('/', '_').replace('^', '')}_{suffix}.png", dpi=150)
    plt.close(fig)


def _save_bland_altman(label: np.ndarray, pred: np.ndarray, name: str, suffix: str) -> None:
    mean = (label + pred) / 2.0
    diff = pred - label
    mean_bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean, diff, alpha=0.7)
    ax.axhline(mean_bias, color="red", linestyle="-")
    ax.axhline(mean_bias + 1.96 * sd, color="orange", linestyle="--")
    ax.axhline(mean_bias - 1.96 * sd, color="orange", linestyle="--")
    ax.set_xlabel("(label + prediction) / 2 (%)")
    ax.set_ylabel("prediction - label (%)")
    ax.set_title(f"Bland-Altman: {name} ({suffix})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"bland_altman_{name.replace('/', '_').replace('^', '')}_{suffix}.png", dpi=150)
    plt.close(fig)


def _plot_worst_three(preds: pd.DataFrame) -> None:
    """Save the LV area curve for the 3 videos with the largest FAC error."""
    err = np.abs(preds["area_ratio_pct"].to_numpy() - preds["ef_label"].to_numpy())
    order = np.argsort(np.where(np.isfinite(err), err, -1))[::-1][:3]
    worst_dir = RESULTS_DIR / "worst_videos"
    worst_dir.mkdir(parents=True, exist_ok=True)
    worst_entries = []
    for rank, idx in enumerate(order, start=1):
        row = preds.iloc[idx]
        video = EVAL_DIR / f"{row['file_name']}.avi"
        try:
            frames = extract_frames(str(video))
            metadata = get_video_metadata(str(video))
            processed = preprocess_frames(frames, denoise=False)
            cfg = PipelineConfig()
            features, target_size = read_sidecar(EvalConfig().weights_path, cfg)
            model = load_model(str(EvalConfig().weights_path), device=None, widths=features)
            masks = segment_frames(model, processed, cfg.threshold, None, cfg.batch_size, target_size)
            contours = track_contours(masks)
            areas = compute_lv_areas(contours)
            result = estimate_ejection_fraction(areas, masks, metadata["fps"], cfg.min_beat_seconds)
            _plot_area_curve(areas, result, worst_dir / f"rank{rank}_{row['file_name']}.png")
            cause = _diagnose(areas, result, row)
        except Exception as exc:
            cause = f"error during replay: {exc}"
        worst_entries.append(
            {
                "rank": rank,
                "file_name": str(row["file_name"]),
                "ef_label": float(row["ef_label"]),
                "fac_pred_pct": float(row["area_ratio_pct"]) if np.isfinite(row["area_ratio_pct"]) else None,
                "abs_error_pp": float(err[idx]) if np.isfinite(err[idx]) else None,
                "cause": cause,
            }
        )
    (RESULTS_DIR / "worst_videos.json").write_text(json.dumps(worst_entries, indent=2), encoding="utf-8")


def _plot_area_curve(areas, result, out_path: Path) -> None:
    frames = np.arange(len(areas))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frames, areas, color="lightsteelblue", label="raw area")
    ax.plot(frames, result["smoothed_areas"], color="steelblue", label="smoothed")
    for pair in result["pairs"]:
        ax.axvline(pair["ed_frame_index"], color="green", linestyle="--")
        ax.axvline(pair["es_frame_index"], color="red", linestyle="--")
    ax.set_xlabel("frame"); ax.set_ylabel("area (pixel^2)"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _diagnose(areas, result, row) -> str:
    """Return a short cause string from the evidence; 'cause unknown' if none fits."""
    arr = np.asarray(areas, dtype=float)
    if len(arr) == 0:
        return "no frames extracted"
    zero_frac = float(np.mean(arr == 0))
    jumps = np.abs(np.diff(arr)) / np.maximum(arr[:-1], 1.0) if len(arr) > 1 else np.array([])
    jump_frac = float(np.mean(jumps > 0.5)) if len(jumps) else 0.0
    if result.get("used_single_pair_fallback"):
        return f"single ED-ES pair fallback (no clean beats found); empty_frac={zero_frac:.2f}"
    if zero_frac > 0.1:
        return f"segmentation drops out on {zero_frac*100:.0f}% of frames"
    if jump_frac > 0.1:
        return f"unstable contour: {jump_frac*100:.0f}% of frames show >50% area jump"
    if result["quality_flags"]["number_of_beats"] == 1:
        return "only one beat detected; limited averaging"
    return "cause unknown"


def main() -> None:
    """Run the evaluation and print the metrics summary."""
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
