"""Phase 3: re-score TEST with the frozen prominence fraction f=0.3.

Timing-aware size pick: measure 10 videos, extrapolate; if the full 1277 TEST
run fits in 90 minutes we score all, else we score 300 (random_state=42) and
disclose. Filters OFF only (filters hurt in the first run; see DECISIONS.md).
"""
from __future__ import annotations

import csv
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig, set_seed
from src.contour_tracking import track_contours
from src.dataset import ZIP_PATH, ZIP_ROOT, build_mask, group_tracings, read_frames
from src.ef_calculation import compute_lv_areas, estimate_ejection_fraction
from src.extract_frames import extract_frames, get_video_metadata
from src.preprocess import preprocess_frames
from src.run import read_sidecar
from src.segment import load_model, segment_frames


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EVAL_DIR = ROOT / "data" / "test_eval"
WEIGHTS = ROOT / "models" / "unet_lv.pth"
SEED = 42
FROZEN_F = 0.3
BUDGET_MINUTES = 300.0  # user widened the cap after a 174-min projection; now score all 1277.
FULL_FALLBACK = 300  # if the full 1277 run would exceed the budget


def _pick(limit: int | None) -> tuple[pd.DataFrame, float, float]:
    """Return (selected rows, TRAIN mean EF, VAL EF label mean) deterministically."""
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(f"{ZIP_ROOT}/FileList.csv") as f:
            fl = pd.read_csv(f)
    test = fl[fl["Split"] == "TEST"].sort_values("FileName").reset_index(drop=True)
    train_mean = float(fl[fl["Split"] == "TRAIN"]["EF"].mean())
    if limit is None or limit >= len(test):
        return test[["FileName", "EF", "FPS", "NumberOfFrames"]].reset_index(drop=True), train_mean, len(test)
    sample = test.sample(n=limit, random_state=SEED).reset_index(drop=True)
    return sample[["FileName", "EF", "FPS", "NumberOfFrames"]], train_mean, len(test)


def _extract(selected: pd.DataFrame) -> None:
    """Stream chosen TEST videos out of the zip one at a time."""
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


def _summary(pred: np.ndarray, label: np.ndarray) -> dict:
    diff = pred - label
    mask = np.isfinite(diff)
    if mask.sum() == 0:
        return {"mae_pp": None, "rmse_pp": None, "bias_pp": None, "pearson_r": None, "n": 0}
    rmask = np.isfinite(pred) & np.isfinite(label)
    r = float("nan") if rmask.sum() < 2 else float(np.corrcoef(pred[rmask], label[rmask])[0, 1])
    return {
        "mae_pp": round(float(np.mean(np.abs(diff[mask]))), 3),
        "rmse_pp": round(float(math.sqrt(np.mean(diff[mask] ** 2))), 3),
        "bias_pp": round(float(np.mean(diff[mask])), 3),
        "pearson_r": round(r, 3),
        "n": int(mask.sum()),
    }


def _dice(pred: np.ndarray, truth: np.ndarray) -> float:
    pb, tb = pred > 0, truth > 0
    inter = np.logical_and(pb, tb).sum()
    denom = pb.sum() + tb.sum()
    return 0.0 if denom == 0 else float(2 * inter / denom)


def _dice_at_labels(video_path: Path, gt: dict[int, np.ndarray], masks: list[np.ndarray]) -> dict:
    if not gt or not masks:
        return {"mean": float("nan"), "n": 0}
    _ = read_frames(video_path, set(gt.keys()))  # consistent with first-run eval code path
    scores = []
    for idx, chords in gt.items():
        if idx >= len(masks):
            continue
        gt_mask = build_mask(chords, shape=masks[idx].shape)
        scores.append(_dice(masks[idx], gt_mask))
    return {"mean": float(np.mean(scores)) if scores else float("nan"), "n": len(scores)}


def _score_rows(selected, tracings, features, target_size, model, cfg):
    """Score every selected TEST video with filters OFF and frozen f."""
    rows = []
    for row in selected.itertuples(index=False):
        video = EVAL_DIR / f"{row.FileName}.avi"
        try:
            frames = extract_frames(str(video))
            metadata = get_video_metadata(str(video))
            processed = preprocess_frames(frames, denoise=False)
            masks = segment_frames(model, processed, cfg.threshold, None, cfg.batch_size, target_size)
            contours = track_contours(masks)
            areas = compute_lv_areas(contours)
            res = estimate_ejection_fraction(areas, masks, metadata["fps"], cfg.min_beat_seconds, min_prominence_fraction=FROZEN_F)
            fac = res["area_ratio"]["median_pct"]
            power = res["area_power_volume_proxy"]["median_pct"]
            disks = res["method_of_disks"]["median_pct"]
            failed = False
        except Exception as exc:
            fac = power = disks = float("nan")
            masks = []
            failed = True
            res = {"error": str(exc)}
        gt = tracings.get(f"{row.FileName}.avi", {})
        dices = _dice_at_labels(video, gt, masks)
        rows.append({
            "file_name": row.FileName,
            "ef_label": float(row.EF),
            "area_ratio_pct": fac,
            "area_power_pct": power,
            "method_of_disks_pct": disks,
            "dice_mean": dices["mean"],
            "dice_count": dices["n"],
            "failed": int(failed),
            "rejected_ed_peaks": int(res.get("quality_flags", {}).get("rejected_ed_peaks", 0)) if not failed else 0,
            "n_beats": int(res.get("quality_flags", {}).get("number_of_beats", 0)) if not failed else 0,
        })
    return rows


def _apply_cal(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply EF = a·x + b calibration fitted on VAL only."""
    return a * x + b


def main() -> None:
    set_seed(SEED)
    cfg = PipelineConfig(denoise=False)
    features, target_size = read_sidecar(WEIGHTS, cfg)
    model = load_model(str(WEIGHTS), device=None, widths=features)
    cal = json.loads((RESULTS_DIR / "val_sweep.json").read_text(encoding="utf-8"))["calibration_fit_on_val"]

    # Timing: run 10 TEST videos first, extrapolate.
    all_rows, train_mean_ef, n_test_total = _pick(None)
    timing_selected = all_rows.head(10).copy()
    _extract(timing_selected)
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(f"{ZIP_ROOT}/VolumeTracings.csv") as f:
            tracings = group_tracings(pd.read_csv(f))

    t0 = time.time()
    _ = _score_rows(timing_selected, tracings, features, target_size, model, cfg)
    seconds_per_video = (time.time() - t0) / len(timing_selected)
    projected_minutes = seconds_per_video * n_test_total / 60.0
    print(f"timing: {seconds_per_video:.2f}s/video, projected {projected_minutes:.1f} min for {n_test_total} videos", flush=True)

    if projected_minutes <= BUDGET_MINUTES:
        n_pick = n_test_total
        selected = all_rows
    else:
        n_pick = FULL_FALLBACK
        selected = all_rows.sample(n=FULL_FALLBACK, random_state=SEED).reset_index(drop=True)

    _extract(selected)
    print(f"scoring {len(selected)} TEST videos with frozen f={FROZEN_F} ...", flush=True)
    t0 = time.time()
    rows = _score_rows(selected, tracings, features, target_size, model, cfg)
    elapsed = (time.time() - t0) / 60.0
    print(f"scored {len(rows)} videos in {elapsed:.1f} min", flush=True)

    preds = pd.DataFrame(rows)
    preds.to_csv(RESULTS_DIR / "predictions_rescore.csv", index=False)

    label = preds["ef_label"].to_numpy()
    fac = preds["area_ratio_pct"].to_numpy()
    pwr = preds["area_power_pct"].to_numpy()
    dks = preds["method_of_disks_pct"].to_numpy()

    # Raw summaries.
    sum_fac = _summary(fac, label)
    sum_pwr = _summary(pwr, label)
    sum_dks = _summary(dks, label)
    # Calibrated summaries (EF = a·x + b fitted on VAL only).
    cal_fac = _summary(_apply_cal(fac, cal["area_ratio"]["a"], cal["area_ratio"]["b"]), label)
    cal_pwr = _summary(_apply_cal(pwr, cal["area_power"]["a"], cal["area_power"]["b"]), label)
    cal_dks = _summary(_apply_cal(dks, cal["method_of_disks"]["a"], cal["method_of_disks"]["b"]), label)
    # TRAIN-mean baseline on this TEST subset.
    baseline = _summary(np.full_like(label, train_mean_ef, dtype=float), label)
    baseline["train_mean_ef_pct"] = round(float(train_mean_ef), 3)
    # Dice.
    d = preds["dice_mean"].to_numpy()
    dice_summary = {
        "mean": round(float(np.nanmean(d)), 4),
        "median": round(float(np.nanmedian(d)), 4),
        "n": int(np.sum(np.isfinite(d))),
    }

    out = {
        "n_test_total_available": int(n_test_total),
        "n_videos_scored": int(len(preds)),
        "selection_mode": "all_test" if n_pick == n_test_total else f"random_{FULL_FALLBACK}_seed42",
        "projected_full_minutes": round(float(projected_minutes), 1),
        "frozen_f": FROZEN_F,
        "filters": "off",
        "n_failed": int(preds["failed"].sum()),
        "test_dice": dice_summary,
        "estimators_raw": {
            "area_ratio": sum_fac,
            "area_power_volume_proxy": sum_pwr,
            "method_of_disks": sum_dks,
        },
        "estimators_calibrated_val_fit": {
            "area_ratio": cal_fac,
            "area_power_volume_proxy": cal_pwr,
            "method_of_disks": cal_dks,
        },
        "train_mean_baseline_on_test": baseline,
        "seconds_per_video": round(float(seconds_per_video), 2),
    }
    (RESULTS_DIR / "metrics_rescore.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
