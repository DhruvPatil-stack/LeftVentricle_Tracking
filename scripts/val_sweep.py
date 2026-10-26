"""Sweep prominence fraction f on 100 VAL videos; fit a VAL-only calibration.

VAL only (never TEST). f is chosen by lowest MAE on area_power_volume_proxy, with
ties within 0.2 pp broken by the lower f. Also records a TRAIN-mean baseline so
we can see how much the estimator beats "predict the TRAIN average".
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
from src.dataset import ZIP_PATH, ZIP_ROOT
from src.ef_calculation import compute_lv_areas, estimate_ejection_fraction
from src.extract_frames import extract_frames, get_video_metadata
from src.preprocess import preprocess_frames
from src.run import read_sidecar
from src.segment import load_model, segment_frames


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
WEIGHTS = ROOT / "models" / "unet_lv.pth"
EVAL_DIR = ROOT / "data" / "val_eval"
N_VAL = 100
F_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)
SEED = 42


def pick_val_videos() -> pd.DataFrame:
    """Pick 100 VAL rows with random_state=SEED."""
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(f"{ZIP_ROOT}/FileList.csv") as f:
            file_list = pd.read_csv(f)
    val = file_list[file_list["Split"] == "VAL"].sort_values("FileName").reset_index(drop=True)
    train_mean_ef = float(file_list[file_list["Split"] == "TRAIN"]["EF"].mean())
    sample = val.sample(n=N_VAL, random_state=SEED).reset_index(drop=True)
    return sample[["FileName", "EF", "FPS", "NumberOfFrames"]], train_mean_ef


def extract_videos(selected: pd.DataFrame) -> None:
    """Stream the selected VAL videos out of the zip; one at a time, no in-memory load."""
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


def _summary(preds: np.ndarray, labels: np.ndarray) -> dict:
    diff = preds - labels
    mask = np.isfinite(diff)
    if mask.sum() == 0:
        return {"mae_pp": None, "rmse_pp": None, "bias_pp": None, "pearson_r": None, "n": 0}
    r_mask = np.isfinite(preds) & np.isfinite(labels)
    r = float("nan") if r_mask.sum() < 2 else float(np.corrcoef(preds[r_mask], labels[r_mask])[0, 1])
    return {
        "mae_pp": round(float(np.mean(np.abs(diff[mask]))), 3),
        "rmse_pp": round(float(math.sqrt(np.mean(diff[mask] ** 2))), 3),
        "bias_pp": round(float(np.mean(diff[mask])), 3),
        "pearson_r": round(r, 3),
        "n": int(mask.sum()),
    }


def _run_variant(areas: list[float], masks, fps: float, cfg: PipelineConfig, use_old_logic: bool, f: float):
    """Score one variant: either the old no-prominence logic (f=0 by setting to 0.0) or the new one with f."""
    prom = 0.0 if use_old_logic else f
    return estimate_ejection_fraction(areas, masks, fps, cfg.min_beat_seconds, min_prominence_fraction=prom)


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected, train_mean_ef = pick_val_videos()
    selected[["FileName"]].to_csv(RESULTS_DIR / "val_selection.csv", index=False)
    extract_videos(selected)

    cfg = PipelineConfig(denoise=False)
    features, target_size = read_sidecar(WEIGHTS, cfg)
    model = load_model(str(WEIGHTS), device=None, widths=features)

    labels = selected["EF"].to_numpy(dtype=float)
    variants = [("old", None)] + [("prom", f) for f in F_VALUES]
    per_variant_preds: dict[str, dict] = {}
    for name, f in variants:
        per_variant_preds[f"{name}_{f}" if f is not None else name] = {
            "area_ratio": np.full(len(labels), np.nan),
            "area_power": np.full(len(labels), np.nan),
            "disks": np.full(len(labels), np.nan),
        }

    t0 = time.time()
    for idx, row in enumerate(selected.itertuples(index=False)):
        video = EVAL_DIR / f"{row.FileName}.avi"
        try:
            frames = extract_frames(str(video))
            metadata = get_video_metadata(str(video))
            processed = preprocess_frames(frames, denoise=False)
            masks = segment_frames(model, processed, cfg.threshold, None, cfg.batch_size, target_size)
            contours = track_contours(masks)
            areas = compute_lv_areas(contours)
            for name, f in variants:
                res = _run_variant(areas, masks, metadata["fps"], cfg, use_old_logic=(name == "old"), f=f or 0.0)
                key = f"{name}_{f}" if f is not None else name
                per_variant_preds[key]["area_ratio"][idx] = res["area_ratio"]["median_pct"] if res["area_ratio"]["median_pct"] is not None else np.nan
                per_variant_preds[key]["area_power"][idx] = res["area_power_volume_proxy"]["median_pct"] if res["area_power_volume_proxy"]["median_pct"] is not None else np.nan
                per_variant_preds[key]["disks"][idx] = res["method_of_disks"]["median_pct"] if res["method_of_disks"]["median_pct"] is not None else np.nan
        except Exception as exc:
            print(f"[warn] {row.FileName}: {exc}", flush=True)
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(selected)} ({time.time() - t0:.1f}s)", flush=True)

    sweep_rows = []
    for name, f in variants:
        key = f"{name}_{f}" if f is not None else name
        preds = per_variant_preds[key]
        for est_name, arr in (("area_ratio", preds["area_ratio"]), ("area_power", preds["area_power"]), ("method_of_disks", preds["disks"])):
            s = _summary(arr, labels)
            sweep_rows.append({"variant": key, "f": f if f is not None else "old", "estimator": est_name, **s})

    with open(RESULTS_DIR / "val_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["variant", "f", "estimator", "mae_pp", "rmse_pp", "bias_pp", "pearson_r", "n"])
        writer.writeheader()
        writer.writerows(sweep_rows)

    # TRAIN-mean baseline on VAL: predict the TRAIN mean EF for every VAL video.
    baseline_pred = np.full_like(labels, train_mean_ef, dtype=float)
    baseline_summary = _summary(baseline_pred, labels)
    baseline_summary["train_mean_ef_pct"] = round(float(train_mean_ef), 3)

    # Choose f: lowest area_power MAE; ties within 0.2 pp broken by lower f.
    power_rows = [row for row in sweep_rows if row["estimator"] == "area_power" and row["f"] != "old"]
    power_rows.sort(key=lambda r: (round(r["mae_pp"], 1), r["f"]))
    best_mae = power_rows[0]["mae_pp"]
    within_tie = [r for r in power_rows if abs(r["mae_pp"] - best_mae) <= 0.2]
    chosen = sorted(within_tie, key=lambda r: r["f"])[0]
    chosen_f = float(chosen["f"])

    # Fit linear calibration EF = a * estimator + b on the chosen variant.
    chosen_key = f"prom_{chosen_f}"
    chosen_power = per_variant_preds[chosen_key]["area_power"]
    chosen_disks = per_variant_preds[chosen_key]["disks"]
    chosen_fac = per_variant_preds[chosen_key]["area_ratio"]
    def _fit(x: np.ndarray, y: np.ndarray):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return None, None, int(m.sum())
        a, b = np.polyfit(x[m], y[m], 1)
        return round(float(a), 4), round(float(b), 4), int(m.sum())
    cal = {
        "area_power": {"a": _fit(chosen_power, labels)[0], "b": _fit(chosen_power, labels)[1], "n": _fit(chosen_power, labels)[2]},
        "method_of_disks": {"a": _fit(chosen_disks, labels)[0], "b": _fit(chosen_disks, labels)[1], "n": _fit(chosen_disks, labels)[2]},
        "area_ratio": {"a": _fit(chosen_fac, labels)[0], "b": _fit(chosen_fac, labels)[1], "n": _fit(chosen_fac, labels)[2]},
    }

    summary = {
        "n_val": int(len(selected)),
        "f_candidates": list(F_VALUES),
        "chosen_f": chosen_f,
        "chosen_reason": f"lowest area_power MAE on VAL ({chosen['mae_pp']} pp); ties within 0.2 pp broken by lower f",
        "train_mean_baseline_on_val": baseline_summary,
        "calibration_fit_on_val": cal,
        "sweep_rows": sweep_rows,
    }
    (RESULTS_DIR / "val_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
