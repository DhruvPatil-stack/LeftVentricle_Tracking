# Left Ventricle Tracking

A student project that trains a small U-Net on the EchoNet-Dynamic apical four-chamber ultrasound videos, tracks the left-ventricle (LV) cavity, and reports three research estimators of ejection-fraction behavior.

This is a student project. It is not a medical device.

## Status

The pipeline runs end to end: build a frame cache from the EchoNet-Dynamic zip, train a U-Net on TRAIN in two stages, pick the ED-peak prominence fraction on 100 VAL videos, score the full EchoNet TEST split (1277 videos) with the frozen setting, and compare raw and VAL-calibrated numbers against a TRAIN-mean baseline. The first 50-video TEST run (before the pairing fix) is also kept for reference.

<!-- GENERATED METRICS START -->
All values in this file come from `results/metrics.json` (first TEST run), `results/metrics_rescore.json` (rescore with the pairing fix), and `results/val_sweep.json` (VAL sweep + baseline + calibration).

The pairing fix adds a prominence filter in `find_ed_es_pairs` (fraction f of the full area swing). It was found while inspecting three TEST failures of the first run, so TEST was scored twice: once before the fix (50 videos) and once with the frozen f (full rescore).

## VAL sweep and chosen f

VAL sweep (100 VAL videos, random_state=42, filters off). Chosen f = 0.3. Reason: lowest area_power MAE on VAL (6.298 pp); ties within 0.2 pp broken by lower f.

| Variant | area_ratio MAE | area_power MAE | method_of_disks MAE |
|---|---|---|---|
| old (no prominence) | 19.02 | 11.26 | 11.32 |
| f = 0.1 | 12.66 | 6.66 | 7.56 |
| f = 0.2 | 12.53 | 6.83 | 7.78 |
| f = 0.3 | 12.27 | 6.30 | 7.30 |
| f = 0.4 | 12.16 | 6.55 | 7.59 |
| f = 0.5 | 11.97 | 6.55 | 7.66 |

TRAIN-mean baseline on VAL (predict constant 55.778 %): MAE 9.13, RMSE 11.76, bias -1.94 pp, N = 100.

VAL-only linear calibration EF = a·x + b:
- area_ratio: a = 0.8097, b = 20.1071 (fit on 100 VAL videos).
- area_power: a = 0.7594, b = 11.9992 (fit on 100 VAL videos).
- method_of_disks: a = 0.7547, b = 10.0822 (fit on 100 VAL videos).

## First TEST run (before the pairing fix): 50 videos

- TEST Dice (first run, mean / median): 0.930 / 0.934 over 50 labeled frames.
- Pipeline failures (filters off): 0
- Pipeline failures (filters on): 0

### First run, filters off

| Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Pearson r | N |
|---|---|---|---|---|---|
| Area ratio (FAC) | 15.31 | 18.01 | -15.11 | 0.64 | 50 |
| Area^(3/2) proxy | 8.13 | 11.46 | -2.57 | 0.62 | 50 |
| Method of disks (single plane) | 8.68 | 11.29 | 0.02 | 0.60 | 50 |

### First run, filters on

| Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Pearson r | N |
|---|---|---|---|---|---|
| Area ratio (FAC) | 15.12 | 18.52 | -14.31 | 0.56 | 50 |
| Area^(3/2) proxy | 9.59 | 14.06 | -1.75 | 0.54 | 50 |
| Method of disks (single plane) | 9.73 | 14.02 | 0.13 | 0.58 | 50 |

## TEST rescore with frozen f = 0.3: 1277 of 1277 videos

- Selection: all_test. Filters: off.
- Projected full run: 150.9 min (timed from 10 videos; 7.09 s/video).
- TEST Dice (rescore, mean / median): 0.921 / 0.929 over 1276 labeled frames.
- Pipeline failures: 0

### Rescore, raw

| Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Pearson r | N |
|---|---|---|---|---|---|
| Area ratio (FAC) | 11.71 | 13.42 | -10.68 | 0.76 | 1277 |
| Area^(3/2) proxy | 6.14 | 8.42 | 2.87 | 0.79 | 1277 |
| Method of disks (single plane) | 7.41 | 9.66 | 5.43 | 0.79 | 1277 |

### Rescore, calibrated with VAL-only linear fit

| Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Pearson r | N |
|---|---|---|---|---|---|
| Area ratio (FAC) | 5.82 | 8.03 | 0.89 | 0.76 | 1277 |
| Area^(3/2) proxy | 5.54 | 7.57 | 0.83 | 0.79 | 1277 |
| Method of disks (single plane) | 5.54 | 7.56 | 0.57 | 0.79 | 1277 |

TRAIN-mean baseline on TEST (predict constant 55.778 %): MAE 9.01, RMSE 12.23, bias 0.28 pp, N = 1277.

<!-- GENERATED METRICS END -->

## What the pipeline reports

The pipeline reports three per-beat numbers, computed from the predicted LV mask area over time:

- Fractional Area Change (FAC): `FAC = (A_ED - A_ES) / A_ED`. This is a 2D area ratio, not EF. It systematically underestimates volume change because area scales more slowly than volume.
- Area-power volume proxy: assumes `V proportional to A^(3/2)`. It assumes equal shrinkage along every axis, which does not hold in a real ventricle.
- Method of disks (single plane): divides the PCA long axis into 20 slabs and sums disk volumes. Clinical Simpson's method is biplane: it uses two orthogonal views (apical 4-chamber and apical 2-chamber). This project has one view, so the single-plane disk sum differs from the clinical biplane method.

Area, area-power, and single-plane disks are all research proxies for EF, not EF.

## Dataset

EchoNet-Dynamic (Ouyang et al., Nature 2020).
DOI: https://doi.org/10.71718/yqp5-y078
Local path: `EchoNet-Dynamic.zip` in the project root (not committed).
Frame size: 112 x 112 pixels. Labels: `VolumeTracings.csv` has `FileName, X1, Y1, X2, Y2, Frame`. The first row of each `Frame` is the long axis; the other rows define 20 short-axis chords. The LV mask polygon walks the `(X1, Y1)` endpoints in order then the `(X2, Y2)` endpoints in reverse.

The zip is streamed one video at a time; it is never loaded into memory or copied.

## Weights

Trained by the author on the EchoNet-Dynamic TRAIN split using `src/train.py`. Not distributed in this repository.

Training settings:
- Model: 4-level U-Net with features `(32, 64, 128, 256)`, in=1, out=1.
- Input: 112 x 112 grayscale, scaled to `[0, 1]`.
- Loss: `BCE + (1 - Dice)`.
- Batch size: 8. Train subset: 2000 TRAIN videos (not the full 7465), 4000 labeled frames.
- Augmentation: small brightness, contrast, and up-to-4-pixel shifts; no flips.
- Seed: 42.
- VAL frames: 2576 (one labeled ED + one labeled ES per VAL video). VAL Dice is reported in `results/train_summary.json`.
- Stage 1: Adam at `1e-3`, 85-minute CPU budget. Best VAL Dice 0.9192 at epoch 5. Snapshot saved to `models/unet_lv_stage1.pth` (kept local, SHA256 in DECISIONS.md).
- Stage 2 (`--resume`): fresh Adam at `3e-4`, 120-minute CPU budget, appended to `results/training_log.csv`. Best VAL Dice 0.9251 at epoch 12. `models/unet_lv.pth` is only overwritten when VAL Dice improves.

Weights live at `models/unet_lv.pth` with a sidecar `models/unet_lv.json` recording the features and target size.

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt scikit-image pandas tqdm
```

## Reproduce

With the EchoNet-Dynamic zip in the project root:

```bash
make test         # unit tests
make cache        # build data/cache from the zip (TRAIN + VAL labeled frames)
make mask-check   # 6 overlay images in results/mask_check/ (not committed; shows dataset frames)
make train        # train the U-Net; writes models/unet_lv.pth and models/unet_lv.json
make evaluate     # first TEST run on 50 videos; filters off/on; writes metrics.json
python scripts/val_sweep.py     # pick the prominence fraction f on 100 VAL videos
python scripts/test_rescore.py  # rescore TEST with the frozen f (filters off)
python scripts/worst_and_plots.py  # scatter / Bland-Altman and worst-case diagnostics
make results      # regenerate the metrics block in this README and results/RESULTS.md
```

The `results/mask_check/` directory holds overlay images on dataset frames and stays local only; it is listed in `.gitignore`. Plots of numbers (scatter, Bland-Altman, area curves in `results/worst_videos*/`) are safe to commit.

Infer on one video:

```bash
python -m src.run --video PATH_TO_VIDEO --weights models/unet_lv.pth --out outputs/run
```

The command saves an overlay video, an area plot, and `result.json`.

## Limitations

- One view: EchoNet is a single apical 4-chamber view. Clinical EF needs at least a biplane method.
- No pixel-to-centimeter calibration. All areas and volumes are in pixel units.
- Short training: 2000 of 7465 TRAIN videos; two CPU stages totaling about 205 minutes.
- FAC is not EF. The reported bias between FAC and labeled EF is expected and documented in the generated metrics block above.
- The disk estimator uses pixel units and a single plane; it does not match clinical Simpson's biplane volumes.
- The TRAIN-mean baseline predicts a constant TRAIN EF on every video; our best estimator is compared against it in the metrics block.
- The ED-peak prominence fraction `f` was chosen on VAL only (100 videos). The pairing fix was found while inspecting three TEST failures of the first run, so TEST was then scored twice (first run before the fix, rescore with f frozen) to avoid using TEST for selection.
- Failure modes observed on EchoNet TEST are listed in `results/worst_videos.json` (first-run worst 3, with before/after numbers) and `results/worst_videos_rescore.json` (rescore worst 5).

## Documents

- Theory: `docs/THEORY.md`
- Decisions: `DECISIONS.md`
- Progress: `results/PROGRESS.md`
- Results: `results/RESULTS.md` (generated)
