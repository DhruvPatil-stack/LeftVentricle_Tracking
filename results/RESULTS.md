# Results

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
