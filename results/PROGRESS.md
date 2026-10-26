# Progress

- [x] Phase 0: Oriented on Windows 11, Python 3.14 at `.venv/Scripts/python.exe`, torch 2.14.1+cpu, no CUDA/MPS. Weights absent from `models/` at start.
- [x] Phase 1: Built labeled frame cache from `EchoNet-Dynamic.zip` (TRAIN + VAL). Wrote 6 overlay images to `results/mask_check/` (kept local; show dataset frames).
- [x] Phase 2: Trained U-Net in two stages. Stage 1 best VAL Dice 0.9192 at epoch 5 (85-min CPU budget, Adam 1e-3). Stage 2 (`--resume`, fresh Adam 3e-4, 120-min budget, append log) best VAL Dice 0.9251 at epoch 12. Only overwrote `models/unet_lv.pth` on improvement.
- [x] Phase 3: Froze weights. `models/unet_lv.pth` SHA256 `0ef521470c5393318964dcc8a34d7acedf5310756c3ba45438cda382f5ceb02a`. Snapshot `models/unet_lv_stage1.pth` SHA256 `6e22420670e5c73a14473063c0b42cd32c2603c9990b9250408b73fb9cb7b3e2`. Ran 3 VAL overlay sanity checks: val1 found 4 beats, val2 found 7 beats, val3 found 3 beats.
- [x] Phase 4: Hardened EF estimators and beat tests (midpoint-rule disk sampling, failed-mask ES exclusion). Tests pass.
- [x] Phase 5: First TEST run on 50 videos. TEST Dice mean 0.9299 / median 0.9341. 0 failures with filters off; 0 with filters on. Metrics in `results/metrics.json`; scatter and Bland-Altman PNGs saved; `results/worst_videos.json` populated.
- [x] Phase 6: Ran `scripts/make_results_md.py` to fill README block and write `results/RESULTS.md`. Ran `scripts/check_claims.py` — passed.
- [x] Phase 7: Pushed Phase 5+6 artifacts to `feat/validation-and-fixes`.
- [x] Pairing fix — Phase 1: Added `min_prominence_fraction` parameter to `find_ed_es_pairs` and a `rejected_ed_peaks` quality flag. Three new tests cover plateau-bump rejection, trough-bump non-degeneracy, and old behavior under `min_prominence_fraction=0`. `make test` passes locally.
- [x] Pairing fix — Phase 2: VAL sweep on 100 random_state=42 videos (filters off). Chosen f = 0.3 (lowest area_power MAE 6.30 pp; ties within 0.2 pp broken by lower f). TRAIN-mean baseline on VAL: MAE 9.13 pp (f=0.3 beats it). Linear calibration EF = a·x + b fit on VAL only. Frozen in `DECISIONS.md`. Weights SHA256 unchanged.
- [x] Pairing fix — Phase 3: Timed 10 TEST videos (7.09 s/video → 150.9 min projected). User raised the budget cap, so scored the full 1277-video TEST split with frozen f = 0.3, filters off. Area_power raw MAE 6.14 pp, calibrated 5.54 pp; FAC calibrated 5.82 pp; disks calibrated 5.54 pp. TRAIN-mean baseline on TEST: MAE 9.01 pp. 0 pipeline failures. Replotted the three old-worst videos under new pairs (errors dropped from 39.19/36.39/35.99 pp to 13.61/7.06/1.59 pp; all three attributed to spurious ED peaks on the old settings). Listed the five new worst from the rescore (one unknown, two single-pair fallbacks, one unstable contour, one single-beat).
- [x] Pairing fix — Phase 4: Un-tracked `results/mask_check/` (overlays on dataset frames) and added it to `.gitignore`. Kept numeric plots (scatter, Bland-Altman, area curves) tracked. Regenerated the metrics block in `README.md` and `results/RESULTS.md` to cover both the first run and the rescore. `scripts/check_claims.py` passes. Weights SHA256 re-checked: unchanged.
- [ ] Pairing fix — Phase 5: small commits per step and `git push` on `feat/validation-and-fixes` only.

## Notes

- Operating under autonomous rules: no questions, safest choice, 1 monitor + 1 wakeup, log checked at most every 28 min, local commits per phase, push once per phase gate, never delete checkpoints, never retrain weights after Phase 2.
- `.gitignore` was expanded in Phase 2 to exclude `EchoNet-Dynamic.zip`, `*.zip`, and `data/cache/`, and in the pairing-fix Phase 4 to exclude `results/mask_check/`.
- No commands blocked this run; `results/BLOCKED.md` is not created.
