# Automated Left Ventricle Tracking

This repository is a research codebase for left-ventricle tracking in apical four-chamber ultrasound videos.

It makes a cavity mask for each frame. It measures the mask area. It detects end diastole and end systole. It reports research estimates of ejection fraction.

The area estimate is a 2D proxy. It is not a validated clinical measurement. The disk estimate needs validated masks and image scale before clinical use.

The repository has no included clinical video, model weight, or clinical result. Synthetic phantom results validate code behavior and estimator bias only. They do not validate clinical accuracy.

For details, see `README.md` and `docs/THEORY.md`.
