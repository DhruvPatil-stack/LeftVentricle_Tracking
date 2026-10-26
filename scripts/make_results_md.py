"""Generate the results Markdown block from metrics files.

Sources:
- results/metrics.json          (first TEST run, 50 videos, before the pairing fix)
- results/metrics_rescore.json  (TEST rescore with frozen f=0.3)
- results/val_sweep.json        (VAL hyperparameter sweep + baseline + calibration)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
RESCORE_PATH = RESULTS_DIR / "metrics_rescore.json"
SWEEP_PATH = RESULTS_DIR / "val_sweep.json"
README = ROOT / "README.md"
RESULTS_MD = RESULTS_DIR / "RESULTS.md"


def _fmt(entry: dict, key: str) -> str:
    raw = entry.get(key)
    if raw is None:
        return "not measured"
    return f"{raw:.2f}"


def _row(name: str, entry: dict) -> str:
    return (
        f"| {name} | {_fmt(entry, 'mae_pp')} | {_fmt(entry, 'rmse_pp')} | "
        f"{_fmt(entry, 'bias_pp')} | {_fmt(entry, 'pearson_r')} | {entry.get('n', 'not measured')} |"
    )


def _dice_line(dice: dict | None, n_frames_label: str) -> str:
    if dice is None or dice.get("mean") is None:
        return f"- TEST Dice ({n_frames_label}): not measured."
    return (
        f"- TEST Dice ({n_frames_label}, mean / median): "
        f"{dice['mean']:.3f} / {dice['median']:.3f} over {dice.get('n', 'not measured')} labeled frames."
    )


def _sweep_table(sweep: dict) -> list[str]:
    rows = sweep.get("sweep_rows", [])
    by_variant: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[row["estimator"]] = row
    chosen_f = sweep.get("chosen_f")
    chosen_reason = sweep.get("chosen_reason", "")
    lines = [
        f"VAL sweep (100 VAL videos, random_state=42, filters off). "
        f"Chosen f = {chosen_f}. Reason: {chosen_reason}.",
        "",
        "| Variant | area_ratio MAE | area_power MAE | method_of_disks MAE |",
        "|---|---|---|---|",
    ]
    order = ["old", "prom_0.1", "prom_0.2", "prom_0.3", "prom_0.4", "prom_0.5"]
    for variant in order:
        if variant not in by_variant:
            continue
        label = "old (no prominence)" if variant == "old" else f"f = {by_variant[variant]['area_ratio']['f']}"
        ar = by_variant[variant].get("area_ratio", {})
        ap = by_variant[variant].get("area_power", {})
        md = by_variant[variant].get("method_of_disks", {})
        lines.append(
            f"| {label} | {_fmt(ar, 'mae_pp')} | {_fmt(ap, 'mae_pp')} | {_fmt(md, 'mae_pp')} |"
        )
    baseline = sweep.get("train_mean_baseline_on_val", {})
    lines.append(
        f"\nTRAIN-mean baseline on VAL (predict constant {baseline.get('train_mean_ef_pct', 'n/a')} %): "
        f"MAE {_fmt(baseline, 'mae_pp')}, RMSE {_fmt(baseline, 'rmse_pp')}, "
        f"bias {_fmt(baseline, 'bias_pp')} pp, N = {baseline.get('n', 'n/a')}."
    )
    cal = sweep.get("calibration_fit_on_val", {})
    if cal:
        lines.append("\nVAL-only linear calibration EF = a·x + b:")
        for name in ("area_ratio", "area_power", "method_of_disks"):
            entry = cal.get(name, {})
            if entry:
                lines.append(
                    f"- {name}: a = {entry['a']:.4f}, b = {entry['b']:.4f} (fit on {entry['n']} VAL videos)."
                )
    return lines


def _first_run_block(metrics: dict) -> list[str]:
    lines = [
        f"## First TEST run (before the pairing fix): {metrics.get('n_videos', 'not measured')} videos",
        "",
        _dice_line(metrics.get("test_dice"), "first run"),
        f"- Pipeline failures (filters off): {metrics.get('n_failed_base', 'not measured')}",
        f"- Pipeline failures (filters on): {metrics.get('n_failed_denoised', 'not measured')}",
        "",
        "### First run, filters off",
        "",
        "| Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Pearson r | N |",
        "|---|---|---|---|---|---|",
    ]
    est_off = metrics.get("estimators_base", {})
    lines.append(_row("Area ratio (FAC)", est_off.get("area_ratio", {})))
    lines.append(_row("Area^(3/2) proxy", est_off.get("area_power_volume_proxy", {})))
    lines.append(_row("Method of disks (single plane)", est_off.get("method_of_disks", {})))
    lines.extend([
        "",
        "### First run, filters on",
        "",
        "| Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Pearson r | N |",
        "|---|---|---|---|---|---|",
    ])
    est_on = metrics.get("estimators_denoised", {})
    lines.append(_row("Area ratio (FAC)", est_on.get("area_ratio", {})))
    lines.append(_row("Area^(3/2) proxy", est_on.get("area_power_volume_proxy", {})))
    lines.append(_row("Method of disks (single plane)", est_on.get("method_of_disks", {})))
    lines.append("")
    return lines


def _rescore_block(rescore: dict) -> list[str]:
    scored = rescore.get("n_videos_scored", "not measured")
    avail = rescore.get("n_test_total_available", "not measured")
    frozen_f = rescore.get("frozen_f", "not measured")
    mode = rescore.get("selection_mode", "not measured")
    lines = [
        f"## TEST rescore with frozen f = {frozen_f}: {scored} of {avail} videos",
        "",
        f"- Selection: {mode}. Filters: {rescore.get('filters', 'not measured')}.",
        f"- Projected full run: {rescore.get('projected_full_minutes', 'not measured')} min "
        f"(timed from 10 videos; {rescore.get('seconds_per_video', 'not measured')} s/video).",
        _dice_line(rescore.get("test_dice"), "rescore"),
        f"- Pipeline failures: {rescore.get('n_failed', 'not measured')}",
        "",
        "### Rescore, raw",
        "",
        "| Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Pearson r | N |",
        "|---|---|---|---|---|---|",
    ]
    est_raw = rescore.get("estimators_raw", {})
    lines.append(_row("Area ratio (FAC)", est_raw.get("area_ratio", {})))
    lines.append(_row("Area^(3/2) proxy", est_raw.get("area_power_volume_proxy", {})))
    lines.append(_row("Method of disks (single plane)", est_raw.get("method_of_disks", {})))
    lines.extend([
        "",
        "### Rescore, calibrated with VAL-only linear fit",
        "",
        "| Estimator | MAE (pp) | RMSE (pp) | Bias (pp) | Pearson r | N |",
        "|---|---|---|---|---|---|",
    ])
    est_cal = rescore.get("estimators_calibrated_val_fit", {})
    lines.append(_row("Area ratio (FAC)", est_cal.get("area_ratio", {})))
    lines.append(_row("Area^(3/2) proxy", est_cal.get("area_power_volume_proxy", {})))
    lines.append(_row("Method of disks (single plane)", est_cal.get("method_of_disks", {})))
    baseline = rescore.get("train_mean_baseline_on_test", {})
    lines.extend([
        "",
        f"TRAIN-mean baseline on TEST (predict constant {baseline.get('train_mean_ef_pct', 'n/a')} %): "
        f"MAE {_fmt(baseline, 'mae_pp')}, RMSE {_fmt(baseline, 'rmse_pp')}, "
        f"bias {_fmt(baseline, 'bias_pp')} pp, N = {baseline.get('n', 'n/a')}.",
    ])
    lines.append("")
    return lines


def _build_block() -> list[str]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    rescore = json.loads(RESCORE_PATH.read_text(encoding="utf-8")) if RESCORE_PATH.exists() else None
    sweep = json.loads(SWEEP_PATH.read_text(encoding="utf-8")) if SWEEP_PATH.exists() else None

    lines: list[str] = [
        "All values in this file come from `results/metrics.json` (first TEST run), "
        "`results/metrics_rescore.json` (rescore with the pairing fix), and "
        "`results/val_sweep.json` (VAL sweep + baseline + calibration).",
        "",
        "The pairing fix adds a prominence filter in `find_ed_es_pairs` (fraction f of the full "
        "area swing). It was found while inspecting three TEST failures of the first run, so TEST "
        "was scored twice: once before the fix (50 videos) and once with the frozen f (full rescore).",
        "",
    ]
    if sweep is not None:
        lines.append("## VAL sweep and chosen f")
        lines.append("")
        lines.extend(_sweep_table(sweep))
        lines.append("")
    lines.extend(_first_run_block(metrics))
    if rescore is not None:
        lines.extend(_rescore_block(rescore))
    return lines


def main() -> None:
    block_lines = _build_block()
    results_lines = ["# Results", ""] + block_lines
    RESULTS_MD.write_text("\n".join(results_lines), encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    marker_start = "<!-- GENERATED METRICS START -->"
    marker_end = "<!-- GENERATED METRICS END -->"
    start = readme_text.index(marker_start)
    end = readme_text.index(marker_end) + len(marker_end)
    new_block = marker_start + "\n" + "\n".join(block_lines) + "\n" + marker_end
    README.write_text(readme_text[:start] + new_block + readme_text[end:], encoding="utf-8")


if __name__ == "__main__":
    main()
